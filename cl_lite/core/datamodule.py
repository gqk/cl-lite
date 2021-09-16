# -*- coding: utf-8 -*-


import os
from copy import deepcopy
from typing import Union, Sequence, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torchvision as tv
import torch.utils.data as td
import pytorch_lightning as pl

from .. import dataset as ds, memory as mm

DEFAULT_ROOT = os.path.join(os.getcwd(), "data")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        dataset: Union[str, Sequence[str]] = "cifar100",
        batch_size: int = 128,
        num_workers: int = 4,
        val_splits: Union[int, float] = 0,
        val_seed: int = 42,
    ):
        super().__init__([], [], [])

        self.root = root
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_splits = val_splits
        self.val_seed = val_seed

    @property
    def num_classes(self):
        return self._source.num_classes

    @property
    def dims(self):
        return self._source.dims

    def load_checkpoint(self, checkpoint):
        pass

    def save_checkpoint(self, checkpoint):
        pass

    def setup(self, stage=None):
        dataset = self.dataset  # TODO multi datasets
        if not isinstance(self.dataset, str):
            dataset = self.dataset[0]

        self._source = ds.create(dataset, root=self.root)
        self._source_transform = self._source.transform
        self._test_set = self._source.test
        self._train_set, self._val_set = self._source.train, None
        if self.val_splits > 0:
            args = [self._source, self.val_splits, self.val_seed]
            self._train_set, self._val_set = ds.split_train_val(*args)

    def sampler(self):
        return None

    def transform_target(self, target: torch.Tensor, *args, **kwargs):
        return target

    def train_dataloader(
        self,
        dataset=None,
        transforms=None,
        mock_test=False,
        **kwargs,
    ):
        dataset = deepcopy(dataset or self._train_set)

        if mock_test:
            return self.test_dataloader(dataset=dataset, **kwargs)

        transforms = transforms or self.train_transforms
        if self._source_transform is not None:
            transforms = [self._source_transform] + transforms
        transform = tv.transforms.Compose(transforms)

        if isinstance(dataset, ds.Subset):
            dataset.dataset.transform = transform
        else:
            dataset.transform = transform

        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("num_workers", self.num_workers)
        kwargs.setdefault("pin_memory", True)
        kwargs.setdefault("drop_last", True)

        if "shuffle" not in kwargs:
            sampler = kwargs.get("sampler", self.sampler())
            kwargs["shuffle"] = sampler is None
            kwargs["sampler"] = sampler

        return td.DataLoader(dataset, **kwargs)

    def val_dataloader(self, dataset=None, **kwargs):
        if self.val_splits == 0:
            return self.test_dataloader(**kwargs)

        transforms = self.val_transforms
        if self._source_transform is not None:
            transforms = [self._source_transform] + transforms
        transform = tv.transforms.Compose(transforms)

        dataset = deepcopy(dataset or self._val_set)
        if isinstance(dataset, ds.Subset):
            dataset.dataset.transform = transform
        else:
            dataset.transform = transform

        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("num_workers", self.num_workers)
        kwargs.setdefault("pin_memory", True)

        return td.DataLoader(dataset, **kwargs)

    def test_dataloader(self, dataset=None, **kwargs):
        transforms = self.test_transforms
        if self._source_transform is not None:
            transforms = [self._source_transform] + transforms
        transform = tv.transforms.Compose(transforms)

        dataset = deepcopy(dataset or self._test_set)
        if isinstance(dataset, ds.Subset):
            dataset.dataset.transform = transform
        else:
            dataset.transform = transform

        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("num_workers", self.num_workers)
        kwargs.setdefault("pin_memory", True)

        return td.DataLoader(dataset, **kwargs)


class SplitedDataModule(DataModule):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        dataset: str = "cifar100",
        batch_size: int = 128,
        num_workers: int = 4,
        val_splits: Union[int, float] = 0,
        val_seed: int = 42,
        num_tasks: int = 5,
        current_task: int = 0,
        class_order: Sequence[int] = [],
        init_task_splits: Union[int, float] = 0,
        task_seed: int = 42,
        test_mode: str = "seen",
        memory_algo: str = "herding",
        memory_size: int = 0,
        dynamic_memory: bool = False,
    ):
        self.num_tasks = num_tasks
        self.current_task = current_task
        self.class_order = class_order
        self.init_task_splits = init_task_splits
        self.task_seed = task_seed

        assert test_mode in ["current", "seen"]
        self.test_mode = test_mode

        self.memory_algo = memory_algo
        self.memory_size = memory_size
        self.dynamic_memory = dynamic_memory

        super().__init__(
            root,
            dataset,
            batch_size,
            num_workers,
            val_splits,
            val_seed,
        )

    @property
    def num_classes(self):
        return self._source[self.current_task].num_classes

    @property
    def dims(self):
        return self._source[self.current_task].dims

    def __getitem__(self, index: int):
        return self._source[index]

    def load_checkpoint(self, checkpoint):
        self.memory = checkpoint.get("datamodule_memory", self.memory)

    def save_checkpoint(self, checkpoint):
        checkpoint["datamodule_memory"] = self.memory

    def setup(self, stage=None):
        self._source, self.class_order = ds.split_task(
            ds.create(self.dataset, root=self.root),
            self.num_tasks,
            self.init_task_splits,
            self.class_order,
            self.task_seed,
        )

        indices = {c: i for i, c in enumerate(self.class_order)}
        indices = [indices[c] for c, _ in enumerate(self.class_order)]
        reindexed = indices != self.class_order
        self._class_indices = torch.tensor(indices) if reindexed else None

        self.switch_task(self.current_task, force=True)

        self.setup_memory()

    def setup_memory(self):
        self.memory = None
        if self.memory_size > 0:
            self.memory = mm.create(
                self.memory_algo,
                self.class_order,
                self.memory_size,
                self.dynamic_memory,
            )

    def update_memory(self, model: nn.Module, **kwargs):
        if self.memory is None:
            return []

        dataloader = self.train_dataloader(with_memory=False, mock_test=True)
        return self.memory.update(model, dataloader, self.num_classes, **kwargs)

    def switch_task(self, task_id: int, force: bool = False):
        assert -1 < task_id < self.num_tasks

        if not force and (task_id == self.current_task):
            return

        self.current_task = task_id
        self._source_transform = self[task_id].transform
        self._train_set, self._val_set = self[task_id].train, None
        if self.val_splits > 0:
            args = [self[task_id], self.val_splits, self.val_seed]
            self._train_set, self._val_set = ds.split_train_val(*args)

        self._test_set = self[task_id].test
        if self.test_mode == "seen":
            for idx in range(task_id - 1, -1, -1):
                self._test_set = self[idx].test + self._test_set

    def memory_set(self, memory: mm.Memory = None, current_task=None):
        memory = memory or self.memory
        if current_task is None:
            current_task = self.current_task

        assert current_task > 0

        dataset, acc_num_classes = None, 0
        for task_id in range(0, current_task):
            subset, num_classes = self[task_id].train, self[task_id].num_classes
            indices = memory[acc_num_classes:][:num_classes]

            assert len(indices) == num_classes

            missings = [
                self.class_order[c + acc_num_classes]
                for c, idxs in enumerate(indices)
                if not idxs
            ]
            if missings:
                warn(f"Missing exemplars of classes: {missings}", UserWarning)

            if len(missings) == num_classes:
                pass
            elif dataset is None:
                dataset = ds.Subset(subset, sum(indices, []))
            else:
                dataset = dataset + ds.Subset(subset, sum(indices, []))

            acc_num_classes += num_classes

        return dataset

    def transform_target(self, target: torch.Tensor, offset: int = 0):
        if self._class_indices is not None:
            target = self._class_indices.to(target.device)[target]

        if offset > 0:
            target = target - offset

        return target

    def train_dataloader(self, dataset=None, with_memory=True, **kwargs):
        if self.memory is None or not with_memory or self.current_task < 1:
            return super().train_dataloader(dataset=dataset, **kwargs)

        dataset = self.memory_set() + (dataset or self._train_set)
        return super().train_dataloader(dataset=dataset, **kwargs)

    def memory_dataloader(self, memory=None, current_task=None, **kwargs):
        kwargs["dataset"] = self.memory_set(memory, current_task)
        kwargs["with_memory"] = False
        return self.train_dataloader(**kwargs)


class MultiTaskDataModule(DataModule):
    def switch_task(self, task_id):
        pass
