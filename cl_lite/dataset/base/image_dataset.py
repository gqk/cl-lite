# -*- coding: utf-8 -*-

from typing import Callable, Sequence, Tuple, Union, Optional

import torch
import torch.utils.data as td
import torchvision as tv

from .subset import Subset


class Dataset:
    dataset_cls = None

    def __init__(self, root: str, **kwargs):
        self.hparams = dict(**kwargs, root=root)
        self._train = self._test = self._num_classes = None

    @property
    def train(self) -> tv.datasets.VisionDataset:
        if self._train is None:
            self._train = self.prepare_data(True)
        return self._train

    @property
    def test(self) -> tv.datasets.VisionDataset:
        if self._test is None:
            self._num_classes = None
            self._test = self.prepare_data()
        return self._test

    @property
    def transform(self) -> Optional[Callable]:
        return None

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            self._num_classes = self.get_targets(self.test).unique().size(0)
        return self._num_classes

    @property
    def dims(self) -> Optional[torch.Size]:
        return None

    def prepare_data(self, train: bool = False) -> tv.datasets.VisionDataset:
        if self.dataset_cls is not None:
            root = self.hparams["root"]
            return self.dataset_cls(root, train=train, download=True)
        raise NotImplementedError

    @staticmethod
    def get_targets(subset: tv.datasets.VisionDataset) -> torch.Tensor:
        indices = None
        if isinstance(subset, Subset):
            subset, indices = subset.dataset, subset.indices

        assert hasattr(subset, "targets")

        targets = torch.tensor(subset.targets)

        if indices is not None:
            targets = targets[indices]

        return targets


class SimpleDataset(Dataset):
    def __init__(
        self,
        train: tv.datasets.VisionDataset,
        test: tv.datasets.VisionDataset,
        transform: Callable = None,
        dims: torch.Size = None,
        **kwargs
    ):
        super().__init__(None, **kwargs)

        self._train = train
        self._test = test
        self._transform = transform
        self._dims = dims

    @property
    def transform(self) -> Optional[Callable]:
        return self._transform

    @property
    def dims(self):
        return self._dims

    def prepare_data(self, train=False):
        pass
