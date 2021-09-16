# -*- coding: utf-8 -*-

from typing import Sequence, Tuple, List, Union

import torch

from .subset import Subset
from .image_dataset import Dataset, SimpleDataset


def split_train_val(
    dataset: Dataset,
    val_splits: Union[int, float] = 0.0,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    generator = torch.Generator().manual_seed(self.seed)
    ys = dataset.get_targets(dataset.train)
    train_indices, val_indices = [], []
    for y in ys.unique():
        indices = (ys == y).nonzero(as_tuple=True)[0]
        perm = torch.randperm(indices.size(0), generator=generator)
        train_splits = indices.size(0) - int(val_splits)
        if val_splits < 1:
            train_splits -= int(val_splits * train_splits)
        train_indices.append(indices[:train_splits])
        val_indices.append(indices[train_splits:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_task(
    dataset: Dataset,
    num_tasks: int,
    initial_splits: Union[int, float] = 0.0,
    class_order: Sequence[int] = [],
    seed: int = 42,
) -> Tuple[List[SimpleDataset], Sequence[int]]:
    assert num_tasks > 1

    num_classes = dataset.num_classes
    if not class_order:
        generator = torch.Generator().manual_seed(seed)
        class_order = torch.randperm(num_classes, generator=generator).tolist()

    assert len(class_order) == num_classes

    if initial_splits >= 1:
        assert (num_classes - initial_splits) % (num_tasks - 1) == 0
    else:
        if initial_splits > 0:
            initial_splits = int(num_classes * initial_splits)
        else:
            initial_splits = num_classes // num_tasks
        initial_splits += (num_classes - initial_splits) % (num_tasks - 1)

    inc_splits = (num_classes - initial_splits) // (num_tasks - 1)
    task_classes = [class_order[:initial_splits]]
    task_classes += [
        class_order[i : i + inc_splits]
        for i in range(initial_splits, num_classes, inc_splits)
    ]

    subsets = [[], []]
    for i, parent in enumerate([dataset.train, dataset.test]):
        ys = dataset.get_targets(parent)
        for classes in task_classes:
            indices = [(ys == y).nonzero(as_tuple=True)[0] for y in classes]
            indices = torch.cat(indices)
            subsets[i].append(Subset(parent, indices))

    subsets = [
        SimpleDataset(train, test, dataset.transform, dataset.dims)
        for train, test in zip(*subsets)
    ]

    return subsets, class_order
