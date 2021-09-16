# -*- coding: utf-8 -*-

import os
from typing import List

from .base import (
    MetaItem,
    ImageMetaDataset,
    Subset,
    Dataset,
    SimpleDataset,
    split_train_val,
    split_task,
)

from .cifar import CIFAR10, CIFAR100
from .tiny_imagenet import TinyImageNet200
from .imagenet import ImageNet100, ImageNet1000


__all__ = [
    "names",
    "create",
    "Subset",
    "Dataset",
    "SimpleDataset",
    "split_train_val",
    "split_task",
]


__factory = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "tiny-imagenet200": TinyImageNet200,
    "imagenet100": ImageNet100,
    "imagenet1000": ImageNet1000,
}


def names() -> List[str]:
    return sorted(__factory.keys())


def create(name: str, root: str, *args, **kwargs) -> Dataset:
    if name not in __factory:
        raise KeyError(f"Unknown dataset: {name}")

    root = os.path.join(root, name)

    return __factory[name](root, *args, **kwargs)
