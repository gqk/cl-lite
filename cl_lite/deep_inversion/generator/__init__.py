# -*- coding: utf-8 -*-

from typing import List
from functools import partial

import torch.nn as nn

from .cifar import CIFAR_GEN
from .tiny_imagenet import TINYIMNET_GEN
from .imagenet import IMNET_GEN


__all__ = ["names", "create"]


__factory = {
    "cifar100": CIFAR_GEN,
    "tiny-imagenet200": TINYIMNET_GEN,
    "imagenet100": IMNET_GEN,
    "imagenet1000": IMNET_GEN,
}


def names() -> List[str]:
    return sorted(__factory.keys())


def create(name: str) -> nn.Module:
    if name not in __factory:
        raise KeyError(f"Unknown dataset: {name}")

    return __factory[name]()
