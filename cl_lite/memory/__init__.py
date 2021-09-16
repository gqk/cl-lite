# -*- coding: utf-8 -*-

from typing import List

from .base import Memory
from .herding import Herding


__all__ = ["names", "create", "Memory"]


__factory = {
    "herding": Herding,
}


def names() -> List[str]:
    return sorted(__factory.keys())


def create(
    name: str,
    classes: List[int],
    size: int,
    dynamic: bool,
    *args,
    **kwargs,
) -> Memory:
    if name not in __factory:
        raise KeyError(f"Unknown Memory Algorithm: {name}")

    return __factory[name](classes, size, dynamic, *args, **kwargs)
