# -*- coding: utf-8 -*-

from typing import Callable, Dict, Union, Sequence

import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(
        self,
        fn: Callable,
        adapter: Union[Sequence[str], Callable] = None,
    ):
        super().__init__()

        self.fn = fn
        self.adapter = adapter

    def forward(
        self,
        module: nn.Module,
        kwargs: dict = {},
        loss_dict: dict = {},
    ):
        if callable(self.adapter):
            args = self.adapter(module, kwargs, loss_dict)
        elif isinstance(self.adapter, (list, tuple)):
            args = [kwargs[key] for key in self.adapter]
        else:
            args = (module, kwargs, loss_dict)

        return self.fn(*args)


class LossMixin:
    device: torch.device
    _losses = nn.ModuleDict()
    _loss_factors: Dict[str, float] = dict()

    def register_loss(
        self,
        name: str,
        fn: Callable,
        adapter: Union[Sequence[str], Callable] = None,
        factor: float = 1.0,
    ):
        if not isinstance(fn, Criterion):
            fn = Criterion(fn, adapter)
        self._losses[name] = fn
        self._loss_factors[name] = factor

        print(f"Register Loss: {name} × {factor}")

    def unregister_loss(self, name):
        if name not in self._losses:
            return None, None

        fn = self._losses.pop(name)
        factor = self._loss_factors.pop(name, None)
        print(f"Unregister Loss: {name}")
        return fn, factor

    def set_loss_factor(self, name: str, factor: float, quiet: bool = False):
        if not quiet:
            print(f"Set the Factor of Loss {name} to {factor}")
        self._loss_factors[name] = factor

    def get_loss_factor(self, name: str):
        return self._loss_factors.get(name, 1.0)

    def move_losses_to_device(self, device=None):
        device = self.device
        for c in self._losses.values():
            c.to(device)

    def compute_loss_item(self, name, loss_dict={}, **kwargs):
        assert name in self._losses
        fn, factor = self._losses[name], self.get_loss_factor(name)
        if factor > 0:
            return factor * fn(self, kwargs, loss_dict)
        return 0.0

    def compute_loss(self, **kwargs):
        loss_dict, loss = {}, 0.0
        for name in self._losses:
            loss_item = self.compute_loss_item(name, loss_dict, **kwargs)
            loss_dict[name] = loss_item
            loss = loss + loss_item
        loss_dict["total"] = loss
        return loss, loss_dict
