# -*- coding: utf-8 -*-

from typing import Any, Dict, List

import torch
import torch.nn as nn

__all__ = ["freeze", "unfreeze", "no_grad_of"]


def freeze(module: nn.Module, mock_training: bool = False):
    """Freeze a torch Module

    1) save all parameters's current requires_grad state,
    2) disable requires_grad,
    3) turn on mock_training
    4) switch to evaluation mode.
    """

    state = {}
    for name, param in module.named_parameters():
        state[name] = param.requires_grad
        param.requires_grad = False
        param.grad = None

    if mock_training and hasattr(module, "mock_training"):
        module.mock_training = True

    module.eval()
    return state


def unfreeze(module: nn.Module, state: Dict[str, bool] = {}):
    """Unfreeze a torch Module

    1) restore all parameters's requires_grad state,
    2) switch to training mode.
    3) turn off mock_training

    """

    default = None if state else True
    for name, param in module.named_parameters():
        requires_grad = state.get(name, default)
        if requires_grad is not None:
            param.requires_grad = requires_grad

    module.train()

    if hasattr(module, "mock_training"):
        module.mock_training = False


class no_grad_of(torch.no_grad):
    """Contex manager that temporarily freeze some modules during training"""

    def __init__(self, *modules: List[nn.Module], mock_training: bool = False):
        super().__init__()
        self.modules = modules
        self.states = None
        self.mock_training = mock_training

    def __enter__(self):
        self.states = [
            freeze(module, mock_training=mock_training)
            for module in self.modules
        ]

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for module, state in zip(self.modules, self.states):
            unfreeze(module, state)
