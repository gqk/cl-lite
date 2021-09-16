# -*- coding: utf-8 -*-

"""
credits:
    https://github.com/arthurdouillard/incremental_learning.pytorch
"""

from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def collapse_dim(input: torch.Tensor, dim: int):
    return input.sum(dim=dim).flatten(-2)


def collapse_channel(input):
    return collapse_dim(input, -3)


def collapse_height(input: torch.Tensor):
    return collapse_dim(input, -2)


def collapse_width(input: torch.Tensor):
    return collapse_dim(input, -1)


def collapse_gap(input: torch.Tensor):
    return input.mean(dim=(-2, -1))


def collapse_spatial(input: torch.Tensor):
    return torch.cat([collapse_height(input), collapse_width(input)], dim=-1)


class PooledOutputDistillationLoss(nn.Module):
    def __init__(self, collapse: str = "spatial", normalize: bool = True):
        assert collapse in ["channel", "width", "height", "gap", "spatial"]

        super().__init__()

        self.collapse = collapse
        self.normalize = normalize

        if collapse == "channel":
            self.collapse_fn = collapse_channel
        elif collapse == "height":
            self.collapse_fn = collapse_height
        elif collapse == "weight":
            self.collapse_fn = collapse_width
        elif collapse == "gap":
            self.collapse_fn = collapse_gap
        elif collapse == "spatial":
            self.collapse_fn = collapse_spatial

    def forward_single(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        assert input.shape == target.shape and input.dim() >= 3

        input = self.collapse_fn(input.pow(2))
        target = self.collapse_fn(target.pow(2))

        if self.normalize:
            input = F.normalize(input, dim=-1)
            target = F.normalize(target, dim=-1)

        return (input - target).norm(dim=-1).mean()

    def forward(
        self,
        input: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            return self.forward_single(input, target)

        outputs = [self.forward_single(i, t) for i, t in zip(input, target)]
        return torch.stack(outputs).mean()
