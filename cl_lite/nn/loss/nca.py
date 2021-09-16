# -*- coding: utf-8 -*-

"""
credits:
    https://github.com/arthurdouillard/incremental_learning.pytorch
"""

from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCALoss(nn.Module):
    def __init__(
        self,
        margin: Union[float, torch.nn.Parameter] = 0.0,
        scale: Union[float, torch.nn.Parameter] = 1.0,
        class_weights: torch.Tensor = None,
        hinge_proxynca: bool = False,
    ):
        super().__init__()

        if isinstance(margin, torch.nn.Parameter):
            assert margin.view(-1).shape[0] == 1

        if isinstance(scale, torch.nn.Parameter):
            assert scale.view(-1).shape[0] == 1

        self.margin = margin
        self.scale = scale
        self.class_weights = class_weights
        self.hinge_proxynca = hinge_proxynca

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # TODO margin vs self.margin
        margin = torch.zeros_like(input)
        margin[torch.arange(margin.shape[0]), target] = self.margin

        output = self.scale * (input - self.margin)
        output = output - output.max(1)[0].view(-1, 1)

        numerator = output[torch.arange(output.shape[0]), target]
        disable_pos = torch.zeros_like(output)
        disable_pos[torch.arange(disable_pos.shape[0]), target] = numerator
        denominator = output - disable_pos

        loss = denominator.exp().sum(-1).log() - numerator
        if self.class_weights is not None:
            loss = self.class_weights[target] * loss

        if self.hinge_proxynca:
            loss = loss.relu()

        loss = loss.mean()

        return loss
