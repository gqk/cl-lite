# -*- coding: utf-8 -*-

"""
DeepInversion

credits:
    https://github.com/NVlabs/DeepInversion
    https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL
"""

import math
import random
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cl_lite.nn.module import freeze

from .feature_hook import DeepInversionFeatureHook


@torch.no_grad()
def output_shape(model, input_dims) -> torch.Size:
    state, device = model.training, next(model.parameters()).device
    model.eval()
    input = torch.randn((1,) + input_dims).to(device)
    shape = model(input).shape
    model.train(state)
    return shape[1:]


class DeepInversion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_dims: Tuple[int, int, int] = (3, 32, 32),
        batch_size: int = 256,
        max_iters: int = 2000,
        lr: float = 0.05,
        alpha_tv: float = 2.5e-5,
        alpha_l2: float = 3e-8,
        alpha_rf: float = 5.0,
    ):
        super().__init__()

        self.model = model
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.lr = lr
        self.alpha_tv = alpha_tv
        self.alpha_l2 = alpha_l2
        self.alpha_rf = alpha_rf

        self.feature_hooks = []
        self.device = next(model.parameters()).device
        self.num_classes = output_shape(model, input_dims)[0]

        self.criterion_ce = nn.CrossEntropyLoss()

        self.setup()

    def register_feature_hooks(self):
        # Remove old before register
        for hook in self.feature_hooks:
            hook.remove()

        ## Create hooks for feature statistics catching
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(DeepInversionFeatureHook(module))

    def criterion_tv(self, input):
        # apply total variation regularization
        diffs = [
            input[:, :, :, :-1] - input[:, :, :, 1:],
            input[:, :, :-1, :] - input[:, :, 1:, :],
            input[:, :, 1:, :-1] - input[:, :, :-1, 1:],
            input[:, :, :-1, :-1] - input[:, :, 1:, 1:],
        ]

        return sum([diff.norm() for diff in diffs])

    def criterion_l2(self, input):
        return input.norm()

    def criterion_rf(self):
        return sum([hook.r_feature_di for hook in self.feature_hooks])

    def augment(self, input):
        lim_0, lim_1 = 2, 2
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        input_jit = torch.roll(input, shifts=(off1, off2), dims=(2, 3))
        return input_jit

    def setup(self):
        freeze(self.model)
        self.register_feature_hooks()

    def train_step(self, input: torch.Tensor, target: torch.LongTensor):
        output = self.model(input)
        loss_ce = self.criterion_ce(output, target)
        loss_tv = self.alpha_tv * self.criterion_tv(input)
        loss_l2 = self.alpha_l2 * self.criterion_l2(input)
        loss_rf = self.alpha_rf * self.criterion_rf()

        loss = loss_ce + loss_tv + loss_l2 + loss_rf
        loss_dict = {
            "loss/ce": loss_ce,
            "loss/tv": loss_tv,
            "loss/l2": loss_l2,
            "loss/rf": loss_rf,
            "loss/total": loss,
        }

        return loss, loss_dict

    def forward(self, target: torch.LongTensor = None):
        if target is None:
            target = torch.randint(self.num_classes, (self.batch_size,))

        assert target.shape[0] == self.batch_size
        assert target.lt(self.num_classes).all()
        assert target.gt(-1).all()

        shape = (target.shape[0],) + self.input_dims
        input = torch.randn(shape).to(self.device).requires_grad_(True)
        target = target.to(self.device)

        optimizer = optim.Adam([input], lr=self.lr)

        best_loss, best_input = float("inf"), input.detach()
        for current_iter in range(self.max_iters):
            optimizer.zero_grad()

            loss, loss_dict = self.train_step(self.augment(input), target)
            if loss.item() < best_loss:
                best_loss, best_input = loss.item(), input.detach()

            loss.backward()
            optimizer.step()

            msg = [f"| {k}: {v:.4f} " for k, v in loss_dict.items()]
            print(f"DeepInversion It {current_iter:03d}", *msg)

        return best_input, target

    @torch.no_grad()
    def sample(self, batch_size: int = None):
        batch_size = self.batch_size if batch_size is None else batch_size

        target = torch.randint(self.num_classes, (batch_size,))

        torch.set_grad_enabled(True)
        input, target = self.forward(target)
        torch.set_grad_enabled(False)

        return input.detach(), target.detach()
