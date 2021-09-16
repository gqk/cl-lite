# -*- coding: utf-8 -*-

"""
DeepInversion

credits:
    https://github.com/NVlabs/DeepInversion
    https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from ..nn.module import freeze, unfreeze

from .feature_hook import DeepInversionFeatureHook
from .gaussian_smoothing import Gaussiansmoothing
from .generator import create as gen_create


class GenerativeInversion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dataset: str,
        input_dims: Tuple[int, int, int] = (3, 32, 32),
        batch_size: int = 256,
        max_iters: int = 5000,
        lr: float = 1e-3,
        tau: float = 1e3,
        alpha_pr: float = 1e-3,
        alpha_rf: float = 5.0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.lr = lr
        self.tau = tau
        self.alpha_pr = alpha_pr
        self.alpha_rf = alpha_rf
        self.feature_hooks = []

        self.model = model
        self.generator = gen_create(dataset)
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        self.criterion_ce = nn.CrossEntropyLoss()

    def setup(self):
        freeze(self.model)
        self.register_feature_hooks()

    def register_feature_hooks(self):
        # Remove old before register
        for hook in self.feature_hooks:
            hook.remove()

        ## Create hooks for feature statistics catching
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(DeepInversionFeatureHook(module))

    def criterion_pr(self, input):
        input_pad = F.pad(input, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.smoothing(input_pad)
        return F.mse_loss(input, input_smooth)

    def criterion_rf(self):
        #  return sum([hook.r_feature for hook in self.feature_hooks])
        return torch.stack([h.r_feature for h in self.feature_hooks]).mean()

    def criterion_cb(self, output: torch.Tensor):
        logit_mu = output.softmax(dim=1).mean(dim=0)
        num_classes = output.shape[1]
        # ignore sign
        entropy = (logit_mu * logit_mu.log() / math.log(num_classes)).sum()
        return 1 + entropy

    @torch.no_grad()
    def sample(self, batch_size: int = None):
        _ = self.model.eval() if self.model.training else None
        batch_size = self.batch_size if batch_size is None else batch_size
        input = self.generator.sample(batch_size)
        target = self.model(input).argmax(dim=1)
        return input, target

    def train_step(self):
        input = self.generator.sample(self.batch_size)
        output = self.model(input)
        target = output.data.argmax(dim=1)

        # content loss
        loss_ce = self.criterion_ce(output / self.tau, target)

        # label diversity loss
        loss_cb = self.criterion_cb(output)

        # locally smooth prior
        loss_pr = self.alpha_pr * self.criterion_pr(input)

        # feature statistics regularization
        loss_rf = self.alpha_rf * self.criterion_rf()

        loss = loss_ce + loss_cb + loss_pr + loss_rf

        loss_dict = {
            "ce": loss_ce,
            "cb": loss_cb,
            "pr": loss_pr,
            "rf": loss_rf,
            "total": loss,
        }

        return loss, loss_dict

    def configure_optimizers(self):
        params = self.generator.parameters()
        return optim.Adam(params, lr=self.lr)

    def forward(self):
        _ = self.setup(), unfreeze(self.generator)
        optimizer = self.configure_optimizers()
        miniters = max(self.max_iters // 100, 1)
        pbar = trange(self.max_iters, miniters=miniters, desc="Inversion")
        for current_iter in pbar:
            loss, loss_dict = self.train_step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (current_iter + 1) % miniters == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        freeze(self.generator)
