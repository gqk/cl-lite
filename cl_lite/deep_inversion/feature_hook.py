# -*- coding: utf-8 -*-

"""
DeepInversion

credits:
    https://github.com/NVlabs/DeepInversion
"""

import math
import random
from typing import Tuple, Callable

import torch


class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute
    a loss on them. Will compute mean and variance, and will use l2 as a loss.
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.value = None

    @property
    def r_feature(self):
        # compute generativeinversion's feature distribution regularization
        if self.value is None:
            return None

        nch = self.value.shape[1]
        mean = self.value.mean([0, 2, 3])
        value = self.value.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8

        r_mean = self.module.running_mean.data.type(var.type())
        r_var = self.module.running_var.data.type(var.type())

        r_feature_item1 = (var / (r_var + 1e-8)).log()
        r_feature_item2 = (r_var + (r_mean - mean).pow(2) + 1e-8) / var
        r_feature = 0.5 * (r_feature_item1 + r_feature_item2 - 1).mean()
        return r_feature

    @property
    def r_feature_di(self):
        # compute deepinversion's feature distribution regularization
        # compute generativeinversion's feature distribution regularization
        if self.value is None:
            return None

        nch = self.value.shape[1]
        mean = self.value.mean([0, 2, 3])
        value = self.value.permute(1, 0, 2, 3).contiguous().view([nch, -1])
        var = value.var(1, unbiased=False) + 1e-8

        r_mean = self.module.running_mean.data.type(var.type())
        r_var = self.module.running_var.data.type(var.type())

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature_item1 = torch.norm(r_mean - mean, 2)
        r_feature_item2 = torch.norm(r_var - var, 2)
        r_feature = r_feature_item1 + r_feature_item2
        return r_feature

    def hook_fn(self, module, input, output):
        self.value = input[0]

    def close(self):
        self.module, self.value = None, None
        self.hook.remove()
