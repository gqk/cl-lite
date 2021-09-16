# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_cosine import DynamicCosineHead


class DynamicLSCHead(DynamicCosineHead):
    def __init__(
        self,
        num_proxy: int = 1,
        reduction: str = "softmax",
        *args,
        **kwargs,
    ):
        assert reduction in ["mean", "max", "min", "softmax"]
        self.num_proxy = num_proxy
        self.reduction = reduction

        super().__init__(*args, **kwargs)

    def _create_classifier(
        self,
        num_features: int,
        num_classes: int,
        bias=True,
    ):
        num_classes = num_classes * self.num_proxy
        return super()._create_classifier(num_features, num_classes, bias)

    def _reduce(self, output: torch.Tensor):
        output = output.view(-1, self.num_classes, self.num_proxy)
        if self.reduction == "mean":
            output = output.mean(dim=-1)
        elif self.reduction == "max":
            output = output.max(dim=-1)
        elif self.reduction == "min":
            output = output.min(dim=-1)
        else:
            attention = output.softmax(dim=-1)
            output = (output * attention).sum(dim=-1)
        return output

    def classify(self, input: torch.Tensor):
        output = super().classify(input)
        output = self._reduce(output)
        return output
