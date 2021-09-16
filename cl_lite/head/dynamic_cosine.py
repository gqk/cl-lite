# -*- coding: utf-8 -*-

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_simple import DynamicSimpleHead


class DynamicCosineHead(DynamicSimpleHead):
    mock_training: bool = False

    def __init__(self, sigma=True, *args, **kwargs):
        super().__init__(bias=False, *args, **kwargs)

        self.register_parameter("sigma", None)
        if sigma:
            self.sigma = nn.Parameter(torch.tensor(1.0))

    def classify(self, input: torch.Tensor):
        output = F.normalize(input, dim=1)
        output = [
            F.linear(output, F.normalize(classifier.weight, dim=1))
            for classifier in self.classifiers
        ]

        output = torch.cat(output, dim=1)

        if self.sigma is not None:
            output = self.sigma * output

        return output
