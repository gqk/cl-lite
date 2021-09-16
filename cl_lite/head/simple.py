# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..nn.init import weights_init_classifier


class SimpleHead(nn.Module):
    feature_mode: bool = False

    def __init__(
        self,
        num_classes: int,
        num_features: int = 2048,
        bias: bool = True,
        neck: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.bias = bias

        self.setup(neck)

    @property
    def embeddings(self):
        return self.classifier.weight

    def setup(self, neck: nn.Module = nn.Identity()):
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = neck

        args = [self.num_features, self.num_classes]
        self.classifier = self._create_classifier(*args, bias=self.bias)

    def classify(self, input: torch.Tensor):
        return self.classifier(input)

    def forward(self, input):
        output = self.pool(input)
        output = output[:, :, 0, 0]
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)
        return output

    def _create_classifier(
        self,
        num_features: int,
        num_classes: int,
        bias=True,
    ):
        assert num_features > 0 and num_classes > 0

        classifier = nn.Linear(num_features, num_classes, bias=bias)
        weights_init_classifier(classifier)

        return classifier
