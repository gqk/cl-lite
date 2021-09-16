# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_simple import DynamicSimpleHead


class DynamicNMEHead(DynamicSimpleHead):
    _class_means: torch.Tensor = None
    nme_mode: bool = True

    @property
    def class_means(self):
        return self._class_means

    @class_means.setter
    def class_means(self, class_means: torch.Tensor):
        assert class_means.shape[0] == self.num_classes
        assert class_means.shape[1] == self.num_features

        self._class_means = class_means.to(next(self.parameters()).device)

    @class_means.deleter
    def class_means(self):
        self._class_means = None

    def append(self, *args, **kwargs):
        del self.class_means
        return super().append(*args, **kwargs)

    def nme_classify(self, input: torch.Tensor):
        output = F.normalize(input, dim=1)
        output = output.matmul(self._class_means.t())
        return output

    def classify(self, input: torch.Tensor):
        if self.training or not self.nme_mode or self.class_means is None:
            return super().classify(input)
        return self.nme_classify(input)
