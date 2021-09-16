# -*- coding: utf-8 -*-

from typing import Any, Callable, Sequence

import torch
import torch.utils.data as td


class Subset(td.Subset):
    def __init__(self, dataset: td.Dataset, indices: Sequence[int]):

        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)

        while isinstance(dataset, td.Subset):
            if isinstance(dataset.indices, torch.Tensor):
                indices = dataset.indices[indices]
            else:
                indices = torch.tensor(dataset.indices)[indices]

            dataset = dataset.dataset

        super().__init__(dataset, indices)

    def __add__(self, other: Any):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return other.__radd__(self)

    def __radd__(self, other: Any):
        if not isinstance(other, self.__class__):
            return NotImplemented

        assert other.dataset == self.dataset

        indices = torch.cat([other.indices, self.indices])
        return Subset(self.dataset, indices)
