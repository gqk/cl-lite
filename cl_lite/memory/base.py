# -*- coding: utf-8 -*-

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.utils.data.dataloader as dl


class Memory(nn.Module):
    def __init__(
        self,
        classes: List[int],
        max_size: int = 2000,
        dynamic: bool = False,
    ):
        super().__init__()

        self.classes = classes
        self.max_size = max_size
        self.dynamic = dynamic

        self.num_classes = len(classes)
        if dynamic:
            assert max_size > self.num_classes > 0
        else:
            self.num_classes = 0

        self._size = None
        self.container = []

    @property
    def size(self):
        if self._size is None:
            self._size = sum([len(item) for item in self.container], 0)
        return self._size

    def __getitem__(self, idx):
        return self.container[idx]

    def select(
        self,
        model: nn.Module,
        dataloader: dl.DataLoader,
        num_exemplars: int,
    ) -> List[List[int]]:
        pass

    def delete(
        self,
        indices: List[int],
        num_exemplars: int,
    ) -> Tuple[List[int], List[int]]:
        return indices[:num_exemplars], indices[num_exemplars:]

    def update(
        self,
        model: nn.Module,
        dataloader: dl.DataLoader,
        num_classes: int = -1,
        num_old_classes: int = -1,
        dry_run: bool = False,
    ) -> List[List[int]]:
        assert num_classes > 0 and self.num_classes >= num_old_classes

        if num_old_classes < 0:
            num_old_classes = len(self.container)

        container, deleted_indices = self.container[:num_old_classes], []
        num_exemplars = self.max_size // (num_old_classes + num_classes)
        if self.dynamic:
            num_exemplars = self.max_size // self.num_classes
        elif num_old_classes > 0:
            results = [self.delete(item, num_exemplars) for item in container]
            container, _ = map(list, zip(*results))

        selected_indices = self.select(model, dataloader, num_exemplars)
        container = container + selected_indices
        if not dry_run:
            self.container, self._size = container, None
            if not self.dynamic:
                self.num_classes = num_old_classes + num_classes

        return container

    def forward(self, *args, **kwargs):
        return self.update(*args, **kwargs)
