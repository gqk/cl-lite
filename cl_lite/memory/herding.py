# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dl

from .base import Memory


class Herding(Memory):
    def class_select(
        self,
        features: torch.Tensor,
        num_exemplars: int,
    ) -> List[int]:
        num_exemplars = min(features.shape[0], num_exemplars)

        assert num_exemplars > 0

        mu_t = mu = features.mean(dim=0)
        selection, num_selected = torch.zeros_like(features[:, 0]), 0

        num_iters = 0
        while num_selected < num_exemplars and num_iters < 1000:
            num_iters += 1
            index = features.matmul(mu_t).argmax()
            if selection[index] == 0:
                num_selected += 1
                selection[index] = num_selected
            mu_t = mu_t + mu - features[index]

        selection[selection == 0] = 10000
        selected_indices = selection.argsort()[:num_exemplars].tolist()

        return selected_indices

    @torch.no_grad()
    def select(
        self,
        model: nn.Module,
        dataloader: dl.DataLoader,
        num_exemplars: int,
    ) -> List[int]:
        assert hasattr(model, "head") and hasattr(model.head, "feature_mode")

        feature_mode, model.head.feature_mode = model.head.feature_mode, True
        device = next(model.parameters()).device

        features, indices, offset = defaultdict(list), defaultdict(list), 0
        for i, batch in enumerate(dataloader):
            input, target = batch
            feature = F.normalize(model(input.to(device)), dim=-1)
            for j in range(target.shape[0]):
                t = target[j].item()
                features[t].append(feature[j])
                indices[t].append(offset + j)
            offset += target.shape[0]

        selected_indices = []
        for c in self.classes:
            if c not in features:
                continue
            fs, idxs = torch.stack(features[c]), torch.tensor(indices[c])
            selected_idxs = self.class_select(fs, num_exemplars)
            selected_indices.append(idxs[torch.tensor(selected_idxs)].tolist())

        model.head.feature_mode = feature_mode

        return selected_indices
