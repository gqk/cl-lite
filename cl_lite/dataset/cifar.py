# -*- coding: utf-8 -*-

import torch
import torchvision as tv

from .base import Dataset


class CIFAR10(Dataset):
    dataset_cls = tv.datasets.CIFAR10

    @property
    def transform(self):
        ts = [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
        return tv.transforms.Compose(ts)

    @property
    def dims(self):
        return torch.Size([3, 32, 32])


class CIFAR100(CIFAR10):
    dataset_cls = tv.datasets.CIFAR100

    @property
    def transform(self):
        ts = [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
        return tv.transforms.Compose(ts)
