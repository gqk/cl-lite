# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, List, Union, NamedTuple, Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

__all__ = ["MetaItem", "ImageMetaDataset"]

MetaItem = NamedTuple("MetaItem", [("img_path", str), ("target", int)])


class ImageMetaDataset(Dataset):
    _num_items: int = None
    _meta: List[MetaItem] = None
    _targets: List[int] = None
    default_transform: Callable[[Any], torch.Tensor] = ToTensor()

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool = None,
        transform: Callable[[Any], torch.Tensor] = None,
        **kwargs
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

    @property
    def meta(self) -> List[MetaItem]:
        if self._meta is None:
            self._meta = self.setup_meta()
        return self._meta

    @meta.setter
    def meta(self, meta: List[MetaItem]):
        self._meta, self._num_items, self._targets = meta, None, None

    @property
    def targets(self) -> List[int]:
        if not self._targets:
            self._targets = [item.target for item in self.meta]
        return self._targets

    def __len__(self):
        if self._num_items is None:
            self._num_items = len(self.meta)
        return self._num_items

    def __getitem__(self, index: Union[int, slice]):
        items = self.meta[index]
        if not isinstance(index, slice):
            return self.process_item(items)
        return list(map(self.process_item, items))

    def __add__(self, other: Any):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other.__radd__(self)

    def __radd__(self, other: Any):
        if not isinstance(other, self.__class__):
            return NotImplemented
        dataset = deepcopy(other)
        dataset.meta = other.meta + self.meta
        return dataset

    def setup_meta(self) -> List[MetaItem]:
        raise NotImplementedError

    def statistic(self, indices: Sequence[int] = None):
        num_imgs, targets = 0, set()
        indices = range(len(self)) if indices is None else indices
        for i in indices:
            num_imgs += 1
            targets.add(self.meta[i].target)
        return len(targets), num_imgs

    def process_item(self, item: MetaItem):
        img = Image.open(item.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.default_transform(img)
        return img, item.target
