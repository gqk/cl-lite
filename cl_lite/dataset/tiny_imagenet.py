# -*- coding: utf-8 -*-

import os
import glob
from typing import List

import torch
import torchvision as tv

from .base import Dataset, ImageMetaDataset, MetaItem


class TinyImageNetMetaDataset(ImageMetaDataset):
    def _get_wnids(self) -> List[str]:
        with open(os.path.join(self.root, "wnids.txt")) as f:
            return f.readlines()

    def setup_meta(self) -> List[MetaItem]:
        wnids = {k[:-1]: v for v, k in enumerate(self._get_wnids())}
        if self.train:
            dir_path = os.path.join(self.root, "train")
            meta = []
            for wnid in wnids:
                files = glob.glob(os.path.join(dir_path, wnid, "**/*.JPEG"))
                meta.extend([MetaItem(file, wnids[wnid]) for file in files])
            return meta

        with open(os.path.join(self.root, "val", "val_annotations.txt")) as f:
            val_list = f.readlines()

        meta, dir_path = [], os.path.join(self.root, "val", "images")
        for item in val_list:
            fname, wnid = item.split("\t")[:2]
            meta.append(MetaItem(os.path.join(dir_path, fname), wnids[wnid]))
        return meta


class TinyImageNet200(Dataset):
    dataset_cls = TinyImageNetMetaDataset

    @property
    def transform(self):
        ts = [
            tv.transforms.Resize(64),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.4803, 0.4481, 0.3976), (0.2764, 0.2688, 0.2816)
            ),
        ]
        return tv.transforms.Compose(ts)

    @property
    def dims(self):
        return torch.Size([3, 64, 64])
