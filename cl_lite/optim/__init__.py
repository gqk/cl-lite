# -*- coding: utf-8 -*-

__all__ = ["ConstantLR", "SequentialLR"]

try:
    from torch.optim.lr_scheduler import ConstantLR, SequentialLR
except:
    from .lr_scheduler import ConstantLR, SequentialLR
