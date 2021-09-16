# -*- coding: utf-8 -*-

""" This module is used to override or extend some modules of pytorch """

from .init import weights_init_kaiming, weights_init_classifier
from .module import freeze, unfreeze, no_grad_of
from .loss import (
    DistillCrossEntropyLoss,
    NCALoss,
    PooledOutputDistillationLoss,
    RKDAngleLoss,
    RKDDistanceLoss,
)
