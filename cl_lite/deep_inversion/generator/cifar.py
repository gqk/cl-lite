# -*- coding: utf-8 -*-

"""
CIFAR Generator

credits:
    https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL
    https://github.com/VainF/Data-Free-Adversarial-Distillation
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super().__init__()

        self.z_dim = zdim
        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        device = next(self.parameters()).device
        z = torch.randn(size, self.z_dim).to(device)
        X = self.forward(z)
        return X


def CIFAR_GEN():
    return Generator(zdim=1000, in_channel=3, img_sz=32)
