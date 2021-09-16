import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillCrossEntropyLoss(nn.Module):
    """Distill cross entropy loss used by lwf

    Reference:
        Zhizhong Li et al. Learning without Forgetting. ECCV 2016.

    Args:
        input (Tensor): (N, C) where C = number of classes
        target (Tensor): (N, C) where C = number of classes
        tau (float): temperature parameter, default is 2.0
    """

    def __init__(self, tau: float = 2.0):
        super().__init__()
        self.tau = tau

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        c = min(input.shape[1], target.shape[1])

        input = input[:, :c].div(self.tau).log_softmax(dim=1)
        target = target[:, :c].div(self.tau).softmax(dim=1)

        return F.kl_div(input, target, reduction="batchmean")
