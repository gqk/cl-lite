# -*- coding: utf-8 -*-

from typing import Union, Sequence

import torch
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix


from .datamodule import DataModule, SplitedDataModule, MultiTaskDataModule


class Evaluator(nn.Module):
    metric: float = 0

    def __init__(self, datamodule: DataModule):
        super().__init__()
        self.datamodule = datamodule
        self.setup()

    def setup(self):
        self.reset()

    def reset(self):
        self.metric = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        pass

    def compute(self) -> Union[float, Sequence[float]]:
        pass

    def summary(self) -> str:
        pass


class SimpleEvaluator(Evaluator):
    def setup(self):
        self.accuracy = Accuracy()

    def reset(self):
        super().reset()
        self.accuracy.reset()

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        self.accuracy.update(prediction, target)

    def compute(self):
        self.metric = self.accuracy.compute() * 100
        return self.metric

    def summary(self):
        acc = self.compute()
        return f"Accuracy: {acc: .2f}"


class ILEvaluator(Evaluator):
    confmat: torch.Tensor = None

    def __init__(
        self,
        datamodule: Union[SplitedDataModule, MultiTaskDataModule],
    ):
        super().__init__(datamodule)

    @property
    def confusion(self):
        if self._confusion is not None:
            return self._confusion

        device = self.accuracy.correct.device
        num_tasks = self.datamodule.current_task
        n = sum([self.datamodule[t].num_classes for t in range(num_tasks + 1)])
        self._confusion = ConfusionMatrix(num_classes=n).to(device)

        return self._confusion

    def setup(self):
        self.accuracy = Accuracy()
        self.accuracy_new = Accuracy()
        self.accuracy_old = Accuracy()
        self._confusion = None

    def reset(self):
        super().reset()
        self.accuracy.reset()
        self.accuracy_new.reset()
        self.accuracy_old.reset()
        self._confusion = None

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        self.accuracy.update(prediction, target)
        self.confusion.update(prediction, target)

        num_tasks = self.datamodule.current_task
        if num_tasks < 1:
            return

        n_old = sum([self.datamodule[t].num_classes for t in range(num_tasks)])
        cond = target < n_old
        if not cond.all():
            self.accuracy_new.update(prediction[~cond], target[~cond])
        if cond.any():
            self.accuracy_old.update(prediction[cond], target[cond])

    def compute(self):
        self.metric = self.accuracy.compute() * 100
        self.confmat = self.confusion.compute().long()
        acc_new, acc_old = 0, 0
        if self.datamodule.current_task > 0:
            acc_new = self.accuracy_new.compute() * 100
            acc_old = self.accuracy_old.compute() * 100
        return self.metric, acc_new, acc_old, self.confmat

    def summary(self, history=False):
        acc, acc_new, acc_old, confmat = self.compute()
        result = "Accuracy (all | new | old): "
        result += f"{acc:.2f} | {acc_new: .2f} | {acc_old: .2f}"
        result += "\nConfusion:"
        for row in confmat.tolist():
            result += "\n" + "|".join([f"{v: 3}" for v in row])
        return result
