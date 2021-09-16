# -*- coding: utf-8 -*-

from typing import Callable, Dict, Optional, Union, Sequence, Type

import torch.nn as nn
import pytorch_lightning as pl

from ..mixin import LoggingMixin, LossMixin
from .datamodule import DataModule
from .evaluator import Evaluator, SimpleEvaluator


class Module(LoggingMixin, LossMixin, pl.LightningModule):
    evaluator_cls: Type[Evaluator] = SimpleEvaluator
    extra_evaluator_clses: Dict[str, Type[Evaluator]] = {}

    @property
    def datamodule(self) -> DataModule:
        return self.__datamodule

    @datamodule.setter
    def datamodule(self, datamodule: DataModule) -> None:
        assert isinstance(datamodule, DataModule)
        self.__datamodule = datamodule
        self.__evaluator = self.evaluator_cls(datamodule)
        self.__extra_evaluators = nn.ModuleDict(
            {k: v(datamodule) for k, v in self.extra_evaluator_clses.items()}
        )

    @property
    def evaluator(self) -> Evaluator:
        return self.__evaluator

    @property
    def extra_evaluators(self) -> Dict[str, Evaluator]:
        return self.__extra_evaluators

    @property
    def example_input_array(self):
        return next(iter(self.datamodule.test_dataloader()))[0]

    def print(self, *args, log_text=True, **kwargs):
        """Override LightningModule's print"""
        # There is a bug in v1.3.8: print not working in on_fit_end
        # Temporarily fall back to v1.1.0
        if self.trainer.is_global_zero:
            print(*args, **kwargs)

        if log_text:
            self.log_text(*args, **kwargs)

    def update_evaluator(self, prediction, target, name: str = None):
        if name is None:
            return self.evaluator.update(prediction, target)
        assert name in self.extra_evaluators
        return self.extra_evaluators[name].update(prediction, target)

    def evaluation_result(self, name: str = None):
        epoch = self.current_epoch
        if name is None:
            name, summary = "default", self.evaluator.summary()
        else:
            assert name in self.extra_evaluators
            summary = self.extra_evaluators[name].summary()
        return f"\n\n=> Evaluation result ({name}) {epoch} \n\n {summary} \n"

    def reset_evaluators(self):
        self.evaluator.reset()
        for evaluator in self.extra_evaluators.values():
            evaluator.reset()

    def on_load_checkpoint(self, checkpoint):
        self.datamodule.load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        self.datamodule.save_checkpoint(checkpoint)

    def on_train_start(self):
        super().on_train_start()
        self.move_losses_to_device()

    def configure_callbacks(self):
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="eval/metric",
            save_last=True,
            save_top_k=1,
            mode="max",
            filename="best_acc",
        )
        learning_rate = pl.callbacks.lr_monitor.LearningRateMonitor()
        return [checkpoint, learning_rate]

    def validation_step(self, batch, batch_idx):
        target = self.datamodule.transform_target(batch[1])
        prediction = self.forward(batch[0]).argmax(dim=1)
        self.update_evaluator(prediction, target)

    def validation_epoch_end(self, validation_step_outputs):
        self.log("eval/metric", self.evaluator.metric)
        self.print(self.evaluation_result())
        self.reset_evaluators()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
