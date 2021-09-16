# -*- coding: utf-8 -*-

import time
from typing import Type, Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from .datamodule import DataModule, SplitedDataModule, MultiTaskDataModule


class App(LightningCLI):
    def __init__(self, module_cls, datamodule_cls: DataModule, **kwargs):
        super().__init__(
            model_class=module_cls,
            datamodule_class=datamodule_cls,
            seed_everything_default=int(time.time()),
            save_config_overwrite=True,
            trainer_defaults=kwargs,
        )

        self.logger_version = self.trainer.logger.version
        self.current_task = self.config.get("data", {}).get("current_task", 0)
        self.num_tasks = self.config.get("data", {}).get("num_tasks", 1)

        self.incremental = self.num_tasks > 1
        if self.incremental:
            inc_dmcs = (SplitedDataModule, MultiTaskDataModule)
            assert issubclass(datamodule_cls, inc_dmcs)

    def before_fit(self) -> None:
        """ Rename super before_fit to before_train"""
        pass

    def fit(self) -> None:
        """ Rename super fit to train"""
        pass

    def after_fit(self) -> None:
        pass

    def before_train(self) -> None:
        super().before_fit()

        if self.incremental:
            version, current_task = self.logger_version, self.current_task
            logger_version = f"version_{version}/task_{current_task}"
            self.config["data"]["current_task"] = current_task
            self.datamodule.switch_task(current_task)

            self.model._logging = None
            self.trainer.logger._version = logger_version
            self.trainer.logger._experiment = None
            self.trainer.logger._prev_step = -1

        self.model.datamodule = self.datamodule
        # TODO remove after fix module.log_text
        self.trainer.logger.log_hyperparams(self.config)

    def train(self) -> None:
        super().fit()

    def after_train(self) -> None:
        super().after_fit()

        last_task_id = self.num_tasks - 1
        if self.incremental and self.current_task < last_task_id:
            keys = ["logger"]  # TODO accelerator
            for key in keys:
                self.config_init["trainer"][key] = getattr(self.trainer, key)

            self.config_init["trainer"]["resume_from_checkpoint"] = None
            self.save_config_callback = None

            self.instantiate_trainer()

        self.current_task += 1

    def main(self) -> None:
        while self.current_task < self.num_tasks:
            self.before_train()
            self.train()
            self.after_train()
