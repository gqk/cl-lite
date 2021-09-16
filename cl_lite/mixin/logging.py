# -*- coding: utf-8 -*-

import os
import logging as pylogging
from typing import Optional

import pytorch_lightning as pl


class LoggingMixin:
    trainer: Optional[pl.Trainer]

    def __init__(self, level: int = pylogging.INFO):
        super().__init__()
        self._logging = None
        self._logging_level = level

    @property
    def logging(self):
        if not getattr(self, "trainer", None):
            print("Warning: no trainer")
            return None

        if self._logging is None:
            log_dir = self.trainer.logger.log_dir
            logger = pylogging.getLogger(log_dir)
            logger.setLevel(self._logging_level)
            if not logger.handlers:
                log_file = os.path.join(log_dir, "log.txt")
                handler = pylogging.FileHandler(log_file)
                logger.addHandler(handler)
            self._logging = logger

        return self._logging

    def log_text(self, *args, level: int = pylogging.INFO, **kwargs):
        if self.trainer.is_global_zero and self.logging is not None:
            self.logging.log(level, *args, **kwargs)
