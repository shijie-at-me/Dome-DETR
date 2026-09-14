"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

__all__ = ["BaseConfig", "Component"]


class Component:
    """
    A training component held by a config: stored under ``_<name>``, type-checked when set from
    outside, and, when ``factory`` is given, built from the config the first time it is read while
    still ``None``. ``YAMLConfig`` overrides these as properties that build the object from the
    yaml, and falls back on this through ``super()``.
    """

    def __init__(self, expected: type | tuple[type, ...] | None = None, factory: Callable[[Any], Any] | None = None):
        self.expected = expected
        self.factory = factory

    def __set_name__(self, owner, name):
        self.name = name
        self.attr = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = getattr(obj, self.attr)
        if value is None and self.factory is not None:
            value = self.factory(obj)
            setattr(obj, self.attr, value)
        return value

    def __set__(self, obj, value):
        if self.expected is not None and not isinstance(value, self.expected):
            raise TypeError(f"{self.name}: expected {self.expected}, got {type(value)}")
        setattr(obj, self.attr, value)


def _default_ema(cfg):
    if cfg.use_ema and cfg.model is not None:
        from ..optim import ModelEMA

        return ModelEMA(cfg.model, cfg.ema_decay, cfg.ema_warmups)
    return None


def _default_scaler(cfg):
    if cfg.use_amp and torch.cuda.is_available():
        return GradScaler("cuda")
    return None


def _default_writer(cfg):
    if cfg.summary_dir:
        return SummaryWriter(cfg.summary_dir)
    if cfg.output_dir:
        return SummaryWriter(Path(cfg.output_dir) / "summary")
    return None


class BaseConfig:
    """
    Everything a solver reads: the runtime settings as plain attributes, and the training
    components (model, criterion, loaders, optimizer, ...) as ``Component`` descriptors. Here the
    components are just held (the EMA, scaler and writer get a default built from the runtime
    settings); ``YAMLConfig`` overrides them to build each one from the yaml the first time it is
    asked for. Public attribute names double as the yaml keys ``YAMLConfig`` copies in.
    """

    model = Component(nn.Module)
    postprocessor = Component(nn.Module)
    criterion = Component(nn.Module)
    optimizer = Component(Optimizer)
    lr_scheduler = Component(LRScheduler)
    lr_warmup_scheduler = Component()
    train_dataloader = Component()
    val_dataloader = Component()
    ema = Component(factory=_default_ema)
    scaler = Component(factory=_default_scaler)
    evaluator = Component(Callable)
    writer = Component(SummaryWriter, factory=_default_writer)

    def __init__(self) -> None:
        super().__init__()

        self.task: str = None

        # components, built lazily by the subclass or set from outside
        self._model: nn.Module = None
        self._postprocessor: nn.Module = None
        self._criterion: nn.Module = None
        self._optimizer: Optimizer = None
        self._lr_scheduler: LRScheduler = None
        self._lr_warmup_scheduler: LRScheduler = None
        self._train_dataloader: DataLoader = None
        self._val_dataloader: DataLoader = None
        self._ema: nn.Module = None
        self._scaler: GradScaler = None
        self._evaluator: Any = None
        self._writer: SummaryWriter = None

        # runtime
        self.resume: str = None
        self.tuning: str = None

        self.epoches: int = None
        self.last_epoch: int = -1

        self.use_amp: bool = False
        self.use_ema: bool = False
        self.ema_decay: float = 0.9999
        self.ema_warmups: int = 2000
        self.ema_restart_decay: float = 0.9999  # decay after the stage-2 restart (see DetSolver)
        self.sync_bn: bool = False
        self.clip_max_norm: float = 0.0
        self.find_unused_parameters: bool = None

        self.seed: int = None
        self.print_freq: int = None
        self.checkpoint_freq: int = 1
        # stage 1 is validated from epoch eval_after on, every eval_freq epochs, and on its last
        # epoch whatever the two say; stage 2 validates every epoch, its patience counts them (see
        # DetSolver)
        self.eval_freq: int = 1
        self.eval_after: int = 0
        # stage 2 reloads the best stage-1 checkpoint after this many evaluated epochs without a new
        # best; 0 never reloads, which is what a run continued past its schedule wants (see DetSolver)
        self.patience: int = 6
        self.output_dir: str = None
        self.summary_dir: str = None
        self.device: str = ""

    def __repr__(self):
        return "".join(f"{k}: {v}\n" for k, v in self.__dict__.items() if not k.startswith("_"))
