"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import atexit
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ..core import BaseConfig
from ..misc import dist_utils
from ..misc.console import tee_console


def load_checkpoint(path: str):
    """A checkpoint from a local path or a URL, on the CPU."""
    if path.startswith("http"):
        return torch.hub.load_state_dict_from_url(path, map_location="cpu")
    return torch.load(path, map_location="cpu")


class BaseSolver:
    """
    Owns the training components a config describes, moved to the device and wrapped for
    distributed training, and knows how to save and restore them. Subclasses implement ``fit``
    and ``val``.

    A checkpoint is the ``state_dict`` of every attribute that has one (model, ema, optimizer,
    schedulers, scaler, ...) keyed by attribute name, plus ``last_epoch`` and a date. A
    checkpoint holding only ``model`` (a converted release checkpoint, say) still resumes: the
    EMA is then initialised from the model weights.
    """

    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg

    def _setup(self):
        """Build the model and its companions; the loaders and optimizer are left to train/eval."""
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = cfg.model
        print("model:", getattr(self.model, "wiring", type(self.model).__name__))

        # tuning weights must be in place before the EMA copies the model
        if cfg.tuning:
            self.load_tuning_state(cfg.tuning)

        self.model = dist_utils.warp_model(
            self.model.to(device),
            sync_bn=cfg.sync_bn,
            find_unused_parameters=cfg.find_unused_parameters,
        )

        self.criterion = self.to(cfg.criterion, device)
        self.postprocessor = self.to(cfg.postprocessor, device)

        self.ema = self.to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = cfg.last_epoch
        # the best validation AP so far and the epoch it came from, carried through a resume so
        # that continuing a run cannot overwrite a better checkpoint with a worse one
        self.best_ap, self.best_epoch = float("-inf"), -1

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if dist_utils.is_main_process():
            tee_console(self.output_dir / "console.log")  # the console, kept next to log.txt
            # the resolved config (includes and command line merged) opens the log and is kept as
            # config.yml, so two runs can be diffed
            config = yaml.safe_dump({**cfg.yaml_cfg, "output_dir": str(self.output_dir)}, sort_keys=False)
            (self.output_dir / "config.yml").write_text(config, encoding="utf-8")
            print("config:\n" + config)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text("config", repr(cfg), 0)

    def cleanup(self):
        if self.writer:
            self.writer.close()

    def _setup_eval(self):
        """The validation loader and the evaluator built on top of it."""
        self.val_dataloader = self._distributed_loader(self.cfg.val_dataloader)
        self.evaluator = self.cfg.evaluator

    def _resume(self):
        if self.cfg.resume:
            self.load_resume_state(self.cfg.resume)

    @staticmethod
    def _distributed_loader(loader):
        return dist_utils.warp_loader(loader, shuffle=loader.shuffle)

    def train(self):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler
        self.train_dataloader = self._distributed_loader(self.cfg.train_dataloader)
        self._setup_eval()
        # last, so that the checkpoint's optimizer and scheduler states land on built objects
        self._resume()

    def eval(self):
        self._setup()
        self._setup_eval()
        self._resume()

    def to(self, module, device):
        return module.to(device) if hasattr(module, "to") else module

    def state_dict(self):
        """Everything needed to resume: every attribute with a state dict, plus the epoch."""
        state = {
            "date": datetime.now().isoformat(),
            "last_epoch": self.last_epoch,
            "best_ap": self.best_ap,
            "best_epoch": self.best_epoch,
        }
        for k, v in self.__dict__.items():
            if hasattr(v, "state_dict"):
                state[k] = dist_utils.de_parallel(v).state_dict()
        return state

    def load_state_dict(self, state):
        """Restore whatever the checkpoint holds; attributes it lacks are reported and kept."""
        if "last_epoch" in state:
            self.last_epoch = state["last_epoch"]
            print("Load last_epoch")
        if "best_ap" in state:  # checkpoints written before this was recorded have none
            self.best_ap, self.best_epoch = state["best_ap"], state.get("best_epoch", -1)
            print(f"Load best_ap {self.best_ap} of epoch {self.best_epoch}")

        for k, v in self.__dict__.items():
            if not hasattr(v, "load_state_dict"):
                continue
            if k in state:
                dist_utils.de_parallel(v).load_state_dict(state[k])
                print(f"Load {k}.state_dict")
            elif k == "ema" and getattr(self, "model", None) is not None:
                # a model-only checkpoint: start the average from the loaded weights
                model_state_dict = dist_utils.remove_module_prefix(self.model.state_dict())
                dist_utils.de_parallel(v).load_state_dict({"module": model_state_dict})
                print(f"Load {k}.state_dict from model.state_dict")
            else:
                print(f"Not load {k}.state_dict")

    def load_resume_state(self, path: str):
        """Resume: the checkpoint's states, the epoch included."""
        print(f"Resume checkpoint from {path}")
        self.load_state_dict(load_checkpoint(path))

    def load_tuning_state(self, path: str):
        """
        Fine-tune from a checkpoint trained elsewhere: load every tensor whose name and shape
        match the current model, and leave the rest (a classification head sized for another
        label set, say) at initialisation. Prefers the EMA weights when the checkpoint has them.
        """
        print(f"Tuning checkpoint from {path}")
        state = load_checkpoint(path)
        pretrained = state["ema"]["module"] if "ema" in state else state["model"]
        pretrained = dist_utils.remove_module_prefix(pretrained)

        module = dist_utils.de_parallel(self.model)
        matched, infos = self._matched_state(module.state_dict(), pretrained)
        module.load_state_dict(matched, strict=False)
        print(f"Load model.state_dict, {infos}")

    @staticmethod
    def _matched_state(state: dict[str, torch.Tensor], params: dict[str, torch.Tensor]):
        """The tensors of ``params`` that fit ``state`` by name and shape, and the names that do not."""
        missed, unmatched, matched = [], [], {}
        for k, v in state.items():
            if k not in params:
                missed.append(k)
            elif v.shape != params[k].shape:
                unmatched.append(k)
            else:
                matched[k] = params[k]
        return matched, {"missed": missed, "unmatched": unmatched}

    def fit(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError
