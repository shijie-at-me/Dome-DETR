"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver, load_checkpoint
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    """
    Detection training in two stages, split at the collate function's ``stop_epoch``.

    Stage 1 trains with augmentation and multi-scale batches, keeping ``last.pth`` and the best
    validation AP so far as ``best_stg1.pth``. Stage 2 reloads that checkpoint's weights, EMA and
    optimizer state, restarts the EMA with ``ema_restart_decay`` and trains on clean single-scale
    batches, keeping its best as ``best_stg2.pth``. When stage 2 goes ``patience`` epochs without
    a new best, it reloads ``best_stg1.pth`` again with a slightly smaller EMA decay and tries
    once more (``patience`` 0 never does, which is what a run continued past its schedule wants).
    A reload leaves the learning rate schedule where training is: restoring the checkpoint's
    schedulers would rewind the rate past its milestones, as upstream did.

    Stage 1 validates from epoch ``eval_after`` on, every ``eval_freq`` epochs, and its last epoch
    whatever the two say (stage 2 reloads that ``best_stg1.pth``); stage 2 validates every epoch,
    its patience counts them.

    ``last.pth`` is written after every epoch of either stage and carries the best AP reached so
    far, so a run can be continued with ``-r <run>/last.pth`` and a larger ``epoches``: the
    continued run keeps validating every epoch, cannot overwrite ``best_stg2.pth`` with something
    worse than the best it already had, and with ``patience`` 0 will not rewind to stage 1. The
    random number generators are not restored, so a continued run is not the run that would have
    trained straight through, only one of the same distribution.
    """

    metric = "coco_eval_bbox"  # the evaluator's stats; AP@[.5:.95] is entry 0

    def fit(self):
        self.train()
        cfg = self.cfg
        stage2_start = self.train_dataloader.collate_fn.stop_epoch

        n_parameters, model_stats = stats(cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        # -inf until an epoch is evaluated, so the first one is saved as best_stg1; a resumed run
        # brings the best it had reached, so continuing cannot overwrite a better checkpoint
        best_ap, best_epoch = self.best_ap, self.best_epoch
        if self.last_epoch > 0:
            test_stats, _ = self._evaluate()  # also a check that the checkpoint loaded into this model
            ap = test_stats[self.metric][0]
            print(f"resumed at epoch {self.last_epoch}: {self.metric} {ap}")
            if ap > best_ap:
                best_ap, best_epoch = ap, self.last_epoch
            print(f"best_stat: {{'epoch': {best_epoch}, '{self.metric}': {best_ap}}}")
        # the stage-2 patience counter is only checked on epochs that fail to improve on the
        # best AP seen since the last reload
        best_since_reload = best_ap
        not_improved = 0

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, cfg.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized() and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)  # a grouped loader's sampler got it from set_epoch above
            stage2 = epoch >= stage2_start

            if epoch == stage2_start:
                self._reload_best_stage1(epoch, ema_decay=cfg.ema_restart_decay)
                not_improved = 0  # the patience counts stage-2 epochs alone

            print("Train starting...")
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=cfg.clip_max_norm,
                print_freq=cfg.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )
            print("Training state finished.")

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch = epoch

            self._save_periodic_checkpoints(epoch, stage2)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if not self._evaluates(epoch, stage2_start):
                print(
                    f"Evaluation skipped (from epoch {cfg.eval_after} every {cfg.eval_freq} epochs, every epoch from {stage2_start})"
                )
                self._append_log(log_stats)
                continue

            print("Evaluate state starting...")
            test_stats, coco_evaluator = self._evaluate()
            ap = test_stats[self.metric][0]
            self._log_test_stats(test_stats, epoch)

            if ap > best_ap:
                best_ap, best_epoch = ap, epoch
                self.best_ap, self.best_epoch = best_ap, best_epoch
                not_improved = 0
                self._save_checkpoint("best_stg2.pth" if stage2 else "best_stg1.pth")
            else:
                not_improved += 1
            print(f"current_stat: {ap}")
            print(f"best_stat: {{'epoch': {best_epoch}, '{self.metric}': {best_ap}}}")

            if ap > best_since_reload:
                best_since_reload = ap
            elif stage2 and cfg.patience > 0:
                if not_improved >= cfg.patience:
                    self._reload_best_stage1(epoch, ema_decay=self.ema.decay - 0.0001 if self.ema else None)
                    not_improved = 0
                    best_since_reload = float("-inf")
                else:
                    print(f"Tolerate undesirable result for patience: {not_improved} / {cfg.patience} ")

            log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
            self._append_log(log_stats)
            self._dump_eval(coco_evaluator, epoch)

        total_time = time.time() - start_time
        print(f"Training time {datetime.timedelta(seconds=int(total_time))}")

    def val(self):
        self.eval()
        _, coco_evaluator = self._evaluate()
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

    def _evaluates(self, epoch: int, stage2_start: int) -> bool:
        """Whether this epoch is validated: in stage 2 always, in stage 1 from ``eval_after`` on every
        ``eval_freq`` epochs (counted from epoch 0) and on the last stage-1 epoch, which stage 2 reloads."""
        cfg = self.cfg
        if epoch >= stage2_start or epoch == stage2_start - 1 or epoch == cfg.epoches - 1:
            return True
        return epoch >= cfg.eval_after and (epoch + 1) % cfg.eval_freq == 0

    def _save_checkpoint(self, name: str):
        if self.output_dir:
            dist_utils.save_on_master(self.state_dict(), self.output_dir / name)

    def _save_periodic_checkpoints(self, epoch: int, stage2: bool):
        """``last.pth`` every epoch, whatever the stage, so a run can always be resumed from where it
        stopped; the numbered copies are stage 1's, where no checkpoint is kept otherwise."""
        self._save_checkpoint("last.pth")
        if not stage2 and (epoch + 1) % self.cfg.checkpoint_freq == 0:
            self._save_checkpoint(f"checkpoint{epoch:04}.pth")

    def _log_test_stats(self, test_stats: dict, epoch: int):
        if self.writer and dist_utils.is_main_process():
            for k, values in test_stats.items():
                for i, v in enumerate(values):
                    self.writer.add_scalar(f"Test/{k}_{i}", v, epoch)

    def _append_log(self, log_stats: dict):
        """One json line per epoch in ``log.txt``, written by the main process (the console goes to ``console.log``)."""
        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    def _dump_eval(self, coco_evaluator, epoch: int):
        """The raw COCOeval results of this epoch under ``eval/``, for offline analysis."""
        if not (self.output_dir and dist_utils.is_main_process() and "bbox" in coco_evaluator.coco_eval):
            return
        (self.output_dir / "eval").mkdir(exist_ok=True)
        filenames = ["latest.pth"]
        if epoch % 50 == 0:
            filenames.append(f"{epoch:03}.pth")
        for name in filenames:
            torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

    def _evaluate(self):
        """Validation of the EMA weights when there are any, else of the model."""
        module = self.ema.module if self.ema else self.model
        return evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, self.evaluator, self.device)

    RELOADED = ("model", "ema", "optimizer")  # what a stage-2 reload takes from the checkpoint

    def _reload_best_stage1(self, epoch, ema_decay):
        """Restart from the best stage-1 checkpoint's weights, EMA and optimizer state with a new EMA decay."""
        path = self.output_dir / "best_stg1.pth"
        if not path.exists():
            raise FileNotFoundError(f"stage 2 starts at epoch {epoch} but there is no {path} to reload")
        dist_utils.barrier()
        print(f"Reload {path} ({', '.join(self.RELOADED)}) at epoch {epoch}")
        state = load_checkpoint(str(path))
        rates = [group["lr"] for group in self.optimizer.param_groups]  # the optimizer state carries its own
        self.load_state_dict({k: v for k, v in state.items() if k in self.RELOADED})
        for group, lr in zip(self.optimizer.param_groups, rates):
            group["lr"] = lr
        if self.ema is not None and ema_decay is not None:
            self.ema.decay = ema_decay
            print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
