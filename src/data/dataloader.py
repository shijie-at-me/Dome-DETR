"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import random
from itertools import product

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as data

from ..core import register

__all__ = [
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "DataLoader",
    "generate_scales",
]


class EpochAware:
    """Something the solver tells the current epoch to; ``epoch`` is -1 until it does."""

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)


@register()
class DataLoader(data.DataLoader, EpochAware):
    """
    torch's DataLoader with the two things the training loop needs on top: ``set_epoch``, which
    forwards the epoch to the dataset and the collate function (both may change behaviour with
    it), and a ``shuffle`` flag that survives construction so the distributed sampler can be
    built with the same setting.
    """

    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        fields = ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]
        body = "".join(f"\n    {n}: {getattr(self, n)}" for n in fields)
        return f"{self.__class__.__name__}({body}\n)"

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        # a worker holds its own copy of the dataset, made when the iterator was created: with
        # persistent workers that copy keeps the epoch it was born with, and every epoch-dependent
        # transform policy silently never fires
        if self.persistent_workers and self.num_workers > 0 and getattr(self.dataset, "_epoch_policy", False):
            raise RuntimeError(
                "the dataset has an epoch-dependent transform policy, which persistent_workers "
                "would freeze at the epoch the workers were created with; set persistent_workers "
                "to false on this loader"
            )
        self.dataset.set_epoch(epoch)
        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)
        if hasattr(self.batch_sampler, "set_epoch"):  # a GroupedBatchSampler reshuffles per epoch
            self.batch_sampler.set_epoch(epoch)

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a bool, got {type(shuffle)}")
        self._shuffle = shuffle


class BaseCollateFunction(EpochAware):
    def __call__(self, items):
        raise NotImplementedError


def _axis_scales(size: int, repeat: int, step: int) -> list[int]:
    """
    The multi-scale sizes for one image side: ``size`` rounded down to a multiple of ``step``,
    then every multiple of ``step`` from 0.75x to 1.25x of it, with the base size itself listed
    ``repeat`` times so that it is drawn more often. Sizes run upwards to the base size and then
    downwards from the top, which is the order the released models were trained with.
    """
    size = (size // step) * step or step
    low = list(range(-(-int(size * 0.75) // step) * step, size + 1, step))  # ceil to a multiple of step
    top = (int(size * 1.25) // step) * step
    high = list(range(top, size - 1, -step)) if top >= size + step else []
    return low + [size] * repeat + high


def generate_scales(base_size, base_size_repeat: int, window_size: int) -> list[tuple[int, int]]:
    """
    The ``(h, w)`` sizes a multi-scale batch is resized to. Every size is a multiple of
    ``8 * window_size`` so that the stride-8 feature map, where MWAS runs, divides into whole
    windows. A scalar ``base_size`` gives square sizes; an ``(h, w)`` pair gives the cartesian
    product of the two sides' scales.
    """
    step = 8 * window_size
    if isinstance(base_size, (list, tuple)):
        h, w = base_size
        return list(product(_axis_scales(h, base_size_repeat, step), _axis_scales(w, base_size_repeat, step)))
    return [(s, s) for s in _axis_scales(base_size, base_size_repeat, step)]


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    """
    Stack the images of a batch and, while ``epoch < stop_epoch`` and ``base_size_repeat`` is
    given, resize the whole batch to a size drawn from ``generate_scales``. The solver also reads
    ``stop_epoch`` as the boundary between its two training stages.

    ``mwas_window_size`` should match the encoder's, so that every drawn size divides into whole
    windows on the stride-8 map.

    With ``pad_to_multiple``, images of different sizes are zero-padded on the bottom and right to
    the largest height and width in the batch, rounded up to that multiple, so that a validation
    batch of aspect-preserving resizes stacks and every side divides into whole MWAS windows. Each
    target then records ``resized_size`` (its image before padding) and ``padded_size`` (after),
    both as (w, h), which the postprocessor needs to map predictions back to the original image.
    """

    def __init__(
        self, stop_epoch=None, base_size=(640, 640), base_size_repeat=None, mwas_window_size=20, pad_to_multiple=None
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.window_size = mwas_window_size
        self.scales = (
            generate_scales(base_size, base_size_repeat, mwas_window_size) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, items):
        targets = [x[1] for x in items]
        if self.pad_to_multiple:
            images = self._pad_to_common_size([x[0] for x in items], targets)
        else:
            images = torch.cat([x[0][None] for x in items], dim=0)

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            # bilinear, not interpolate's nearest default: a batch drawn at 0.8x loses every fifth
            # row and column under nearest, which on an 8 px object is most of what it is
            images = F.interpolate(images, size=sz, mode="bilinear", align_corners=False)
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")

        return images, targets

    def _pad_to_common_size(self, images, targets):
        m = self.pad_to_multiple
        pad_h = -(-max(im.shape[-2] for im in images) // m) * m
        pad_w = -(-max(im.shape[-1] for im in images) // m) * m
        padded = []
        for im, tg in zip(images, targets):
            h, w = im.shape[-2:]
            padded.append(F.pad(im, (0, pad_w - w, 0, pad_h - h)))
            tg["resized_size"] = torch.tensor([w, h])
            tg["padded_size"] = torch.tensor([pad_w, pad_h])
        return torch.stack(padded, dim=0)
