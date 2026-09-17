"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Batches of images with similar object counts. Under a per-image query budget the batch runs
with the largest budget in it and the other images are padded to it; with the images grouped
by the bucket of their count the padding is gone, and the budget of a batch is the budget
of its images.
"""

import bisect
import math

import torch
from torch.utils.data import Sampler

from ..core import register
from ..misc import dist_utils

__all__ = ["GroupedBatchSampler", "object_counts"]


def object_counts(dataset) -> list[int]:
    """
    The objects of every image of ``dataset`` in index order, crowd (ignore) boxes left out.

    Read from the annotation columns, not from ``dataset.coco``: building the COCO ground truth of
    the training split costs 7 s and 231 MB resident for AI-TOD's 376k annotations, and nothing
    else in training ever reads it (the evaluator scores against the validation split's).
    """
    meta = getattr(dataset, "hf_meta", None)
    if meta is not None:
        counts = []
        for row in meta:
            _, _, iscrowd = dataset.parse_objects(row["objects"])
            counts.append(int((iscrowd == 0).sum()))
        return counts
    coco = dataset.coco
    return [sum(1 for a in coco.imgToAnns.get(i, ()) if not a.get("iscrowd", 0)) for i in range(len(dataset))]


@register()
class GroupedBatchSampler(Sampler):
    """
    Yields batches of indices whose images fall in the same count bucket, the buckets cut at
    ``edges`` (``[100, 200, 300, 600]``: 0 .. 100, ..., 600 and above; keep them equal to the
    decoder's ``count_edges`` so a batch shares one budget). Every epoch (``set_epoch``) the images of each bucket are shuffled and cut into batches, the
    batches of every bucket then shuffled together; the images a bucket has left over are
    pooled, sorted by bucket and batched among themselves (the only batches that mix
    neighbouring buckets), the last of those dropped when ``drop_last`` or incomplete under
    several ranks. Distributed, the ranks take the batches in turn: the same shuffle on every
    rank, every rank the same number of batches.
    """

    def __init__(self, dataset, batch_size, edges=(100, 200, 300, 600), shuffle=True, drop_last=False, seed=0):
        edges = sorted(edges)
        self.groups = [bisect.bisect_right(edges, c) for c in object_counts(dataset)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.rank = dist_utils.get_rank() if dist_utils.is_dist_available_and_initialized() else 0
        self.world_size = dist_utils.get_world_size() if dist_utils.is_dist_available_and_initialized() else 1
        by_group = {}
        for i, g in enumerate(self.groups):
            by_group.setdefault(g, []).append(i)
        self.by_group = dict(sorted(by_group.items()))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        batches, leftovers = [], []
        for members in self.by_group.values():
            order = torch.randperm(len(members), generator=gen).tolist() if self.shuffle else range(len(members))
            members = [members[j] for j in order]
            full = len(members) // self.batch_size * self.batch_size
            batches += [members[j : j + self.batch_size] for j in range(0, full, self.batch_size)]
            leftovers += members[full:]  # in bucket order, so a leftover batch mixes neighbouring buckets only
        for j in range(0, len(leftovers), self.batch_size):
            batch = leftovers[j : j + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        if self.shuffle:
            batches = [batches[j] for j in torch.randperm(len(batches), generator=gen).tolist()]
        if self.world_size > 1:  # every rank the same number of batches, the incomplete round dropped
            n = len(batches) // self.world_size * self.world_size
            batches = batches[self.rank : n : self.world_size]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        full = sum(len(m) // self.batch_size for m in self.by_group.values())
        left = sum(len(m) % self.batch_size for m in self.by_group.values())
        n = full + (left // self.batch_size if self.drop_last else math.ceil(left / self.batch_size))
        return n // self.world_size if self.world_size > 1 else n

    def summary(self) -> str:
        return ", ".join(f"bucket {g}: {len(m)} images" for g, m in self.by_group.items())
