"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..core import register
from ..misc.box_ops import box_cxcywh_to_xyxy, elementwise_generalized_box_iou, gaussian_box_similarity

__all__ = ["FlatMatches", "HungarianMatcher", "PaddedTargets", "padded_targets", "topk_matching", "topk_matching_flat"]

# the assignments of a batch run in parallel: scipy releases the GIL in linear_sum_assignment
_ASSIGN_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


class PaddedTargets(NamedTuple):
    """A batch's ground truths padded to ``[B, M]``, with the counts the padding hides, on the host."""

    labels: Tensor  # [B, M], zeros past each image's count
    boxes: Tensor  # [B, M, 4]
    gt_valid: Tensor  # [B, M]
    q_valid: Tensor  # [B, Q], the real (unpadded) queries of every image
    num_gt: list[int]  # ground truths per image
    num_q: list[int]  # real queries per image


class FlatMatches(NamedTuple):
    """
    The matches of one or more prediction sets as flat index tensors, sorted by set and image:
    pair ``k`` matches query ``query_idx[k]`` of image ``batch_idx[k]`` to that image's ground
    truth ``target_idx[k]``. ``counts`` is the pairs per set, on the host.
    """

    batch_idx: Tensor
    query_idx: Tensor
    target_idx: Tensor
    counts: list[int]


@torch.no_grad()
def padded_targets(targets, num_queries, batch_queries_num=None) -> PaddedTargets:
    """
    A batch's targets as padded tensors: labels ``[B, M]`` and boxes ``[B, M, 4]`` (zeros past
    each image's count), the ground truths' validity ``[B, M]`` and the queries' ``[B, Q]``
    (every query, or the first ``batch_queries_num[i]`` of image ``i``). ``M`` is 0 for a batch
    without ground truths. Two host-to-device copies, no sync.
    """
    device = targets[0]["boxes"].device
    num_gt = [t["boxes"].shape[0] for t in targets]
    m = max(num_gt)
    if m:
        labels = pad_sequence([t["labels"] for t in targets], batch_first=True)
        boxes = pad_sequence([t["boxes"] for t in targets], batch_first=True)
    else:
        labels = torch.zeros((len(targets), 0), dtype=torch.long, device=device)
        boxes = torch.zeros((len(targets), 0, 4), dtype=targets[0]["boxes"].dtype, device=device)
    num_q = [num_queries] * len(targets) if batch_queries_num is None else list(batch_queries_num)
    counts = torch.tensor([num_gt, num_q]).to(device, non_blocking=True)  # [2, B]
    gt_valid = torch.arange(m, device=device)[None, :] < counts[0][:, None]
    q_valid = torch.arange(num_queries, device=device)[None, :] < counts[1][:, None]
    return PaddedTargets(labels, boxes, gt_valid, q_valid, num_gt, num_q)


def _batch_idx(lengths: list[int], device) -> Tensor:
    """Image index of every pair, ``lengths`` pairs per image in order, built on the host."""
    return (
        torch.arange(len(lengths))
        .repeat_interleave(torch.tensor(lengths, dtype=torch.long))
        .to(device, non_blocking=True)
    )


def _split_flat(flat: FlatMatches, lengths: list[list[int]]):
    """Per-set, per-image ``(pred_idx, target_idx)`` lists from flat matches (``lengths[set][image]`` pairs)."""
    all_lengths = [n for lens in lengths for n in lens]
    queries, targets = flat.query_idx.split(all_lengths), flat.target_idx.split(all_lengths)
    indices = list(zip(queries, targets))
    b = len(lengths[0])
    return [indices[k * b : (k + 1) * b] for k in range(len(lengths))]


@torch.no_grad()
def topk_matching_flat(outputs, padded: PaddedTargets, k) -> FlatMatches:
    """
    One-to-many assignment for a prediction set whose predictions are candidates rather than
    detections (the encoder's queries): every ground truth takes its ``k`` most similar real
    queries by ``box_ops.gaussian_box_similarity`` of the boxes, and a query wanted by several
    ground truths goes to the one it is most similar to, so a query has at most one target and a
    ground truth at most ``k`` queries. The batch is assigned in one go: padded queries and
    ground truths take part with similarity -inf and are dropped at the end. One host sync.
    """
    boxes = outputs["pred_boxes"]  # [B, Q, 4]
    b, q = boxes.shape[:2]
    empty = torch.zeros(0, dtype=torch.long, device=boxes.device)
    if padded.boxes.shape[1] == 0 or q == 0:
        return FlatMatches(empty, empty, empty, [0])
    sim = gaussian_box_similarity(boxes[:, :, None, :], padded.boxes[:, None, :, :])  # [B, Q, M]
    sim = sim.masked_fill(~(padded.q_valid[:, :, None] & padded.gt_valid[:, None, :]), float("-inf"))
    wanted = torch.zeros_like(sim, dtype=torch.bool).scatter_(1, sim.topk(min(k, q), dim=1).indices, True)
    wanted &= padded.gt_valid[:, None, :]  # a padded ground truth wants nothing
    best = sim.masked_fill(~wanted, -1.0).max(dim=2)  # each query's best ground truth among those wanting it
    pairs = (best.values >= 0).nonzero()  # [K, 2] (image, query), sorted by image; the sync
    batch_idx, src = pairs.unbind(1)
    return FlatMatches(batch_idx, src, best.indices[batch_idx, src], [pairs.shape[0]])


@torch.no_grad()
def topk_matching(outputs, targets, k, batch_queries_num=None):
    """``topk_matching_flat`` as per-image ``(pred_idx, target_idx)`` pairs, as ``HungarianMatcher.forward`` returns them."""
    padded = padded_targets(targets, outputs["pred_boxes"].shape[1], batch_queries_num)
    flat = topk_matching_flat(outputs, padded, k)
    lengths = torch.bincount(flat.batch_idx, minlength=len(targets)).tolist()
    return _split_flat(flat, [lengths])[0]


@register()
class HungarianMatcher(nn.Module):
    """
    One-to-one assignment of predictions to ground-truth boxes, per image, by the Hungarian
    algorithm on a cost of ``cost_class * class + cost_bbox * L1 + cost_giou * (-GIoU) +
    cost_gaussian * (1 - Gaussian similarity)`` (``weight_dict``; a missing or zero weight
    skips the term). The class cost is the focal-style cost of the target class when
    ``use_focal_loss`` is set (the config's top-level flag), else ``-softmax probability``. The
    Gaussian term (``box_ops.gaussian_box_similarity``) still tells apart candidates that do not
    overlap a tiny ground truth, where GIoU saturates and the normalized L1 is negligible.
    Predictions left unmatched are background.
    """

    __share__ = ["use_focal_loss"]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict.get("cost_class", 0)
        self.cost_bbox = weight_dict.get("cost_bbox", 0)
        self.cost_giou = weight_dict.get("cost_giou", 0)
        self.cost_gaussian = weight_dict.get("cost_gaussian", 0)
        assert any((self.cost_class, self.cost_bbox, self.cost_giou, self.cost_gaussian)), "all costs cant be 0"

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def _cost(self, logits, boxes, tgt_ids, tgt_bbox):
        """
        The cost matrices ``[B, Q, M]`` of a batch's queries (``logits [B, Q, C]``, ``boxes
        [B, Q, 4]``) against its padded ground truths (``tgt_ids [B, M]``, ``tgt_bbox [B, M, 4]``);
        the entries of padded queries and ground truths are meaningless and left to the caller.
        """
        prob = F.sigmoid(logits) if self.use_focal_loss else logits.softmax(-1)
        out_prob = prob.gather(-1, tgt_ids[:, None, :].expand(-1, logits.shape[1], -1))  # the target class's
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        cost = self.cost_class * cost_class + self.cost_bbox * torch.cdist(boxes, tgt_bbox, p=1)
        if self.cost_giou:
            giou = elementwise_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes)[:, :, None, :], box_cxcywh_to_xyxy(tgt_bbox)[:, None, :, :]
            )
            cost = cost - self.cost_giou * giou
        if self.cost_gaussian:
            cost = cost + self.cost_gaussian * (
                1 - gaussian_box_similarity(boxes[:, :, None, :], tgt_bbox[:, None, :, :])
            )
        return cost

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets, batch_queries_num=None):
        """
        Args:
            outputs: ``pred_logits`` ``[B, Q, C]`` and ``pred_boxes`` ``[B, Q, 4]`` (normalized cxcywh).
            targets: one dict per image with ``labels`` ``[N_i]`` and ``boxes`` ``[N_i, 4]``.
            batch_queries_num: the real query count of every image; the queries past it are
                padding and never matched.

        Returns:
            ``{"indices": [(pred_idx, target_idx), ...]}``, one pair of int64 index tensors per
            image on the predictions' device, each of length ``min(Q_i, N_i)``.
        """
        return {"indices": self.match_sets([outputs], targets, batch_queries_num)[0]}

    @torch.no_grad()
    def match_sets(self, outputs_list, targets, batch_queries_num=None):
        """``match_sets_flat`` as one list of per-image ``(pred_idx, target_idx)`` pairs per set."""
        padded = padded_targets(targets, outputs_list[0]["pred_logits"].shape[1], batch_queries_num)
        flat, lengths = self.match_sets_flat(outputs_list, padded)
        return _split_flat(flat, lengths)

    @torch.no_grad()
    def match_sets_flat(self, outputs_list, padded: PaddedTargets) -> tuple[FlatMatches, list[list[int]]]:
        """
        The matching of several prediction sets (each as in ``forward``, all with the same number
        of queries) against the same padded targets in one go: the costs are computed on the
        device in one call over every set and image (the sets stacked along the batch), the real
        sub-matrices reach the host in one copy, the assignments run in parallel threads and the
        indices go back in one copy. Returns the matches, flat and sorted by set and image, and
        the pairs per set and image. One host sync (the index of the real entries) besides the copy.
        """
        device = outputs_list[0]["pred_logits"].device
        b, q = outputs_list[0]["pred_logits"].shape[:2]
        m = padded.boxes.shape[1]
        s = len(outputs_list)
        assert all(o["pred_logits"].shape[1] == q for o in outputs_list), "every set has the queries of the first"
        real = (padded.q_valid[:, :, None] & padded.gt_valid[:, None, :]).flatten().nonzero().squeeze(1)  # [K]
        logits = torch.cat([o["pred_logits"] for o in outputs_list])  # [S * B, Q, C]
        boxes = torch.cat([o["pred_boxes"] for o in outputs_list])
        cost = self._cost(logits, boxes, padded.labels.repeat(s, 1), padded.boxes.repeat(s, 1, 1))  # [S * B, Q, M]
        # set by set, image by image, each sub-matrix row-major: the same entries in every set
        real = (real[None, :] + torch.arange(s, device=device)[:, None] * (b * q * m)).flatten()
        shapes = [(nq, ng) for _ in outputs_list for nq, ng in zip(padded.num_q, padded.num_gt)]
        flat = torch.nan_to_num(cost.flatten()[real], nan=1.0).cpu()  # the one device-to-host copy
        mats = [x.view(shape).numpy() for x, shape in zip(flat.split([nq * ng for nq, ng in shapes]), shapes)]
        pairs = list(_ASSIGN_POOL.map(linear_sum_assignment, mats))

        lengths = [[len(i) for i, _ in pairs[k * b : (k + 1) * b]] for k in range(s)]
        packed = torch.from_numpy(np.concatenate([np.stack([i, j]) for i, j in pairs], axis=1)).to(
            device, non_blocking=True
        )
        batch_idx = torch.cat([_batch_idx(lens, device) for lens in lengths])
        return FlatMatches(batch_idx, packed[0], packed[1], [sum(lens) for lens in lengths]), lengths
