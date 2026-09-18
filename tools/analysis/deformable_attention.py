"""
Where the decoder's deformable cross-attention looks, layer by layer. A trained model is run
on validation images with every layer's sampling locations and attention weights captured,
and each ground-truth object is followed through its detection's query (the highest-scoring
query whose final box reaches the IoU threshold, as ``fdr_refinement.py`` does), so the
``object_<k>.png`` figures of the two tools show the same objects.

Written into ``<run>/attention/``:

* ``object_<k>.png``: one object, one panel per layer: the crop, the box the layer starts from,
  the ground truth, and every sampling point of the layer (all heads), coloured by feature
  level and sized by its attention weight.
* ``cells_<k>.png``: the same object, layers as rows and levels as columns, with the level's
  cell grid drawn, each head's points in its own colour, and per panel whether the head's
  points sit in one cell or one bilinear footprint, how alike the features they read are, and
  how alike the map itself is one cell over.
* ``trend.png`` and ``trend.md``: over a sample of images, per object size and layer, the
  attention-weighted density of sampling points in ground-truth box units (where the model
  reads relative to the object), the share of weight per feature level, the share inside the
  box, and how far out the points reach; then per level the box the layer starts from in cells,
  how far apart each head's points of that level land in cells, the weight carried by heads
  whose points all fall within one cell or one bilinear footprint (the coarse-level collapse
  ``min_sample_cells`` floors), the cosine similarity between the features a head's points read
  there, and the map's own similarity one and two cells over (the calibration: a collapse only
  costs something where the map changes from cell to cell).

    python tools/analysis/deformable_attention.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/deformable_attention.py <run> --image 0000001_02999_d_0000005 --objects 8 --num-images 100
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import (  # noqa: E402
    ANCHOR,
    GRID,
    GT,
    INK,
    MUTED,
    SIZE_BUCKETS,
    SURFACE,
    Sample,
    bucket_of,
    draw_box,
    find_decoder,
    load_model,
    pick_checkpoint,
    resolve_image,
)

from src.core import YAMLConfig  # noqa: E402
from src.misc.box_ops import box_iou  # noqa: E402

LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # one per feature level
REACH = 3.0  # the density plots span this many box sizes to either side of the centre


class SamplingRecorder:
    """
    Records what every decoder layer's cross-attention samples: per layer ``(locations
    [B, Q, H, P, 2] normalized to the input, weights [B, Q, H, P] softmaxed per head, level [P])``.
    Wraps the attention core, so the recorded locations are exactly the ones sampled, floors and
    all; with a fine level (``fine_dim`` > 0) its points go through ``_read_fine`` before the
    core sees the rest, and the two are joined back in point order (the fine level first).
    """

    def __init__(self, decoder):
        self.decoder = decoder
        self.records = []
        self.raw = []  # per layer: (raw offsets [B, Q, H, P, 2] as the linear layer outputs them, reference boxes [B, Q, 4])
        self.null = []  # per layer: the weight of every head's null entry [B, Q, H], or None without null_point
        self.reads = []  # per layer, per level: what the points read (see read_level), batch element 0
        self.originals = []
        self.handles = []

    def __enter__(self):
        for layer in self.decoder.layers:
            attn = layer.cross_attn
            original = attn.ms_deformable_attn_core

            pending = []  # the fine level's (shape, locations, weights) of the call in progress

            def wrapped(value, shapes, locations, weights, num_points_list, _original=original, _attn=attn, _p=pending):
                all_shapes, all_locations, all_weights = [tuple(s) for s in shapes], locations, weights
                reads = []
                if _p:
                    fine_shape, fine_locations, fine_weights, fine_value = _p.pop()
                    all_shapes = [fine_shape, *all_shapes]
                    all_locations = torch.cat([fine_locations, locations], dim=-2)
                    all_weights = torch.cat([fine_weights, weights], dim=-1)
                    reads.append(read_level(fine_value[:1], fine_shape, fine_locations[:1], fine=True))
                for level, n in enumerate(num_points_list):
                    start = sum(num_points_list[:level])
                    reads.append(
                        read_level(value[level][:1], all_shapes[len(reads)], locations[:1, :, :, start : start + n])
                    )
                self.reads.append(reads)
                self.records.append(
                    (
                        all_locations.detach().float().cpu(),
                        all_weights.detach().float().cpu(),
                        _attn.point_level.cpu(),
                        all_shapes,
                    )
                )
                return _original(value, shapes, locations, weights, num_points_list)

            attn.ms_deformable_attn_core = wrapped
            self.originals.append((attn, "ms_deformable_attn_core", original))
            if getattr(attn, "fine_dim", 0) > 0:
                original_fine = attn._read_fine

                def wrapped_fine(value, hw, locations, weights, _original=original_fine, _p=pending):
                    _p.append((tuple(hw), locations, weights, value))
                    return _original(value, hw, locations, weights)

                attn._read_fine = wrapped_fine
                self.originals.append((attn, "_read_fine", original_fine))

            def hook(module, args, output, _raw=self.raw, _null=self.null):
                query, reference_points = args[0], args[1]
                raw = module.sampling_offsets(query).reshape(*query.shape[:2], module.num_heads, -1, 2)
                _raw.append((raw.detach().float().cpu(), reference_points[:, :, 0].detach().float().cpu()))
                if getattr(module, "null_point", False):
                    logits = module.attention_weights(query).reshape(*query.shape[:2], module.num_heads, -1)
                    _null.append(logits.softmax(-1)[..., -1].detach().float().cpu())
                else:
                    _null.append(None)

            self.handles.append(attn.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for attn, name, original in self.originals:
            setattr(attn, name, original)
        for handle in self.handles:
            handle.remove()


def _bilinear(value, hw, locations):
    """``value`` [N, C, h*w] sampled at ``locations`` [N, M, 2] in [0, 1], as the attention samples: [N, M, C]."""
    h, w = hw
    grid = (2 * locations - 1).reshape(locations.shape[0], -1, 1, 2)
    out = F.grid_sample(
        value.reshape(-1, value.shape[1], h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return out[..., 0].transpose(1, 2)


def read_level(value, hw, locations, fine=False):
    """
    What every head's points read at one level, batch element 0. ``value`` is ``[1, H, c, h*w]``
    (a raw fine level: ``[1, C, h*w]``, every head sampling the same map), ``locations``
    ``[1, Q, H, P, 2]`` in [0, 1]. Returns, on the CPU:

    * ``cos_within`` [Q, H]: the mean cosine similarity between a head's P samples (1: the
      points read the same feature);
    * ``same_footprint`` [Q, H]: whether all of a head's points fall in the same 2x2 bilinear
      footprint (the same four cells, different weights: the reads are mixtures of one another);
    * ``cos_shift`` [Q, H, P, 2]: the cosine between each point's sample and the sample one and
      two cells to its right (the map's own smoothness: how different a neighbouring cell is).
    """
    _, q, heads, p, _ = locations.shape
    h, w = hw
    loc = locations[0].float()  # [Q, H, P, 2]
    if fine:
        v = value.float()  # [1, C, h*w]

        def sample(l):
            return _bilinear(v, hw, l.reshape(1, -1, 2)).reshape(q, heads, p, -1)

    else:
        v = value[0].float()  # [H, c, h*w]

        def sample(l):
            return (
                _bilinear(v, hw, l.permute(1, 0, 2, 3).reshape(heads, -1, 2))
                .reshape(heads, q, p, -1)
                .permute(1, 0, 2, 3)
            )

    s = sample(loc)  # [Q, H, P, C]
    shift = torch.tensor([1.0 / w, 0.0], device=loc.device)
    s1, s2 = sample(loc + shift), sample(loc + 2 * shift)
    n = F.normalize(s, dim=-1, eps=1e-8)
    gram = n @ n.transpose(-1, -2)  # [Q, H, P, P]
    off = 1 - torch.eye(p, device=gram.device)
    cos_within = (gram * off).sum((-1, -2)) / max(p * (p - 1), 1)
    cell = (
        loc * torch.tensor([w, h], device=loc.device, dtype=loc.dtype) - 0.5
    ).floor()  # the footprint's top-left cell
    same_footprint = (cell == cell[:, :, :1]).all(-1).all(-1)
    cos_shift = torch.stack(
        [F.cosine_similarity(s, s1, dim=-1, eps=1e-8), F.cosine_similarity(s, s2, dim=-1, eps=1e-8)], -1
    )
    return {
        "cos_within": cos_within.detach().cpu(),
        "same_footprint": same_footprint.detach().cpu(),
        "cos_shift": cos_shift.detach().cpu(),
    }


def read_stats(sample, layer, q):
    """
    Per level, weighted by the attention the query's heads put there: the mean cosine between a
    head's samples, the share of weight on heads whose points share one bilinear footprint, and
    the cosine between a sample and the map one and two cells away.
    """
    _, weights, level, _ = sample.attention[layer]
    w = weights[0, q]  # [H, P]
    out = {"cos_within": [], "same_footprint": [], "cos_shift1": [], "cos_shift2": []}
    for lv, read in enumerate(sample.reads[layer]):
        m = level == lv
        wl = w[:, m].sum(-1)  # [H]
        total = float(wl.sum())
        if total <= 0:
            for v in out.values():
                v.append(0.0)
            continue
        wp = w[:, m]  # [H, n]
        out["cos_within"].append(float((wl * read["cos_within"][q]).sum() / total))
        out["same_footprint"].append(float(wl[read["same_footprint"][q]].sum() / total))
        out["cos_shift1"].append(float((wp * read["cos_shift"][q][..., 0]).sum() / total))
        out["cos_shift2"].append(float((wp * read["cos_shift"][q][..., 1]).sum() / total))
    return out


def sample_with_attention(dataset, collate, index, model, decoder, device):
    with SamplingRecorder(decoder) as recorder:
        sample = Sample(dataset, collate, index, model, decoder, device)
    if len(recorder.records) != decoder.num_layers:
        raise RuntimeError(f"recorded {len(recorder.records)} cross-attention calls for {decoder.num_layers} layers")
    sample.attention = recorder.records
    sample.raw_offsets = recorder.raw
    sample.null_weights = recorder.null
    sample.reads = recorder.reads
    return sample


def query_points(sample, layer, q):
    """The layer's sampling points of query ``q``: positions [H*P, 2] in input px, weights [H*P] (sum H), levels [H*P]."""
    locations, weights, level, _ = sample.attention[layer]
    points = locations[0, q] * sample.size  # [H, P, 2]
    heads = points.shape[0]
    return points.reshape(-1, 2), weights[0, q].reshape(-1), level.repeat(heads)


def entering_box(sample, layer, q):
    """The reference box the layer's attention is centred on: the proposal for layer 0, else the previous layer's box."""
    return sample.boxes[0, q] if layer == 0 else sample.boxes[2 + layer - 1, q]


def object_stats(sample, layer, q, g):
    """Per level the share of attention weight, the share inside the ground truth, and the weighted reach in box units and px."""
    points, weights, levels = query_points(sample, layer, q)
    gt = sample.gt[g]
    total = float(weights.sum())
    center = (gt[:2] + gt[2:]) / 2
    size = gt[2:] - gt[:2]
    rel = (points - center) / size  # box units, centre 0, edges at +-0.5
    inside = (rel.abs() <= 0.5).all(-1)
    reach_units = rel.abs().max(-1).values  # Chebyshev distance in box units
    reach_px = (points - center).abs().max(-1).values
    num_levels = int(levels.max()) + 1
    return {
        "level_share": [float(weights[levels == lv].sum() / total) for lv in range(num_levels)],
        "inside": float(weights[inside].sum() / total),
        "within_1": float(weights[reach_units <= 1.0].sum() / total),
        "reach_units": float((weights * reach_units).sum() / total),
        "reach_px": float((weights * reach_px).sum() / total),
        "rel": rel,
        "weights": weights / total,
    }


def cell_stats(sample, layer, q, strides):
    """
    Per level: the half-size of the box the layer starts from in cells of that level, the
    attention-weighted spread (largest Chebyshev distance between a head's points of the level,
    in cells) and the share of the level's weight on heads whose points all lie within one cell.
    """
    locations, weights, level, _ = sample.attention[layer]
    points = locations[0, q] * sample.size  # [H, P, 2] input px
    w = weights[0, q]  # [H, P]
    box = entering_box(sample, layer, q)  # xyxy input px
    half_px = (box[2:] - box[:2]) / 2  # [2]
    half_cells, spread, collapsed, effective = [], [], [], []
    for lv in range(int(level.max()) + 1):
        m = level == lv
        pts = points[:, m]  # [H, n, 2]
        wl = w[:, m].sum(-1)  # [H] the level's weight per head
        prob = w[:, m] / wl.clamp_min(1e-12)[:, None]  # [H, n] the head's weights within the level
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1)  # [H]
        distance = (pts[:, :, None] - pts[:, None]).abs().max(-1).values  # [H, n, n] Chebyshev, px
        s = distance.flatten(1).max(-1).values / strides[lv]  # [H] cells
        total = float(wl.sum())
        half_cells.append(float(half_px.mean() / strides[lv]))
        spread.append(float((wl * s).sum() / total) if total > 0 else 0.0)
        collapsed.append(float(wl[s < 1.0].sum() / total) if total > 0 else 0.0)
        effective.append(float((wl * entropy.exp()).sum() / total) if total > 0 else 0.0)
    return {"half_cells": half_cells, "spread": spread, "collapsed": collapsed, "effective": effective}


INIT_OFFSET = (
    2.5  # the mean Chebyshev magnitude of the initial raw offsets (points at 1, 2, 3, 4 along a head's direction)
)


def offset_records_of(sample, score_threshold):
    """
    One record per detection (a query whose final score reaches ``score_threshold``): its size
    bucket by its final box, and per layer and level the mean Chebyshev magnitude of the raw
    offsets (the linear layer's output, before the box scaling) relative to the initialization.
    The queries need no ground truth: the question is whether the network asks for larger raw
    offsets when the box is small.
    """
    keep = torch.where(sample.scores[-1] >= score_threshold)[0]
    if len(keep) == 0:
        return []
    boxes = sample.boxes[-1, keep]  # xyxy px
    sizes = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp_min(0).sqrt() * sample.orig_scale
    level = sample.attention[0][2]
    num_levels = int(level.max()) + 1
    per_layer = []
    for raw, _ in sample.raw_offsets:
        magnitude = raw[0, keep].abs().max(-1).values  # [n, H, P] Chebyshev, in the unit of the initialization
        per_layer.append(torch.stack([magnitude[:, :, level == lv].mean(dim=(1, 2)) for lv in range(num_levels)], -1))
    gain = torch.stack(per_layer, 1) / INIT_OFFSET  # [n, L, levels]
    return [{"bucket": bucket_of(float(sizes[i])), "gain": gain[i].tolist()} for i in range(len(keep))]


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def object_figure(sample, g, q, strides, path, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    num_layers = len(sample.attention)
    gt = sample.gt[g]
    size = float((gt[2:] - gt[:2]).max())
    margin = max(REACH * size / 2, 24.0)
    h, w = sample.image.shape[:2]
    cx, cy = float((gt[0] + gt[2]) / 2), float((gt[1] + gt[3]) / 2)
    x1, y1 = max(0, int(cx - margin)), max(0, int(cy - margin))
    x2, y2 = min(w, int(cx + margin) + 1), min(h, int(cy + margin) + 1)

    label = names.get(int(sample.gt_labels[g]), str(int(sample.gt_labels[g])))
    final_iou = float(box_iou(sample.boxes[-1, q][None], gt[None])[0][0, 0])
    header = (
        f"{sample.name}: query {q}, {label}, {sample.gt_size(g):.0f}px, final IoU {final_iou:.2f}, "
        f"score {sample.scores[-1, q]:.2f}. Marker area follows the attention weight; all heads shown."
    )
    fig, axes = plt.subplots(1, num_layers, figsize=(4.6 * num_layers, 5.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for layer, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(np.clip(sample.image[y1:y2, x1:x2], 0, 1), extent=(x1, x2, y2, y1), interpolation="nearest")
        points, weights, levels = query_points(sample, layer, q)
        for lv in range(int(levels.max()) + 1):
            m = levels == lv
            ax.scatter(
                points[m, 0],
                points[m, 1],
                s=4 + 300 * weights[m],  # weights sum to the head count over all points
                c=LEVEL_COLORS[lv % len(LEVEL_COLORS)],
                alpha=0.75,
                linewidths=0,
            )
        draw_box(ax, entering_box(sample, layer, q), ANCHOR, "--", 1.2)
        draw_box(ax, gt, GT, ":", 1.4)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        st = object_stats(sample, layer, q, g)
        shares = ", ".join(f"s{strides[lv]} {100 * s:.0f}%" for lv, s in enumerate(st["level_share"]))
        ax.set_title(
            (header if layer == 0 else "")
            + f"\nlayer {layer}: {shares}\ninside box {100 * st['inside']:.0f}%, within 1 box {100 * st['within_1']:.0f}%, reach {st['reach_units']:.2f} box ({st['reach_px']:.1f}px)",
            fontsize=8,
            color=INK,
            loc="left",
        )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=LEVEL_COLORS[lv], label=f"stride {s}")
        for lv, s in enumerate(strides)
    ]
    handles += [
        Line2D([], [], color=ANCHOR, linestyle="--", label="box entering the layer"),
        Line2D([], [], color=GT, linestyle=":", label="ground truth"),
    ]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left", labelcolor=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


HEAD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b52c9", "#008b8b", "#9c5b2f"]


def cells_figure(sample, g, q, strides, path, names):
    """
    One object, layers as rows and levels as columns: the crop with the level's cell grid drawn
    over it, every head's points of that level (one colour per head, area by attention weight),
    the ground truth, and per panel the level's numbers: the head-weighted share of weight on
    heads whose points share one bilinear footprint, the cosine between a head's samples, and the
    map's own similarity one cell over. Whether the points of a head sit in one cell is then
    visible, and whether that matters is next to it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    num_layers, num_levels = len(sample.attention), len(strides)
    gt = sample.gt[g]
    size = float((gt[2:] - gt[:2]).max())
    h, w = sample.image.shape[:2]
    cx, cy = float((gt[0] + gt[2]) / 2), float((gt[1] + gt[3]) / 2)
    label = names.get(int(sample.gt_labels[g]), str(int(sample.gt_labels[g])))
    fig, axes = plt.subplots(
        num_layers, num_levels, figsize=(3.4 * num_levels, 3.6 * num_layers), constrained_layout=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for layer in range(num_layers):
        locations, weights, level, _ = sample.attention[layer]
        points = locations[0, q] * sample.size  # [H, P, 2] input px
        wq = weights[0, q]  # [H, P]
        reads, cells = read_stats(sample, layer, q), cell_stats(sample, layer, q, strides)
        for lv, ax in enumerate(axes[layer]):
            stride = strides[lv]
            margin = max(size, 2.5 * stride, 12.0)
            x1, y1 = max(0, int(cx - margin)), max(0, int(cy - margin))
            x2, y2 = min(w, int(cx + margin) + 1), min(h, int(cy + margin) + 1)
            ax.imshow(np.clip(sample.image[y1:y2, x1:x2], 0, 1), extent=(x1, x2, y2, y1), interpolation="nearest")
            for x in range(int(x1 // stride) * stride, x2 + stride, stride):
                ax.axvline(x, color=GRID, linewidth=0.6, alpha=0.8)
            for y in range(int(y1 // stride) * stride, y2 + stride, stride):
                ax.axhline(y, color=GRID, linewidth=0.6, alpha=0.8)
            m = level == lv
            for head in range(points.shape[0]):
                ax.scatter(
                    points[head, m, 0],
                    points[head, m, 1],
                    s=6 + 250 * wq[head, m],
                    c=HEAD_COLORS[head % len(HEAD_COLORS)],
                    alpha=0.8,
                    linewidths=0.4,
                    edgecolors="white",
                )
            draw_box(ax, gt, GT, ":", 1.4)
            ax.set_xlim(x1, x2)
            ax.set_ylim(y2, y1)
            ax.set_xticks([])
            ax.set_yticks([])
            share = float(wq[:, m].sum() / wq.sum())
            ax.set_title(
                f"layer {layer}, stride {stride}: {share:.0%} of the weight\n"
                f"one cell {cells['collapsed'][lv]:.0%}, one footprint {reads['same_footprint'][lv]:.0%}\n"
                f"cos within head {reads['cos_within'][lv]:.2f}, map 1 cell over {reads['cos_shift1'][lv]:.2f}",
                fontsize=7,
                color=INK,
                loc="left",
            )
    fig.suptitle(
        f"{sample.name}: query {q}, {label}, {sample.gt_size(g):.0f}px. Grid: the level's cells; one colour per head, "
        f"marker area by attention weight; dotted: the ground truth",
        fontsize=8,
        color=INK,
        x=0.01,
        ha="left",
    )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=HEAD_COLORS[i], label=f"head {i}")
        for i in range(points.shape[0])
    ]
    axes[0][0].legend(handles=handles, fontsize=6, frameon=False, loc="lower left", labelcolor=INK, ncol=2)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def trend_figure(records, num_layers, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    buckets = [b[0] for b in SIZE_BUCKETS if any(r["bucket"] == b[0] for r in records)]
    bins = np.linspace(-REACH, REACH, 61)
    fig, axes = plt.subplots(
        len(buckets), num_layers, figsize=(3.4 * num_layers, 3.4 * len(buckets)), constrained_layout=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for row, name in enumerate(buckets):
        rows = [r for r in records if r["bucket"] == name]
        for layer in range(num_layers):
            ax = axes[row, layer]
            rel = np.concatenate([r["rel"][layer] for r in rows])
            weights = np.concatenate([r["weights"][layer] for r in rows]) / len(rows)
            hist, _, _ = np.histogram2d(rel[:, 0], rel[:, 1], bins=[bins, bins], weights=weights)
            ax.imshow(hist.T, extent=(-REACH, REACH, REACH, -REACH), cmap="Blues", interpolation="nearest", vmin=0)
            ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor=GT, linestyle=":", linewidth=1.2))
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.tick_params(colors=MUTED, labelsize=7, length=0)
            for side in ax.spines.values():
                side.set_color(GRID)
            inside = np.mean([r["inside"][layer] for r in rows])
            ax.set_title(
                f"{name}, layer {layer}: {100 * inside:.0f}% inside (n={len(rows)})", fontsize=8, color=INK, loc="left"
            )
            if layer == 0:
                ax.set_ylabel("box heights from the centre", fontsize=7, color=MUTED)
            if row == len(buckets) - 1:
                ax.set_xlabel("box widths from the centre", fontsize=7, color=MUTED)
    fig.suptitle(
        "Attention-weighted density of sampling points in ground-truth box units (dotted: the box)",
        fontsize=9,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trend_table(records, num_layers, strides):
    lines = []
    names = [b[0] for b in SIZE_BUCKETS] + ["all"]
    for key, title, fmt in (
        ("inside", "Share of attention weight inside the ground-truth box", "{:.0%}"),
        ("within_1", "Share within one box size of the centre (Chebyshev)", "{:.0%}"),
        ("reach_units", "Weighted mean reach, in box units", "{:.2f}"),
        ("reach_px", "Weighted mean reach, input pixels", "{:.1f}"),
    ):
        lines.append(f"### {title}\n")
        lines.append("| size | n | " + " | ".join(f"layer {i}" for i in range(num_layers)) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in range(num_layers)) + " |")
        for name in names:
            rows = records if name == "all" else [r for r in records if r["bucket"] == name]
            if not rows:
                continue
            values = np.array([r[key] for r in rows]).mean(0)
            lines.append(f"| {name} | {len(rows)} | " + " | ".join(fmt.format(v) for v in values) + " |")
        lines.append("")
    lines.append("### Share of attention weight per feature level\n")
    lines.append("| size | n | layer | " + " | ".join(f"stride {s}" for s in strides) + " |")
    lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in strides) + " |")
    for name in names:
        rows = records if name == "all" else [r for r in records if r["bucket"] == name]
        if not rows:
            continue
        for layer in range(num_layers):
            values = np.array([r["level_share"][layer] for r in rows]).mean(0)
            lines.append(f"| {name} | {len(rows)} | {layer} | " + " | ".join(f"{v:.0%}" for v in values) + " |")
    lines.append("")
    for key, title, fmt in (
        ("half_cells", "Half-size of the box the layer starts from, in cells of the level", "{:.2f}"),
        ("spread", "Weighted spread of a head's points of the level, in cells (largest Chebyshev distance)", "{:.2f}"),
        ("collapsed", "Share of the level's weight on heads whose points all lie within one cell", "{:.0%}"),
        (
            "effective",
            "Effective points of a head at the level (exp of the entropy of its weights there, 1 to the level's points)",
            "{:.2f}",
        ),
        (
            "same_footprint",
            "Share of the level's weight on heads whose points all fall in the same 2x2 bilinear footprint (the same four cells)",
            "{:.0%}",
        ),
        (
            "cos_within",
            "Mean cosine similarity between the features a head's points read at the level (1: the same feature)",
            "{:.2f}",
        ),
        (
            "cos_shift1",
            "Cosine similarity between a point's feature and the map one cell to its right (the map's own smoothness)",
            "{:.2f}",
        ),
        ("cos_shift2", "The same, two cells to the right", "{:.2f}"),
    ):
        lines.append(f"### {title}\n")
        lines.append("| size | n | layer | " + " | ".join(f"stride {s}" for s in strides) + " |")
        lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in strides) + " |")
        for name in names:
            rows = records if name == "all" else [r for r in records if r["bucket"] == name]
            if not rows:
                continue
            for layer in range(num_layers):
                values = np.array([r[key][layer] for r in rows]).mean(0)
                lines.append(f"| {name} | {len(rows)} | {layer} | " + " | ".join(fmt.format(v) for v in values) + " |")
        lines.append("")
    return "\n".join(lines)


def null_table(records, num_layers):
    """The weight on the null entry per object size and layer (mean over heads); empty without the option."""
    if not records or any(np.isnan(records[0]["null"])):
        return ""
    names = [b[0] for b in SIZE_BUCKETS] + ["all"]
    lines = [
        "### Weight on the null entry (a head abstaining), mean over heads\n",
        "| size | n | " + " | ".join(f"layer {i}" for i in range(num_layers)) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in range(num_layers)) + " |",
    ]
    for name in names:
        rows = records if name == "all" else [r for r in records if r["bucket"] == name]
        if not rows:
            continue
        values = np.array([r["null"] for r in rows]).mean(0)
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(f"{v:.1%}" for v in values) + " |")
    lines.append("")
    return "\n".join(lines)


def offset_table(records, num_layers, strides, score_threshold):
    names = [b[0] for b in SIZE_BUCKETS] + ["all"]
    lines = [
        f"### Raw offset magnitude relative to the initialization, detections scoring at least {score_threshold} (by their own box)\n",
        "| size | n | layer | " + " | ".join(f"stride {s}" for s in strides) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in strides) + " |",
    ]
    for name in names:
        rows = records if name == "all" else [r for r in records if r["bucket"] == name]
        if not rows:
            continue
        for layer in range(num_layers):
            values = np.array([r["gain"][layer] for r in rows]).mean(0)
            lines.append(f"| {name} | {len(rows)} | {layer} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines.append("")
    return "\n".join(lines)


def records_of(sample, iou_threshold, strides):
    records = []
    num_layers = len(sample.attention)
    for g, q, _ in sample.match(iou_threshold):
        stats = [object_stats(sample, layer, q, g) for layer in range(num_layers)]
        cells = [cell_stats(sample, layer, q, strides) for layer in range(num_layers)]
        reads = [read_stats(sample, layer, q) for layer in range(num_layers)]
        null = [float(n[0, q].mean()) if n is not None else float("nan") for n in sample.null_weights]
        records.append(
            {
                "bucket": bucket_of(sample.gt_size(g)),
                "null": null,
                "half_cells": [c["half_cells"] for c in cells],
                "spread": [c["spread"] for c in cells],
                "collapsed": [c["collapsed"] for c in cells],
                "effective": [c["effective"] for c in cells],
                "cos_within": [r["cos_within"] for r in reads],
                "same_footprint": [r["same_footprint"] for r in reads],
                "cos_shift1": [r["cos_shift1"] for r in reads],
                "cos_shift2": [r["cos_shift2"] for r in reads],
                "inside": [s["inside"] for s in stats],
                "within_1": [s["within_1"] for s in stats],
                "reach_units": [s["reach_units"] for s in stats],
                "reach_px": [s["reach_px"] for s in stats],
                "level_share": [s["level_share"] for s in stats],
                "rel": [s["rel"].numpy() for s in stats],
                "weights": [s["weights"].numpy() for s in stats],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output directory of train.py (config.yml and checkpoints)")
    parser.add_argument("--checkpoint", help="checkpoint file in the run (default: best_stg2, best_stg1, then last)")
    parser.add_argument(
        "--image",
        help="validation image for the object figures: a row index or a unique part of the image id (default: row 0)",
    )
    parser.add_argument(
        "--objects", type=int, default=6, help="objects of that image to draw, spread over sizes (default 6)"
    )
    parser.add_argument(
        "--num-images", type=int, default=50, help="images sampled for the trend statistics (default 50, 0 to skip)"
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="final IoU a query needs to be a candidate for an object; the highest-scoring one is taken (default 0.5)",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.3,
        help="score a query needs to count as a detection in the raw offset table (default 0.3)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", help="output directory (default <run>/attention)")
    parser.add_argument(
        "--min-sample-cells",
        type=float,
        default=None,
        help="override every cross-attention layer's min_sample_cells (an inference-time geometric floor); "
        "with the model's trained value use it to isolate whether the floor spreads points out of one cell",
    )
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.run, "attention")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    print(f"checkpoint {checkpoint}")

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    decoder = find_decoder(model)
    if args.min_sample_cells is not None:
        for layer in decoder.layers:
            layer.cross_attn.min_sample_cells = args.min_sample_cells
        print(f"override: min_sample_cells set to {args.min_sample_cells} on every layer")
    strides = list(model.decoder.feat_strides)
    if getattr(model.decoder, "fine_channels", 0) > 0:  # the fine level, one stride finer, is the first value level
        strides = [strides[0] // 2, *strides]
    loader = cfg.val_dataloader
    dataset, collate = loader.dataset, loader.collate_fn
    names = dict(getattr(dataset, "CATEGORIES", []))
    attn = decoder.layers[0].cross_attn
    print(
        f"decoder: {decoder.num_layers} layers, {attn.num_heads} heads, points per level {attn.num_points_list}, strides {strides}, offset_scale {attn.offset_scale}, min_sample_cells {attn.min_sample_cells}"
    )

    index = resolve_image(dataset, args.image)
    sample = sample_with_attention(dataset, collate, index, model, decoder, args.device)
    matched = sorted(sample.match(args.match_iou), key=lambda m: sample.gt_size(m[0]))
    picks = (
        np.unique(np.linspace(0, len(matched) - 1, min(args.objects, len(matched))).round().astype(int))
        if matched
        else []
    )
    for k, i in enumerate(picks):
        g, q, iou = matched[i]
        object_figure(sample, g, q, strides, os.path.join(out_dir, f"object_{k}.png"), names)
        cells_figure(sample, g, q, strides, os.path.join(out_dir, f"cells_{k}.png"), names)
        print(
            f"  object_{k}.png: gt {g} ({names.get(int(sample.gt_labels[g]), '?')}, {sample.gt_size(g):.0f}px), query {q}, final IoU {iou:.2f}"
        )

    if args.num_images > 0:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), min(args.num_images, len(dataset)))
        records, offset_records = [], []
        for i, idx in enumerate(indices):
            s = sample if idx == index else sample_with_attention(dataset, collate, idx, model, decoder, args.device)
            records += records_of(s, args.match_iou, strides)
            offset_records += offset_records_of(s, args.score)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(indices)} images, {len(records)} objects")
        if records:
            trend_figure(records, decoder.num_layers, os.path.join(out_dir, "trend.png"))
            text = (
                f"# Deformable cross-attention of `{os.path.relpath(checkpoint).replace(os.sep, '/')}`\n\n"
                f"{len(records)} ground-truth objects of {len(indices)} validation images, each followed through its detection's query "
                f"(the highest-scoring query whose final box reaches IoU {args.match_iou}). Every layer samples "
                f"{attn.num_heads} heads x {sum(attn.num_points_list)} points; weights are softmaxed per head and pooled over heads. "
                f"Reach is the Chebyshev distance of a sampling point from the box centre, in box sizes or input pixels. "
                f"The per-level tables measure the box a layer starts from (the proposal for layer 0, else the previous "
                f"layer's box) and each head's {attn.num_points_list[0]} points of a level in cells of that level; points "
                f"within one cell read the same bilinear neighbourhood. The raw offset table looks at the sampling offsets "
                f"before they are scaled by the box (a Chebyshev magnitude of 1.0 is the initialization's star of points at "
                f"1, 2, 3, 4), over every detection rather than matched objects. Sizes are square-root areas in original pixels."
                f"\n\n![trend](trend.png)\n\n"
                + trend_table(records, decoder.num_layers, strides)
                + "\n"
                + offset_table(offset_records, decoder.num_layers, strides, args.score)
                + null_table(records, decoder.num_layers)
            )
            with open(os.path.join(out_dir, "trend.md"), "w", encoding="utf-8") as f:
                f.write(text)
            print(text)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
