"""
Where a detector loses on the validation set. One pass of a run's checkpoint caches every
query's box and class scores, the encoder proposals and the ground truth of every image
(``<run>/probe/val_predictions.pt``); the probes below then work on the cache and write
``<run>/probe/report.md`` with their figures.

* recall ladder: per object size and per class, how many ground-truth boxes survive each
  stage (an encoder proposal near the box, a decoder query near it, a same-class score, the
  postprocessor's top-k, the greedy matching, a score threshold)
* false positives at a score threshold: duplicates, class confusions, localization errors,
  background, and per detection size
* class confusion between the classes
* score quality: precision per score bin, and what AP would be if the scores ranked
  detections by their IoU (the ranking headroom)
* density and position: recall against the number of boxes in the image, against the box's
  position in the frame and against how crowded its neighbourhood is
* levers scored with the real evaluator from the cache: class-aware NMS, clamping boxes to the
  image, dropping low scores

    python tools/analysis/probe_val.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/probe_val.py <run> --checkpoint best_stg1.pth --score 0.3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import SIZE_BUCKETS, bucket_of, find_decoder, load_model, pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.data.dataset.visdrone_eval import detections_in_ignore_regions  # noqa: E402
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e6e5e1"


# --------------------------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def collect(cfg, model, decoder, device, evaluator):
    """Every validation image's queries, proposals and ground truth, boxes in input pixels."""
    captured = {}

    def hook(module, args, kwargs, output):
        captured["proposal"] = torch.sigmoid(args[1])

    handle = decoder.register_forward_hook(hook, with_kwargs=True)
    loader = cfg.val_dataloader
    images_out, seen = [], 0
    try:
        for images, targets in loader:
            out = model(images.to(device))
            scores = out["pred_logits"].float().sigmoid().cpu()  # [B, Q, C]
            boxes = box_cxcywh_to_xyxy(out["pred_boxes"].float().cpu())  # normalized to the padded input
            proposals = box_cxcywh_to_xyxy(captured["proposal"].float().cpu())
            for b, t in enumerate(targets):
                padded = t.get("padded_size", t["orig_size"]).float()
                resized = t.get("resized_size", t["orig_size"]).float()
                scale = padded.repeat(2)
                to_orig = (t["orig_size"].float() / resized).repeat(2)  # input px -> original px
                image_id = int(t["image_id"])
                # VisDrone marks ignore regions; other evaluators (AI-TOD, COCO) have none
                ignore = getattr(evaluator, "ignore_regions", {}).get(image_id)
                images_out.append(
                    {
                        "image_id": image_id,
                        "input_size": padded,  # (w, h) the network saw
                        "content_size": resized,  # (w, h) of the image inside it
                        "to_orig": to_orig,
                        "gt": t["boxes"].as_subclass(torch.Tensor).float(),
                        "labels": t["labels"].clone(),
                        "ignore": ignore / to_orig if ignore is not None else torch.zeros(0, 4),
                        "boxes": boxes[b] * scale,
                        "scores": scores[b],
                        "proposals": proposals[b] * scale,
                    }
                )
            seen += len(targets)
            if seen % 160 == 0:
                print(f"  {seen} images")
    finally:
        handle.remove()
    return images_out


# --------------------------------------------------------------------------------------------
# detections, matching
# --------------------------------------------------------------------------------------------


def postprocess(rec, top_k, num_classes_offset=0):
    """The postprocessor's detections: the top-k (query, class) pairs, then the evaluator's ignore-region filter."""
    scores = rec["scores"]
    q, c = scores.shape
    flat_scores, index = scores.flatten().topk(min(top_k, q * c))
    query, label = index // c, index % c
    boxes = rec["boxes"][query]
    keep = torch.ones(len(boxes), dtype=torch.bool)
    if len(rec["ignore"]):
        keep = ~detections_in_ignore_regions(boxes, rec["ignore"], 0.5)
    return {
        "boxes": boxes[keep],
        "scores": flat_scores[keep],
        "labels": label[keep],
        "query": query[keep],
        "in_ignore": int((~keep).sum()),
    }


def greedy_match(det, gt, gt_labels, iou_threshold):
    """
    COCO-style matching at one IoU threshold: detections in score order, each takes the
    unmatched same-class ground truth it overlaps most (at least the threshold). Returns per
    detection the matched ground-truth index (-1: none), its best IoU with a same-class box and
    with any box, and per ground truth the detection that matched it (-1: none).
    """
    d = len(det["boxes"])
    matched = torch.full((d,), -1, dtype=torch.long)
    gt_of = torch.full((len(gt),), -1, dtype=torch.long)
    if d == 0 or len(gt) == 0:
        return matched, torch.zeros(d), torch.zeros(d), gt_of
    iou, _ = box_iou(det["boxes"], gt)  # [D, G]
    same = det["labels"][:, None] == gt_labels[None, :]
    iou_same = torch.where(same, iou, torch.zeros_like(iou))
    order = det["scores"].argsort(descending=True)
    taken = torch.zeros(len(gt), dtype=torch.bool)
    iou_np, same_np, taken_np = iou.numpy(), same.numpy(), taken.numpy()
    for i in order.tolist():
        cand = iou_np[i] * same_np[i]
        cand[taken_np] = 0
        g = int(cand.argmax())
        if cand[g] >= iou_threshold:
            matched[i] = g
            taken_np[g] = True
            gt_of[g] = i
    return matched, iou_same.max(1).values, iou.max(1).values, gt_of


# --------------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------------


def gt_sizes(rec):
    """Square-root areas of the ground-truth boxes in original pixels."""
    gt = rec["gt"]
    return ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).sqrt() * rec["to_orig"][0]


def ladder(records, dets, matches, iou_threshold, score_threshold, names):
    """Per size and per class, the share of ground-truth boxes that survive each stage."""
    stages = [
        "proposal near",
        "query near",
        "same class, score>=0.05",
        "in top-k",
        "matched (any score)",
        f"matched, score>={score_threshold}",
    ]
    rows = []  # (bucket, class, stage flags)
    for rec, det, (_, _, _, gt_of) in zip(records, dets, matches):
        gt, labels = rec["gt"], rec["labels"]
        if len(gt) == 0:
            continue
        prop_iou = box_iou(gt, rec["proposals"])[0].max(1).values
        query_iou, _ = box_iou(gt, rec["boxes"])  # [G, Q]
        near = query_iou >= iou_threshold
        class_scores = rec["scores"][:, labels].T  # [G, Q]
        best_same = torch.where(near, class_scores, torch.full_like(class_scores, -1.0)).max(1)
        top_k_pairs = set(zip(det["query"].tolist(), det["labels"].tolist()))
        in_topk = [
            (int(best_same.indices[g]), int(labels[g])) in top_k_pairs and best_same.values[g] >= 0
            for g in range(len(gt))
        ]
        matched_any = gt_of >= 0
        matched_thr = matched_any & (det["scores"][gt_of.clamp(min=0)] >= score_threshold)
        sizes = gt_sizes(rec)
        for g in range(len(gt)):
            flags = [
                bool(prop_iou[g] >= iou_threshold),
                bool(near[g].any()),
                bool(best_same.values[g] >= 0.05),
                in_topk[g],
                bool(matched_any[g]),
                bool(matched_thr[g]),
            ]
            rows.append((bucket_of(float(sizes[g])), names.get(int(labels[g]), str(int(labels[g]))), flags))
    return stages, rows


def ladder_table(stages, rows, key):
    groups = [b[0] for b in SIZE_BUCKETS] if key == "size" else sorted({r[1] for r in rows})
    lines = [
        "| " + key + " | GT | " + " | ".join(stages) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in stages) + " |",
    ]
    for name in groups + ["all"]:
        rs = rows if name == "all" else [r for r in rows if (r[0] if key == "size" else r[1]) == name]
        if not rs:
            continue
        flags = np.array([r[2] for r in rs])
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(f"{100 * v:.1f}%" for v in flags.mean(0)) + " |")
    return "\n".join(lines)


def false_positives(records, dets, matches, iou_threshold, score_threshold):
    """Every detection above the score threshold that matched nothing, by kind and by its own size."""
    kinds = ["duplicate", "class confusion", "localization", "background"]
    rows = []
    in_ignore = 0
    for rec, det, (matched, iou_same, iou_any, _) in zip(records, dets, matches):
        in_ignore += det["in_ignore"]
        keep = det["scores"] >= score_threshold
        sizes = ((det["boxes"][:, 2] - det["boxes"][:, 0]) * (det["boxes"][:, 3] - det["boxes"][:, 1])).clamp(
            min=0
        ).sqrt() * rec["to_orig"][0]
        for i in keep.nonzero().flatten().tolist():
            if matched[i] >= 0:
                rows.append((bucket_of(float(sizes[i])), "true positive"))
            elif iou_same[i] >= iou_threshold:
                rows.append((bucket_of(float(sizes[i])), "duplicate"))
            elif iou_any[i] >= iou_threshold:
                rows.append((bucket_of(float(sizes[i])), "class confusion"))
            elif iou_same[i] >= 0.1:
                rows.append((bucket_of(float(sizes[i])), "localization"))
            else:
                rows.append((bucket_of(float(sizes[i])), "background"))
    lines = [
        "| detection size | detections | precision | " + " | ".join(kinds) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in kinds) + " |",
    ]
    for name in [b[0] for b in SIZE_BUCKETS] + ["all"]:
        rs = rows if name == "all" else [r for r in rows if r[0] == name]
        if not rs:
            continue
        n = len(rs)
        tp = sum(1 for r in rs if r[1] == "true positive")
        lines.append(
            f"| {name} | {n} | {100 * tp / n:.1f}% | "
            + " | ".join(f"{100 * sum(1 for r in rs if r[1] == k) / n:.1f}%" for k in kinds)
            + " |"
        )
    return "\n".join(lines), in_ignore


def confusion(records, iou_threshold, names):
    """Ground-truth class against the class of the best-scoring query near the box."""
    ids = sorted(names)
    index = {c: i for i, c in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for rec in records:
        gt, labels = rec["gt"], rec["labels"]
        if len(gt) == 0:
            continue
        iou, _ = box_iou(gt, rec["boxes"])
        near = iou >= iou_threshold
        scores = rec["scores"]  # [Q, C]
        for g in range(len(gt)):
            qs = near[g].nonzero().flatten()
            if len(qs) == 0:
                continue
            best = scores[qs].flatten().argmax()
            predicted = int(best % scores.shape[1])
            if predicted in index and int(labels[g]) in index:
                matrix[index[int(labels[g])], index[predicted]] += 1
    return ids, matrix


def confusion_by_size(records, iou_threshold, names):
    """
    The confusion per object size: for every size bucket the ground truths with a query near
    them, the share the best-scoring pair labels correctly, and the three most frequent wrong
    (ground truth -> predicted) pairs with their share of the bucket's confusions.
    """
    ids = sorted(names)
    per_bucket = {b[0]: {"n": 0, "correct": 0, "pairs": {}} for b in SIZE_BUCKETS}
    for rec in records:
        gt, labels = rec["gt"], rec["labels"]
        if len(gt) == 0:
            continue
        sizes = gt_sizes(rec)
        iou, _ = box_iou(gt, rec["boxes"])
        near = iou >= iou_threshold
        scores = rec["scores"]
        for g in range(len(gt)):
            qs = near[g].nonzero().flatten()
            truth = int(labels[g])
            if len(qs) == 0 or truth not in ids:
                continue
            predicted = int(scores[qs].flatten().argmax() % scores.shape[1])
            b = per_bucket[bucket_of(float(sizes[g]))]
            b["n"] += 1
            if predicted == truth:
                b["correct"] += 1
            else:
                b["pairs"][(truth, predicted)] = b["pairs"].get((truth, predicted), 0) + 1
    lines = [
        "| size | GT with a query near | correct class | confusions | most frequent (share of the bucket's confusions) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, b in per_bucket.items():
        if b["n"] == 0:
            continue
        wrong = b["n"] - b["correct"]
        top = sorted(b["pairs"].items(), key=lambda kv: -kv[1])[:3]
        top_text = ", ".join(f"{names[t]} -> {names[p]} {100 * c / max(wrong, 1):.0f}%" for (t, p), c in top)
        lines.append(
            f"| {name} | {b['n']} | {100 * b['correct'] / b['n']:.1f}% | {100 * wrong / b['n']:.1f}% | {top_text} |"
        )
    return "\n".join(lines)


def confusion_table(ids, matrix, names):
    lines = [
        "| GT \\ predicted | " + " | ".join(names[c][:6] for c in ids) + " | correct |",
        "| --- | " + " | ".join("---:" for _ in ids) + " | ---: |",
    ]
    for i, c in enumerate(ids):
        total = matrix[i].sum()
        cells = " | ".join(f"{100 * v / total:.0f}%" if total else "-" for v in matrix[i])
        lines.append(
            f"| {names[c]} | {cells} | {100 * matrix[i, i] / total:.0f}% |"
            if total
            else f"| {names[c]} | {cells} | - |"
        )
    return "\n".join(lines)


def calibration(dets, matches):
    """Precision at IoU 0.5 per score bin, and the Spearman correlation of score with IoU among near-hits."""
    from scipy.stats import spearmanr

    scores = torch.cat([d["scores"] for d in dets])
    hit = torch.cat([m[0] >= 0 for m in matches])
    iou = torch.cat([m[1] for m in matches])
    bins = np.linspace(0, 1, 11)
    lines = ["| score bin | detections | precision (IoU 0.5) | mean IoU of hits |", "| --- | ---: | ---: | ---: |"]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if m.sum() == 0:
            continue
        lines.append(
            f"| {lo:.1f}-{hi:.1f} | {int(m.sum())} | {100 * float(hit[m].float().mean()):.1f}% | {float(iou[m & hit].mean()) if (m & hit).any() else float('nan'):.3f} |"
        )
    near = iou >= 0.5
    rho = spearmanr(scores[near].numpy(), iou[near].numpy()).correlation if near.sum() > 2 else float("nan")
    return "\n".join(lines), rho


def density_position(records, matches):
    """Recall against the image's box count, the box's distance to the frame border and its crowding."""
    per_gt = []  # (count, border px orig, crowd IoU, size, recalled)
    for rec, (_, _, _, gt_of) in zip(records, matches):
        gt = rec["gt"]
        if len(gt) == 0:
            continue
        w, h = rec["content_size"]
        border = torch.stack([gt[:, 0], gt[:, 1], w - gt[:, 2], h - gt[:, 3]], 1).min(1).values * rec["to_orig"][0]
        iou, _ = box_iou(gt, gt)
        iou.fill_diagonal_(0)
        crowd = iou.max(1).values if len(gt) > 1 else torch.zeros(len(gt))
        sizes = gt_sizes(rec)
        for g in range(len(gt)):
            per_gt.append((len(gt), float(border[g]), float(crowd[g]), float(sizes[g]), bool(gt_of[g] >= 0)))
    arr = np.array(per_gt, dtype=float)

    def table(title, values, edges, fmt):
        lines = [f"### {title}\n", "| bin | GT | recall |", "| --- | ---: | ---: |"]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            if m.sum():
                lines.append(f"| {fmt(lo, hi)} | {int(m.sum())} | {100 * arr[m, 4].mean():.1f}% |")
        return "\n".join(lines) + "\n"

    out = table(
        "Recall (matched at any score) by the number of ground-truth boxes in the image",
        arr[:, 0],
        [0, 25, 50, 100, 200, 400, 10000],
        lambda lo, hi: f"{int(lo)}-{int(hi) - 1}" if hi < 10000 else f">={int(lo)}",
    )
    out += table(
        "Recall by distance to the image border (original px)",
        arr[:, 1],
        [0, 4, 16, 64, 100000],
        lambda lo, hi: f"{int(lo)}-{int(hi)}" if hi < 100000 else f">={int(lo)}",
    )
    out += table(
        "Recall by crowding: the box's highest IoU with another ground-truth box",
        arr[:, 2],
        [0, 1e-6, 0.1, 0.3, 0.5, 1.01],
        lambda lo, hi: "isolated" if hi <= 1e-6 else f"{lo:.1f}-{hi:.1f}",
    )
    return out


def evaluate_with(evaluator, records, make_predictions):
    """AP of the predictions ``make_predictions(record)`` returns (boxes in input px, scores, labels), through the real evaluator."""
    import contextlib

    evaluator.cleanup()
    for rec in records:
        boxes, scores, labels = make_predictions(rec)
        evaluator.update({rec["image_id"]: {"boxes": boxes * rec["to_orig"], "scores": scores, "labels": labels}})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        evaluator.summarize()
    return evaluator.coco_eval["bbox"].stats


def levers(evaluator, records, top_k, labels_fmt):
    """The evaluator's AP for the plain postprocessor output and a few cheap changes to it."""

    def plain(rec):
        det = postprocess(rec, top_k)
        return det["boxes"], det["scores"], det["labels"]

    def clamped(rec):
        det = postprocess(rec, top_k)
        w, h = rec["content_size"]
        boxes = det["boxes"].clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(w))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(h))
        return boxes, det["scores"], det["labels"]

    def nms(threshold):
        def run(rec):
            det = postprocess(rec, top_k)
            keep = torchvision.ops.batched_nms(det["boxes"], det["scores"], det["labels"], threshold)
            return det["boxes"][keep], det["scores"][keep], det["labels"][keep]

        return run

    def min_score(threshold):
        def run(rec):
            det = postprocess(rec, top_k)
            keep = det["scores"] >= threshold
            return det["boxes"][keep], det["scores"][keep], det["labels"][keep]

        return run

    def one_per_query(rec):
        scores, labels = rec["scores"].max(1)
        order = scores.argsort(descending=True)[:top_k]
        boxes = rec["boxes"][order]
        keep = torch.ones(len(boxes), dtype=torch.bool)
        if len(rec["ignore"]):
            keep = ~detections_in_ignore_regions(boxes, rec["ignore"], 0.5)
        return boxes[keep], scores[order][keep], labels[order][keep]

    variants = [
        ("postprocessor output (top-k pairs)", plain),
        ("boxes clamped to the image", clamped),
        ("one detection per query (best class only)", one_per_query),
        ("class-aware NMS 0.7", nms(0.7)),
        ("class-aware NMS 0.5", nms(0.5)),
        ("drop scores < 0.05", min_score(0.05)),
        ("drop scores < 0.3", min_score(0.3)),
    ]
    lines = ["| variant | " + " | ".join(labels_fmt) + " |", "| --- | " + " | ".join("---:" for _ in labels_fmt) + " |"]
    for name, fn in variants:
        stats = evaluate_with(evaluator, records, fn)
        lines.append(f"| {name} | " + " | ".join(f"{100 * s:.1f}" for s in stats[: len(labels_fmt)]) + " |")
        print(f"  {name}: AP {100 * stats[0]:.1f}")
    return "\n".join(lines)


def oracle_ranking(evaluator, records, top_k, labels_fmt):
    """AP if every detection's score were its IoU with the best same-class ground truth (perfect ranking, same boxes)."""

    def oracle(rec):
        det = postprocess(rec, top_k)
        if len(rec["gt"]) == 0:
            return det["boxes"], torch.zeros_like(det["scores"]), det["labels"]
        iou, _ = box_iou(det["boxes"], rec["gt"])
        same = det["labels"][:, None] == rec["labels"][None, :]
        best = torch.where(same, iou, torch.zeros_like(iou)).max(1).values
        return det["boxes"], best, det["labels"]

    def oracle_class(rec):
        # the label of every detection replaced by that of the box it overlaps most (if at IoU 0.5), scores kept
        det = postprocess(rec, top_k)
        if len(rec["gt"]) == 0:
            return det["boxes"], det["scores"], det["labels"]
        iou, _ = box_iou(det["boxes"], rec["gt"])
        best, g = iou.max(1)
        labels = torch.where(best >= 0.5, rec["labels"][g], det["labels"])
        return det["boxes"], det["scores"], labels

    def oracle_both(rec):
        # relabelling makes the (query, class) pairs of one query identical; one of each is kept
        det = postprocess(rec, top_k)
        boxes, _, labels = oracle_class(rec)
        if len(rec["gt"]) == 0:
            return boxes, torch.zeros(len(boxes)), labels
        iou, _ = box_iou(boxes, rec["gt"])
        same = labels[:, None] == rec["labels"][None, :]
        scores = torch.where(same, iou, torch.zeros_like(iou)).max(1).values
        key = det["query"] * 1000 + labels
        _, first = np.unique(key.numpy(), return_index=True)
        keep = torch.from_numpy(np.sort(first))
        return boxes[keep], scores[keep], labels[keep]

    lines = []
    for name, fn in (
        ("oracle class: labels replaced by the overlapped box's class, scores kept", oracle_class),
        ("oracle score: scores replaced by IoU with the same-class ground truth", oracle),
        ("oracle class and score", oracle_both),
    ):
        stats = evaluate_with(evaluator, records, fn)
        lines.append(f"| {name} | " + " | ".join(f"{100 * s:.1f}" for s in stats[: len(labels_fmt)]) + " |")
        print(f"  {name}: AP {100 * stats[0]:.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)


def ladder_figure(stages, rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for i, (name, _, _) in enumerate(SIZE_BUCKETS):
        rs = [r for r in rows if r[0] == name]
        if not rs:
            continue
        flags = np.array([r[2] for r in rs]).mean(0) * 100
        ax.plot(
            range(len(stages)),
            flags,
            color=SERIES[i],
            linewidth=1.6,
            marker="o",
            markersize=4,
            label=f"{name} (n={len(rs)})",
        )
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel("% of ground-truth boxes", fontsize=8, color=MUTED)
    ax.set_title("recall ladder: where ground-truth boxes are lost", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_figure(ids, matrix, names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    share = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels([names[c] for c in ids], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([names[c] for c in ids], fontsize=8)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if share[i, j] >= 0.02:
                ax.text(
                    j,
                    i,
                    f"{100 * share[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK if share[i, j] < 0.6 else SURFACE,
                )
    ax.set_xlabel("predicted class of the best-scoring query near the box", fontsize=8, color=MUTED)
    ax.set_ylabel("ground-truth class", fontsize=8, color=MUTED)
    ax.set_title("class confusion (% of each ground-truth class)", fontsize=10, color=INK, loc="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output directory of train.py (config.yml and checkpoints)")
    parser.add_argument("--checkpoint", help="checkpoint file in the run (default: best_stg2, best_stg1, then last)")
    parser.add_argument("--iou", type=float, default=0.5, help="matching IoU of the probes (default 0.5)")
    parser.add_argument(
        "--score", type=float, default=0.3, help="score threshold of the thresholded probes (default 0.3)"
    )
    parser.add_argument("--recollect", action="store_true", help="run the model again even if the cache exists")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = os.path.join(args.run, "probe")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "val_predictions.pt")
    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    evaluator = cfg.evaluator
    dataset = cfg.val_dataloader.dataset
    names = dict(getattr(dataset, "CATEGORIES", []))
    top_k = min(
        cfg.postprocessor.num_top_queries,
        cfg.yaml_cfg.get(cfg.yaml_cfg["DOME"]["decoder"], {}).get("num_queries", 10**9),
    )

    if os.path.exists(cache) and not args.recollect:
        records = torch.load(cache, weights_only=False)
        print(f"loaded {len(records)} images from {cache}")
    else:
        model = load_model(cfg, checkpoint, args.device)
        print(f"checkpoint {checkpoint}: collecting")
        records = collect(cfg, model, find_decoder(model), args.device, evaluator)
        torch.save(records, cache)
        print(f"cached {len(records)} images to {cache}")

    print("matching")
    dets = [postprocess(rec, top_k) for rec in records]
    matches = [greedy_match(det, rec["gt"], rec["labels"], args.iou) for rec, det in zip(records, dets)]
    num_gt = sum(len(r["gt"]) for r in records)

    report = [f"# Validation probes of `{os.path.relpath(checkpoint).replace(os.sep, '/')}`\n"]
    report.append(
        f"{len(records)} images, {num_gt} ground-truth boxes; the postprocessor keeps the top {top_k} (query, class) pairs per image, detections inside ignore regions are dropped as the evaluator does. Matching is COCO-style at IoU {args.iou}. Sizes are square-root areas in original pixels.\n"
    )

    stages, rows = ladder(records, dets, matches, args.iou, args.score, names)
    ladder_figure(stages, rows, os.path.join(out_dir, "ladder.png"))
    report.append("## Recall ladder\n")
    report.append(
        "Share of ground-truth boxes that survive each stage. *proposal near*: one of the encoder's proposals reaches the IoU; *query near*: a final decoder box does; *same class, score>=0.05*: such a query scores the box's class at least 0.05; *in top-k*: the best-scoring of them is among the postprocessor's pairs; *matched*: the greedy matching assigns it a detection, at any score or at the threshold.\n"
    )
    report.append("![ladder](ladder.png)\n")
    report.append(ladder_table(stages, rows, "size") + "\n")
    report.append(ladder_table(stages, rows, "class") + "\n")

    fp_table, in_ignore = false_positives(records, dets, matches, args.iou, args.score)
    report.append(f"## Detections with score >= {args.score}\n")
    report.append(
        f"By what each detection is: a true positive, a *duplicate* (a same-class box at IoU >= {args.iou} that another detection already took), a *class confusion* (a box of another class at that IoU), a *localization* error (same-class box at IoU 0.1-{args.iou}) or *background* (nothing near). {in_ignore} detections per pass fell inside ignore regions and were dropped before this.\n"
    )
    report.append(fp_table + "\n")

    ids, matrix = confusion(records, args.iou, names)
    confusion_figure(ids, matrix, names, os.path.join(out_dir, "confusion.png"))
    report.append("## Class confusion\n")
    report.append(
        f"For every ground-truth box with a query at IoU >= {args.iou}, the class of the best-scoring (query, class) pair among those queries.\n"
    )
    report.append("![confusion](confusion.png)\n")
    report.append(confusion_table(ids, matrix, names) + "\n")
    report.append(
        "Per object size: how often the best-scoring pair near a ground truth has its class, and which confusions dominate.\n"
    )
    report.append(confusion_by_size(records, args.iou, names) + "\n")

    cal_table, rho = calibration(dets, matches)
    report.append("## Score quality\n")
    report.append(
        f"Precision of the postprocessor's detections per score bin, and among detections at IoU >= 0.5 with a same-class box the Spearman correlation of score with IoU: {rho:.3f}.\n"
    )
    report.append(cal_table + "\n")

    report.append(density_position(records, matches))

    labels_fmt = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"]
    print("scoring the levers with the evaluator")
    lever_table = levers(evaluator, records, top_k, labels_fmt)
    oracle_row = oracle_ranking(evaluator, records, top_k, labels_fmt)
    report.append("## Levers, scored with the evaluator from the cache\n")
    report.append(
        "The plain postprocessor output should reproduce the run's validation AP; the rows below change only the postprocessing. The oracles keep the boxes: *oracle class* corrects the label of every detection that overlaps a box at IoU 0.5 (what perfect classification would give), *oracle score* replaces every score by the box's IoU with the same-class ground truth (what perfect ranking would give).\n"
    )
    report.append(lever_table + "\n" + oracle_row + "\n")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"wrote {out_dir}/report.md")


if __name__ == "__main__":
    main()
