"""
For every ground-truth box of the validation set, the model's best candidate for it: among
all queries, the one whose box overlaps the ground truth most, and that query's score for the
ground-truth class. This says whether a miss is a localization miss (no query near the box)
or a scoring miss (a query sits on the box but scores low), independently of the greedy
matching and the score ordering that decide TP/FP in the COCO protocol.

Written into ``<run>/gt_candidates.csv`` (one row per ground-truth box) and
``<run>/gt_candidates.png`` (best IoU against its score, per object size), with a summary
table on the console.

Columns: image, gt (index in the image), label, class, size_px (square-root area in original
pixels), best_iou, best_iou_score (that query's score for the class), best_iou_in_topk (whether
the postprocessor's top-k output contains that (query, class) pair), best_score_at_iou (the
highest class score among queries reaching ``--iou`` with the box, -1 if none), its query's IoU
``best_score_iou`` and whether that (query, class) pair is in the top-k output, ``best_score_in_topk``.
The figure plots the best-overlapping query; the summary table's score columns use the
best-scoring query at the IoU threshold, which is what the postprocessor would rank (the two differ
where the best-overlapping query is a suppressed duplicate).

    python tools/analysis/gt_candidates.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/gt_candidates.py <run> --checkpoint best_stg1.pth --iou 0.5 --score 0.3
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import SIZE_BUCKETS, bucket_of, load_model, pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e6e5e1"


@torch.no_grad()
def candidates_of_batch(model, postprocessor, images, targets, iou_threshold, device):
    """One record per ground-truth box of the batch: its best-overlapping query and the scores around it."""
    out = model(images.to(device))
    logits, boxes = out["pred_logits"].float(), out["pred_boxes"].float()  # [B, Q, C], [B, Q, 4] normalized
    scores = logits.sigmoid()
    k = min(postprocessor.num_top_queries, logits.shape[1])
    topk = scores.flatten(1).topk(k, dim=-1).indices  # the (query, class) pairs the postprocessor outputs
    num_classes = scores.shape[-1]

    records = []
    for b, target in enumerate(targets):
        gt = target["boxes"].as_subclass(torch.Tensor).float().to(device)  # xyxy in input pixels
        if len(gt) == 0:
            continue
        size = target.get("padded_size", target["orig_size"]).float().to(device).repeat(2)
        pred = box_cxcywh_to_xyxy(boxes[b]) * size
        iou, _ = box_iou(gt, pred)  # [G, Q]
        labels = target["labels"].to(device)
        class_scores = scores[b][:, labels].T  # [G, Q]: every query's score for each box's class
        best_iou, best_q = iou.max(1)
        in_topk = torch.zeros_like(scores[b], dtype=torch.bool).flatten()
        in_topk[topk[b]] = True
        in_topk = in_topk.view(-1, num_classes)
        reach = iou >= iou_threshold
        best_score, best_score_q = torch.where(reach, class_scores, -1.0).max(1)
        scale = (target["orig_size"].float() / target.get("resized_size", target["orig_size"]).float()).mean()
        sizes = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).sqrt() * scale.to(device)
        for g in range(len(gt)):
            q = int(best_q[g])
            records.append(
                {
                    "image": int(target["image_id"]),
                    "gt": g,
                    "label": int(labels[g]),
                    "size_px": float(sizes[g]),
                    "best_iou": float(best_iou[g]),
                    "best_iou_score": float(class_scores[g, q]),
                    "best_iou_in_topk": bool(in_topk[q, labels[g]]),
                    "best_score_at_iou": float(best_score[g]),
                    "best_score_iou": float(iou[g, best_score_q[g]]) if best_score[g] >= 0 else -1.0,
                    "best_score_in_topk": bool(in_topk[best_score_q[g], labels[g]]) if best_score[g] >= 0 else False,
                }
            )
    return records


def summary(records, iou_threshold, score_threshold):
    rows = []
    for name, _, _ in SIZE_BUCKETS + [("all", 0, 1e9)]:
        rs = records if name == "all" else [r for r in records if bucket_of(r["size_px"]) == name]
        if not rs:
            continue
        n = len(rs)
        localized = [r for r in rs if r["best_iou"] >= iou_threshold]
        scored = [r for r in localized if r["best_score_at_iou"] >= score_threshold]
        low = [r for r in localized if r["best_score_at_iou"] < score_threshold]
        dropped = [r for r in localized if not r["best_score_in_topk"]]
        rows.append(
            [
                name,
                n,
                f"{100 * len(localized) / n:.1f}%",
                f"{100 * len(scored) / n:.1f}%",
                f"{100 * len(low) / n:.1f}%",
                f"{100 * len(dropped) / n:.1f}%",
                f"{np.median([r['best_iou'] for r in rs]):.3f}",
                f"{np.median([r['best_iou_score'] for r in rs]):.3f}",
            ]
        )
    headers = [
        "size",
        "GT",
        f"IoU>={iou_threshold}",
        f"and score>={score_threshold}",
        f"and score<{score_threshold}",
        "its query not in top-k",
        "median best IoU",
        "median its score",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| --- | " + " | ".join("---:" for _ in headers[1:]) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def figure(records, iou_threshold, score_threshold, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = [b for b in SIZE_BUCKETS if any(bucket_of(r["size_px"]) == b[0] for r in records)]
    fig, axes = plt.subplots(1, len(buckets), figsize=(4.6 * len(buckets), 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, _, _) in zip(np.atleast_1d(axes), buckets):
        rs = [r for r in records if bucket_of(r["size_px"]) == name]
        x = [r["best_iou"] for r in rs]
        y = [r["best_iou_score"] for r in rs]
        ax.set_facecolor(SURFACE)
        h = ax.hist2d(x, y, bins=[np.linspace(0, 1, 26), np.linspace(0, 1, 26)], cmap="Blues", cmin=1)
        ax.axvline(iou_threshold, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.axhline(score_threshold, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        low = sum(1 for r in rs if r["best_iou"] >= iou_threshold and r["best_iou_score"] < score_threshold)
        ax.set_title(
            f"{name}: {len(rs)} GT; {100 * low / len(rs):.1f}% have the best box at IoU>={iou_threshold} scoring under {score_threshold}",
            fontsize=8,
            color=INK,
            loc="left",
        )
        ax.set_xlabel("best IoU of any query with the GT box", fontsize=8, color=MUTED)
        ax.set_ylabel("that query's score for the GT class", fontsize=8, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        for side in ax.spines.values():
            side.set_color(GRID)
        fig.colorbar(h[3], ax=ax, shrink=0.8).ax.tick_params(colors=MUTED, labelsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output directory of train.py (config.yml and checkpoints)")
    parser.add_argument("--checkpoint", help="checkpoint file in the run (default: best_stg2, best_stg1, then last)")
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU at which a query counts as localizing a box (default 0.5)"
    )
    parser.add_argument(
        "--score", type=float, default=0.3, help="score below which a localized box counts as scored away (default 0.3)"
    )
    parser.add_argument(
        "--num-images", type=int, default=0, help="stop after this many images (default: the whole split)"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    postprocessor = cfg.postprocessor
    loader = cfg.val_dataloader
    names = dict(getattr(loader.dataset, "CATEGORIES", []))
    print(
        f"checkpoint {checkpoint}; the postprocessor keeps {min(postprocessor.num_top_queries, getattr(model.decoder, 'num_queries', postprocessor.num_top_queries))} detections per image"
    )

    records, seen = [], 0
    for images, targets in loader:
        records += candidates_of_batch(model, postprocessor, images, targets, args.iou, args.device)
        seen += len(targets)
        if seen % 160 == 0 or (args.num_images and seen >= args.num_images):
            print(f"  {seen} images, {len(records)} boxes")
        if args.num_images and seen >= args.num_images:
            break

    csv_path = os.path.join(args.run, "gt_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "gt",
                "label",
                "class",
                "size_px",
                "best_iou",
                "best_iou_score",
                "best_iou_in_topk",
                "best_score_at_iou",
                "best_score_iou",
                "best_score_in_topk",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    **r,
                    "class": names.get(r["label"], r["label"]),
                    "size_px": f"{r['size_px']:.1f}",
                    "best_iou": f"{r['best_iou']:.4f}",
                    "best_iou_score": f"{r['best_iou_score']:.4f}",
                    "best_score_at_iou": f"{r['best_score_at_iou']:.4f}",
                    "best_score_iou": f"{r['best_score_iou']:.4f}",
                }
            )
    figure(records, args.iou, args.score, os.path.join(args.run, "gt_candidates.png"))

    print(
        f"\n{len(records)} ground-truth boxes of {seen} images. Per box, the query overlapping it most and the highest class score among queries reaching IoU {args.iou}:\n"
    )
    print(summary(records, args.iou, args.score))
    print(f"\nwrote {csv_path} and gt_candidates.png")


if __name__ == "__main__":
    main()
