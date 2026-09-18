"""
How well the scores separate true from false positives, from probe_val.py's cache
(``<run>/probe/val_predictions.pt``): per object size, the score quantiles of the true
positives and of the false positives, the AUC of the score as a TP-versus-FP classifier (the
probability that a random TP outscores a random FP), and how many FPs outscore the median TP.
False positives are split into duplicates (a same-class box at the IoU that another
detection already took) and the rest (background, localization and class errors), since the
duplicates are what one-to-one matching trains down on purpose.

    python tools/analysis/tp_fp_scores.py outputs/dfine_s_aitod/<run> --aitod
    python tools/analysis/tp_fp_scores.py outputs/dfine_s_visdrone/<run>
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_val import greedy_match, postprocess  # noqa: E402

from src.core import YAMLConfig  # noqa: E402

VISDRONE = [("tiny <16px", 0, 16), ("small 16-32px", 16, 32), ("medium 32-96px", 32, 96), ("large >96px", 96, 1e9)]
AITOD = [("vt <8", 0, 8), ("t 8-16", 8, 16), ("s 16-32", 16, 32), ("m 32-64", 32, 64), ("l >64", 64, 1e9)]


def auc(pos, neg):
    """Mann-Whitney AUC: P(score of a random positive > score of a random negative)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    ranks = x.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = x.argsort()
    xs = x[order]
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    ranks = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--aitod", action="store_true", help="AI-TOD size buckets (default VisDrone's)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    buckets = AITOD if args.aitod else VISDRONE

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    top_k = min(
        cfg.postprocessor.num_top_queries,
        cfg.yaml_cfg.get(cfg.yaml_cfg["DOME"]["decoder"], {}).get("num_queries", 10**9),
    )
    records = torch.load(os.path.join(args.run, "probe", "val_predictions.pt"), weights_only=False)
    print(f"{len(records)} images, top-{top_k} pairs per image, matching at IoU {args.iou}")

    rows = []  # (kind, size_px, score, iou_same)
    for rec in records:
        det = postprocess(rec, top_k)
        matched, iou_same, iou_any, _ = greedy_match(det, rec["gt"], rec["labels"], args.iou)
        b = det["boxes"]
        det_size = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=0).sqrt() * rec["to_orig"][0]
        gt = rec["gt"]
        gt_size = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).clamp(min=0).sqrt() * rec["to_orig"][0]
        for i in range(len(b)):
            if matched[i] >= 0:
                rows.append(("tp", float(gt_size[matched[i]]), float(det["scores"][i]), float(iou_same[i])))
            elif iou_same[i] >= args.iou:
                rows.append(("dup", float(det_size[i]), float(det["scores"][i]), float(iou_same[i])))
            else:
                rows.append(("fp", float(det_size[i]), float(det["scores"][i]), float(iou_same[i])))
    kind = np.array([r[0] for r in rows])
    size = np.array([r[1] for r in rows])
    score = np.array([r[2] for r in rows])

    print(
        "\n| bucket | TP | dup | other FP | TP score p10/p50/p90 | dup p50/p90 | other FP p50/p90/p99 | AUC TP vs other FP | AUC TP vs all FP | other FPs above TP median, per 100 TP | TPs below 0.3 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, lo, hi in buckets + [("all", 0, 1e9)]:
        m = (size >= lo) & (size < hi)
        tp, dup, fp = score[m & (kind == "tp")], score[m & (kind == "dup")], score[m & (kind == "fp")]
        if len(tp) == 0:
            continue
        med = np.median(tp)
        print(
            f"| {name} | {len(tp)} | {len(dup)} | {len(fp)} | {np.percentile(tp, 10):.2f}/{med:.2f}/{np.percentile(tp, 90):.2f} | "
            f"{np.percentile(dup, 50) if len(dup) else float('nan'):.2f}/{np.percentile(dup, 90) if len(dup) else float('nan'):.2f} | "
            f"{np.percentile(fp, 50):.3f}/{np.percentile(fp, 90):.2f}/{np.percentile(fp, 99):.2f} | {auc(tp, fp):.3f} | {auc(tp, np.concatenate([dup, fp])):.3f} | "
            f"{100 * (fp > med).sum() / len(tp):.1f} | {100 * (tp < 0.3).mean():.1f}% |"
        )

    # precision-recall at score thresholds, all sizes: what a threshold costs in recall and buys in precision
    num_gt = sum(len(r["gt"]) for r in records)
    print(f"\n| score >= | TP | recall (IoU {args.iou}) | precision | other FP | dup |")
    print("|---|---:|---:|---:|---:|---:|")
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        s = score >= t
        tp, dup, fp = (s & (kind == "tp")).sum(), (s & (kind == "dup")).sum(), (s & (kind == "fp")).sum()
        print(f"| {t:.2f} | {tp} | {100 * tp / num_gt:.1f}% | {100 * tp / max(1, tp + dup + fp):.1f}% | {fp} | {dup} |")


if __name__ == "__main__":
    main()
