"""
Where the AP is lost, from probe_val.py's cache (``<run>/probe/val_predictions.pt``), through
the run's own evaluator: AP at every IoU threshold, then oracles that fix one thing at a time
and keep the rest as the model left it. Unlike probe_val's "oracle score" (which scores every
same-class overlap by its IoU and so promotes the duplicates too), these oracles act on the
COCO-style greedy matching at IoU 0.5, so duplicates stay where the model scored them.

* ranking: every true positive outscores every false positive (TP score + 1), the order among
  the TPs and among the FPs unchanged: what the TP/FP mixing costs
* tightness: the TPs are ordered by their IoU (score = 1 + IoU), FPs unchanged: what a score
  that tracked box quality would give at the higher IoU thresholds
* boxes: every TP's box is replaced by its ground truth (scores and FPs unchanged): the ceiling
  of localization precision for the objects already found
* boxes halfway: every TP's box moved halfway to its ground truth
* near misses: the same-class detections at IoU 0.1-0.5 with a still unmatched ground truth are
  snapped to it: the recall that lies in loose boxes
* background gone: the FPs with no same-class box at IoU 0.1 are removed

    python tools/analysis/ap_decomposition.py outputs/dfine_s_aitod/<run>
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_val import evaluate_with, greedy_match, postprocess  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.misc.box_ops import box_iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    evaluator = cfg.evaluator
    top_k = min(
        cfg.postprocessor.num_top_queries,
        cfg.yaml_cfg.get(cfg.yaml_cfg["DOME"]["decoder"], {}).get("num_queries", 10**9),
    )
    records = torch.load(os.path.join(args.run, "probe", "val_predictions.pt"), weights_only=False)
    print(f"{len(records)} images, top-{top_k} pairs, matching at IoU {args.iou}")

    print("matching")
    dets = [postprocess(rec, top_k) for rec in records]
    matches = [greedy_match(det, rec["gt"], rec["labels"], args.iou) for rec, det in zip(records, dets)]
    by_id = {rec["image_id"]: i for i, rec in enumerate(records)}

    def variant(kind):
        def fn(rec):
            i = by_id[rec["image_id"]]
            det, (matched, iou_same, _iou_any, gt_of) = dets[i], matches[i]
            boxes, scores, labels = det["boxes"].clone(), det["scores"].clone(), det["labels"]
            tp = matched >= 0
            if kind == "plain":
                pass
            elif kind == "ranking":
                scores[tp] += 1.0
            elif kind == "tightness":
                gt = rec["gt"][matched[tp]]
                iou = box_iou(boxes[tp], gt)[0].diagonal()
                scores[tp] = 1.0 + iou
            elif kind in ("boxes", "halfway"):
                gt = rec["gt"][matched[tp]]
                boxes[tp] = gt if kind == "boxes" else 0.5 * (boxes[tp] + gt)
            elif kind == "near":
                if len(rec["gt"]):
                    iou, _ = box_iou(boxes, rec["gt"])
                    same = labels[:, None] == rec["labels"][None, :]
                    free = gt_of < 0
                    cand = torch.where(same & free[None, :], iou, torch.zeros_like(iou))
                    best, g = cand.max(1)
                    order = scores.argsort(descending=True)
                    taken = torch.zeros(len(rec["gt"]), dtype=torch.bool)
                    for d in order.tolist():
                        if tp[d] or best[d] < 0.1 or best[d] >= args.iou or taken[g[d]]:
                            continue
                        boxes[d] = rec["gt"][g[d]]
                        taken[g[d]] = True
            elif kind == "background":
                keep = tp | (iou_same >= 0.1)
                return boxes[keep], scores[keep], labels[keep]
            return boxes, scores, labels

        return fn

    names = ["AP", "AP50", "AP75"] + [f"AP_{s}" for s in ("vt", "t", "s", "m")]
    print("\n| variant | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for kind, title in (
        ("plain", "postprocessor output"),
        ("ranking", "ranking: every TP outscores every FP"),
        ("tightness", "tightness: TPs ordered by IoU, FPs unchanged"),
        ("halfway", "boxes: every TP moved halfway to its ground truth"),
        ("boxes", "boxes: every TP replaced by its ground truth"),
        ("near", "near misses: loose same-class boxes (IoU 0.1-0.5) snapped to the unmatched ground truth"),
        ("background", "background gone: FPs with no same-class box at IoU 0.1 removed"),
    ):
        stats = evaluate_with(evaluator, records, variant(kind))
        if kind == "plain":
            ce = evaluator.coco_eval["bbox"]
            prec = ce.eval["precision"]  # [T, R, K, A, M]
            thr = ce.params.iouThrs
            per_t = []
            for t in range(len(thr)):
                p = prec[t, :, :, 0, -1]
                per_t.append(float(np.mean(p[p > -1])) if (p > -1).any() else float("nan"))
            print(
                "| AP per IoU threshold | " + " | ".join(f"{100 * v:.1f}@{th:.2f}" for v, th in zip(per_t, thr)) + " |"
            )
        print(f"| {title} | " + " | ".join(f"{100 * s:.1f}" for s in stats[: len(names)]) + " |", flush=True)


if __name__ == "__main__":
    main()
