#!/usr/bin/env python
"""
Re-run the official COCO eval straight from a saved ``tp_fp_raw.npz`` and print
the standard 12-number summary, so you can check it matches what `test.sh`
printed during the actual eval run.

Why this is a faithful check (not an approximation):
  * `tp_fp_raw.npz` stores the *exact* post-processed detections the test run fed
    to COCO -- they were recovered verbatim from `coco_evaluator.cocoDt`
    (det_engine.py), i.e. AFTER post-processing and (for VisDrone) ignore-region
    filtering. We rebuild that same prediction list and `loadRes` it directly.
  * We evaluate with the SAME evaluator the test used. The dataset-specific
    evaluator (VisdroneCocoEvaluator / AitodCocoEvaluator) carries custom area
    ranges and maxDets=[1,10,100,500]; we obtain it via `cfg.evaluator`, the same
    code path as training/eval, so params + coco_gt are identical.

So the numbers here should match the test.log summary line-for-line. The single
IoU=0.5 greedy TP/FP arrays in the npz are NOT used for the metric (they're for
size-binned analysis only); we use the raw det_* records as the prediction set.

Usage (run from the repo `code/` dir):
    python tools/coco_eval_from_npz.py <dataset> <size> <type>
        e.g. python tools/coco_eval_from_npz.py visdrone m baseline

Overrides:
    --npz     PATH   explicit npz (default: output/aiiou-<size>-<dataset>/<type>/eval/tp_fp_raw.npz)
    --config  PATH   explicit config (default: configs/dome/Dome-<S|M|L>-<AITOD|VisDrone>.yml)
"""
import argparse
import os
import sys

import numpy as np

# --- make `import src...` work the same way train.py does -------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # tools/.. -> code/
os.chdir(REPO)                                     # so relative cfg paths / cwd match train.py
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(REPO))

from src.core import YAMLConfig  # noqa: E402

DATASET_PROPER = {"aitod": "AITOD", "visdrone": "VisDrone"}


def derive_paths(dataset, size, type_):
    dataset = dataset.lower()
    if dataset not in DATASET_PROPER:
        sys.exit(f"ERROR: dataset must be aitod|visdrone (got {dataset!r})")
    size_u = size.upper()
    if size_u not in ("S", "M", "L"):
        sys.exit(f"ERROR: size must be s|m|l (got {size!r})")
    config = f"configs/dome/Dome-{size_u}-{DATASET_PROPER[dataset]}.yml"
    outdir = f"output/aiiou-{size.lower()}-{dataset}/{type_}"
    npz = os.path.join(outdir, "eval", "tp_fp_raw.npz")
    return config, npz


def load_predictions(npz_path):
    """Rebuild the COCO-format detection list from the saved per-detection arrays."""
    d = np.load(npz_path)
    img = d["det_image_id"]
    cat = d["det_category_id"]
    sc = d["det_score"]
    x, y, w, h = d["det_x"], d["det_y"], d["det_w"], d["det_h"]
    preds = [
        {
            "image_id": int(img[i]),
            "category_id": int(cat[i]),
            "bbox": [float(x[i]), float(y[i]), float(w[i]), float(h[i])],
            "score": float(sc[i]),
        }
        for i in range(img.shape[0])
    ]
    return preds


def _mean_valid(x):
    """COCO convention: entries == -1 mean 'not computed' and are excluded."""
    x = np.asarray(x, dtype=float)
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def _iou_idx(iou_thrs, target):
    """Index of `target` IoU in params.iouThrs, or None if absent."""
    diffs = np.abs(np.asarray(iou_thrs) - target)
    j = int(diffs.argmin())
    return j if diffs[j] < 1e-6 else None


def summarize_per_iou(coco_eval):
    """Overall AP at each IoU threshold (area=all, largest maxDets)."""
    p = coco_eval.eval["precision"]            # [T, R, K, A, M]
    prm = coco_eval.params
    a = prm.areaRngLbl.index("all") if "all" in prm.areaRngLbl else 0
    m = len(prm.maxDets) - 1
    print(f"\nAP per IoU threshold (area=all, maxDets={prm.maxDets[m]}):")
    cells = [f"{iouv:.2f}:{_mean_valid(p[t, :, :, a, m]):.4f}"
             for t, iouv in enumerate(prm.iouThrs)]
    for i in range(0, len(cells), 5):
        print("  " + "  ".join(cells[i:i + 5]))


def summarize_per_area(coco_eval, coco_gt):
    """Overall AP / AR (+ GT counts) for every area range the evaluator defines.

    Reads the evaluator's OWN params.areaRng/areaRngLbl, so the size categories
    here are identical to what the live evaluator (test.sh / training) reports --
    e.g. VisDrone's verytiny/tiny/small/medium/large, AITOD's verytiny/tiny/...
    """
    p = coco_eval.eval["precision"]            # [T, R, K, A, M]
    r = coco_eval.eval["recall"]               # [T, K, A, M]
    prm = coco_eval.params
    m = len(prm.maxDets) - 1
    i50, i75 = _iou_idx(prm.iouThrs, 0.5), _iou_idx(prm.iouThrs, 0.75)
    ann_ids = coco_gt.getAnnIds(catIds=prm.catIds)  # only evaluated categories
    areas = np.array([a["area"] for a in coco_gt.loadAnns(ann_ids)], dtype=float)
    # AP is IoU=.50:.95; AP50/AP75 are at those IoUs; AR is IoU=.50:.95 -- all per size.
    print(f"\nAP / AR per area range (maxDets={prm.maxDets[m]}):")
    print(f"  {'area':<12}{'#GT':>9}{'AP':>9}{'AP50':>9}{'AP75':>9}{'AR':>9}")
    print("  " + "-" * 57)
    for a, lbl in enumerate(prm.areaRngLbl):
        lo, hi = prm.areaRng[a]                 # COCO area bounds (px^2); ranges may overlap
        ngt = int(((areas >= lo) & (areas < hi)).sum())
        ap = _mean_valid(p[:, :, :, a, m])
        ap50 = _mean_valid(p[i50, :, :, a, m]) if i50 is not None else float("nan")
        ap75 = _mean_valid(p[i75, :, :, a, m]) if i75 is not None else float("nan")
        ar = _mean_valid(r[:, :, a, m])
        print(f"  {lbl:<12}{ngt:>9}{ap:>9.4f}{ap50:>9.4f}{ap75:>9.4f}{ar:>9.4f}")


def summarize_per_category(coco_eval, coco_gt, csv_path=None):
    """Per-class AP / AP50 / AP75 / AR (area=all, largest maxDets) + GT counts.

    The 'mean' row averages per-category AP over classes with GT, which equals
    the overall mAP printed by summarize() -- a useful self-consistency check.
    """
    p = coco_eval.eval["precision"]            # [T, R, K, A, M]
    r = coco_eval.eval["recall"]               # [T, K, A, M]
    prm = coco_eval.params
    a = prm.areaRngLbl.index("all") if "all" in prm.areaRngLbl else 0
    m = len(prm.maxDets) - 1
    i50, i75 = _iou_idx(prm.iouThrs, 0.5), _iou_idx(prm.iouThrs, 0.75)
    cat_ids = list(prm.catIds)
    names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    rows = []
    for k, cid in enumerate(cat_ids):
        ngt = len(coco_gt.getAnnIds(catIds=[cid]))
        ap = _mean_valid(p[:, :, k, a, m])
        ap50 = _mean_valid(p[i50, :, k, a, m]) if i50 is not None else float("nan")
        ap75 = _mean_valid(p[i75, :, k, a, m]) if i75 is not None else float("nan")
        ar = _mean_valid(r[:, k, a, m])
        rows.append((names.get(cid, str(cid)), ngt, ap, ap50, ap75, ar))

    print(f"\nPer-category metrics (area=all, maxDets={prm.maxDets[m]}):")
    print(f"  {'category':<18}{'#GT':>9}{'AP':>9}{'AP50':>9}{'AP75':>9}{'AR':>9}")
    print("  " + "-" * 63)
    for name, ngt, ap, ap50, ap75, ar in rows:
        print(f"  {name:<18}{ngt:>9}{ap:>9.4f}{ap50:>9.4f}{ap75:>9.4f}{ar:>9.4f}")
    print("  " + "-" * 63)
    print(f"  {'mean':<18}{sum(x[1] for x in rows):>9}"
          f"{np.nanmean([x[2] for x in rows]):>9.4f}"
          f"{np.nanmean([x[3] for x in rows]):>9.4f}"
          f"{np.nanmean([x[4] for x in rows]):>9.4f}"
          f"{np.nanmean([x[5] for x in rows]):>9.4f}")

    if csv_path:
        import csv as _csv
        with open(csv_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["category", "num_gt", "AP", "AP50", "AP75", "AR"])
            for name, ngt, ap, ap50, ap75, ar in rows:
                w.writerow([name, ngt, f"{ap:.6f}", f"{ap50:.6f}", f"{ap75:.6f}", f"{ar:.6f}"])
        print(f"  [csv] wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="visdrone", help="aitod | visdrone")
    ap.add_argument("size", nargs="?", default="m", help="s | m | l")
    ap.add_argument("type", nargs="?", default="baseline", help="baseline | <variant name>")
    ap.add_argument("--npz", default=None, help="override path to tp_fp_raw.npz")
    ap.add_argument("--config", default=None, help="override config .yml")
    ap.add_argument("--csv", action="store_true",
                    help="also dump per-category metrics to <npz dir>/per_category_ap.csv")
    args = ap.parse_args()

    config, npz = derive_paths(args.dataset, args.size, args.type)
    if args.config:
        config = args.config
    if args.npz:
        npz = args.npz

    if not os.path.isfile(config):
        sys.exit(f"ERROR: config not found: {config}")
    if not os.path.isfile(npz):
        sys.exit(f"ERROR: npz not found: {npz}")

    print("=" * 60)
    print("COCO eval from saved npz")
    print(f"  config: {config}")
    print(f"  npz:    {npz}")
    print("=" * 60)

    # Build the SAME evaluator the test used (correct class, coco_gt, params).
    cfg = YAMLConfig(config)
    evaluator = cfg.evaluator                       # VisdroneCocoEvaluator / AitodCocoEvaluator / ...
    coco_gt = evaluator.coco_gt
    coco_eval = evaluator.coco_eval["bbox"]

    preds = load_predictions(npz)
    print(f"[load] {len(preds)} detections; "
          f"{len(coco_gt.getImgIds())} images, {len(coco_gt.getCatIds())} categories")

    # Feed the saved predictions verbatim and run the standard evaluate/accumulate/summarize.
    # (Bypass evaluator.update()'s prepare() so we don't re-apply ignore-region filtering --
    #  the saved dets are already post-filter, recovered from the test run's cocoDt.)
    coco_dt = coco_gt.loadRes(preds)
    coco_eval.cocoDt = coco_dt
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("IoU metric: bbox")
    coco_eval.summarize()

    print("\nstats (coco_eval_bbox) =", [round(float(s), 4) for s in coco_eval.stats])

    # --- extra, more detailed breakdowns (all from the same coco_eval) -------
    summarize_per_iou(coco_eval)
    summarize_per_area(coco_eval, coco_gt)
    csv_path = os.path.join(os.path.dirname(npz), "per_category_ap.csv") if args.csv else None
    summarize_per_category(coco_eval, coco_gt, csv_path=csv_path)


if __name__ == "__main__":
    main()
