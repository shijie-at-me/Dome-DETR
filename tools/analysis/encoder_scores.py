"""
What the encoder's token scores look like on the validation split, so that a ``query_budget:
threshold`` rule can be set from data rather than guessed: the score distribution itself, how
many tokens clear a grid of thresholds against the image's object count, the score of the token
at rank "one per object", and the coverage-against-cost curve of every (threshold, margin) pair.

Works on any run, whatever its query_budget: the encoder's score head is what the rule reads.
The threshold only decides anything on the images whose object count reaches the budget's floor,
so ``--dense`` probes those and a random sample of the rest, which is much faster than the split.

    python tools/analysis/encoder_scores.py outputs/dfine_s_aitod/<run>
    python tools/analysis/encoder_scores.py <run> --dense 150 --sample 1500
    python tools/analysis/encoder_scores.py <run> --checkpoint last.pth --limit 2000

Writes <run>/encoder_scores.csv (one row per image) and prints the tables.
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import load_model, pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.data.dataloader import BatchImageCollateFunction  # noqa: E402
from src.zoo.dome.dfine_decoder import DFINETransformer  # noqa: E402

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)
MARGINS = (0, 50, 100, 200, 300)
RANKS = (100, 300, 500, 1000)
MAX_RANK = 3000  # the ranks read per image; AI-TOD's densest tile holds 2667 objects


def object_counts(ds):
    """The number of annotations of every image, without decoding any of them."""
    if hasattr(ds, "coco"):
        return np.array([len(ds.coco.getAnnIds(imgIds=i)) for i in range(len(ds))])
    return np.array([len(ds[i][1]["labels"]) for i in range(len(ds))])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory (config.yml and checkpoints)")
    parser.add_argument("--checkpoint", help="checkpoint file (default: best_stg2, best_stg1 or last)")
    parser.add_argument("--limit", type=int, default=0, help="images to probe, in order (default: the whole split)")
    parser.add_argument("--dense", type=int, default=0, help="probe every image with at least this many objects")
    parser.add_argument("--sample", type=int, default=0, help="with --dense, that many random images besides")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--round", type=int, default=100, help="the budget's rounding step")
    parser.add_argument("--range", type=int, nargs=2, default=(300, 1500), help="the budget's clamp")
    parser.add_argument("--no-amp", action="store_true", help="run the model in fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = load_model(cfg, pick_checkpoint(args.run, args.checkpoint), args.device)
    decoder = next(m for m in model.modules() if isinstance(m, DFINETransformer))

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        dataset = cfg.val_dataloader.dataset
    counts = object_counts(dataset)
    if args.dense:
        idx = np.flatnonzero(counts >= args.dense)
        if args.sample:
            rest = np.flatnonzero(counts < args.dense)
            rng = np.random.default_rng(0)
            idx = np.concatenate([idx, rng.choice(rest, min(args.sample, rest.size), replace=False)])
        subset = torch.utils.data.Subset(dataset, sorted(idx.tolist()))
    else:
        subset = dataset if not args.limit else torch.utils.data.Subset(dataset, range(args.limit))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=BatchImageCollateFunction(pad_to_multiple=32),
    )
    print(f"{len(subset)} of {len(dataset)} images, batch {args.batch}, {'fp32' if args.no_amp else 'amp'}")

    grabbed = []

    def hook(_m, _i, out):
        # [B, N, num_classes] encoder logits, one row per token; everything below stays on the GPU
        s = torch.sigmoid(out.detach().float())
        grabbed.append(s.squeeze(-1) if s.shape[-1] == 1 else s.max(-1).values)

    handle = decoder.enc_score_head.register_forward_hook(hook)
    taus = torch.tensor(THRESHOLDS, device=args.device)

    rows = []
    autocast = torch.autocast("cuda", enabled=not args.no_amp and args.device.startswith("cuda"))
    with torch.no_grad(), autocast, contextlib.redirect_stdout(open(os.devnull, "w")):
        for samples, targets in loader:
            grabbed.clear()
            model(samples.to(args.device, non_blocking=True))
            scores = grabbed[0]  # the first call is the query selection's
            above = (scores[:, None, :] > taus[None, :, None]).sum(-1)  # [B, len(THRESHOLDS)]
            top = torch.topk(scores, min(MAX_RANK, scores.shape[1]), dim=1).values  # [B, MAX_RANK]
            soft, peak = scores.sum(1), top[:, 0]
            above, top, soft, peak = (t.cpu().numpy() for t in (above, top, soft, peak))
            for i, t in enumerate(targets):
                g = len(t["labels"])
                row = {"gt": g, "tokens": scores.shape[1], "soft": float(soft[i]), "max": float(peak[i])}
                row.update({f"n{tau}": int(above[i, j]) for j, tau in enumerate(THRESHOLDS)})
                row.update({f"s@{r}": float(top[i, min(r, top.shape[1]) - 1]) for r in RANKS})
                row["s@gt"] = float(top[i, min(g, top.shape[1]) - 1]) if g else float("nan")
                rows.append(row)
    handle.remove()

    keys = list(rows[0])
    data = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    out = os.path.join(args.run, "encoder_scores.csv")
    with open(out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")

    gt = data["gt"]
    print(f"{len(rows)} images, {int(gt.sum())} objects, {data['tokens'][0]:.0f} tokens an image")
    print(f"median objects {np.median(gt):.0f}, mean {gt.mean():.1f}, max {gt.max():.0f}\n")

    print("the score of the token at a given rank (percentiles over images)")
    print(f"{'rank':>8} {'p5':>7} {'p25':>7} {'median':>7} {'p75':>7} {'p95':>7}")
    for key in [f"s@{r}" for r in RANKS] + ["s@gt"]:
        v = data[key][~np.isnan(data[key])]
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"{key:>8} " + " ".join(f"{x:7.4f}" for x in q))

    print("\ntokens above a threshold against the object count")
    print(f"{'tau':>6} {'mean n':>8} {'median n':>9} {'n>=gt':>7} {'corr':>6} {'median n/gt':>12}")
    for tau in THRESHOLDS:
        n, ok = data[f"n{tau}"], gt > 0
        print(
            f"{tau:6.2f} {n.mean():8.1f} {np.median(n):9.0f} {100 * (n >= gt).mean():6.1f}% "
            f"{np.corrcoef(n[ok], gt[ok])[0, 1]:6.3f} {np.median(n[ok] / gt[ok]):12.2f}"
        )

    dense = gt >= max(args.dense, args.range[0])
    if dense.sum():
        print(
            f"\nthe images the threshold decides anything on ({int(dense.sum())} with gt >= {max(args.dense, args.range[0])})"
        )
        print(f"{'tau':>6} {'median n':>9} {'median n/gt':>12} {'n>=gt':>7} {'p10 n/gt':>9}")
        for tau in THRESHOLDS:
            n = data[f"n{tau}"][dense]
            print(
                f"{tau:6.2f} {np.median(n):9.0f} {np.median(n / gt[dense]):12.2f} "
                f"{100 * (n >= gt[dense]).mean():6.1f}% {np.percentile(n / gt[dense], 10):9.2f}"
            )

    lo, hi = args.range
    print(f"\nthe budget rule: round_up(n(tau) + margin, {args.round}) clamped to [{lo}, {hi}]")
    print(f"{'tau':>6} {'margin':>7} {'mean budget':>12} {'covers gt':>10} {'gt missed':>10}")
    for tau in THRESHOLDS:
        n = data[f"n{tau}"]
        for m in MARGINS:
            b = np.clip(np.ceil((n + m) / args.round) * args.round, lo, hi)
            missed = np.clip(gt - b, 0, None).sum()
            print(f"{tau:6.2f} {m:7d} {b.mean():12.1f} {100 * (b >= gt).mean():9.2f}% {100 * missed / gt.sum():9.2f}%")
    print("\nfixed budgets, for comparison")
    for k in (300, 500, 1000, 1500):
        missed = np.clip(gt - k, 0, None).sum() / gt.sum() * 100
        print(f"{'fixed':>6} {k:7d} {k:12d} {100 * (k >= gt).mean():9.2f}% {missed:9.2f}%")
    print(f"\nper-image rows written to {out}")


if __name__ == "__main__":
    main()
