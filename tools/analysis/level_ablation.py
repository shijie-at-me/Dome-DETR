"""
What the decoder's deformable cross-attention needs, by ablation at inference: a trained model
is evaluated through the run's own evaluator with its attention weights edited, and the AP per
object size is compared to the untouched model. Two edits: whole levels zeroed (``--drop``), and
each head keeping only its k highest-weighted points per level (``--points``), which tells
whether a head's points are redundant, the case when they collapse into one cell. The rest is
renormalized per head. The model was trained with everything in place, so a drop is an upper
bound on what the removed part contributes.

Written into ``<run>/probe/level_ablation.md``.

    python tools/analysis/level_ablation.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/level_ablation.py <run> --drop 32 16,32 --points 1 2 1@8,16,32 --layers 1 2
"""

import argparse
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import find_decoder, load_model, pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.data.dataset.aitod_eval import AITODCOCOeval  # noqa: E402
from src.data.dataset.visdrone_eval import VisDroneCOCOeval  # noqa: E402
from src.solver.det_engine import evaluate  # noqa: E402

# finer size buckets than COCO's, in original-image pixels (square root of the area)
EXTRA_AREAS = {"tiny": [0, 16**2], "16-32": [16**2, 32**2]}


def add_area_ranges():
    """VisDroneCOCOeval scores the extra buckets too (its summarize is unchanged)."""
    original = VisDroneCOCOeval.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for label, rng in EXTRA_AREAS.items():
            self.params.areaRng.append(rng)
            self.areaRngLbl.append(label)
        self.params.areaRngLbl = self.areaRngLbl

    VisDroneCOCOeval.__init__ = patched


class AttentionEditor:
    """
    Edits the cross-attention weights of the chosen decoder layers: the points of ``drop_levels``
    are zeroed, and on ``topk_levels`` every head keeps only its ``topk`` highest-weighted points
    per level. The weights are then renormalized per head.
    """

    def __init__(self, decoder, layers, drop_levels=(), topk=None, topk_levels=()):
        self.decoder = decoder
        self.layers = layers
        self.drop_levels = list(drop_levels)
        self.topk = topk
        self.topk_levels = list(topk_levels)
        self.originals = []

    def __enter__(self):
        for i, layer in enumerate(self.decoder.layers):
            if i not in self.layers or not (self.drop_levels or self.topk):
                continue
            attn = layer.cross_attn
            device = attn.point_level.device
            n_points = len(attn.point_level)  # the null entry, if any, sits after the points and is left alone
            keep = ~torch.isin(attn.point_level, torch.tensor(self.drop_levels, device=device))
            groups = [torch.where(attn.point_level == lv)[0] for lv in self.topk_levels] if self.topk else []

            def edit(weights, _keep=keep, _groups=groups, _n=n_points):
                # weights: [bs, Q, H, P (+1)], softmaxed per head over every level (and the null entry)
                points = weights[..., :_n] * _keep.to(weights.dtype)
                for idx in _groups:  # the top-k of every head's points of the level
                    sub = points[..., idx]
                    mask = torch.zeros_like(sub).scatter(-1, sub.topk(min(self.topk, len(idx)), dim=-1).indices, 1.0)
                    points = points.clone()
                    points[..., idx] = sub * mask
                weights = torch.cat([points, weights[..., _n:]], dim=-1)
                return weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)

            attn.edit_weights = edit
            self.originals.append(attn)
        return self

    def __exit__(self, *exc):
        for attn in self.originals:
            attn.edit_weights = None


def run_eval(cfg, model, evaluator, device):
    """AP and AR per size bucket, in percent, of one validation pass (VisDrone's buckets or AI-TOD's)."""
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        stats, _ = evaluate(model, cfg.criterion, cfg.postprocessor, cfg.val_dataloader, evaluator, device)
        coco = evaluator.coco_eval["bbox"]
        s = stats["coco_eval_bbox"]
        if isinstance(coco, AITODCOCOeval):
            row = {
                "AP": s[0],
                "AP50": s[1],
                "AP75": s[2],
                "AP vt": s[3],
                "AP t": s[4],
                "AP s": s[5],
                "AP m": s[6],
                "AR vt": s[10],
                "AR t": s[11],
                "AR s": s[12],
                "AR": s[9],
            }
        else:
            max_det = coco.params.maxDets[-1]
            row = {
                "AP": s[0],
                "AP50": s[1],
                "AP75": s[2],
                "AP tiny": coco._summarize(1, areaRng="tiny", maxDets=max_det),
                "AP 16-32": coco._summarize(1, areaRng="16-32", maxDets=max_det),
                "AP_s": s[3],
                "AP_m": s[4],
                "AP_l": s[5],
                "AR tiny": coco._summarize(0, areaRng="tiny", maxDets=max_det),
                "AR 16-32": coco._summarize(0, areaRng="16-32", maxDets=max_det),
                "AR": s[9],
            }
    return {k: 100 * v for k, v in row.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output directory of train.py (config.yml and checkpoints)")
    parser.add_argument("--checkpoint", help="checkpoint file in the run (default: best_stg2, best_stg1, then last)")
    parser.add_argument(
        "--drop",
        nargs="*",
        default=["4", "8", "16", "32", "16,32", "8,16,32"],
        help="stride sets to drop, one condition each, comma-separated strides (default: each level, then 16+32 and 8+16+32)",
    )
    parser.add_argument(
        "--points",
        nargs="*",
        default=["1", "2", "1@8,16,32"],
        help="points every head keeps per level, one condition each: k, or k@strides to edit those levels only (default: 1, 2, 1@8,16,32)",
    )
    parser.add_argument("--layers", type=int, nargs="*", help="decoder layers to drop from (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    print(f"checkpoint {checkpoint}")
    add_area_ranges()
    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    decoder = find_decoder(model)
    evaluator = cfg.evaluator
    strides = list(model.decoder.feat_strides)
    if getattr(model.decoder, "fine_channels", 0) > 0:  # the fine level is the cross-attention's first
        strides = [strides[0] // 2, *strides]
    layers = list(range(decoder.num_layers)) if not args.layers else args.layers
    print(f"decoder: {decoder.num_layers} layers, strides {strides}; dropping from layers {layers}")

    rows = [("none", run_eval(cfg, model, evaluator, args.device))]
    tiny_key = "AP tiny" if "AP tiny" in rows[-1][1] else "AP t"
    print(f"  none: AP {rows[-1][1]['AP']:.1f}, tiny {rows[-1][1][tiny_key]:.1f}")
    for spec in args.drop:
        dropped = [int(s) for s in spec.split(",")]
        levels = [strides.index(s) for s in dropped]
        with AttentionEditor(decoder, layers, drop_levels=levels):
            row = run_eval(cfg, model, evaluator, args.device)
        rows.append((f"drop stride {' + '.join(map(str, dropped))}", row))
        print(f"  drop {spec}: AP {row['AP']:.1f}, tiny {row[tiny_key]:.1f}")
    for spec in args.points:
        k, _, which = spec.partition("@")
        edited = [int(s) for s in which.split(",")] if which else strides
        with AttentionEditor(decoder, layers, topk=int(k), topk_levels=[strides.index(s) for s in edited]):
            row = run_eval(cfg, model, evaluator, args.device)
        name = f"top-{k} point{'s' if int(k) > 1 else ''} per head" + (
            f" on stride {' + '.join(map(str, edited))}" if which else ""
        )
        rows.append((name, row))
        print(f"  points {spec}: AP {row['AP']:.1f}, tiny {row[tiny_key]:.1f}")

    keys = list(rows[0][1].keys())
    base = rows[0][1]
    lines = [
        "| edit | " + " | ".join(keys) + " |",
        "| --- | " + " | ".join("---:" for _ in keys) + " |",
    ]
    for name, row in rows:
        cells = [f"{row[c]:.1f}" if name == "none" else f"{row[c]:.1f} ({row[c] - base[c]:+.1f})" for c in keys]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = (
        f"# Cross-attention level ablation of `{os.path.relpath(checkpoint).replace(os.sep, '/')}`\n\n"
        f"Edits to the cross-attention weights of decoder layers {layers}, the rest renormalized per head; nothing is "
        f"retrained. 'drop' zeroes a level's points; 'top-k points' keeps, for every head and level, only its k "
        f"highest-weighted points of the {decoder.layers[0].cross_attn.num_points_list[0]} it has there. Sizes are square "
        f"roots of the COCO area in original pixels: tiny < 16, 16-32, then COCO's small (< 32), medium and large. "
        f"Differences to the untouched model in brackets.\n\n" + "\n".join(lines) + "\n"
    )
    out = os.path.join(args.run, "probe")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "level_ablation.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
