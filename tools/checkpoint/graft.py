"""
Graft a trained checkpoint onto a model of another config: every tensor whose name and shape
match is copied, a deformable attention's sampling offsets and attention weights are copied
level by level when the levels or the points per level differ, and the rest keeps the new
model's initialisation. The result is a resumable checkpoint (model, EMA, epoch, best AP; no
optimizer state) of the new config, so that a switch can be trained from a baseline's
stage-1 checkpoint instead of from scratch.

    python tools/checkpoint/graft.py -c configs/dome/ablation/DFine-S-AITOD-2-fine-from120.yml \
        --source outputs/dfine_s_aitod/<run>/checkpoint0119.pth -o <run dir>/best_stg1.pth

A deformable attention's offsets and weights are laid out ``[heads, sum(points), 2]`` and
``[heads, sum(points) (+1 null)]``, the points level by level. The source levels are matched to
the target's last ones, so a target with one extra level in front keeps the source on its old
levels. Copied offsets are rescaled by the ratio of the points per level, because the forward
divides an offset by its level's point count, so the grafted model samples where the source did.
A new point's weight logit bias starts at the head's lowest copied one, so the new points draw
little until they learn.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.core import YAMLConfig, yaml_utils  # noqa: E402
from src.misc import dist_utils  # noqa: E402
from src.nn.deformable_attention import MSDeformableAttention  # noqa: E402

ATTENTION_KEYS = (
    "sampling_offsets.weight",
    "sampling_offsets.bias",
    "attention_weights.weight",
    "attention_weights.bias",
)


def level_slices(points: list[int]) -> list[slice]:
    """The slice of each level's points in the ``sum(points)`` axis."""
    out, start = [], 0
    for n in points:
        out.append(slice(start, start + n))
        start += n
    return out


def graft_attention(module: MSDeformableAttention, prefix: str, target: dict, source: dict, source_points: list[int]):
    """Copy the source's offsets and weights into ``target``'s tensors of ``prefix`` level by level."""
    heads = module.num_heads
    tgt_points = module.num_points_list
    shift = len(tgt_points) - len(source_points)
    assert shift >= 0, (
        f"{prefix}: the source has more levels ({len(source_points)}) than the target ({len(tgt_points)})"
    )
    src_slices, tgt_slices = level_slices(source_points), level_slices(tgt_points)

    # offsets: weight [heads * P * 2, C] -> [heads, P, 2, C], bias [heads * P * 2] -> [heads, P, 2]
    ow = target[f"{prefix}.sampling_offsets.weight"].view(heads, sum(tgt_points), 2, -1)
    ob = target[f"{prefix}.sampling_offsets.bias"].view(heads, sum(tgt_points), 2)
    sw = source[f"{prefix}.sampling_offsets.weight"].view(heads, sum(source_points), 2, -1)
    sb = source[f"{prefix}.sampling_offsets.bias"].view(heads, sum(source_points), 2)
    # weights: [heads * (P + null), C] and [heads * (P + null)]
    n_tgt = sum(tgt_points) + (1 if module.null_point else 0)
    aw = target[f"{prefix}.attention_weights.weight"].view(heads, n_tgt, -1)
    ab = target[f"{prefix}.attention_weights.bias"].view(heads, n_tgt)
    n_src = source[f"{prefix}.attention_weights.bias"].numel() // heads
    saw = source[f"{prefix}.attention_weights.weight"].view(heads, n_src, -1)
    sab = source[f"{prefix}.attention_weights.bias"].view(heads, n_src)

    copied = torch.zeros(sum(tgt_points), dtype=torch.bool)
    for src_level, src_slice in enumerate(src_slices):
        tgt_level = src_level + shift
        n = min(source_points[src_level], tgt_points[tgt_level])
        t = slice(tgt_slices[tgt_level].start, tgt_slices[tgt_level].start + n)
        s = slice(src_slice.start, src_slice.start + n)
        scale = tgt_points[tgt_level] / source_points[src_level]  # the forward divides by the level's count
        ow[:, t] = sw[:, s] * scale
        ob[:, t] = sb[:, s] * scale
        aw[:, t] = saw[:, s]
        ab[:, t] = sab[:, s]
        copied[t] = True
    new = ~copied
    if new.any():
        points_bias = ab[:, : sum(tgt_points)]
        floor = points_bias[:, copied].min(dim=1).values  # [heads]
        for head in range(heads):
            points_bias[head, new] = floor[head]
    return int(copied.sum()), int(new.sum())


def graft(model: torch.nn.Module, source: dict, source_points: list[int]) -> tuple[dict, dict]:
    """The model's state dict with the source's tensors grafted in, and what happened to each name."""
    target = {k: v.clone() for k, v in model.state_dict().items()}
    report = {"copied": [], "grafted": [], "new": [], "unmatched": []}
    handled = set()
    for name, module in model.named_modules():
        if not isinstance(module, MSDeformableAttention):
            continue
        keys = [f"{name}.{p}" for p in ATTENTION_KEYS]
        if all(k in source for k in keys) and any(source[k].shape != target[k].shape for k in keys):
            n_copied, n_new = graft_attention(module, name, target, source, source_points)
            report["grafted"].append(f"{name} ({n_copied} points copied, {n_new} new)")
            handled.update(keys)
    for k, v in target.items():
        if k in handled:
            continue
        if k not in source:
            report["new"].append(k)
        elif source[k].shape != v.shape:
            report["unmatched"].append(k)
        else:
            target[k] = source[k].clone()
            report["copied"].append(k)
    return target, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="the config of the model to graft onto")
    parser.add_argument(
        "--source", required=True, help="the checkpoint to graft (a solver checkpoint with model and ema)"
    )
    parser.add_argument("-o", "--output", required=True, help="where to write the grafted checkpoint")
    parser.add_argument(
        "--source-points", default="4,4,4,4", help="the source decoder's points per level (default: D-FINE's 4,4,4,4)"
    )
    parser.add_argument("-u", "--update", nargs="+", default=[], help="config overrides, as for train.py")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, **yaml_utils.parse_cli(args.update))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    source_points = [int(x) for x in args.source_points.split(",")]

    state = torch.load(args.source, map_location="cpu")
    out = {"date": state.get("date"), "last_epoch": state["last_epoch"], "grafted_from": os.path.abspath(args.source)}
    for k in ("best_ap", "best_epoch"):
        if k in state:
            out[k] = state[k]
    src_model = dist_utils.remove_module_prefix(state["model"])
    out["model"], report = graft(model, src_model, source_points)
    if "ema" in state:
        ema_state, _ = graft(model, dist_utils.remove_module_prefix(state["ema"]["module"]), source_points)
        out["ema"] = {"module": ema_state, "updates": state["ema"].get("updates", 0)}

    print(f"copied {len(report['copied'])} tensors")
    for key in ("grafted", "new", "unmatched"):
        print(f"{key} ({len(report[key])}):")
        for name in report[key]:
            print(f"  {name}")
    n_new = sum(out["model"][k].numel() for k in report["new"])
    print(f"new parameters: {n_new / 1e6:.3f} M of {sum(v.numel() for v in out['model'].values()) / 1e6:.3f} M")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"wrote {args.output} (last_epoch {out['last_epoch']}, best_ap {out.get('best_ap')})")


if __name__ == "__main__":
    main()
