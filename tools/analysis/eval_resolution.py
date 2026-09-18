"""
The validation AP of a checkpoint at other input resolutions, through the run's own evaluator:
the short side of the aspect-preserving resize is swept (the long-side cap scales with it), and
the paper's squashed square protocol can be added for reference. Tells whether the tiny
objects are pixel-limited, at what cost per image.

    python tools/analysis/eval_resolution.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/eval_resolution.py <run> --short 640 800 960 1120 --squash 800
"""

import argparse
import contextlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.solver.det_engine import evaluate  # noqa: E402


def resize_op(cfg):
    ops = cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    op = next(o for o in ops if o["type"] == "Resize")
    return op


def run(config_path, weights, device, edit, label):
    cfg = YAMLConfig(config_path)
    edit(cfg.yaml_cfg)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    model.load_state_dict(weights)
    model = model.to(device).eval()
    loader = cfg.val_dataloader
    torch.cuda.synchronize() if device.startswith("cuda") else None
    start = time.time()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        stats, _ = evaluate(model, cfg.criterion, cfg.postprocessor, loader, cfg.evaluator, device)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    seconds = (time.time() - start) / len(loader.dataset)
    return stats["coco_eval_bbox"], seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--short",
        type=int,
        nargs="+",
        default=[640, 800, 960, 1120, 1280],
        help="short sides of the aspect-preserving resize",
    )
    parser.add_argument(
        "--squash",
        type=int,
        nargs="*",
        default=[800],
        help="square sizes every image is squashed to (the paper's protocol)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    config_path = os.path.join(args.run, "config.yml")
    base = YAMLConfig(config_path).yaml_cfg
    op = resize_op(base)
    # a square resize (AI-TOD's tiles): the sweep is over the square's side
    square_run = isinstance(op["size"], (list, tuple))
    ratio = 1.0 if square_run else op.get("max_size", op["size"] * 1333 / 800) / op["size"]
    print(f"checkpoint {checkpoint}; the run validated with {op}")

    rows = []
    for short in args.short:

        def edit(cfg, short=short):
            o = resize_op(cfg)
            if square_run:
                o["size"] = [short, short]
            else:
                o["size"], o["max_size"] = short, int(round(short * ratio))
            if short > 960:  # the batch would not fit the memory the run's batch size was chosen for
                cfg["val_dataloader"]["total_batch_size"] = max(1, cfg["val_dataloader"]["total_batch_size"] // 4)

        stats, seconds = run(config_path, weights, args.device, edit, f"short {short}")
        name = f"{short}x{short}" if square_run else f"short side {short}, long side <= {int(round(short * ratio))}"
        rows.append((name, stats, seconds))
        print(f"  short {short}: AP {100 * stats[0]:.1f}, {1000 * seconds:.0f} ms/img")
    for square in args.squash:

        def edit(cfg, square=square):
            o = resize_op(cfg)
            o["size"] = [square, square]
            o.pop("max_size", None)
            cfg["val_dataloader"]["collate_fn"].pop("pad_to_multiple", None)

        stats, seconds = run(config_path, weights, args.device, edit, f"squash {square}")
        rows.append((f"squashed to {square}x{square}", stats, seconds))
        print(f"  squash {square}: AP {100 * stats[0]:.1f}, {1000 * seconds:.0f} ms/img")

    labels = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"]
    lines = [
        "| validation input | " + " | ".join(labels) + " | AR (max det) | ms / image |",
        "| --- | " + " | ".join("---:" for _ in labels) + " | ---: | ---: |",
    ]
    for name, stats, seconds in rows:
        lines.append(
            f"| {name} | "
            + " | ".join(f"{100 * s:.1f}" for s in stats[:6])
            + f" | {100 * stats[9]:.1f} | {1000 * seconds:.0f} |"
        )
    text = (
        f"# Validation resolution sweep of `{os.path.relpath(checkpoint).replace(os.sep, '/')}`\n\nThe model is unchanged; only the validation resize differs. Timing is the whole evaluation loop per image on {args.device}.\n\n"
        + "\n".join(lines)
        + "\n"
    )
    out = os.path.join(args.run, "probe")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "resolution.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
