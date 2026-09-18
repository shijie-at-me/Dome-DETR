"""
What the training pipeline shows the model against what validation asks of it: the object
sizes in network-input pixels after the training transforms (with the augmentations on and
off, and after the multi-scale collate) next to the sizes at the validation resize, and how
many ground-truth boxes the crops drop. A gap here is a domain shift the model pays for at
validation.

    python tools/analysis/train_scale_probe.py outputs/dfine_s_visdrone/2026-09-09_17-13-58 --num-images 300
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core import YAMLConfig  # noqa: E402

EDGES = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 1e9]


def sizes_after_train(dataset, collate, indices, epoch, rng):
    """Object sizes (sqrt area, input px) after the transforms at ``epoch`` and one multi-scale draw, and the boxes kept per image."""
    dataset.set_epoch(epoch)
    collate.set_epoch(epoch)
    sizes, kept, total = [], 0, 0
    for i in indices:
        image, target = dataset[i]
        _, h, w = image.shape
        scale = 1.0
        if collate.scales is not None and epoch < collate.stop_epoch:
            sh, sw = rng.choice(collate.scales)
            scale = ((sh / h) * (sw / w)) ** 0.5
        boxes = target["boxes"].as_subclass(torch.Tensor)  # normalized cxcywh after ConvertBoxes
        area = boxes[:, 2] * w * boxes[:, 3] * h
        sizes.append(area.sqrt() * scale)
        kept += len(boxes)
        total += len(dataset.load_item(i)[1]["boxes"])
    return torch.cat(sizes).numpy(), kept, total


def sizes_at_val(dataset, indices):
    sizes = []
    for i in indices:
        image, target = dataset[i]
        boxes = target["boxes"].as_subclass(torch.Tensor)  # xyxy in resized px
        sizes.append(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sqrt())
    return torch.cat(sizes).numpy()


def sizes_original(dataset, indices):
    sizes = []
    for i in indices:
        boxes = dataset.load_item(i)[1]["boxes"].as_subclass(torch.Tensor)
        sizes.append(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sqrt())
    return torch.cat(sizes).numpy()


def describe(name, sizes):
    q = np.percentile(sizes, [10, 25, 50, 75, 90])
    hist = np.histogram(sizes, EDGES)[0] / len(sizes)
    return [name, len(sizes), *[f"{v:.1f}" for v in q], *[f"{100 * v:.1f}%" for v in hist]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    train_loader, val_loader = cfg.train_dataloader, cfg.val_dataloader
    train, val = train_loader.dataset, val_loader.dataset
    collate = train_loader.collate_fn
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train_idx = rng.sample(range(len(train)), min(args.num_images, len(train)))
    val_idx = rng.sample(range(len(val)), min(args.num_images, len(val)))
    stop = collate.stop_epoch
    print(
        f"train transforms: {[op['type'] for op in cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops']]}"
    )
    print(f"multi-scale sizes: {collate.scales}, until epoch {stop}")

    rows = []
    rows.append(describe("train, original pixels", sizes_original(train, train_idx)))
    on, kept_on, total_on = sizes_after_train(train, collate, train_idx, 0, rng)
    rows.append(describe("train input, augmentation + multi-scale on (epoch 0)", on))
    off, kept_off, total_off = sizes_after_train(train, collate, train_idx, stop, rng)
    rows.append(describe(f"train input, augmentation off (epoch {stop})", off))
    rows.append(describe("val, original pixels", sizes_original(val, val_idx)))
    at_val = sizes_at_val(val, val_idx)
    rows.append(describe("val input (the validation resize)", at_val))

    headers = ["objects", "n", "p10", "p25", "p50", "p75", "p90"] + [
        f"<{int(hi)}" if hi < 1e9 else f">={int(lo)}" for lo, hi in zip(EDGES[:-1], EDGES[1:])
    ]
    lines = ["| " + " | ".join(headers) + " |", "| --- | " + " | ".join("---:" for _ in headers[1:]) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    text = (
        f"# Object sizes through the pipelines of `{os.path.relpath(args.run).replace(os.sep, '/')}`\n\n"
        f"{len(train_idx)} training and {len(val_idx)} validation images. Sizes are square-root box areas; the percentile columns are pixels, the rest the share of boxes in each pixel band.\n\n"
        + "\n".join(lines)
        + f"\n\nBoxes kept by the training crops: {100 * kept_on / total_on:.1f}% with the augmentations on, {100 * kept_off / total_off:.1f}% with them off (of the boxes in the original images).\n"
    )
    out = os.path.join(args.run, "probe")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "train_scale.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
