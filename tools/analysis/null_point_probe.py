"""
What the null entry of the cross-attention heads (``null_point``) does on a trained run, without
retraining: the weight the heads put on it, split by what the query is (a detection or
background, and per decoder layer), and the run's AP when it is switched off at inference in
two ways: the null column zeroed and the remaining points renormalized (the heads sample as if
the entry never existed), and the null vector zeroed while the abstaining weight stays (the
heads still abstain, but nothing is added). Both through the run's own evaluator.

    python tools/analysis/null_point_probe.py outputs/dfine_s_visdrone_ours/<run>
"""

import argparse
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import load_model, pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.nn.deformable_attention import MSDeformableAttention  # noqa: E402
from src.solver.det_engine import evaluate  # noqa: E402


def cross_attentions(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, MSDeformableAttention) and m.null_point]


def run_eval(cfg, model, device):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        stats, _ = evaluate(model, cfg.criterion, cfg.postprocessor, cfg.val_dataloader, cfg.evaluator, device)
    return stats["coco_eval_bbox"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint")
    ap.add_argument("--stat-images", type=int, default=64, help="images for the per-query statistics")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    attns = cross_attentions(model)
    if not attns:
        sys.exit("the run has no null_point cross-attention")
    print(f"checkpoint {checkpoint}: {len(attns)} cross-attentions with a null entry")

    # 1. the null weight per query, against the query's final score
    records = {i: [] for i in range(len(attns))}

    def recorder(i):
        def edit(w):  # [bs, len_q, heads, P+1]
            records[i].append(w[..., -1].float().mean(-1).cpu())  # mean over heads, [bs, len_q]
            return w

        return edit

    for i, (_, m) in enumerate(attns):
        m.edit_weights = recorder(i)
    loader = cfg.val_dataloader
    scores, seen = [], 0
    with torch.no_grad():
        for images, targets in loader:
            out = model(images.to(args.device))
            scores.append(out["pred_logits"].float().sigmoid().max(-1).values.cpu())
            seen += len(targets)
            if seen >= args.stat_images:
                break
    for _, m in attns:
        m.edit_weights = None
    scores = torch.cat(scores).flatten()
    print(f"\n### weight on the null entry, mean over heads ({seen} validation images, {len(scores)} queries)\n")
    print(
        "| layer | all queries | background (score < 0.1) | uncertain (0.1-0.3) | detections (score >= 0.3) | uniform share |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for i, (name, m) in enumerate(attns):
        w = torch.cat(records[i]).flatten()
        bins = [(scores < 0.1), (scores >= 0.1) & (scores < 0.3), scores >= 0.3]
        cells = " | ".join(f"{100 * float(w[b].mean()):.1f}%" for b in bins)
        print(f"| {name.split('.')[-2]} | {100 * float(w.mean()):.1f}% | {cells} | {100 / (m.total_points + 1):.1f}% |")

    # 2. AP with the null entry disabled two ways
    def renormalize(w):
        w = w.clone()
        w[..., -1] = 0
        return w / w.sum(-1, keepdim=True).clamp_min(1e-6)

    labels = ["AP", "AP50", "AP75"]
    print("\n| variant | " + " | ".join(labels) + " | AR |")
    print("|---|" + "---:|" * (len(labels) + 1))
    rows = [
        ("as trained", None, False),
        ("null column zeroed, points renormalized", renormalize, False),
        ("null vector zeroed, weights as trained", None, True),
    ]
    saved = [m.null_value.detach().clone() for _, m in attns]
    for title, edit, zero_value in rows:
        for (_, m), v in zip(attns, saved):
            m.edit_weights = edit
            with torch.no_grad():
                m.null_value.copy_(torch.zeros_like(v) if zero_value else v)
        stats = run_eval(cfg, model, args.device)
        print(
            f"| {title} | " + " | ".join(f"{100 * s:.1f}" for s in stats[:3]) + f" | {100 * stats[9]:.1f} |", flush=True
        )
    for (_, m), v in zip(attns, saved):
        m.edit_weights = None
        with torch.no_grad():
            m.null_value.copy_(v)


if __name__ == "__main__":
    main()
