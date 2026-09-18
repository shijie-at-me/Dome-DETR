"""
Does the encoder's score say what the selection needs on AI-TOD? On a trained baseline, per
ground-truth box: the encoder proposal's IoU against the final box's IoU of the same query
(the VFL target under enc_quality_source own vs final), the rank correlation of the encoder's
score with either, and the selection headroom (queries ranked 300..999 that end up on a box
no top-300 query reaches).

    python tools/analysis/enc_quality_probe.py outputs/dfine_s_aitod/<run> [--num-images 800] [--batch 8] [--checkpoint best_stg2.pth]
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "analysis"))

from fdr_refinement import find_decoder, load_model, pick_checkpoint, run_all_layers  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou  # noqa: E402

BUCKETS = [("vt <8", 0, 8), ("t 8-16", 8, 16), ("s 16-32", 16, 32), ("m 32-64", 32, 64), ("l >64", 64, 1e9)]
NAMES = [b[0] for b in BUCKETS] + ["all"]


def bucket_of(size):
    return next(n for n, lo, hi in BUCKETS if lo <= size < hi)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def probe_batch(model, decoder, images, targets, device, enc_scores_holder):
    out = run_all_layers(model, decoder, images.to(device))
    B, Q = out["proposal"].shape[:2]
    size = torch.tensor([images.shape[-1], images.shape[-2]], dtype=torch.float32).repeat(2)
    prop = box_cxcywh_to_xyxy(out["proposal"]) * size  # [B,Q,4] px
    final = box_cxcywh_to_xyxy(out["boxes"][:, -1]) * size
    final_scores = out["logits"][:, -1].sigmoid()  # [B,Q,C]
    enc_full = enc_scores_holder["logits"]  # [B,T,C]
    enc_sel = torch.topk(enc_full.max(-1).values, Q, dim=-1).values.sigmoid().cpu()  # [B,Q] sorted desc
    gt_records, q_records = [], []
    for b in range(B):
        gt = targets[b]["boxes"].as_subclass(torch.Tensor).float()
        labels = targets[b]["labels"]
        if len(gt) == 0:
            continue
        iou_p, _ = box_iou(gt, prop[b])  # [G,Q]
        iou_f, _ = box_iou(gt, final[b])
        sizes = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).clamp(min=0).sqrt()
        # the detection of every gt: highest class score among queries with final IoU >= 0.5
        cls_scores = final_scores[b][:, labels]  # [Q,G]
        det_score, det_q = torch.where(iou_f.T >= 0.5, cls_scores, -1.0).max(0)  # [G]
        best_p, best_pq = iou_p.max(1)
        best_f, best_fq = iou_f.max(1)
        for g in range(len(gt)):
            q = int(det_q[g])
            gt_records.append(
                {
                    "size": float(sizes[g]),
                    "n_gt": int(len(gt)),
                    "best_final_enc": float(enc_sel[b, best_fq[g]]),
                    "cut_enc": float(enc_sel[b, min(299, Q - 1)]),
                    "bucket": bucket_of(float(sizes[g])),
                    "best_prop": float(best_p[g]),
                    "final_of_best_prop": float(iou_f[g, best_pq[g]]),
                    "best_final": float(best_f[g]),
                    "prop_of_best_final": float(iou_p[g, best_fq[g]]),
                    "rank_best_final": int(best_fq[g]),
                    "det": bool(det_score[g] >= 0),
                    "det_prop": float(iou_p[g, q]) if det_score[g] >= 0 else float("nan"),
                    "det_final": float(iou_f[g, q]) if det_score[g] >= 0 else float("nan"),
                    "det_score": float(det_score[g]) if det_score[g] >= 0 else float("nan"),
                    "det_rank": q if det_score[g] >= 0 else -1,
                    "det_enc_score": float(enc_sel[b, q]) if det_score[g] >= 0 else float("nan"),
                }
            )
        # per query: its encoder score, its own (proposal) IoU and its final IoU with the gt its final box overlaps most
        fq_best, fq_g = iou_f.max(0)  # [Q]
        own_same = iou_p[fq_g, torch.arange(Q)]
        for q in range(Q):
            q_records.append(
                {
                    "enc": float(enc_sel[b, q]),
                    "n_gt": int(len(gt)),
                    "own": float(own_same[q]),
                    "final": float(fq_best[q]),
                    "final_cls": float(final_scores[b, q].max()),
                    "bucket": bucket_of(float(sizes[fq_g[q]])),
                }
            )
    return gt_records, q_records


def gt_table(records, title):
    print(f"\n### {title}: per ground-truth box, top-{records[0]['Q']} queries\n")
    print(
        "| bucket | boxes | prop IoU>=.5 | final IoU>=.5 | detected | mean prop / final IoU of the detection | det with prop<.5 | det with prop<.3 | det rank p50 / p90 | det rank>=200 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in NAMES:
        rs = [r for r in records if name == "all" or r["bucket"] == name]
        if not rs:
            continue
        det = [r for r in rs if r["det"]]
        n = len(rs)
        p = np.array([r["det_prop"] for r in det])
        f = np.array([r["det_final"] for r in det])
        ranks = np.array([r["det_rank"] for r in det])
        print(
            f"| {name} | {n} | {100 * np.mean([r['best_prop'] >= 0.5 for r in rs]):.1f}% | {100 * np.mean([r['best_final'] >= 0.5 for r in rs]):.1f}% | "
            f"{100 * len(det) / n:.1f}% | {p.mean() if len(det) else float('nan'):.3f} / {f.mean() if len(det) else float('nan'):.3f} | "
            f"{100 * (p < 0.5).mean() if len(det) else float('nan'):.1f}% | {100 * (p < 0.3).mean() if len(det) else float('nan'):.1f}% | "
            f"{np.percentile(ranks, 50) if len(det) else float('nan'):.0f} / {np.percentile(ranks, 90) if len(det) else float('nan'):.0f} | "
            f"{100 * (ranks >= 200).mean() if len(det) else float('nan'):.1f}% |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint")
    ap.add_argument("--num-images", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--wide", type=int, default=1000, help="the wider budget of the headroom pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    print("checkpoint", checkpoint)
    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    decoder = find_decoder(model)
    loader = cfg.val_dataloader
    dataset, collate = loader.dataset, loader.collate_fn
    holder = {}
    model.decoder.enc_score_head.register_forward_hook(lambda m, i, o: holder.__setitem__("logits", o.detach().float()))

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.num_images)
    print(
        f"{len(dataset)} test images, probing {len(indices)}; num_queries {model.decoder.num_queries}, budget {model.decoder.query_budget}"
    )

    results = {}
    for Q in (model.decoder.num_queries, args.wide):
        model.decoder.num_queries = Q
        gt_all, q_all = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(indices), args.batch):
                items = [dataset[i] for i in indices[s : s + args.batch]]
                images, targets = collate(items)
                g, q = probe_batch(model, decoder, images, targets, args.device, holder)
                for r in g:
                    r["Q"] = Q
                gt_all += g
                q_all += q
        results[Q] = (gt_all, q_all)
        print(f"Q={Q}: {len(gt_all)} gt boxes, {len(q_all)} queries")

    Q0 = model.decoder.num_queries = list(results)[0]
    gt0, q0 = results[Q0]
    gt_table(gt0, "trained budget")
    gt1, _ = results[args.wide]
    gt_table(gt1, "wide budget")

    print(f"\n### selection headroom: top-{Q0} vs top-{args.wide}\n")
    print(
        "| bucket | boxes | final>=.5 @Q0 | final>=.5 @wide | gained | gained, detected @wide | best-final rank >= Q0 @wide |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in NAMES:
        idx = [i for i, r in enumerate(gt0) if name == "all" or r["bucket"] == name]
        if not idx:
            continue
        a = np.array([gt0[i]["best_final"] >= 0.5 for i in idx])
        b = np.array([gt1[i]["best_final"] >= 0.5 for i in idx])
        det_b = np.array([gt1[i]["det"] for i in idx])
        far = np.array([gt1[i]["rank_best_final"] >= Q0 for i in idx])
        print(
            f"| {name} | {len(idx)} | {100 * a.mean():.1f}% | {100 * b.mean():.1f}% | {100 * (b & ~a).mean():.1f}% | {100 * (b & ~a & det_b).mean():.1f}% | {100 * far.mean():.1f}% |"
        )

    print("\n### selection headroom by tile density (objects per tile)\n")
    print(
        "| tile density | boxes | share of boxes | final>=.5 @Q0 | final>=.5 @wide | gained | gained: best-final rank p50/p90 @wide | gained: enc score gap below the cut, median |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    DENS = [("<=50", 0, 50), ("51-150", 51, 150), ("151-300", 151, 300), (">300", 301, 10**6)]
    for name, lo, hi in DENS:
        idx = [i for i, r in enumerate(gt0) if lo <= r["n_gt"] <= hi]
        if not idx:
            continue
        a = np.array([gt0[i]["best_final"] >= 0.5 for i in idx])
        b = np.array([gt1[i]["best_final"] >= 0.5 for i in idx])
        gained = [i for i in idx if gt1[i]["best_final"] >= 0.5 and gt0[i]["best_final"] < 0.5]
        ranks = np.array([gt1[i]["rank_best_final"] for i in gained]) if gained else np.array([0])
        gaps = np.array([gt1[i]["cut_enc"] - gt1[i]["best_final_enc"] for i in gained]) if gained else np.array([0.0])
        print(
            f"| {name} | {len(idx)} | {100 * len(idx) / len(gt0):.1f}% | {100 * a.mean():.1f}% | {100 * b.mean():.1f}% | {100 * (b & ~a).mean():.1f}% | {np.percentile(ranks, 50):.0f}/{np.percentile(ranks, 90):.0f} | {np.median(gaps):.3f} |"
        )

    print("\n### the encoder's ranking inside sparse tiles (<=150 objects): detections' rank\n")
    print("| bucket | detections | rank p50 / p90 / p99 | rank >= 200 | rank >= 280 |")
    print("|---|---:|---:|---:|---:|")
    for name in NAMES:
        det = [r for r in gt0 if r["det"] and r["n_gt"] <= 150 and (name == "all" or r["bucket"] == name)]
        if len(det) < 5:
            continue
        ranks = np.array([r["det_rank"] for r in det])
        print(
            f"| {name} | {len(det)} | {np.percentile(ranks, 50):.0f} / {np.percentile(ranks, 90):.0f} / {np.percentile(ranks, 99):.0f} | {100 * (ranks >= 200).mean():.1f}% | {100 * (ranks >= 280).mean():.1f}% |"
        )

    print("\n### what the encoder's score tracks, sparse tiles only (<=150 objects)\n")
    print("| bucket | queries | Spearman(enc, own) | Spearman(enc, final) | own<.5 & final>=.5 |")
    print("|---|---:|---:|---:|---:|")
    for name in NAMES:
        rs = [r for r in q0 if (name == "all" or r["bucket"] == name) and r["final"] >= 0.1 and r["n_gt"] <= 150]
        if len(rs) < 10:
            continue
        own = np.array([r["own"] for r in rs])
        fin = np.array([r["final"] for r in rs])
        enc = np.array([r["enc"] for r in rs])
        print(
            f"| {name} | {len(rs)} | {spearman(enc, own):.3f} | {spearman(enc, fin):.3f} | {100 * ((own < 0.5) & (fin >= 0.5)).mean():.1f}% |"
        )

    print(f"\n### what the encoder's score tracks (top-{Q0} queries, final IoU >= 0.1 with some box)\n")
    print(
        "| bucket | queries | Spearman(own, final) | Spearman(enc, own IoU) | Spearman(enc, final IoU) | Spearman(final cls, final IoU) | mean own / final IoU | own<.5 & final>=.5 | own>=.5 & final<.5 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in NAMES:
        rs = [r for r in q0 if (name == "all" or r["bucket"] == name) and r["final"] >= 0.1]
        if len(rs) < 10:
            continue
        own = np.array([r["own"] for r in rs])
        fin = np.array([r["final"] for r in rs])
        enc = np.array([r["enc"] for r in rs])
        fc = np.array([r["final_cls"] for r in rs])
        print(
            f"| {name} | {len(rs)} | {spearman(own, fin):.3f} | {spearman(enc, own):.3f} | {spearman(enc, fin):.3f} | {spearman(fc, fin):.3f} | {own.mean():.3f} / {fin.mean():.3f} | "
            f"{100 * ((own < 0.5) & (fin >= 0.5)).mean():.1f}% | {100 * ((own >= 0.5) & (fin < 0.5)).mean():.1f}% |"
        )

    # the VFL target of the encoder for the detections: own vs final, by bucket
    print("\n### the encoder's VFL target on the detections (own = proposal IoU, final = last layer's IoU)\n")
    print("| bucket | detections | own p25/p50/p75 | final p25/p50/p75 | final - own mean | own < 0.3 |")
    print("|---|---:|---:|---:|---:|---:|")
    for name in NAMES:
        det = [r for r in gt0 if r["det"] and (name == "all" or r["bucket"] == name)]
        if not det:
            continue
        p = np.array([r["det_prop"] for r in det])
        f = np.array([r["det_final"] for r in det])
        print(
            f"| {name} | {len(det)} | {np.percentile(p, 25):.2f}/{np.percentile(p, 50):.2f}/{np.percentile(p, 75):.2f} | "
            f"{np.percentile(f, 25):.2f}/{np.percentile(f, 50):.2f}/{np.percentile(f, 75):.2f} | {(f - p).mean():+.3f} | {100 * (p < 0.3).mean():.1f}% |"
        )


if __name__ == "__main__":
    main()
