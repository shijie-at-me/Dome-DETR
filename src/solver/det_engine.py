"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
import sys
from typing import Iterable

import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tools.visualize_image_annotation import visualize_detection
from tools.concatenate_images import concatenate_images
import os
import concurrent.futures
import time

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA, Warmup

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get('SAVE_INTERMEDIATE_VISUALIZE_RESULT', 'false').lower() == 'true'
SAVE_TP_FP_ANALYSIS = os.environ.get('SAVE_TP_FP_ANALYSIS', 'false').lower() == 'true'

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        no_gt = False
        num_gts = [len(t["labels"]) for t in targets]
        max_gt_num = max(num_gts)
        if max_gt_num == 0: # no gt for denoising will cause error in model forward
            no_gt = True
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            
            for b, target in enumerate(targets):
                image = samples[b].cpu()
                _, H, W = image.shape
                target_cpu = {}
                for k, v in target.items():
                    if k == 'boxes':
                        target_cpu[k] = v.cpu().detach().clone() * torch.tensor([W, H, W, H])
                    else:
                        target_cpu[k] = v.cpu().detach().clone()
                visualize_detection(image, target_cpu, f"sample_gt", return_image=False, type="xywh")

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    output_dir=None,
):
    SAVE_TEST_VISUALIZE_RESULT = os.environ.get('SAVE_TEST_VISUALIZE_RESULT', 'False') == 'True'
    if SAVE_TEST_VISUALIZE_RESULT:
        os.makedirs("visualize_all", exist_ok=True)
        print("Saving visualize results to visualize_all/")
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    # For defe Accuracy calculation
    if model.encoder.use_defe:
        total_defe_samples = 0
        ample_defe_predictions = 0
        total_anchor_num = 0

    MAX_PENDING_TASKS = 256
    predictions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        pending_futures = []
        
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            image_ids = [t["image_id"].item() for t in targets]
            coco = data_loader.dataset.coco
            file_names = [coco.loadImgs(id)[0]['file_name'] for id in image_ids]

            outputs = model(samples, targets=None)
            criterion(outputs, targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            if SAVE_TEST_VISUALIZE_RESULT:
                process_args = []
                scale_factor = float(samples[0].shape[1] / orig_target_sizes[0][0])
                for i in range(len(targets)):
                    sample_cpu = samples[i].cpu()
                    target_cpu = {k: v.cpu() for k, v in targets[i].items()}
                    result_cpu = {k: v.cpu() for k, v in results[i].items()}
                    process_args.append((
                        sample_cpu,
                        target_cpu,
                        result_cpu,
                        file_names[i],
                        scale_factor
                    ))
                
                if len(pending_futures) >= MAX_PENDING_TASKS:
                    while len(pending_futures) > 0:
                        done_futures = []
                        for future in pending_futures:
                            if future.done():
                                done_futures.append(future)
                        
                        for future in done_futures:
                            pending_futures.remove(future)
                        
                        if not done_futures:
                            time.sleep(0.1)

                for args in process_args:
                    future = executor.submit(process_image_pair, args)
                    pending_futures.append(future)

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if SAVE_TP_FP_ANALYSIS:
                coco_dt_existing = coco_evaluator.coco_eval['bbox'].cocoDt
                if coco_dt_existing is None or len(coco_dt_existing.anns) == 0:
                    raise RuntimeError(
                        "coco_evaluator has no accumulated predictions. "
                        "Make sure synchronize_between_processes() was called."
                    )
                for ann in coco_dt_existing.anns.values():
                    predictions.append({
                        "image_id": int(ann["image_id"]),
                        "category_id": int(ann["category_id"]),
                        "bbox": [float(x) for x in ann["bbox"]],
                        "score": float(ann["score"]),
                    })
                print(f"[recovered] {len(predictions)} predictions from coco_evaluator")

            if model.encoder.use_defe:
                # For defe Ample Rate calculation
                pred_defe = outputs['batch_queries_num'][0]
                if pred_defe >= targets[0]['labels'].shape[0]:
                    ample_defe_predictions += 1
                total_defe_samples += 1
                total_anchor_num += outputs['batch_queries_num'][0]

        concurrent.futures.wait(pending_futures)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if model.encoder.use_defe:
        print("defe Ample Rate:", ample_defe_predictions / total_defe_samples)
        print("defe Average Anchor Number:", total_anchor_num / total_defe_samples)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        if SAVE_TP_FP_ANALYSIS:
            # inference only DUMPS raw per-detection records; all analysis/plotting is
            # done later by dedicated code on the saved file (mirrors the vfl_stats workflow).
            raw = extract_tp_fp_raw(
                coco_gt=coco_evaluator.coco_gt,
                predictions=predictions,
                iou_thr=0.5,
                max_dets=500,
            )
            save_dir = str(output_dir) if output_dir is not None else "."
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "tp_fp_raw.npz")
            np.savez(save_path, **raw)
            print(f"[tp_fp] saved {int(raw['det_score'].shape[0])} detections / "
                  f"{int(raw['gt_area'].shape[0])} GTs -> {save_path}")

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator


def process_image_pair(args):
    sample, target, result, filename, scale_factor = args
    sample_img = visualize_detection(sample, target, f"sample_{filename}", return_image=True)
    result_img = visualize_detection(sample, result, f"result_{filename}", 
                                   scale_factor=scale_factor, return_image=True)
    concatenate_images(sample_img, result_img, output_path=f"visualize_all/{filename}")


import numpy as np
from collections import defaultdict

def _box_iou_xywh(b1, b2):
    """IoU between two xywh boxes."""
    x1a, y1a, w1, h1 = b1
    x1b, y1b, w2, h2 = b2
    x2a, y2a = x1a + w1, y1a + h1
    x2b, y2b = x1b + w2, y1b + h2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def extract_tp_fp_raw(coco_gt, predictions, iou_thr=0.5, max_dets=500):
    """COCO-style greedy TP/FP matching at a single IoU threshold, returning FLAT
    per-detection records (NO size binning, NO plotting). Downstream analysis bins /
    computes PR on the saved file, mirroring the vfl_stats workflow.

    Matching, per (image, category): sort detections by score desc, take top `max_dets`,
    greedily assign each to the highest-IoU unclaimed GT with IoU >= iou_thr.

    Deviations from official COCOeval (fine for analysis, NOT a metric replacement):
      * iscrowd GTs are skipped (no crowd-ignore logic);
      * max_dets is applied per (image, category), not per image;
      * a single IoU threshold (no .5:.95 averaging), stored in the output.

    Coordinates: predictions and GT are both in ORIGINAL-image pixels (xywh), since the
    postprocessor rescales boxes to orig_size -- so every `*_area` is original-image px^2
    (the benchmark AP_S/AP_vt convention; this differs from vfl_stats' input space).

    Returns equal-length per-detection arrays (`det_*`) plus a separate GT table (`gt_*`)
    so recall and any size binning can be recomputed downstream.
    """
    if len(predictions) == 0:
        raise RuntimeError("No predictions to analyze.")

    preds_by_key = defaultdict(list)
    for p in predictions:
        preds_by_key[(p['image_id'], p['category_id'])].append(p)

    gts_by_key = defaultdict(list)
    for ann in coco_gt.anns.values():
        if ann.get('iscrowd', 0):
            continue
        gts_by_key[(ann['image_id'], ann['category_id'])].append(ann)

    valid_cats = set(coco_gt.getCatIds())
    valid_imgs = set(coco_gt.getImgIds())

    # per-detection record columns
    d_img, d_cat, d_score = [], [], []
    d_x, d_y, d_w, d_h = [], [], [], []
    d_is_tp, d_iou, d_pred_area, d_match_gt_area = [], [], [], []
    # matched-GT box (xywh, original-image px) for TPs -- enables exact AI-IoU
    # (scaled-box) re-scoring downstream; nan for FPs.
    d_mgt_x, d_mgt_y, d_mgt_w, d_mgt_h = [], [], [], []

    # union of (img, cat): visit keys with GT and pred-only keys (which yield FPs)
    all_keys = set(preds_by_key.keys()) | set(gts_by_key.keys())
    for img_id, cat_id in all_keys:
        if img_id not in valid_imgs or cat_id not in valid_cats:
            continue
        preds = sorted(preds_by_key.get((img_id, cat_id), []), key=lambda p: -p['score'])[:max_dets]
        gts = gts_by_key.get((img_id, cat_id), [])
        gt_matched = [False] * len(gts)

        for p in preds:
            bb = p['bbox']  # xywh, original-image px
            best_iou, best_g = iou_thr, -1
            for g_idx, gt in enumerate(gts):
                if gt_matched[g_idx]:
                    continue
                iou = _box_iou_xywh(bb, gt['bbox'])
                if iou >= best_iou:
                    best_iou, best_g = iou, g_idx
            is_tp = best_g >= 0
            if is_tp:
                gt_matched[best_g] = True
            d_img.append(int(img_id)); d_cat.append(int(cat_id)); d_score.append(float(p['score']))
            d_x.append(float(bb[0])); d_y.append(float(bb[1])); d_w.append(float(bb[2])); d_h.append(float(bb[3]))
            d_is_tp.append(bool(is_tp))
            d_iou.append(float(best_iou) if is_tp else 0.0)
            d_pred_area.append(float(bb[2] * bb[3]))
            d_match_gt_area.append(float(gts[best_g]['area']) if is_tp else float('nan'))
            mgt = gts[best_g]['bbox'] if is_tp else (float('nan'),) * 4
            d_mgt_x.append(float(mgt[0])); d_mgt_y.append(float(mgt[1]))
            d_mgt_w.append(float(mgt[2])); d_mgt_h.append(float(mgt[3]))

    # GT table (all non-crowd GTs in scope), for recall / size binning downstream
    g_img, g_cat, g_area = [], [], []
    for ann in coco_gt.anns.values():
        if ann.get('iscrowd', 0):
            continue
        if ann['image_id'] not in valid_imgs or ann['category_id'] not in valid_cats:
            continue
        g_img.append(int(ann['image_id'])); g_cat.append(int(ann['category_id'])); g_area.append(float(ann['area']))

    return {
        'det_image_id': np.asarray(d_img, dtype=np.int64),
        'det_category_id': np.asarray(d_cat, dtype=np.int64),
        'det_score': np.asarray(d_score, dtype=np.float32),
        'det_x': np.asarray(d_x, dtype=np.float32),
        'det_y': np.asarray(d_y, dtype=np.float32),
        'det_w': np.asarray(d_w, dtype=np.float32),
        'det_h': np.asarray(d_h, dtype=np.float32),
        'det_is_tp': np.asarray(d_is_tp, dtype=bool),
        'det_iou': np.asarray(d_iou, dtype=np.float32),           # matched IoU (>= iou_thr) for TP, else 0
        'det_pred_area': np.asarray(d_pred_area, dtype=np.float32),
        'det_matched_gt_area': np.asarray(d_match_gt_area, dtype=np.float32),  # GT area for TP, else nan
        'det_matched_gt_x': np.asarray(d_mgt_x, dtype=np.float32),  # matched GT box xywh for TP, else nan
        'det_matched_gt_y': np.asarray(d_mgt_y, dtype=np.float32),
        'det_matched_gt_w': np.asarray(d_mgt_w, dtype=np.float32),
        'det_matched_gt_h': np.asarray(d_mgt_h, dtype=np.float32),
        'gt_image_id': np.asarray(g_img, dtype=np.int64),
        'gt_category_id': np.asarray(g_cat, dtype=np.int64),
        'gt_area': np.asarray(g_area, dtype=np.float32),
        'iou_thr': np.float32(iou_thr),
        'max_dets': np.int64(max_dets),
    }


