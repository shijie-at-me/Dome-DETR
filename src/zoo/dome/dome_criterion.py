"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dome_utils import bbox2distance


@register()
class DomeCriterion(nn.Module):
    """This class computes the loss for Dome-DETR."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        defe_density_map_weight=4,
        density_recall_penalty=0.3,
        mal_alpha=None,
        use_uni_set=True,
        aiiou_variant="none",
        aiiou_s_ref=32.0,
        aiiou_lam=1.0,
        aiiou_alpha=0.5,
        aiiou_B=0.2,
        aiiou_size_space="orig",
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in Dome-DETR.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.defe_density_map_weight = defe_density_map_weight
        self.density_recall_penalty = density_recall_penalty
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        # Adaptive Inner-IoU (AI-IoU) soft-target variant. 'none' = baseline (raw IoU).
        #   mult     : scaled-box IoU, r = max(1, s_ref/s_gt)          (the main method)
        #   smooth   : scaled-box IoU, r = sqrt(1 + s_ref^2/s_gt^2)    (no kink)
        #   partial  : scaled-box IoU, s_eff = max(s_gt, lam*s_ref)    (weaker floor)
        #   convex   : alpha*q_ai(mult) + (1-alpha)*q                  (blend)
        #   additive : clip(q + B*(1 - s_gt/s_ref)_+, 0, 1)            (slope-preserving)
        self.aiiou_variant = aiiou_variant
        self.aiiou_s_ref = aiiou_s_ref
        self.aiiou_lam = aiiou_lam
        self.aiiou_alpha = aiiou_alpha
        self.aiiou_B = aiiou_B
        self.aiiou_size_space = aiiou_size_space  # 'orig' (AP_S px, default) | 'input'

        self._vfl_buffer = {
            # 核心字段
            "ious": [],
            "l1": [],
            "gious": [],
            "scores": [],  # GT class 对应的预测分数
            "max_scores": [],  # 最大预测分数
            # GT box 几何信息
            "areas": [],  # GT 面积
            "widths": [],  # GT 宽
            "heights": [],  # GT 高
            "pred_areas": [],  # 预测框面积
            # 训练信号
            "residuals": [],  # score - IoU
            # 类别信息
            "target_classes": [],
            "is_correct_class": [],  # 预测类别是否正确
            # 位置信息
            "batch_indices": [],
        }
        self.extract_vfl_stats = False

    # ---------------- Adaptive Inner-IoU (AI-IoU) soft-target variants ----------------
    def _aiiou_gt_size_px(self, targets, indices, target_boxes):
        """Per-matched-box GT size sqrt(area) in pixels. Default space 'orig' ties s_ref
        to the COCO AP_S boundary (original px); 'input' uses model-input px. Boxes are
        normalised cxcywh, so px size = sqrt((w*W)*(h*H))."""
        space = getattr(self, "aiiou_size_space", "orig")
        ws, hs = [], []
        for t, (_, J) in zip(targets, indices):
            n = len(J)
            if space == "input" and "size" in t:        # size is [H, W] (Resize/Pad)
                h, w = float(t["size"][0]), float(t["size"][1])
            elif "orig_size" in t:                       # orig_size is [W, H]
                w, h = float(t["orig_size"][0]), float(t["orig_size"][1])
            elif "size" in t:
                h, w = float(t["size"][0]), float(t["size"][1])
            else:
                w = h = 1.0
            ws.append(torch.full((n,), w)); hs.append(torch.full((n,), h))
        dev = target_boxes.device
        W = torch.cat(ws).to(dev) if ws else torch.zeros(0, device=dev)
        H = torch.cat(hs).to(dev) if hs else torch.zeros(0, device=dev)
        area = (target_boxes[:, 2] * W) * (target_boxes[:, 3] * H)
        return torch.sqrt(torch.clamp(area, min=1e-6))

    def _aiiou_ratio(self, s_gt):
        s_ref = self.aiiou_s_ref
        if self.aiiou_variant == "smooth":
            return torch.sqrt(1.0 + (s_ref ** 2) / torch.clamp(s_gt ** 2, min=1e-6))
        lam = self.aiiou_lam if self.aiiou_variant == "partial" else 1.0
        return torch.clamp((lam * s_ref) / s_gt, min=1.0)   # s_eff = max(s_gt, lam*s_ref)

    def _aiiou_scaled_iou(self, src, tgt, r):
        """IoU of src/tgt after scaling both about their centers by r (cxcywh: *w,*h)."""
        rs = r.unsqueeze(-1)
        src_s = src.clone(); src_s[:, 2:] = src[:, 2:] * rs
        tgt_s = tgt.clone(); tgt_s[:, 2:] = tgt[:, 2:] * rs
        iou, _ = box_iou(box_cxcywh_to_xyxy(src_s), box_cxcywh_to_xyxy(tgt_s))
        return torch.diag(iou)

    def _apply_aiiou(self, ious, outputs, targets, indices, idx):
        """Replace the raw-IoU soft target q with an AI-IoU variant (detached). variant
        'none' is the identity. Acts on matched positives only."""
        var = getattr(self, "aiiou_variant", "none")
        if var in (None, "none") or ious.numel() == 0:
            return ious
        with torch.no_grad():
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            s_gt = self._aiiou_gt_size_px(targets, indices, target_boxes)
            q = ious
            if var == "additive":
                bonus = self.aiiou_B * torch.clamp(1.0 - s_gt / self.aiiou_s_ref, min=0.0)
                q_ai = torch.clamp(q + bonus, 0.0, 1.0)
            else:
                q_scaled = self._aiiou_scaled_iou(src_boxes, target_boxes, self._aiiou_ratio(s_gt))
                if var == "convex":
                    q_ai = self.aiiou_alpha * q_scaled + (1.0 - self.aiiou_alpha) * q
                else:  # mult / smooth / partial
                    q_ai = q_scaled
            return q_ai.to(ious.dtype)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, batch_queries_num=None):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )

        if batch_queries_num is not None:
            mask = torch.arange(src_logits.shape[1], device=src_logits.device)[None, :] < \
                   torch.as_tensor(batch_queries_num, device=src_logits.device)[:, None]
            loss = loss * mask.unsqueeze(-1)

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, batch_queries_num=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        if self.extract_vfl_stats:
            self.extract_vfl_stats_func(ious, outputs["pred_logits"], idx, targets, indices)

        ious = self._apply_aiiou(ious, outputs, targets, indices, idx)

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )

        if batch_queries_num is not None:
            mask = torch.arange(src_logits.shape[1], device=src_logits.device)[None, :] < \
                   torch.as_tensor(batch_queries_num, device=src_logits.device)[:, None]
            loss = loss * mask.unsqueeze(-1)

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}
    
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None, batch_queries_num=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        ious = self._apply_aiiou(ious, outputs, targets, indices, idx)

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')

        if batch_queries_num is not None:
            mask = torch.arange(src_logits.shape[1], device=src_logits.device)[None, :] < \
                   torch.as_tensor(batch_queries_num, device=src_logits.device)[:, None]
            loss = loss * mask.unsqueeze(-1)

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        if self.extract_vfl_stats:
            buf = self._vfl_buffer
            buf["l1"].append(loss_bbox.cpu().float().numpy())
            buf["gious"].append(loss_giou.cpu().float().numpy())

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, **kwargs):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            "local": self.loss_local
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        batch_queries_num = outputs.get("batch_queries_num")

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs and self.training:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert not self.training, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local']) and self.training
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, batch_queries_num=batch_queries_num, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs and self.training:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local']) and self.training
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, batch_queries_num=batch_queries_num, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dome
        if 'pre_outputs' in outputs and self.training:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local']) and self.training
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, batch_queries_num=batch_queries_num, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs and self.training:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes') and self.training
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, batch_queries_num=batch_queries_num, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs and self.training:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dome
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of defe Category losses.
        if "defe" in outputs and self.training:
            # Calculate defe Regression Loss
            min_num_select = outputs["defe"]["min_num_select"]
            max_num_select = outputs["defe"]["max_num_select"]

            reg_targets = []

            for i in range(len(targets)):
                tgt_num = targets[i]['labels'].shape[0]
                if tgt_num < min_num_select:
                    tgt_num = min_num_select
                elif tgt_num > max_num_select:
                    tgt_num = max_num_select
                reg_targets.append((tgt_num - min_num_select) / (max_num_select - min_num_select))

            reg_targets = torch.tensor(reg_targets, dtype=torch.int64).to(outputs["defe"]["reg_value"].device)
            reg_value = outputs["defe"]["reg_value"]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                diff = reg_value - reg_targets
                penalty_weights = torch.where(diff < 0, 2.0, 1.0).to(diff.device)
                defe_reg_loss = (penalty_weights * (diff ** 2)).mean() 
            del reg_value, reg_targets
            torch.cuda.empty_cache()
            losses["defe_reg_loss"] = defe_reg_loss

            # Calculate defe Density Map Loss with emphasis on high GT regions
            density_map = outputs["defe"]["defe_feature"]
            gt_density_map = outputs["defe"]["gt_density_map"]
            with torch.amp.autocast('cuda', dtype=torch.float16):
                diff = density_map - gt_density_map
                underestimation_mask = (density_map < gt_density_map).float()
                penalty_weight = 1 + self.density_recall_penalty * gt_density_map * underestimation_mask
                defe_density_loss = (penalty_weight * (diff ** 2)).mean() * self.defe_density_map_weight
            del density_map, gt_density_map
            torch.cuda.empty_cache()
            losses["defe_density_loss"] = defe_density_loss

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list

    def extract_vfl_stats_func(self, ious, pred_logits, idx, targets, indices):
        """Harvest per-matched-positive diagnostics for the confidence-suppression study (E1).

        COORDINATE-SYSTEM CONTRACT (critical for downstream size analysis):
          * `widths`/`heights` are NORMALISED cx,cy,w,h box sides in [0,1] w.r.t. the
            MODEL-INPUT image -- they are NOT pixels.
          * `areas` (= target["area"]) is rescaled by the Resize transform, so it is in
            MODEL-INPUT pixel^2. For the unresized case (e.g. AI-TOD) input == original.
          * For an unambiguous, explicit size axis we additionally store per-pair input and
            original image dims, plus pixel areas recomputed from the SAME normalised box:
              - `area_in_px`   = (w*Win)*(h*Hin)   -> MECHANISM axis (size the detector sees;
                                                      where the IoU geometry actually plays out)
              - `area_orig_px` = (w*Worig)*(h*Horig) -> BENCHMARK axis (COCO/AI-TOD AP_S/AP_vt
                                                      buckets are defined on original pixels)
            These coincide when there is no (or isotropic) resize, and differ under the
            anisotropic 800x800 resize used by VisDrone -- do NOT mix the two.
          * `image_ids` are global COCO ids; the legacy `batch_indices` is per-batch-local
            (0..bs-1, repeats every batch) and is kept only for back-compat -- prefer image_ids.

        NOTE: this only enriches FUTURE dumps; existing .npz files must be regenerated
        (re-run the val() dump) to gain the new keys.
        """
        with torch.no_grad():
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_areas = torch.cat([t["area"][i] for t, (_, i) in zip(targets, indices)], dim=0).float()
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

            widths = target_boxes[:, 2]   # NORMALISED w (not pixels)
            heights = target_boxes[:, 3]  # NORMALISED h (not pixels)

            # ── per-image dims, broadcast to each matched pair ───────────────────────
            def _dims(t):
                # input (model) H,W: target["size"] is [H,W] (set by Resize/Pad);
                # if no such transform ran (e.g. AI-TOD), input == original.
                if "size" in t:
                    h_in, w_in = float(t["size"][0]), float(t["size"][1])
                elif "orig_size" in t:  # orig_size is stored as [W,H]
                    w_in, h_in = float(t["orig_size"][0]), float(t["orig_size"][1])
                else:
                    h_in = w_in = float("nan")
                if "orig_size" in t:    # [W,H]
                    w_o, h_o = float(t["orig_size"][0]), float(t["orig_size"][1])
                else:
                    w_o, h_o = w_in, h_in
                return h_in, w_in, h_o, w_o

            dev = target_boxes.device
            ih, iw, oh, ow, iid = [], [], [], [], []
            for t, (_, J) in zip(targets, indices):
                n = len(J)
                h_in, w_in, h_o, w_o = _dims(t)
                ih.append(torch.full((n,), h_in, device=dev))
                iw.append(torch.full((n,), w_in, device=dev))
                oh.append(torch.full((n,), h_o, device=dev))
                ow.append(torch.full((n,), w_o, device=dev))
                _id = int(t["image_id"].item()) if "image_id" in t else -1
                iid.append(torch.full((n,), _id, device=dev, dtype=torch.long))
            input_h = torch.cat(ih) if ih else torch.zeros(0, device=dev)
            input_w = torch.cat(iw) if iw else torch.zeros(0, device=dev)
            orig_h = torch.cat(oh) if oh else torch.zeros(0, device=dev)
            orig_w = torch.cat(ow) if ow else torch.zeros(0, device=dev)
            image_ids = torch.cat(iid) if iid else torch.zeros(0, device=dev, dtype=torch.long)

            area_in_px = (widths * input_w) * (heights * input_h)      # mechanism axis
            area_orig_px = (widths * orig_w) * (heights * orig_h)      # benchmark axis

            # ── predicted scores ────────────────────────────────────────────────────
            matched_scores = torch.sigmoid(pred_logits[idx])  # [N, num_classes]
            gt_class_scores = matched_scores[torch.arange(len(target_classes_o)), target_classes_o]  # GT-class p
            max_scores, max_pred_classes = matched_scores.max(dim=-1)  # ranking score used by NMS/top-K
            is_correct_class = max_pred_classes == target_classes_o

            residuals = gt_class_scores.cpu().float() - ious.cpu().float()

            # ── write buffer ─────────────────────────────────────────────────────────
            # existing keys (kept for back-compat with prior dumps / analysis)
            buf = self._vfl_buffer
            buf["ious"].append(ious.cpu().float().numpy())                    # q = VFL soft target
            buf["scores"].append(gt_class_scores.cpu().float().numpy())       # p = GT-class score
            buf["max_scores"].append(max_scores.cpu().float().numpy())
            buf["areas"].append(target_areas.cpu().float().numpy())           # input-space px^2 (Resize-scaled)
            buf["widths"].append(widths.cpu().float().numpy())                # NORMALISED
            buf["heights"].append(heights.cpu().float().numpy())              # NORMALISED
            buf["residuals"].append(residuals.numpy())
            buf["target_classes"].append(target_classes_o.cpu().numpy())
            buf["is_correct_class"].append(is_correct_class.cpu().numpy())
            buf["batch_indices"].append(idx[0].cpu().numpy())                 # per-batch local (legacy)
            # new, unambiguous geometry / identity (setdefault avoids touching __init__)
            buf.setdefault("image_ids", []).append(image_ids.cpu().numpy())
            buf.setdefault("input_h", []).append(input_h.cpu().float().numpy())
            buf.setdefault("input_w", []).append(input_w.cpu().float().numpy())
            buf.setdefault("orig_h", []).append(orig_h.cpu().float().numpy())
            buf.setdefault("orig_w", []).append(orig_w.cpu().float().numpy())
            buf.setdefault("area_in_px", []).append(area_in_px.cpu().float().numpy())
            buf.setdefault("area_orig_px", []).append(area_orig_px.cpu().float().numpy())

    def save_vfl_stats(self, output_dir):
        import numpy as np
        if not hasattr(self, '_vfl_buffer') or not self._vfl_buffer["ious"]:
            print("No stats collected.")
            return

        save_dict = {
            k: np.concatenate(v)
            for k, v in self._vfl_buffer.items()
            if len(v) > 0
        }

        np.savez(output_dir, **save_dict)

        n = len(save_dict["ious"])
        print(f"Saved vfl_stats.npz | total matched pairs: {n}")
        print(f"Fields: {list(save_dict.keys())}")
        for k in self._vfl_buffer:
            self._vfl_buffer[k] = []





