"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The Dome decoder: ``DFINETransformer`` with Progressive Adaptive Query Initialization (PAQI) in
place of the fixed top-k query selection. Each image gets a core set of the ``min_num_select``
best encoder tokens plus, from the next ``max_num_select - min_num_select``, those that fall in
a window the density map marks as populated and survive a class-wise NMS whose IoU threshold
rises with the local density. Images in a batch therefore have different query counts. The
density map comes from ``DomeHybridEncoder``; with any other encoder the decoder raises.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy
from .dfine_decoder import DecoderInput, DFINETransformer
from .dynamic_nms import dynamic_nms_batch

__all__ = ["DomeTransformer"]


@register()
class DomeTransformer(DFINETransformer):
    """
    Args (on top of ``DFINETransformer``'s; ``num_queries`` is replaced by ``max_num_select``):
        min_num_select / max_num_select: PAQI's core query count and the pool the adaptive
            queries are drawn from.
        nms_iou_low / nms_iou_high: the dynamic NMS threshold runs from ``low`` where the density
            map is 0 to ``high`` where it is 1 (the paper's IoU_N and IoU_M).
    """

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        feat_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32, 64, 128),
        num_levels=5,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        min_num_select=300,
        max_num_select=1500,
        nms_iou_low=0.4,
        nms_iou_high=0.9,
        min_sample_cells=0.0,
        min_refine_cells=0.0,
        anchor_grid_size=0.05,
        fine_channels=0,
        fine_key_aware=False,
        null_point=False,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            num_queries=max_num_select,
            min_sample_cells=min_sample_cells,
            min_refine_cells=min_refine_cells,
            anchor_grid_size=anchor_grid_size,
            fine_channels=fine_channels,
            fine_key_aware=fine_key_aware,
            null_point=null_point,
        )
        self.min_num_select = min_num_select
        self.max_num_select = max_num_select
        self.nms_iou_low = nms_iou_low
        self.nms_iou_high = nms_iou_high

    # ------------------------------------------------------------------ PAQI: query initialization

    @staticmethod
    def _in_marked_windows(anchors_unact, window_mask):
        """Which anchors (logit cxcywh) have their centre in a window ``window_mask`` ``[B, rows, cols]`` marks."""
        b = anchors_unact.shape[0]
        n_rows, n_cols = window_mask.shape[1], window_mask.shape[2]
        cx, cy = F.sigmoid(anchors_unact[..., 0]), F.sigmoid(anchors_unact[..., 1])
        col = (cx * n_cols).long().clamp(0, n_cols - 1)
        row = (cy * n_rows).long().clamp(0, n_rows - 1)
        return window_mask[torch.arange(b, device=anchors_unact.device).view(-1, 1), row, col]

    def _density_nms(self, boxes_cxcywh, logits, density_map, valid):
        """
        Class-wise NMS over the batch's candidate boxes ``[B, N, 4]`` (normalized cxcywh; only
        the ``valid`` ones take part) with an IoU threshold that rises from ``nms_iou_low`` to
        ``nms_iou_high`` with the density under each box's centre (``density_map`` is
        ``[B, 1, h, w]``). Returns the ``[B, N]`` mask of the boxes to keep.
        """
        b = boxes_cxcywh.shape[0]
        cx, cy = boxes_cxcywh[..., 0], boxes_cxcywh[..., 1]
        h, w = density_map.shape[2:]
        row = (cy * (h - 1)).long().clamp(0, h - 1)
        col = (cx * (w - 1)).long().clamp(0, w - 1)
        density = density_map[torch.arange(b, device=row.device)[:, None], 0, row, col].detach()
        iou_thresholds = self.nms_iou_low + (self.nms_iou_high - self.nms_iou_low) * density
        scores, class_ids = logits.max(dim=-1)
        return dynamic_nms_batch(box_cxcywh_to_xyxy(boxes_cxcywh), scores, class_ids, iou_thresholds, valid)

    def _get_decoder_input(self, memory, spatial_shapes, encoder_out, targets=None):
        """
        PAQI. Returns the initial query contents and boxes (as logits, both detached and padded
        to the largest query count in the batch), the encoder-side predictions for the auxiliary
        loss, and the real query count of every image.

        The whole batch goes through the window filter, the NMS and the padding together: the
        one host round trip is the NMS's (its sequential pass runs on the host), and the query
        counts come back with it.
        """
        defe = encoder_out.get("defe")
        if defe is None or "defe_window_mask" not in defe:
            raise ValueError("DomeTransformer needs DomeHybridEncoder's density map and window mask (use_mwas: True)")
        # the criterion reads the query budget from here
        defe["min_num_select"] = self.min_num_select
        defe["max_num_select"] = self.max_num_select
        defe_window_mask = defe["defe_window_mask"]
        defe_feature = defe["density_map_pooled"]

        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        topk_memory, topk_logits, topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.max_num_select
        )
        b, n = topk_anchors.shape[:2]
        min_num = self.min_num_select
        device = memory.device

        # the core queries are kept as they are; the rest must sit in a populated window, and
        # among those the NMS decides (the core queries take part as suppressors, never suppressed)
        core = torch.arange(n, device=device)[None, :] < min_num  # [1, N]
        candidate = torch.cat(
            [core.expand(b, -1)[:, :min_num], self._in_marked_windows(topk_anchors[:, min_num:], defe_window_mask)], 1
        )
        bbox_unact = self.enc_bbox_head(topk_memory) + topk_anchors
        kept = self._density_nms(F.sigmoid(bbox_unact), topk_logits, defe_feature, candidate)
        selected = core | (candidate & kept)  # [B, N]

        # every image's selected queries first, in their top-k order, padded to the largest count
        batch_queries_num = selected.sum(1).tolist()
        max_total = max(batch_queries_num)
        order = torch.sort((~selected).to(torch.int8), dim=1, stable=True).indices[:, :max_total]
        counts = torch.tensor(batch_queries_num).to(device, non_blocking=True)  # no host sync
        real = torch.arange(max_total, device=device)[None, :] < counts[:, None]

        def take(x):
            x = x.gather(1, order[..., None].expand(-1, -1, x.shape[-1])).float()
            return x.masked_fill(~real[..., None], 0.0)

        padded_memory, padded_logits, padded_bbox_unact = take(topk_memory), take(topk_logits), take(bbox_unact)

        # the criterion masks the padded entries with batch_queries_num
        enc_topk_bboxes_list = [F.sigmoid(padded_bbox_unact)]
        enc_topk_logits_list = [padded_logits]
        return DecoderInput(
            padded_memory.detach(),
            padded_bbox_unact.detach(),
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            batch_queries_num,
        )
