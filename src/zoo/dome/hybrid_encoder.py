"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

D-FINE's hybrid encoder: a channel projection per backbone level, an intra-scale transformer on
the listed (top) levels, and a CSP-ELAN feature pyramid across levels. It differs from upstream
only in upsampling to the finer level's exact size, which a four-level pyramid on arbitrary input
sizes needs. It has one hook, ``enhance``, that runs on the projected levels before the transformer
and may add entries to the output dict; here it does nothing, so this is the baseline to build on.
``dome_encoder.DomeHybridEncoder`` fills the hook with Dome's DeFE and MWAS.
"""

import copy
from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...core import register
from ...misc.visualizer import dump_feature_map
from ...nn.blocks import ConvNormLayerFuse, LightFusion, RepNCSPELAN4, RepSepFusion, SCDown, SeparableConv
from ...nn.checkpoint import checkpoint_module
from ...nn.position_encoding import build_2d_sincos_position_embedding
from ...nn.transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = ["HybridEncoder"]


class ResidualSequential(nn.Sequential):
    """A Sequential whose input is added to its output (the members keep Sequential's state-dict names)."""

    def forward(self, x):
        return x + super().forward(x)


@register()
class HybridEncoder(nn.Module):
    """
    Args:
        in_channels / feat_strides: the backbone levels, lowest stride first.
        hidden_dim: every level is projected to this width.
        use_encoder_idx / num_encoder_layers / nhead / dim_feedforward / dropout / enc_act /
            pe_temperature: the intra-scale transformer applied to the listed levels.
        expansion / depth_mult / act: width, depth and activation of the pyramid's fusion blocks.
        fine_fusion: the top-down fusion block of the finest level, where the maps are largest:
            ``elan`` (the same ``RepNCSPELAN4`` as the other levels), ``slim`` (the ELAN block
            with its inner width halved to ``hidden_dim``), ``light`` (``blocks.LightFusion``:
            a 1x1 fuse, a depthwise 3x3 and a 1x1 with a residual, a third of the ELAN block's
            FLOPs) or ``separable`` (``blocks.SeparableConv``, EfficientDet's BiFPN block as
            published: a depthwise 3x3 and a pointwise 1x1, a fifth of the ELAN block's FLOPs).
        use_hybrid: run the top-down / bottom-up pyramid; off, the projected levels are returned.
        checkpoint_fusion: in training, recompute the fusion blocks' (and the fine level's
            blocks') activations in the backward pass instead of keeping them (the stride-4
            level's are most of the encoder's memory).
        eval_spatial_size: (h, w) at evaluation, to precompute the position embeddings; unset,
            they are built for whatever size arrives.
        fine_in_channels / fine_dim / fine_blocks: the fine level, a map one stride finer than
            the pyramid built from the backbone's stem map (``HGNetv2(return_stem=True)``, the
            first of ``feats``, ``fine_in_channels`` wide): a 1x1 on the stem map plus the
            finest pyramid level's semantics (a 1x1, upsampled), then ``fine_blocks`` depthwise
            3x3 / 1x1 pairs, ``fine_dim`` wide. It stays out of the pyramid and the encoder's
            levels: the decoder reads it as values only (``DFINETransformer(fine_channels)``).
            ``fine_in_channels`` 0 (default): no fine level. ``fine_residual`` gives each
            block a skip from its input (free; off by default so the configs of the runs
            trained without it still describe them). ``fine_groups`` (with the decoder's, set to
            num_heads) mixes each head's channels only within its own group, so the level is a
            stack of per-head sub-maps as the coarse levels are split across heads.

    ``forward(feats, img_inputs, targets)`` returns a dict with ``feats`` (the pyramid, one
    tensor per level), ``img_inputs`` (the image, passed through for the decoder's dumps) and,
    with a fine level, ``fine``, plus whatever ``enhance`` adds.
    """

    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        use_hybrid=True,
        checkpoint_fusion=False,
        fine_fusion="elan",
        fine_fusion_hidden=256,
        fine_fusion_depth=2,
        fine_in_channels=0,
        fine_dim=64,
        fine_blocks=2,
        fine_residual=False,
        fine_groups=1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.pos_embeds = []
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_hybrid = use_hybrid
        self.checkpoint_fusion = checkpoint_fusion
        assert fine_fusion in ("elan", "slim", "light", "separable", "repsep"), fine_fusion
        self.fine_fusion = fine_fusion
        self.dim_feedforward = dim_feedforward
        self.fine_in_channels = fine_in_channels

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ]
                    )
                )
            )

        # intra-scale transformer, one per listed level
        if self.num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act
            )
            self.encoder = nn.ModuleList(
                [
                    TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                    for _ in range(len(use_encoder_idx))
                ]
            )

        if self.use_hybrid:
            fusion = dict(c3=hidden_dim * 2, c4=round(expansion * hidden_dim // 2), n=round(3 * depth_mult), act=act)
            # top-down: lateral 1x1 on the coarser level, upsample, fuse with the finer one; the
            # last block fuses into the finest level
            self.lateral_convs = nn.ModuleList()
            self.fpn_blocks = nn.ModuleList()
            for i in range(len(in_channels) - 1):
                self.lateral_convs.append(ConvNormLayerFuse(hidden_dim, hidden_dim, 1, 1))
                finest = i == len(in_channels) - 2
                if finest and fine_fusion == "light":
                    self.fpn_blocks.append(LightFusion(hidden_dim * 2, hidden_dim, act=act))
                elif finest and fine_fusion == "separable":
                    self.fpn_blocks.append(SeparableConv(hidden_dim * 2, hidden_dim, act=act))
                elif finest and fine_fusion == "repsep":
                    self.fpn_blocks.append(
                        RepSepFusion(
                            hidden_dim * 2, hidden_dim, hidden=fine_fusion_hidden, depth=fine_fusion_depth, k=5, act=act
                        )
                    )
                elif finest and fine_fusion == "slim":
                    self.fpn_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **{**fusion, "c3": hidden_dim}))
                else:
                    self.fpn_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **fusion))
            # bottom-up: downsample the finer level, fuse with the coarser one
            self.downsample_convs = nn.ModuleList()
            self.pan_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1):
                self.downsample_convs.append(nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2)))
                self.pan_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **fusion))

        # the fine level: the stem map and the finest pyramid level's semantics, summed, then
        # depthwise 3x3 / 1x1 pairs (the maps are four times the stride-4 level's: nothing wide)
        if fine_in_channels > 0:
            self.fine_lateral = ConvNormLayerFuse(fine_in_channels, fine_dim, 1, 1)
            self.fine_top_down = ConvNormLayerFuse(hidden_dim, fine_dim, 1, 1)
            assert fine_dim % fine_groups == 0, "fine_dim must divide by fine_groups"
            block = ResidualSequential if fine_residual else nn.Sequential
            self.fine_blocks = nn.Sequential(
                *(
                    block(
                        ConvNormLayerFuse(fine_dim, fine_dim, 3, 1, g=fine_dim, act=act),
                        # grouped 1x1: with fine_groups == num_heads each head's channels mix only
                        # within their own group, so the map is num_heads independent sub-maps
                        ConvNormLayerFuse(fine_dim, fine_dim, 1, 1, g=fine_groups, act=act),
                    )
                    for _ in range(fine_blocks)
                )
            )

        self._build_eval_pos_embeds()

    def _build_eval_pos_embeds(self):
        """Position embeddings for the evaluation size, one per intra-scale level, computed once."""
        if not self.eval_spatial_size:
            return
        for idx in self.use_encoder_idx:
            stride = self.feat_strides[idx]
            self.pos_embeds.append(
                build_2d_sincos_position_embedding(
                    ceil(self.eval_spatial_size[1] / stride),
                    ceil(self.eval_spatial_size[0] / stride),
                    self.hidden_dim,
                    self.pe_temperature,
                )
            )

    def enhance(self, proj_feats: list[torch.Tensor], img_inputs, targets) -> dict:
        """
        Hook on the projected levels, before the transformer and the pyramid. May modify
        ``proj_feats`` in place and returns entries to add to the output dict. Nothing here.
        """
        return {}

    def _pos_embed(self, w: int, h: int, device) -> torch.Tensor:
        """The position embedding of a ``w`` x ``h`` grid on ``device``, built once per size (multi-scale training draws a handful)."""
        cache = self.__dict__.setdefault("_pos_embed_cache", {})  # a plain attribute: not a buffer, not saved
        key = (w, h, str(device))
        if key not in cache:
            if len(cache) >= 64:
                cache.clear()
            cache[key] = build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(device)
        return cache[key]

    def _fine(self, stem: torch.Tensor, finest: torch.Tensor) -> torch.Tensor:
        """
        The fine level from the stem map and the finest pyramid level (upsampled to the stem map's
        size). ``align_corners=False`` is the convention that matches the maps' own geometry: a
        stride-4 cell covers the input pixel ``4i + 2`` and a stride-2 cell ``2j + 1``, so cell j
        reads ``0.5j - 0.25``, which is what False computes exactly and True misses by a ramp of
        plus or minus one input pixel across the map. Measured on the trained fine level the two
        score the same (31.7 AP either way, the very tiny bucket 15.4 against 15.5), because the
        branch being shifted is the upsampled one, whose own resolution is four pixels; the stem
        branch, which carries the detail, is not shifted. Correct rather than better.
        """
        semantics = F.interpolate(self.fine_top_down(finest), size=stem.shape[2:], mode="bilinear", align_corners=False)
        x = self.fine_lateral(stem) + semantics
        if self.checkpoint_fusion and self.training and torch.is_grad_enabled():
            return checkpoint_module(self.fine_blocks, x)
        return self.fine_blocks(x)

    def forward(self, feats, img_inputs, targets=None):
        stem = None
        if self.fine_in_channels > 0:
            stem, feats = feats[0], feats[1:]
            assert stem.shape[1] == self.fine_in_channels, (stem.shape, self.fine_in_channels)
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        dump_feature_map("backbone_output_0", proj_feats[0])

        out = {"img_inputs": img_inputs}
        out.update(self.enhance(proj_feats, img_inputs, targets))

        # intra-scale transformer
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B, HW, C]
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self._pos_embed(w, h, src_flatten.device)
                else:
                    pos_embed = self.pos_embeds[i].to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        if not self.use_hybrid:
            out["feats"] = proj_feats
            if stem is not None:
                out["fine"] = self._fine(stem, proj_feats[0])
            return out

        def fuse(block, feats):
            if self.checkpoint_fusion and self.training and torch.is_grad_enabled():
                return checkpoint_module(block, *feats, fn=lambda *f: block(torch.concat(f, dim=1)))
            return block(torch.concat(feats, dim=1))

        # top-down: coarsest level first, each finer level fused with the upsampled result
        inner_outs = [proj_feats[-1]]
        for i, idx in enumerate(range(len(self.in_channels) - 1, 0, -1)):
            feat_high = self.lateral_convs[i](inner_outs[0])
            feat_low = proj_feats[idx - 1]
            inner_outs[0] = feat_high
            # align_corners=False: the maps' own geometry, as in _fine
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[2:], mode="bilinear", align_corners=False)
            inner_outs.insert(0, fuse(self.fpn_blocks[i], [upsample_feat, feat_low]))

        # bottom-up: finest level first, each coarser level fused with the downsampled result
        outs = [inner_outs[0]]
        for i in range(len(self.in_channels) - 1):
            downsample_feat = self.downsample_convs[i](outs[-1])
            outs.append(fuse(self.pan_blocks[i], [downsample_feat, inner_outs[i + 1]]))

        out["feats"] = outs
        if stem is not None:
            out["fine"] = self._fine(stem, outs[0])
        return out
