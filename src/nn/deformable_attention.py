"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

Multi-scale deformable attention (Deformable DETR), in the pure-PyTorch form D-FINE uses: every
query samples ``num_points`` locations per head on every feature level around its reference box
and averages the sampled values with learned weights.
"""

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.nn.init as init

__all__ = ["MSDeformableAttention", "ms_deformable_attention_core"]


def ms_deformable_attention_core(
    value: list[torch.Tensor],
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method: str = "default",
):
    """
    Args:
        value: one ``[bs, n_head, c, h*w]`` tensor per level.
        value_spatial_shapes: the ``(h, w)`` of each level.
        sampling_locations: ``[bs, query_length, n_head, sum(num_points), 2]`` in [0, 1].
        attention_weights: ``[bs, query_length, n_head, sum(num_points)]``.
        num_points_list: points sampled per level.
        method: ``default`` samples bilinearly with ``grid_sample``; ``discrete`` rounds each
            location to the nearest cell and gathers it.

    Returns:
        ``[bs, query_length, n_head * c]``
    """
    bs, n_head, c, _ = value[0].shape
    _, len_q, _, _, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1 if method == "default" else sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, len_q, sum(num_points_list))
    attn_weights_list = attn_weights.split(num_points_list, dim=-1)

    # the weighted sum accumulated level by level: no [bs * n_head, c, len_q, sum(num_points)]
    # concatenation of the samples, nor its split in the backward pass
    output = None
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)
            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, len_q * num_points_list[level], 2)

            s_idx = torch.arange(sampling_coord.shape[0], device=value_l.device).unsqueeze(-1)
            s_idx = s_idx.repeat(1, sampling_coord.shape[1])
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]  # n l c
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, len_q, num_points_list[level])
        else:
            raise ValueError(f"unknown deformable attention method {method!r}")

        weighted = (sampling_value_l * attn_weights_list[level]).sum(-1)  # [bs * n_head, c, len_q]
        output = weighted if output is None else output + weighted

    return output.reshape(bs, n_head * c, len_q).permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    """
    Multi-scale deformable attention. ``num_points`` is one int for every level or a list with
    one entry per level; ``offset_scale`` scales the predicted offsets relative to the reference
    box size. ``min_sample_cells`` (0: off) floors the sampling radius, per level, at so many
    cells of the level (1: the points reach the neighbouring cells): a box smaller than a cell
    then still spreads its sampling points over its surroundings on every level instead of
    reading the same cell with all of them. With ``method='discrete'`` the sampling offsets are
    frozen.

    ``fine_dim`` (0: off) makes the first level a raw one: a ``fine_dim``-channel map (not
    ``embed_dim``, not split across heads) that every head samples whole at its own points; the
    weighted sum of a head's samples is then lifted to the head's width by a per-head linear
    and added to what the head read from the other levels. Sampling before projecting keeps
    the dense map at ``fine_dim`` channels (its fp32 copy and the dense gradient buffers of
    ``grid_sample``'s backward pass shrink with it) and runs the projection on the samples only.

    ``null_point`` gives every head one more entry in its softmax that samples nothing: a
    learned per-head vector (zero at first) weighted by that entry is added to what the head
    read. Without it a head's weights sum to one and it must return a full-magnitude mixture of
    its samples whether or not any of them is informative (the probes found flat weights over
    points that read the same cell); with it a head can abstain, and the vector is an explicit
    "nothing here" the next blocks can recognize. One entry per head, not per level: the softmax
    already spans the levels. Its logit starts at the others' (weight 1 / (P + 1)).
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
        min_sample_cells=0.0,
        fine_dim=0,
        fine_key_aware=False,
        fine_groups=1,
        null_point=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        self.min_sample_cells = min_sample_cells
        self.null_point = null_point

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, "num_points needs one entry per level"
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]
        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))
        point_level = [level for level, n in enumerate(num_points_list) for _ in range(n)]
        self.register_buffer("point_level", torch.tensor(point_level), persistent=False)

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        # one weight per sampling point, plus the null entry per head
        self.attention_weights = nn.Linear(embed_dim, self.total_points + (num_heads if null_point else 0))
        if null_point:
            self.null_value = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.fine_dim = fine_dim
        self.fine_key_aware = fine_key_aware and fine_dim > 0
        # fine_groups > 1 (only num_heads is supported) splits the fine map into num_heads
        # contiguous channel groups and gives each head its own group, as the coarse levels split
        # their value into num_heads slices; each head then lifts fine_dim // num_heads channels
        # instead of the whole map (1: the map is shared, every head lifts all fine_dim channels).
        self.fine_groups = fine_groups if fine_dim > 0 else 1
        assert self.fine_groups in (1, num_heads), "fine_groups must be 1 or num_heads"
        fine_group_dim = fine_dim // self.fine_groups
        if fine_dim > 0:
            assert method == "default", "a raw fine level needs bilinear sampling"
            assert fine_dim % self.fine_groups == 0, "fine_dim must divide by fine_groups"
            # per head: fine_group_dim -> head_dim, on the weighted sum of the head's fine samples
            # (no bias: the output projection's covers it, and a bias here would not commute with
            # the zero padding outside the map and the attention weights' sum below 1)
            self.fine_lift = nn.Parameter(torch.empty(num_heads, fine_group_dim, self.head_dim))
        if self.fine_key_aware:
            # the fine level's logits read what was sampled: a per-head key of every sample and a
            # per-head query, their scaled dot product added to the query-predicted logit. The key
            # starts at zero, so the layer begins exactly where it begins without this
            self.fine_key = nn.Parameter(torch.zeros(num_heads, fine_group_dim, self.head_dim))
            self.fine_query = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(ms_deformable_attention_core, method=self.method)

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling offsets start as a star of points around the reference, one direction per head
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        if self.fine_dim > 0:
            for head in range(self.num_heads):
                init.xavier_uniform_(self.fine_lift.data[head])
        if self.fine_key_aware:
            init.xavier_uniform_(self.fine_query.weight)
            init.constant_(self.fine_query.bias, 0)
            init.constant_(self.fine_key, 0)  # zero logits at first, an unchanged softmax

    def _sample_fine(self, value, hw, sampling_locations):
        """
        The raw fine level ``value`` ``[bs, fine_dim, h * w]`` sampled at ``sampling_locations``
        ``[bs, len_q, heads, P, 2]`` (in [0, 1]). Shared (``fine_groups`` 1): the map is sampled
        once for all heads, ``[bs, fine_dim, len_q, heads, P]``. Grouped (``fine_groups`` ==
        heads): each head samples only its own channel group, so ``grid_sample`` reads
        ``fine_dim // heads`` channels per head rather than the whole map (and its backward
        buffer is heads-fold smaller), returning ``[bs, fine_dim // heads, len_q, heads, P]``.
        """
        bs, len_q, heads, p, _ = sampling_locations.shape
        h, w = hw
        grid = 2 * sampling_locations - 1
        if self.fine_groups == 1:
            grid = grid.reshape(bs, len_q * heads * p, 1, 2)
            return F.grid_sample(
                value.view(bs, self.fine_dim, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=False
            ).view(bs, self.fine_dim, len_q, heads, p)
        # per head: batch the map by head so head h reads only group h's gdim channels at its own
        # points; the view is free (groups are contiguous in the channel axis)
        gdim = self.fine_dim // heads
        value = value.reshape(bs * heads, gdim, h, w)
        grid = grid.permute(0, 2, 1, 3, 4).reshape(bs * heads, len_q * p, 1, 2)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return sampled.view(bs, heads, gdim, len_q, p).permute(0, 2, 3, 1, 4)  # [bs, gdim, len_q, heads, P]

    def _fine_logits(self, samples, query):
        """
        The scaled dot product of every head's query with the key of each of its fine samples,
        ``[bs, len_q, heads, P]``. Without it a head's weights come from the query alone and
        cannot tell an informative sample from a background one; the probes found a head's six
        fine points reading features only 0.61 alike yet carrying 4.6 of 6 effective points.
        """
        bs, c, len_q, heads, p = samples.shape
        with torch.autocast(device_type=samples.device.type, enabled=False):
            flat = samples.float().permute(0, 3, 2, 4, 1).reshape(bs, heads, len_q * p, c)
            keys = torch.matmul(flat, self.fine_key).view(bs, heads, len_q, p, self.head_dim)
        q = self.fine_query(query).view(bs, len_q, heads, 1, self.head_dim).transpose(1, 2)
        logits = (keys.to(q.dtype) * q).sum(-1) * self.head_dim**-0.5  # [bs, heads, len_q, p]
        return logits.transpose(1, 2)

    def _combine_fine(self, samples, attention_weights):
        """
        Every head's weighted sum of its fine ``samples`` with ``attention_weights``
        ``[bs, len_q, heads, P]``, lifted per head to ``[bs, len_q, C]``. The weighted sum is a
        multiply and a reduction as in the core, and the lift is one batched matmul in fp32 like
        the sampling (with the one transposition copy it needs) and a copy into the output layout.
        """
        bs, _, len_q, heads, _ = samples.shape
        read = (samples * attention_weights.unsqueeze(1)).sum(-1)  # [bs, fine_dim, len_q, heads]
        with torch.autocast(device_type=read.device.type, enabled=False):
            lifted = torch.matmul(read.permute(0, 3, 2, 1), self.fine_lift)  # [bs, heads, len_q, head_dim]
        return lifted.transpose(1, 2).reshape(bs, len_q, heads * self.head_dim)

    def _min_cells(self, value_spatial_shapes, dtype, device) -> torch.Tensor:
        """
        The ``[P, 2]`` floor of the offsets' unit ``wh``: the unit whose ``offset_scale`` half is a
        radius of ``min_sample_cells`` cells of the point's level. Built once per (shapes, dtype,
        device): multi-scale training draws a handful of shapes, and every layer asks.
        """
        key = (tuple(tuple(int(x) for x in hw) for hw in value_spatial_shapes), dtype, str(device))
        cache = self.__dict__.setdefault("_min_cells_cache", {})  # a plain attribute: not a buffer, not saved
        if key not in cache:
            if len(cache) >= 64:
                cache.clear()
            hw = torch.tensor(value_spatial_shapes, dtype=dtype).to(device, non_blocking=True)  # [L, 2]
            cache[key] = (self.min_sample_cells / self.offset_scale / hw.flip(-1))[self.point_level]
        return cache[key]

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, value, value_spatial_shapes):
        """
        Args:
            query: ``[bs, query_length, C]``
            reference_points: ``[bs, query_length, 1, 4]`` normalized cxcywh boxes (the
                2-coordinate point form of Deformable DETR is not supported here).
            value: the per-level values from ``TransformerDecoder.value_op``.
            value_spatial_shapes: the ``(h, w)`` of each level.

        Returns:
            ``[bs, query_length, C]``
        """
        bs, len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, len_q, self.num_heads, sum(self.num_points_list), 2)

        logits = self.attention_weights(query).reshape(bs, len_q, self.num_heads, -1)

        if reference_points.shape[-1] != 4:
            # See: https://github.com/lyuwenyu/RT-DETR/issues/505 for the 2-coordinate form
            raise NotImplementedError(
                f"reference points must be cxcywh boxes, got last dim {reference_points.shape[-1]}"
            )

        num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        wh = reference_points[:, :, None, :, 2:]  # [bs, len_q, 1, 1, 2], the offsets' unit
        if self.min_sample_cells > 0:
            wh = torch.maximum(wh, self._min_cells(value_spatial_shapes, wh.dtype, wh.device))
        offset = sampling_offsets * num_points_scale * wh * self.offset_scale
        sampling_locations = reference_points[:, :, None, :, :2] + offset

        p = self.num_points_list[0]
        samples = None
        if self.fine_dim > 0:
            samples = self._sample_fine(value[0], value_spatial_shapes[0], sampling_locations[:, :, :, :p])
            if self.fine_key_aware:
                # the fine level's logits are the query-predicted one plus what the sample says;
                # the softmax below still spans every level, so the levels stay comparable
                logits = torch.cat([logits[..., :p] + self._fine_logits(samples, query), logits[..., p:]], dim=-1)

        attention_weights = F.softmax(logits, dim=-1)  # over the head's points, and its null entry
        null_weight = None
        if self.null_point:
            attention_weights, null_weight = attention_weights[..., :-1], attention_weights[..., -1:]

        if self.fine_dim == 0:
            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
            )
        else:
            # the raw fine level's points come first; the other levels go through the core
            fine = self._combine_fine(samples, attention_weights[..., :p])
            rest = self.ms_deformable_attn_core(
                value[1:],
                value_spatial_shapes[1:],
                sampling_locations[:, :, :, p:],
                attention_weights[..., p:],
                self.num_points_list[1:],
            )
            output = rest + fine
        if null_weight is not None:
            # [bs, len_q, heads, 1] x [heads, head_dim] -> the output's head-major layout
            output = output + (null_weight * self.null_value.to(output.dtype)).reshape(bs, len_q, -1)
        return output
