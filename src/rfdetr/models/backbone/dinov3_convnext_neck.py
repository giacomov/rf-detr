# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Necks for the DINOv3-ConvNeXt backbone.

``MultiScaleProjector`` (``rfdetr.models.backbone.projector``) was built for DINOv2's case: several feature maps
all at the *same* spatial resolution, extracted from different transformer depths, which it then resamples into an
artificial P3/P4/P5 pyramid. A true CNN backbone like ConvNeXt doesn't have that problem -- its stage2/stage3/
stage4 outputs are already a *real* multi-resolution pyramid (strides 8/16/32), which is exactly RF-DETR's P3/P4/P5
convention with zero resampling needed. The necks here only ever change channel count, never spatial resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr.models.backbone.projector import C2f, ConvX, get_norm


class LateralNeck(nn.Module):
    """Per-level 1x1 conv + norm, no cross-level fusion.

    Each input level is already at its target P-level resolution (ConvNeXt stage2/3/4 == P3/P4/P5 natively), so
    this only projects channel counts -- no resampling, no mixing between levels.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            # ConvX.act is typed as `str`, but its own get_activation(name, inplace) accepts and handles
            # `None` (-> nn.Identity()) at runtime -- a pre-existing stub/runtime mismatch in projector.py.
            ConvX(c, out_channels, kernel=1, act=None, layer_norm=layer_norm, rms_norm=rms_norm)  # type: ignore[arg-type]
            for c in in_channels
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Project each level to ``out_channels``, preserving its spatial shape and level order."""
        if len(feats) != len(self.lateral):
            raise ValueError(f"LateralNeck built for {len(self.lateral)} levels, got {len(feats)}.")
        return [lat(f) for lat, f in zip(self.lateral, feats)]


class FPNNeck(nn.Module):
    """Lateral 1x1 projections, then classic top-down fusion from the coarsest level down.

    Each level is laterally projected to ``out_channels``, then (from the coarsest level to the finest) upsampled
    and added into the next-finer level's lateral, each sum refined by a ``C2f`` block before a final channel
    ``LayerNorm``. Unlike ``LateralNeck``, every output level mixes information from every coarser level -- the
    classic FPN top-down pathway -- while still preserving each input level's own spatial shape (only the
    coarser level's feature map is resampled, up to the finer level's resolution, never the other way around).
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int = 1,
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            ConvX(c, out_channels, kernel=1, act=None, layer_norm=layer_norm, rms_norm=rms_norm)  # type: ignore[arg-type]
            for c in in_channels
        )
        self.fuse = nn.ModuleList(
            C2f(out_channels, out_channels, num_blocks, layer_norm=layer_norm, rms_norm=rms_norm) for _ in in_channels
        )
        self.norm = nn.ModuleList(cast(nn.Module, get_norm("LN", out_channels)) for _ in in_channels)

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Fuse every level top-down, finest-level output first (same order as the input levels)."""
        if len(feats) != len(self.lateral):
            raise ValueError(f"FPNNeck built for {len(self.lateral)} levels, got {len(feats)}.")

        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        outs: list[Tensor] = [laterals[-1]]
        for lateral in reversed(laterals[:-1]):
            up = F.interpolate(outs[0], size=lateral.shape[-2:], mode="nearest")
            outs.insert(0, lateral + up)
        return [cast(Tensor, norm(fuse(out))) for norm, fuse, out in zip(self.norm, self.fuse, outs)]
