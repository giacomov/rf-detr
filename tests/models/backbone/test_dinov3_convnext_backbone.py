# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Offline, network-free tests for the DINOv3-ConvNeXt neck(s) and the ``Backbone`` subclass that wires them to
the encoder. All tests build with ``load_dinov2_weights=False`` (random init), so no network access is needed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

import rfdetr.models.backbone.dinov3_convnext_backbone as backbone_module
from rfdetr.models.backbone import Joiner, build_backbone
from rfdetr.models.backbone.dinov3_convnext_backbone import DINOv3ConvNextEncoderBackbone, _parse_name
from rfdetr.models.backbone.dinov3_convnext_neck import LateralNeck
from rfdetr.utilities.tensors import NestedTensor


def _random_feats(channels: list[int], sizes: list[int], batch: int = 2) -> list[Tensor]:
    return [torch.randn(batch, c, s, s) for c, s in zip(channels, sizes)]


class TestLateralNeck:
    """Shape contract and no-cross-level-mixing property."""

    def test_projects_every_level_to_out_channels_preserving_shape(self) -> None:
        neck = LateralNeck(in_channels=[96, 192, 384], out_channels=16)
        feats = _random_feats([96, 192, 384], [32, 16, 8])

        outs = neck(feats)

        assert [tuple(o.shape) for o in outs] == [(2, 16, 32, 32), (2, 16, 16, 16), (2, 16, 8, 8)]

    def test_wrong_number_of_input_levels_raises(self) -> None:
        neck = LateralNeck(in_channels=[96, 192, 384], out_channels=16)

        with pytest.raises(ValueError, match="built for 3 levels, got 2"):
            neck(_random_feats([96, 192], [32, 16]))

    def test_levels_do_not_mix(self) -> None:
        """Changing one input level must not change any other output level (pure per-level projection)."""
        neck = LateralNeck(in_channels=[96, 192, 384], out_channels=16)
        neck.eval()
        feats_a = _random_feats([96, 192, 384], [32, 16, 8])
        feats_b = [feats_a[0], torch.randn_like(feats_a[1]), feats_a[2]]

        with torch.no_grad():
            outs_a = neck(feats_a)
            outs_b = neck(feats_b)

        assert torch.equal(outs_a[0], outs_b[0])
        assert not torch.equal(outs_a[1], outs_b[1])
        assert torch.equal(outs_a[2], outs_b[2])


class TestParseName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("dinov3_convnext_tiny_lateral", ("tiny", "lateral"), id="tiny-lateral"),
            pytest.param("dinov3_convnext_small_lateral", ("small", "lateral"), id="small-lateral"),
            pytest.param("dinov3_convnext_base_lateral", ("base", "lateral"), id="base-lateral"),
        ],
    )
    def test_valid_names(self, name: str, expected: tuple[str, str]) -> None:
        assert _parse_name(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("dinov3_convnext_tiny", id="missing-neck"),
            pytest.param("dinov2_windowed_small", id="wrong-family"),
            pytest.param("dinov3_convnext_huge_lateral", id="unknown-size"),
            pytest.param("dinov3_convnext_tiny_fpn", id="unregistered-neck"),
        ],
    )
    def test_invalid_names_raise(self, name: str) -> None:
        with pytest.raises(ValueError):
            _parse_name(name)


def _build(encoder: str, **overrides: Any) -> Joiner:
    """Call ``build_backbone`` for *encoder* with small test arguments, mirroring
    ``tests/models/backbone/test_backbone_registry.py``'s helper of the same name.
    """
    kwargs: dict[str, Any] = dict(
        encoder=encoder,
        vit_encoder_num_layers=12,
        pretrained_encoder=None,
        window_block_indexes=None,
        drop_path=0.0,
        out_channels=16,
        out_feature_indexes=[2, 3, 4],
        projector_scale=["P3", "P4", "P5"],
        use_cls_token=False,
        hidden_dim=16,
        position_embedding="sine",
        freeze_encoder=False,
        layer_norm=True,
        target_shape=(256, 256),
        rms_norm=False,
        backbone_lora=False,
        force_no_pretrain=False,
        gradient_checkpointing=False,
        load_dinov2_weights=False,
        patch_size=32,
        num_windows=1,
        positional_encoding_size=8,
    )
    kwargs.update(overrides)
    return build_backbone(**kwargs)


class TestDINOv3ConvNextEncoderBackbone:
    def test_builds_through_the_registry(self) -> None:
        joiner = _build("dinov3_convnext_tiny_lateral")

        assert isinstance(joiner, Joiner)
        assert type(joiner[0]) is DINOv3ConvNextEncoderBackbone

    def test_forward_produces_one_level_per_projector_scale_entry(self) -> None:
        joiner = _build("dinov3_convnext_tiny_lateral")
        images = torch.randn(2, 3, 256, 256)
        mask = torch.zeros(2, 256, 256, dtype=torch.bool)
        mask[1, :, 128:] = True

        features, poss, cross = joiner(NestedTensor(images, mask))

        assert cross is None
        assert [tuple(f.tensors.shape) for f in features] == [
            (2, 16, 32, 32),
            (2, 16, 16, 16),
            (2, 16, 8, 8),
        ]
        # Padding mask: True (padded) for the right half of batch element 1, nothing padded for element 0 --
        # downsampled onto each level's own grid by Backbone._level_mask, inherited unchanged.
        assert features[0].mask[1, :, 16:].all() and not features[0].mask[0].any()
        assert tuple(poss[0].shape) == (2, 16, 32, 32)

    def test_dual_projector_shares_spatial_shapes_with_primary(self) -> None:
        joiner = _build("dinov3_convnext_tiny_lateral", dual_projector=True)
        images = torch.randn(2, 3, 256, 256)
        mask = torch.zeros(2, 256, 256, dtype=torch.bool)

        features, _poss, cross = joiner(NestedTensor(images, mask))

        assert cross is not None
        assert [tuple(f.tensors.shape[-2:]) for f in features] == [tuple(c.tensors.shape[-2:]) for c in cross]

    def test_freeze_encoder_stops_gradients_to_the_backbone(self) -> None:
        joiner = _build("dinov3_convnext_tiny_lateral", freeze_encoder=True)

        assert all(not p.requires_grad for p in joiner[0].encoder.parameters())
        assert all(p.requires_grad for p in joiner[0].projector.parameters())

    def test_mismatched_projector_scale_length_raises(self) -> None:
        with pytest.raises(ValueError, match="one ConvNeXt stage per projector_scale level"):
            _build("dinov3_convnext_tiny_lateral", out_feature_indexes=[4], projector_scale=["P3", "P4", "P5"])

    def test_unsorted_projector_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="ascending-order subset"):
            _build("dinov3_convnext_tiny_lateral", projector_scale=["P5", "P3", "P4"])

    def test_missing_out_feature_indexes_raises(self) -> None:
        with pytest.raises(ValueError, match="requires out_feature_indexes"):
            _build("dinov3_convnext_tiny_lateral", out_feature_indexes=None)

    def test_registered_for_every_size(self) -> None:
        import rfdetr.models.backbone as backbone_pkg

        for size in ("tiny", "small", "base"):
            assert backbone_pkg._BACKBONE_REGISTRY[f"dinov3_convnext_{size}_lateral"] is DINOv3ConvNextEncoderBackbone
        # "fpn" isn't registered yet -- added alongside FPNNeck in a later step.
        assert "dinov3_convnext_tiny_fpn" not in backbone_pkg._BACKBONE_REGISTRY

    def test_module_is_registered_without_reimporting(self) -> None:
        """Guards against the registration call silently not running at import time."""
        assert backbone_module.DINOv3ConvNextEncoderBackbone is DINOv3ConvNextEncoderBackbone
