# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Offline, network-free tests for the vendored C-RADIOv3-B encoder."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models.backbone.c_radio_v3 import (
    CPE_GRID_SIZE,
    EMBED_DIM,
    LAYER_SCALE_INIT,
    MLP_RATIO,
    NUM_PREFIX_TOKENS,
    PATCH_SIZE,
    CRadioV3,
    _cpe_pos_embed_eval,
    _imagenet_to_clip_affine,
    remap_c_radio_v3_state_dict,
)


class TestCRadioV3Shapes:
    """Shape / channel contract with random init."""

    def test_forward_shapes_square(self) -> None:
        encoder = CRadioV3(
            out_feature_indexes=[3, 6, 9, 12],
            target_shape=(128, 128),
            load_pretrained_weights=False,
        )
        assert encoder._out_feature_channels == [EMBED_DIM] * 4

        feats = encoder(torch.randn(2, 3, 128, 128))

        assert len(feats) == 4
        for feat in feats:
            assert tuple(feat.shape) == (2, EMBED_DIM, 128 // PATCH_SIZE, 128 // PATCH_SIZE)

    def test_forward_shapes_nonsquare(self) -> None:
        encoder = CRadioV3(
            out_feature_indexes=[6, 12],
            target_shape=(64, 96),
            load_pretrained_weights=False,
        )
        feats = encoder(torch.randn(1, 3, 64, 96))

        assert [tuple(f.shape) for f in feats] == [
            (1, EMBED_DIM, 4, 6),
            (1, EMBED_DIM, 4, 6),
        ]

    def test_prefix_tokens_are_stripped(self) -> None:
        """Output spatial token count must be H/ps * W/ps, not that plus the 8 prefix tokens."""
        encoder = CRadioV3(
            out_feature_indexes=[12],
            target_shape=(64, 64),
            load_pretrained_weights=False,
        )
        feats = encoder(torch.randn(1, 3, 64, 64))
        assert feats[0].shape[-2:] == (4, 4)
        assert feats[0].numel() == 1 * EMBED_DIM * 4 * 4

    def test_invalid_indexes_raise(self) -> None:
        with pytest.raises(ValueError, match="1..12"):
            CRadioV3(out_feature_indexes=[0, 3], load_pretrained_weights=False)
        with pytest.raises(ValueError, match="ascending"):
            CRadioV3(out_feature_indexes=[12, 3], load_pretrained_weights=False)
        with pytest.raises(ValueError, match="non-empty"):
            CRadioV3(out_feature_indexes=[], load_pretrained_weights=False)

    def test_non_divisible_input_raises(self) -> None:
        encoder = CRadioV3(out_feature_indexes=[12], target_shape=(64, 64), load_pretrained_weights=False)
        with pytest.raises(ValueError, match="divisible"):
            encoder(torch.randn(1, 3, 65, 64))


class TestCpePosEmbed:
    def test_matches_reference_interpolate_and_crop(self) -> None:
        pe = torch.randn(1, CPE_GRID_SIZE * CPE_GRID_SIZE, 8)
        height_tokens, width_tokens = 5, 7
        got = _cpe_pos_embed_eval(pe, height_tokens, width_tokens)

        grid = pe.reshape(1, CPE_GRID_SIZE, CPE_GRID_SIZE, -1).permute(0, 3, 1, 2)
        max_dim = max(height_tokens, width_tokens)
        grid = F.interpolate(grid.float(), size=(max_dim, max_dim), mode="bilinear", align_corners=True)
        grid = grid[..., :height_tokens, :width_tokens]
        expected = grid.flatten(2).permute(0, 2, 1)

        assert torch.allclose(got, expected)

    def test_set_export_shape_rebakes(self) -> None:
        encoder = CRadioV3(
            out_feature_indexes=[12],
            target_shape=(64, 64),
            load_pretrained_weights=False,
        )
        first = encoder.baked_pos_embed.clone()
        encoder.set_export_shape((96, 128))
        assert encoder._baked_token_hw == (96 // PATCH_SIZE, 128 // PATCH_SIZE)
        assert encoder.baked_pos_embed.shape == (1, (96 // PATCH_SIZE) * (128 // PATCH_SIZE), EMBED_DIM)
        assert not torch.equal(first, encoder.baked_pos_embed)


class TestAffineNormalization:
    def test_affine_matches_denorm_then_clip(self) -> None:
        scale, shift = _imagenet_to_clip_affine()
        x = torch.randn(2, 3, 8, 8)
        y = x * scale + shift

        im_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        im_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        pixel = x * im_std + im_mean
        expected = (pixel - clip_mean) / clip_std

        assert torch.allclose(y, expected, atol=1e-6)


class TestRemapStateDict:
    def test_remap_patch_embed_and_grandma(self) -> None:
        w = torch.randn(EMBED_DIM, 3 * PATCH_SIZE * PATCH_SIZE)
        pe = torch.randn(1, CPE_GRID_SIZE * CPE_GRID_SIZE, EMBED_DIM)
        pref = torch.randn(NUM_PREFIX_TOKENS, EMBED_DIM)
        raw = {
            "radio_model.model.patch_generator.embedder.weight": w,
            "radio_model.model.patch_generator.pos_embed": pe,
            "radio_model.model.patch_generator.cls_token.token": pref,
            "radio_model.model.blocks.0.attn.qkv.weight": torch.randn(EMBED_DIM * 3, EMBED_DIM),
            "radio_model.model.blocks.0.attn.qkv.bias": torch.randn(EMBED_DIM * 3),
            "radio_model.model.blocks.0.attn.proj.weight": torch.randn(EMBED_DIM, EMBED_DIM),
            "radio_model.model.blocks.0.attn.proj.bias": torch.randn(EMBED_DIM),
            "radio_model.model.blocks.0.norm1.weight": torch.ones(EMBED_DIM),
            "radio_model.model.blocks.0.norm1.bias": torch.zeros(EMBED_DIM),
            "radio_model.model.blocks.0.norm2.weight": torch.ones(EMBED_DIM),
            "radio_model.model.blocks.0.norm2.bias": torch.zeros(EMBED_DIM),
            "radio_model.model.blocks.0.mlp.fc1.weight": torch.randn(int(EMBED_DIM * MLP_RATIO), EMBED_DIM),
            "radio_model.model.blocks.0.mlp.fc1.bias": torch.randn(int(EMBED_DIM * MLP_RATIO)),
            "radio_model.model.blocks.0.mlp.fc2.weight": torch.randn(EMBED_DIM, int(EMBED_DIM * MLP_RATIO)),
            "radio_model.model.blocks.0.mlp.fc2.bias": torch.randn(EMBED_DIM),
            "radio_model.model.blocks.0.ls1.grandma": torch.ones(EMBED_DIM) * LAYER_SCALE_INIT,
            "radio_model.model.blocks.0.ls2.grandma": torch.ones(EMBED_DIM) * LAYER_SCALE_INIT,
            "radio_model.model.reg_token": torch.randn(1, 4, EMBED_DIM),
            "radio_model.summary_idxs": torch.tensor([0, 1, 2], dtype=torch.int64),
            "radio_model.input_conditioner.norm_mean": torch.tensor([0.48, 0.45, 0.40]).view(3, 1, 1),
            "radio_model.input_conditioner.norm_std": torch.tensor([0.26, 0.26, 0.27]).view(3, 1, 1),
        }
        out = remap_c_radio_v3_state_dict(raw)

        assert out["patch_embed.weight"].shape == (EMBED_DIM, 3, PATCH_SIZE, PATCH_SIZE)
        assert torch.equal(out["patch_embed.weight"].flatten(1), w)
        assert "blocks.0.ls1.gamma" in out
        assert "reg_token" not in out
        assert "summary_idxs" not in out

        # A model with only block 0 still accepts the remapped subset under strict=False; full checkpoints
        # cover all 12 blocks. Here we only assert the remapped keys that exist on a fresh encoder.
        encoder = CRadioV3(out_feature_indexes=[12], target_shape=(64, 64), load_pretrained_weights=False)
        encoder.load_state_dict(out, strict=False)
