# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Integration parity check: vendored CRadioV3 vs NVIDIA's Hub ``RADIOModel`` (requires timm)."""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.backbone.c_radio_v3 import (
    EMBED_DIM,
    HUB_REPO_ID,
    HUB_REVISION,
    NUM_PREFIX_TOKENS,
    PATCH_SIZE,
    CRadioV3,
)
from rfdetr.utilities.imports import _IS_TIMM_INSTALLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _IS_TIMM_INSTALLED, reason="timm is required to load NVIDIA's RADIOModel reference"),
]


def test_vendored_features_match_hub_reference() -> None:
    """Compare last-block spatial features against ``AutoModel(..., trust_remote_code=True)``.

    Both sides receive CLIP-normalized ``[0, 1]``-style inputs at the reference's conditioner boundary: the Hub
    model runs its own ``input_conditioner``, while the vendored encoder is fed the ImageNet-normalized tensor
    that RF-DETR would produce from the same pixels (so the folded ImageNet→CLIP affine is also exercised).
    """
    from transformers import AutoModel

    torch.manual_seed(0)
    height = width = 256
    # Synthetic "pixels" in [0, 1].
    pixels = torch.rand(1, 3, height, width)

    im_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imagenet_normalized = (pixels - im_mean) / im_std

    reference = AutoModel.from_pretrained(HUB_REPO_ID, revision=HUB_REVISION, trust_remote_code=True)
    reference.eval()
    vendored = CRadioV3(
        out_feature_indexes=[12],
        target_shape=(height, width),
        load_pretrained_weights=True,
    )
    vendored.eval()

    with torch.no_grad():
        # Hub RADIO forward: input_conditioner expects [0, 1] RGB.
        _summary, spatial = reference(pixels)
        # spatial is (B, T, D); drop nothing — Hub already strips prefix tokens.
        ref_map = spatial.transpose(1, 2).reshape(1, EMBED_DIM, height // PATCH_SIZE, width // PATCH_SIZE)

        ours = vendored(imagenet_normalized)[0]

    assert ours.shape == ref_map.shape
    # Loose enough for float32 Hub vs vendored SDPA differences; tight enough to catch remap bugs.
    assert torch.allclose(ours, ref_map, rtol=1e-3, atol=1e-4), (
        f"max abs diff={(ours - ref_map).abs().max().item():.6g}"
    )


def test_hub_reference_strips_prefix_token_count() -> None:
    """Sanity: Hub spatial token count equals H/ps * W/ps (prefix already removed)."""
    from transformers import AutoModel

    height = width = 128
    pixels = torch.rand(1, 3, height, width)
    reference = AutoModel.from_pretrained(HUB_REPO_ID, revision=HUB_REVISION, trust_remote_code=True)
    reference.eval()
    with torch.no_grad():
        _summary, spatial = reference(pixels)
    assert spatial.shape[1] == (height // PATCH_SIZE) * (width // PATCH_SIZE)
    assert spatial.shape[1] != (height // PATCH_SIZE) * (width // PATCH_SIZE) + NUM_PREFIX_TOKENS
