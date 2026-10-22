# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end, network-free test of the DINOv3-ConvNeXt backbone through the real deformable
transformer/decoder, with a real (random-init) forward + backward pass.

This also doubles as the first real exerciser of a genuine 3-level ``projector_scale=["P3","P4","P5"]`` forward
pass anywhere in the test suite: every shipped DINOv2 config uses only 1-2 levels, and the only existing 3-level
coverage (``tests/export/test_prepare.py``) only exercises output naming against a stub, never a real forward
pass through the transformer.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from rfdetr.config import RFDETRDINOv3ConvNextTinyLateralConfig, TrainConfig
from rfdetr.models import MODEL_DEFAULTS, build_criterion_from_config, build_model_from_config
from rfdetr.models.lwdetr import LWDETR

#: `pretrain_weights=None` (no trained RF-DETR checkpoint exists for this backbone yet) would otherwise make
#: `build_model_from_config` try to download the (gated) upstream DINOv3-ConvNeXt weights -- force random init so
#: these tests run offline, with no network/Hub-auth dependency.
_NO_PRETRAIN = replace(MODEL_DEFAULTS, force_no_pretrain=True)
_TRAIN_CONFIG = TrainConfig(dataset_dir=".", output_dir=".")


def _tiny_targets(batch_size: int, num_classes: int) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.randint(0, num_classes, (2,), dtype=torch.int64),
            "boxes": torch.rand(2, 4).clamp(0.05, 0.95),
        }
        for _ in range(batch_size)
    ]


class TestDINOv3ConvNextEndToEnd:
    def test_builds_a_real_lwdetr_model(self) -> None:
        mc = RFDETRDINOv3ConvNextTinyLateralConfig(resolution=128, num_classes=3)

        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)

        assert isinstance(model, LWDETR)
        assert model.class_embed.out_features == 4  # num_classes + background

    def test_forward_pass_produces_expected_shapes(self) -> None:
        mc = RFDETRDINOv3ConvNextTinyLateralConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        model.eval()
        images = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            outputs = model(images)

        assert outputs["pred_logits"].shape == (2, mc.num_queries, 4)
        assert outputs["pred_boxes"].shape == (2, mc.num_queries, 4)

    def test_forward_backward_through_real_decoder_with_3_levels(self) -> None:
        """Exercises the genuine ``projector_scale=["P3","P4","P5"]`` path end-to-end: real backbone, real
        lateral neck, real deformable transformer/decoder, real criterion, real backward pass."""
        mc = RFDETRDINOv3ConvNextTinyLateralConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        criterion, _postprocess = build_criterion_from_config(mc, _TRAIN_CONFIG, defaults=_NO_PRETRAIN)
        model.train()

        images = torch.randn(2, 3, 128, 128)
        targets = _tiny_targets(batch_size=2, num_classes=3)

        outputs = model(images, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        assert torch.isfinite(loss)
        loss.backward()

        # Gradients must reach both the ConvNeXt encoder and the lateral neck -- confirms the whole chain
        # (backbone -> neck -> transformer -> decoder -> criterion) is actually connected, not just shape-correct.
        backbone = model.backbone[0]
        encoder_grad_norms = [p.grad.norm().item() for p in backbone.encoder.parameters() if p.grad is not None]
        neck_grad_norms = [p.grad.norm().item() for p in backbone.projector.parameters() if p.grad is not None]

        assert encoder_grad_norms, "No gradients reached the DINOv3-ConvNeXt encoder."
        assert any(g > 0 for g in encoder_grad_norms)
        assert neck_grad_norms, "No gradients reached the lateral neck."
        assert any(g > 0 for g in neck_grad_norms)

    def test_one_optimizer_step_keeps_loss_finite(self) -> None:
        """A tiny synthetic training step: one forward + backward + optimizer.step(), finite loss throughout."""
        mc = RFDETRDINOv3ConvNextTinyLateralConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        criterion, _postprocess = build_criterion_from_config(mc, _TRAIN_CONFIG, defaults=_NO_PRETRAIN)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(2):
            images = torch.randn(2, 3, 128, 128)
            targets = _tiny_targets(batch_size=2, num_classes=3)

            optimizer.zero_grad()
            outputs = model(images, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
