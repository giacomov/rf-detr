# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end, network-free tests of the C-RADIOv3-B backbone through the real decoder."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rfdetr.config import RFDETRCRadioV3BaseConfig, RFDETRSegCRadioV3BaseConfig, TrainConfig
from rfdetr.models import MODEL_DEFAULTS, build_criterion_from_config, build_model_from_config
from rfdetr.models.lwdetr import LWDETR

_NO_PRETRAIN = replace(MODEL_DEFAULTS, force_no_pretrain=True)
_TRAIN_CONFIG = TrainConfig(dataset_dir=".", output_dir=".")


def _tiny_targets(batch_size: int, num_classes: int) -> list[dict[str, torch.Tensor]]:
    """Build a tiny batch of detection targets for a forward/backward smoke test.

    Examples:
        >>> targets = _tiny_targets(2, 3)
        >>> len(targets)
        2
        >>> targets[0]["labels"].dtype
        torch.int64
        >>> targets[0]["boxes"].shape
        torch.Size([2, 4])
    """
    return [
        {
            "labels": torch.randint(0, num_classes, (2,), dtype=torch.int64),
            "boxes": torch.rand(2, 4).clamp(0.05, 0.95),
        }
        for _ in range(batch_size)
    ]


class TestCRadioV3EndToEnd:
    def test_builds_a_real_lwdetr_model(self) -> None:
        mc = RFDETRCRadioV3BaseConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)

        assert isinstance(model, LWDETR)
        assert model.class_embed.out_features == 4

    def test_forward_pass_produces_expected_shapes(self) -> None:
        mc = RFDETRCRadioV3BaseConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        model.eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 128, 128))

        assert outputs["pred_logits"].shape == (2, mc.num_queries, 4)
        assert outputs["pred_boxes"].shape == (2, mc.num_queries, 4)

    def test_forward_backward_reaches_encoder_and_projector(self) -> None:
        mc = RFDETRCRadioV3BaseConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        criterion, _ = build_criterion_from_config(mc, _TRAIN_CONFIG, defaults=_NO_PRETRAIN)
        model.train()

        targets = _tiny_targets(2, 3)
        outputs = model(torch.randn(2, 3, 128, 128), targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        assert torch.isfinite(loss)
        loss.backward()

        backbone = model.backbone[0]
        encoder_grads = [p.grad.norm().item() for p in backbone.encoder.parameters() if p.grad is not None]
        projector_grads = [p.grad.norm().item() for p in backbone.projector.parameters() if p.grad is not None]
        assert encoder_grads and any(g > 0 for g in encoder_grads)
        assert projector_grads and any(g > 0 for g in projector_grads)

    def test_segmentation_config_builds_and_runs(self) -> None:
        mc = RFDETRSegCRadioV3BaseConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        assert model.segmentation_head is not None
        model.eval()

        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 128, 128))

        assert outputs["pred_logits"].shape == (1, mc.num_queries, 4)
        assert "pred_masks" in outputs

    def test_drop_path_schedule_applied(self) -> None:
        mc = RFDETRCRadioV3BaseConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        model.update_drop_path(drop_path_rate=0.2, vit_encoder_num_layers=12)

        rates = [block.drop_path.drop_prob for block in model.backbone[0].encoder.blocks]
        assert rates[0] == pytest.approx(0.0)
        assert rates[-1] == pytest.approx(0.2)
        assert rates[0] <= rates[len(rates) // 2] <= rates[-1]

    def test_lr_groups_key_on_blocks(self) -> None:
        mc = RFDETRCRadioV3BaseConfig(resolution=128, num_classes=3)
        model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
        args = SimpleNamespace(
            out_feature_indexes=mc.out_feature_indexes,
            lr_encoder=1e-4,
            lr_vit_layer_decay=0.8,
            lr_component_decay=1.0,
            weight_decay=0.05,
        )
        pairs = model.backbone[0].get_named_param_lr_pairs(args)
        block_keys = [name for name in pairs if ".blocks." in name]
        assert block_keys, "Expected per-block LR groups keyed on .blocks."
        # Deeper blocks should get a higher LR than shallower ones under decay < 1.
        shallow = next(v["lr"] for k, v in pairs.items() if ".blocks.0." in k)
        deep = next(v["lr"] for k, v in pairs.items() if ".blocks.11." in k)
        assert deep > shallow
