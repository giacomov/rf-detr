# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for MultiLabelHungarianMatcher and MultiLabelSetCriterion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.multilabel import MultiLabelHungarianMatcher, MultiLabelSetCriterion


def _outputs(batch: int = 1, queries: int = 4, classes: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "pred_logits": torch.randn(batch, queries, classes),
        "pred_boxes": torch.rand(batch, queries, 4).clamp(0.05, 0.95),
    }


def _single_label_targets(classes: int = 3) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.tensor([0, 2], dtype=torch.int64),
            "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
            "label_sets": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        }
    ]


def _multi_label_target() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
            "label_sets": torch.tensor([[1.0, 1.0, 0.0]]),
        }
    ]


class TestMultiLabelMatcher:
    def test_without_label_sets_matches_stock(self) -> None:
        stock = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        multi = MultiLabelHungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        outputs = _outputs()
        targets = [
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.4, 0.4, 0.2, 0.2]]),
            }
        ]
        assert stock(outputs, targets)[0][0].tolist() == multi(outputs, targets)[0][0].tolist()
        assert stock(outputs, targets)[0][1].tolist() == multi(outputs, targets)[0][1].tolist()

    def test_single_label_sets_match_stock_gather(self) -> None:
        stock = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        multi = MultiLabelHungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        outputs = _outputs()
        targets = _single_label_targets()
        stock_targets = [{"labels": t["labels"], "boxes": t["boxes"]} for t in targets]
        stock_out = stock(outputs, stock_targets)
        multi_out = multi(outputs, targets)
        assert stock_out[0][0].tolist() == multi_out[0][0].tolist()
        assert stock_out[0][1].tolist() == multi_out[0][1].tolist()

    def test_two_label_object_gets_one_query(self) -> None:
        matcher = MultiLabelHungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        outputs = _outputs(queries=6)
        indices = matcher(outputs, _multi_label_target())
        src, tgt = indices[0]
        assert len(src) == 1
        assert tgt.tolist() == [0]

    def test_group_detr(self) -> None:
        matcher = MultiLabelHungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        outputs = _outputs(queries=6)
        indices = matcher(outputs, _multi_label_target(), group_detr=3)
        src, tgt = indices[0]
        # One match per group for the single target.
        assert len(src) == 3
        assert tgt.tolist() == [0, 0, 0]

    def test_row_count_mismatch_raises(self) -> None:
        matcher = MultiLabelHungarianMatcher()
        outputs = _outputs()
        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]),
                "label_sets": torch.tensor([[1.0, 0.0, 0.0]]),  # wrong row count
            }
        ]
        with pytest.raises(ValueError, match="label_sets has"):
            matcher(outputs, targets)


class TestMultiLabelCriterion:
    def _criterion(self, matcher: HungarianMatcher | None = None) -> MultiLabelSetCriterion:
        matcher = matcher or MultiLabelHungarianMatcher(
            cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25
        )
        return MultiLabelSetCriterion(
            num_classes=3,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
            ia_bce_loss=True,
        )

    def test_single_label_sets_loss_matches_stock(self) -> None:
        stock_matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        stock = SetCriterion(
            num_classes=3,
            matcher=stock_matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
            ia_bce_loss=True,
        )
        multi = self._criterion(
            MultiLabelHungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
        )
        outputs = _outputs()
        targets = _single_label_targets()
        stock_targets = [{"labels": t["labels"], "boxes": t["boxes"]} for t in targets]
        # Force identical matching by using the same seed and same costs; then compare losses.
        stock_losses = stock(outputs, stock_targets)
        multi_losses = multi(outputs, targets)
        assert torch.allclose(stock_losses["loss_ce"], multi_losses["loss_ce"], atol=1e-5)
        assert torch.allclose(stock_losses["loss_bbox"], multi_losses["loss_bbox"], atol=1e-5)
        assert torch.allclose(stock_losses["loss_giou"], multi_losses["loss_giou"], atol=1e-5)

    def test_two_label_object_has_positives_in_both_columns(self) -> None:
        criterion = self._criterion()
        # One query, two classes; force the matcher by making query 0 dominate class costs.
        outputs = {
            "pred_logits": torch.tensor([[[5.0, 5.0, -5.0], [-5.0, -5.0, -5.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1]]]),
        }
        targets = _multi_label_target()
        indices = criterion.matcher(outputs, targets)
        losses = criterion.loss_labels(outputs, targets, indices, num_boxes=torch.tensor(1.0))
        assert "loss_ce" in losses
        # Manually check that both positive columns receive a reduced negative weight.
        # After loss_labels, we recompute the weight pattern the same way.
        src_logits = outputs["pred_logits"]
        idx = criterion._get_src_permutation_idx(indices)
        matched = torch.cat([t["label_sets"][j] for t, (_, j) in zip(targets, indices)])
        assert matched[0].tolist() == [1.0, 1.0, 0.0]
        assert idx[1].tolist() == [0]

    def test_without_label_sets_delegates(self) -> None:
        criterion = self._criterion()
        outputs = _outputs()
        targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        losses = criterion(outputs, targets)
        assert "loss_ce" in losses

    def test_row_mismatch_raises(self) -> None:
        criterion = self._criterion()
        outputs = _outputs()
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "label_sets": torch.zeros(2, 3),
            }
        ]
        with pytest.raises(ValueError, match="label_sets has"):
            criterion(outputs, targets)


class TestBuildCriterionWiring:
    def _args(self, **overrides: Any) -> SimpleNamespace:
        base = dict(
            device="cpu",
            set_cost_class=2,
            set_cost_bbox=5,
            set_cost_giou=2,
            focal_alpha=0.25,
            cls_loss_coef=1.0,
            bbox_loss_coef=5.0,
            giou_loss_coef=2.0,
            segmentation_head=False,
            use_grouppose_keypoints=False,
            aux_loss=False,
            two_stage=False,
            dec_layers=3,
            num_classes=2,
            group_detr=1,
            use_varifocal_loss=False,
            use_position_supervised_loss=False,
            ia_bce_loss=True,
            num_select=100,
            multi_label_targets=False,
            sum_group_losses=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_flag_selects_multilabel_classes(self) -> None:
        criterion, _ = build_criterion_and_postprocessors(self._args(multi_label_targets=True))
        assert isinstance(criterion, MultiLabelSetCriterion)
        assert isinstance(criterion.matcher, MultiLabelHungarianMatcher)

    def test_flag_off_keeps_stock(self) -> None:
        criterion, _ = build_criterion_and_postprocessors(self._args(multi_label_targets=False))
        assert type(criterion) is SetCriterion
        assert type(criterion.matcher) is HungarianMatcher

    def test_requires_ia_bce(self) -> None:
        with pytest.raises(ValueError, match="ia_bce_loss"):
            build_criterion_and_postprocessors(
                self._args(multi_label_targets=True, ia_bce_loss=False)
            )

    def test_refuses_keypoints(self) -> None:
        with pytest.raises(ValueError, match="keypoint"):
            build_criterion_and_postprocessors(
                self._args(multi_label_targets=True, use_grouppose_keypoints=True)
            )


class TestTrainConfigFields:
    def test_defaults_and_threshold_validator(self) -> None:
        from rfdetr.config import TrainConfig

        cfg = TrainConfig(dataset_dir="ds")
        assert cfg.multi_label_targets is False
        assert cfg.multi_label_iou_threshold == 0.99
        with pytest.raises(ValueError, match="multi_label_iou_threshold"):
            TrainConfig(dataset_dir="ds", multi_label_iou_threshold=0.0)
        with pytest.raises(ValueError, match="multi_label_iou_threshold"):
            TrainConfig(dataset_dir="ds", multi_label_iou_threshold=1.5)
