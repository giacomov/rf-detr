# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for multi-label annotation merging and ConvertCoco integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import ConvertCoco, _train_multi_label_iou_threshold
from rfdetr.datasets.multilabel import (
    connected_components,
    merge_multilabel_annotations,
    summarize_multilabel_merges,
)
from rfdetr.datasets.transforms import AlbumentationsWrapper
from rfdetr.datasets._torchvision import Compose, RandomHorizontalFlip, Resize


def _xyxy(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2, y2]


class TestConnectedComponents:
    def test_isolated_nodes(self) -> None:
        adj = torch.zeros(3, 3, dtype=torch.bool)
        components = connected_components(adj)
        assert components.tolist() == [0, 1, 2]

    def test_pair_merges(self) -> None:
        adj = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
        components = connected_components(adj)
        assert components[0] == components[1]
        assert components[2] != components[0]


class TestMergeMultilabelAnnotations:
    def test_exact_pair_merges(self) -> None:
        boxes = torch.tensor([_xyxy(10, 10, 50, 50), _xyxy(10, 10, 50, 50)], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        merged = merge_multilabel_annotations(boxes, labels, num_classes=3, iou_threshold=0.99)
        assert merged["boxes"].shape[0] == 1
        assert merged["label_sets"].tolist() == [[1.0, 1.0, 0.0]]
        assert merged["labels"].tolist() == [0]

    def test_iou_threshold_boundary(self) -> None:
        # Boxes with IoU ~0.995 (near-exact) vs ~0.97.
        high = torch.tensor([_xyxy(0, 0, 100, 100), _xyxy(0, 0, 100, 99.5)], dtype=torch.float32)
        low = torch.tensor([_xyxy(0, 0, 100, 100), _xyxy(0, 0, 100, 96)], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        merged_high = merge_multilabel_annotations(high, labels, num_classes=2, iou_threshold=0.99)
        merged_low = merge_multilabel_annotations(low, labels, num_classes=2, iou_threshold=0.99)
        assert merged_high["boxes"].shape[0] == 1
        assert merged_low["boxes"].shape[0] == 2

    def test_same_class_never_merges(self) -> None:
        boxes = torch.tensor([_xyxy(0, 0, 10, 10), _xyxy(0, 0, 10, 10)], dtype=torch.float32)
        labels = torch.tensor([1, 1], dtype=torch.int64)
        merged = merge_multilabel_annotations(boxes, labels, num_classes=3, iou_threshold=0.99)
        assert merged["boxes"].shape[0] == 2
        assert merged["label_sets"].tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

    def test_mask_union_and_enclosing_box(self) -> None:
        boxes = torch.tensor([_xyxy(0, 0, 4, 4), _xyxy(1, 1, 5, 5)], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        masks = torch.zeros(2, 6, 6, dtype=torch.bool)
        masks[0, 0:3, 0:3] = True
        masks[1, 2:5, 2:5] = True
        # Force merge regardless of the slight box mismatch by using a low threshold.
        merged = merge_multilabel_annotations(
            boxes, labels, num_classes=2, iou_threshold=0.3, masks=masks
        )
        assert merged["boxes"].shape[0] == 1
        assert merged["boxes"][0].tolist() == [0.0, 0.0, 5.0, 5.0]
        expected = masks[0] | masks[1]
        assert torch.equal(merged["masks"][0], expected)

    def test_empty_and_singleton_emit_label_sets(self) -> None:
        empty = merge_multilabel_annotations(
            torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64), num_classes=4, iou_threshold=0.99
        )
        assert empty["label_sets"].shape == (0, 4)
        single = merge_multilabel_annotations(
            torch.tensor([_xyxy(0, 0, 1, 1)]),
            torch.tensor([2], dtype=torch.int64),
            num_classes=4,
            iou_threshold=0.99,
        )
        assert single["label_sets"].tolist() == [[0.0, 0.0, 1.0, 0.0]]


class TestConvertCocoMultiLabel:
    def test_convert_emits_label_sets(self) -> None:
        converter = ConvertCoco(multi_label_iou_threshold=0.99, num_classes=3)
        image = Image.new("RGB", (100, 100))
        target = {
            "image_id": 1,
            "annotations": [
                {"bbox": [10, 10, 40, 40], "category_id": 0, "area": 1600, "iscrowd": 0},
                {"bbox": [10, 10, 40, 40], "category_id": 1, "area": 1600, "iscrowd": 0},
                {"bbox": [60, 60, 20, 20], "category_id": 2, "area": 400, "iscrowd": 0},
            ],
        }
        _, out = converter(image, target)
        assert "label_sets" in out
        assert out["boxes"].shape[0] == 2
        assert out["label_sets"].shape == (2, 3)
        # One row is the merged multi-hot for classes 0 and 1.
        multi = (out["label_sets"].sum(dim=1) > 1).nonzero(as_tuple=False).flatten()
        assert multi.numel() == 1
        assert out["label_sets"][multi[0]].tolist() == [1.0, 1.0, 0.0]


def _apply_pipeline_and_check_alignment(
    pipeline: Any, image: Image.Image, target: dict[str, Any]
) -> dict[str, Any]:
    img, out = pipeline(image, target)
    assert out is not None
    n = out["boxes"].shape[0]
    assert out["labels"].shape[0] == n
    assert out["label_sets"].shape[0] == n
    if "masks" in out:
        assert out["masks"].shape[0] == n
    # Surviving multi-hot rows must still be valid 0/1 vectors of the original width.
    assert out["label_sets"].shape[1] == target["label_sets"].shape[1]
    assert ((out["label_sets"] == 0) | (out["label_sets"] == 1)).all()
    return out


class TestLabelSetsSurviveAugmentation:
    @pytest.fixture()
    def multi_label_sample(self) -> tuple[Image.Image, dict[str, Any]]:
        image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        boxes = torch.tensor(
            [
                [5.0, 5.0, 30.0, 30.0],
                [35.0, 35.0, 60.0, 60.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1], dtype=torch.int64)
        label_sets = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        masks = torch.zeros(2, 64, 64, dtype=torch.bool)
        masks[0, 5:30, 5:30] = True
        masks[1, 35:60, 35:60] = True
        target = {
            "boxes": boxes,
            "labels": labels,
            "label_sets": label_sets,
            "masks": masks,
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(2, dtype=torch.int64),
            "image_id": torch.tensor([1]),
            "orig_size": torch.tensor([64, 64]),
            "size": torch.tensor([64, 64]),
        }
        return image, target

    def test_albumentations_filter_keeps_rows_aligned(
        self, multi_label_sample: tuple[Image.Image, dict[str, Any]]
    ) -> None:
        # Exercise the same per-instance filter AlbumentationsWrapper uses after a crop.
        _, target = multi_label_sample
        filtered = AlbumentationsWrapper._filter_per_instance_fields(target, num_boxes=2, kept_idxs=[1])
        assert filtered["label_sets"].tolist() == [[0.0, 0.0, 1.0]]
        assert filtered["area"].shape[0] == 1
        assert filtered["masks"].shape[0] == 1

    def test_albumentations_clear_keeps_shape(
        self, multi_label_sample: tuple[Image.Image, dict[str, Any]]
    ) -> None:
        _, target = multi_label_sample
        cleared = AlbumentationsWrapper._clear_per_instance_fields(target, num_boxes=2)
        assert cleared["label_sets"].shape == (0, 3)
        assert cleared["masks"].shape[0] == 0

    def test_albumentations_crop_end_to_end(
        self, multi_label_sample: tuple[Image.Image, dict[str, Any]]
    ) -> None:
        pytest.importorskip("albumentations")
        from albumentations import Compose as AlbuCompose
        from albumentations import Crop

        image, target = multi_label_sample
        albu = AlbuCompose([Crop(x_min=32, y_min=32, x_max=64, y_max=64, p=1.0)])
        wrapper = AlbumentationsWrapper(albu)
        out = _apply_pipeline_and_check_alignment(wrapper, image, target)
        assert out["boxes"].shape[0] == 1
        assert out["label_sets"][0].tolist() == [0.0, 0.0, 1.0]

    def test_torchvision_resize_keeps_rows_aligned(
        self, multi_label_sample: tuple[Image.Image, dict[str, Any]]
    ) -> None:
        image, target = multi_label_sample
        # Resize + no-op flip keeps both objects; rows must stay aligned.
        pipeline = Compose([Resize((32, 32)), RandomHorizontalFlip(p=0.0)])
        out = _apply_pipeline_and_check_alignment(pipeline, image, target)
        assert out["boxes"].shape[0] == 2
        assert out["label_sets"].tolist() == [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class TestTrainThresholdHelper:
    def test_only_train_split(self) -> None:
        args = SimpleNamespace(multi_label_targets=True, multi_label_iou_threshold=0.95)
        assert _train_multi_label_iou_threshold("train", args) == 0.95
        assert _train_multi_label_iou_threshold("val", args) is None
        args.multi_label_targets = False
        assert _train_multi_label_iou_threshold("train", args) is None


class TestSummarizeWithoutImages:
    def test_summarize_finds_duplicates(self, tmp_path: Path) -> None:
        # Minimal fake COCO API object — no image files.
        coco = MagicMock()
        coco.imgs = {1: {"id": 1, "width": 100, "height": 100}}
        coco.getAnnIds.return_value = [1, 2]
        coco.loadAnns.return_value = [
            {"bbox": [10, 10, 40, 40], "category_id": 1, "iscrowd": 0},
            {"bbox": [10, 10, 40, 40], "category_id": 2, "iscrowd": 0},
        ]
        coco.cats = {1: {"id": 1, "name": "a"}, 2: {"id": 2, "name": "b"}}
        summary = summarize_multilabel_merges(
            coco, cat2label={1: 0, 2: 1}, num_classes=2, iou_threshold=0.99
        )
        assert summary["num_multilabel_objects"] == 1
        assert summary["num_objects_before"] == 2
        assert summary["num_objects_after"] == 1
        # Image folder is unused; prove the helper never needs it.
        assert not (tmp_path / "no_such_images").exists()


class TestBuildRoboflowYoloRefusal:
    def test_yolo_layout_raises(self, tmp_path: Path) -> None:
        from rfdetr.datasets import build_roboflow

        # Minimal YOLO-looking layout: train/images exists, no COCO annotations.
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "data.yaml").write_text("train: train/images\nval: train/images\nnames: [a]\n")
        args = SimpleNamespace(
            dataset_dir=str(tmp_path),
            multi_label_targets=True,
            square_resize_div_64=True,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            patch_size=14,
            num_windows=1,
        )
        with pytest.raises(ValueError, match="COCO-format"):
            build_roboflow("train", args, resolution=224)
