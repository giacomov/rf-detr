# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Merge co-located multi-label COCO annotations into joint targets before augmentation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import Tensor

from rfdetr.utilities.box_ops import box_iou


def connected_components(adjacency: Tensor) -> Tensor:
    """Return a component id per row of a symmetric boolean adjacency matrix.

    Uses min-label propagation. ``N`` is the number of boxes in one image (typically small).

    Args:
        adjacency: Boolean matrix of shape ``[N, N]``. The diagonal is ignored; each node is
            always connected to itself.

    Returns:
        Integer tensor of shape ``[N]`` with values in ``[0, M)`` for ``M`` components.
    """
    n = adjacency.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.int64, device=adjacency.device)
    labels = torch.arange(n, device=adjacency.device, dtype=torch.int64)
    # Force self-edges so isolated nodes keep their own id.
    adj = adjacency.clone()
    adj.fill_diagonal_(True)
    for _ in range(max(n - 1, 1)):
        # For each node, take the minimum label among its neighbours (including itself).
        neighbour_labels = labels.unsqueeze(0).expand(n, n).masked_fill(~adj, n)
        new_labels = neighbour_labels.min(dim=1).values
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    # Densify to [0, M).
    unique = torch.unique(labels, sorted=True)
    remap = torch.empty(int(labels.max().item()) + 1, dtype=torch.int64, device=adjacency.device)
    remap[unique] = torch.arange(unique.numel(), device=adjacency.device, dtype=torch.int64)
    return remap[labels]


def merge_multilabel_annotations(
    boxes: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
    iou_threshold: float,
    masks: Tensor | None = None,
    area: Tensor | None = None,
    iscrowd: Tensor | None = None,
) -> dict[str, Tensor]:
    """Collapse same-image boxes with IoU >= ``iou_threshold`` and different labels.

    Groups are connected components of that adjacency. Each group becomes one object whose
    box encloses every member, whose mask is the logical OR of the members (when present),
    and whose ``label_sets`` row is the multi-hot union of the member labels. Same-class
    boxes are never merged with each other through this rule.

    Args:
        boxes: Absolute ``[N, 4]`` boxes in ``xyxy``.
        labels: Integer class ids of shape ``[N]``.
        num_classes: Width of the emitted ``label_sets`` multi-hot matrix.
        iou_threshold: Minimum IoU for two different-class boxes to form an edge.
        masks: Optional ``[N, H, W]`` boolean (or uint8) masks.
        area: Optional ``[N]`` areas.
        iscrowd: Optional ``[N]`` crowd flags.

    Returns:
        Dict with ``boxes``, ``labels``, ``label_sets``, and any of ``masks`` / ``area`` /
        ``iscrowd`` that were supplied. ``labels`` / ``iscrowd`` keep the first member of
        each component (lowest index). ``area`` keeps the maximum member area.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    n = boxes.shape[0]
    device = boxes.device
    if n == 0:
        empty: dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "label_sets": torch.zeros((0, num_classes), dtype=torch.float32, device=device),
        }
        if masks is not None:
            empty["masks"] = masks
        if area is not None:
            empty["area"] = area
        if iscrowd is not None:
            empty["iscrowd"] = iscrowd
        return empty

    if int(labels.max().item()) >= num_classes or int(labels.min().item()) < 0:
        raise ValueError(
            f"labels must lie in [0, {num_classes}), got min={int(labels.min().item())} "
            f"max={int(labels.max().item())}"
        )

    if n == 1:
        label_sets = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
        label_sets[0, labels[0]] = 1.0
        out: dict[str, Tensor] = {"boxes": boxes, "labels": labels, "label_sets": label_sets}
        if masks is not None:
            out["masks"] = masks
        if area is not None:
            out["area"] = area
        if iscrowd is not None:
            out["iscrowd"] = iscrowd
        return out

    iou, _ = box_iou(boxes, boxes)
    different = labels.unsqueeze(0) != labels.unsqueeze(1)
    adjacency = (iou >= iou_threshold) & different
    components = connected_components(adjacency)
    num_components = int(components.max().item()) + 1

    merged_boxes = torch.empty((num_components, 4), dtype=boxes.dtype, device=device)
    merged_labels = torch.empty((num_components,), dtype=labels.dtype, device=device)
    label_sets = torch.zeros((num_components, num_classes), dtype=torch.float32, device=device)
    merged_masks: Tensor | None = None
    if masks is not None:
        merged_masks = torch.zeros((num_components, *masks.shape[1:]), dtype=masks.dtype, device=device)
    merged_area: Tensor | None = None
    if area is not None:
        merged_area = torch.zeros((num_components,), dtype=area.dtype, device=device)
    merged_iscrowd: Tensor | None = None
    if iscrowd is not None:
        merged_iscrowd = torch.zeros((num_components,), dtype=iscrowd.dtype, device=device)

    # Process components in ascending id order so the output rows are stable.
    for component_id in range(num_components):
        members = (components == component_id).nonzero(as_tuple=False).flatten()
        member_boxes = boxes[members]
        merged_boxes[component_id, 0] = member_boxes[:, 0].min()
        merged_boxes[component_id, 1] = member_boxes[:, 1].min()
        merged_boxes[component_id, 2] = member_boxes[:, 2].max()
        merged_boxes[component_id, 3] = member_boxes[:, 3].max()
        # First member (lowest original index) supplies the primary label / iscrowd.
        first = int(members[0].item())
        merged_labels[component_id] = labels[first]
        label_sets[component_id, labels[members]] = 1.0
        if merged_masks is not None and masks is not None:
            # Logical OR across members; works for bool and {0,1} uint8.
            merged_masks[component_id] = masks[members].amax(dim=0)
        if merged_area is not None and area is not None:
            merged_area[component_id] = area[members].max()
        if merged_iscrowd is not None and iscrowd is not None:
            merged_iscrowd[component_id] = iscrowd[first]

    result: dict[str, Tensor] = {
        "boxes": merged_boxes,
        "labels": merged_labels,
        "label_sets": label_sets,
    }
    if merged_masks is not None:
        result["masks"] = merged_masks
    if merged_area is not None:
        result["area"] = merged_area
    if merged_iscrowd is not None:
        result["iscrowd"] = merged_iscrowd
    return result


def summarize_multilabel_merges(
    coco: Any,
    *,
    cat2label: dict[int, int] | None,
    num_classes: int,
    iou_threshold: float,
) -> dict[str, Any]:
    """Run the merge over every image's annotations without reading image files.

    Args:
        coco: A pycocotools ``COCO`` API object (``dataset.coco``).
        cat2label: Optional ``category_id`` → contiguous label mapping. ``None`` uses raw
            category ids as labels.
        num_classes: Width of ``label_sets``.
        iou_threshold: Merge threshold.

    Returns:
        Summary dict with ``num_images``, ``num_objects_before``, ``num_objects_after``,
        ``num_multilabel_objects``, and ``combination_counts`` mapping sorted label-id
        tuples to occurrence counts.
    """
    num_objects_before = 0
    num_objects_after = 0
    num_multilabel = 0
    combination_counts: Counter[tuple[int, ...]] = Counter()
    image_ids = list(coco.imgs.keys())
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        info = coco.imgs[image_id]
        width = float(info["width"])
        height = float(info["height"])
        box_rows: list[list[float]] = []
        label_rows: list[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(width, float(x) + float(w))
            y2 = min(height, float(y) + float(h))
            if x2 <= x1 or y2 <= y1:
                continue
            category_id = int(ann["category_id"])
            if cat2label is not None:
                if category_id not in cat2label:
                    continue
                label_rows.append(cat2label[category_id])
            else:
                label_rows.append(category_id)
            box_rows.append([x1, y1, x2, y2])
        if not box_rows:
            continue
        boxes = torch.tensor(box_rows, dtype=torch.float32)
        labels = torch.tensor(label_rows, dtype=torch.int64)
        num_objects_before += boxes.shape[0]
        merged = merge_multilabel_annotations(
            boxes,
            labels,
            num_classes=num_classes,
            iou_threshold=iou_threshold,
        )
        label_sets = merged["label_sets"]
        num_objects_after += label_sets.shape[0]
        for row in label_sets:
            positive = tuple(sorted(int(i) for i in row.nonzero(as_tuple=False).flatten().tolist()))
            if len(positive) > 1:
                num_multilabel += 1
                combination_counts[positive] += 1
    return {
        "num_images": len(image_ids),
        "num_objects_before": num_objects_before,
        "num_objects_after": num_objects_after,
        "num_multilabel_objects": num_multilabel,
        "combination_counts": dict(combination_counts),
    }
