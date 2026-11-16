# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Multi-label matching and classification loss for co-located ground-truth labels."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.models import _assignment
from rfdetr.models.criterion import SetCriterion, _MatchedTargets
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.utilities import box_ops
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _assert_label_sets_aligned(targets: list[dict[str, Tensor]]) -> None:
    """Raise if any target's ``label_sets`` row count disagrees with ``labels``."""
    for index, target in enumerate(targets):
        label_sets = target["label_sets"]
        labels = target["labels"]
        if label_sets.shape[0] != labels.shape[0]:
            raise ValueError(
                f"targets[{index}] label_sets has {label_sets.shape[0]} rows but labels has "
                f"{labels.shape[0]}; an augmentation path likely dropped rows from one field only."
            )


def _pad_label_sets(label_sets: Tensor, num_logit_classes: int) -> Tensor:
    """Zero-pad ``label_sets`` on the class axis to match the model's logit width.

    Args:
        label_sets: Multi-hot tensor of shape ``[N, C_data]``.
        num_logit_classes: ``pred_logits.shape[-1]``.

    Returns:
        Tensor of shape ``[N, num_logit_classes]``.

    Raises:
        ValueError: If ``label_sets`` is wider than the logit table.
    """
    width = label_sets.shape[-1]
    if width == num_logit_classes:
        return label_sets
    if width > num_logit_classes:
        raise ValueError(
            f"label_sets width {width} exceeds pred_logits class count {num_logit_classes}."
        )
    padded = label_sets.new_zeros(label_sets.shape[0], num_logit_classes)
    padded[:, :width] = label_sets
    return padded


class MultiLabelHungarianMatcher(HungarianMatcher):
    """Hungarian matcher whose classification cost averages over each target's label set.

    When targets carry no ``label_sets`` key (validation, or stock training), behaviour is
    identical to :class:`HungarianMatcher`. Keypoints and fixed-size target padding are not
    supported in the multi-label path.
    """

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        group_detr: int = 1,
        target_side_safety: Any | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Match predictions to targets, averaging the class cost over multi-hot label sets.

        Args:
            outputs: Model outputs with ``pred_logits`` and ``pred_boxes``.
            targets: Per-image targets. Multi-label mode requires ``label_sets``.
            group_detr: Number of query groups.
            target_side_safety: Forwarded to the stock path when ``label_sets`` is absent.

        Returns:
            Per-image ``(src_indices, tgt_indices)`` pairs.
        """
        if not targets or "label_sets" not in targets[0]:
            return super().forward(
                outputs, targets, group_detr=group_detr, target_side_safety=target_side_safety
            )

        _assert_label_sets_aligned(targets)
        if "valid" in targets[0]:
            raise NotImplementedError(
                "Multi-label matching does not support pad_targets_to / fixed-size target padding."
            )
        if "pred_keypoints" in outputs and "keypoints" in targets[0]:
            raise NotImplementedError("Multi-label matching does not support keypoints.")

        bs, num_queries = outputs["pred_logits"].shape[:2]
        num_logit_classes = outputs["pred_logits"].shape[-1]
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        label_sets = torch.cat(
            [_pad_label_sets(target["label_sets"].to(flat_pred_logits.dtype), num_logit_classes) for target in targets]
        )
        tgt_bbox = torch.cat([target["boxes"] for target in targets])

        # Average focal cost over each target's positive labels.
        # cost_class[q, t] = sum_c focal(logits[q, c]) * (label_sets[t, c] / sum_c label_sets[t, c])
        per_class_cost = self._focal_classification_cost(flat_pred_logits)  # [B*Q, C]
        weights = label_sets / label_sets.sum(dim=1, keepdim=True).clamp(min=1.0)
        cost_class = per_class_cost @ weights.transpose(0, 1)

        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou
        l1_cost = self._resolve_l1_cost() if out_bbox.dtype == tgt_bbox.dtype else None
        if l1_cost is None:
            from rfdetr.utilities.box_ops import pairwise_box_l1_cost

            cost_bbox = pairwise_box_l1_cost(out_bbox, tgt_bbox)
        else:
            cost_bbox = l1_cost(out_bbox, tgt_bbox)

        cost_matrix: Tensor = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        masks_present = "masks" in targets[0]
        if masks_present:
            cost_mask_ce, cost_mask_dice = self._compute_mask_costs(outputs, targets)
            cost_matrix = cost_matrix + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice

        cost_matrix = cost_matrix.view(bs, num_queries, -1).float()
        all_finite = torch.isfinite(cost_matrix).all().item()
        if not all_finite:
            cost_matrix = cost_matrix.cpu()
            if not self._warned_non_finite_costs:
                logger.warning(
                    "Non-finite values detected in multi-label matcher cost matrix; "
                    "replacing with finite sentinel. Check for numerical instability."
                )
                self._warned_non_finite_costs = True
            cost_matrix = self._sanitize_cost_matrix(cost_matrix)

        sizes = [len(target["boxes"]) for target in targets]
        target_offsets = [0]
        for size in sizes:
            target_offsets.append(target_offsets[-1] + size)
        diagonal_cost_matrix: Tensor = torch.cat(
            [cost_matrix[i, :, target_offsets[i] : target_offsets[i + 1]] for i in range(bs)], dim=-1
        )
        if diagonal_cost_matrix.is_cuda:
            return _assignment.assign_many_bucketed([diagonal_cost_matrix], sizes, group_detr)[0]
        return self._assign_compact_cost_matrix(diagonal_cost_matrix.cpu(), sizes, group_detr)


class MultiLabelSetCriterion(SetCriterion):
    """SetCriterion with multi-column IA-BCE positives for ``label_sets`` targets.

    When targets carry no ``label_sets``, behaviour matches :class:`SetCriterion`. The
    compiled batched detection-loss fast path is skipped automatically because
    ``type(self) is not SetCriterion``.
    """

    def loss_labels(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
        log: bool = True,
        matched_targets: _MatchedTargets | None = None,
    ) -> dict[str, Tensor]:
        """IA-BCE classification loss with positives in every column of each label set."""
        if not targets or "label_sets" not in targets[0]:
            return super().loss_labels(
                outputs, targets, indices, num_boxes, log=log, matched_targets=matched_targets
            )

        _assert_label_sets_aligned(targets)
        if not self.ia_bce_loss:
            raise NotImplementedError(
                "Multi-label classification loss requires ia_bce_loss=True "
                "(the TrainConfig default)."
            )
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        valid_mask = self._matched_valid_mask(targets, indices)
        if valid_mask is not None:
            raise NotImplementedError(
                "Multi-label classification loss does not support pad_targets_to / fixed-size "
                "target padding."
            )

        if matched_targets is None:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        else:
            idx = matched_targets.source_indices
            target_boxes = matched_targets.boxes
        matched_label_sets = torch.cat(
            [t["label_sets"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )
        matched_label_sets = _pad_label_sets(matched_label_sets, src_logits.shape[-1])

        alpha = self.focal_alpha
        gamma = 2
        src_boxes = outputs["pred_boxes"][idx]
        iou_targets, _ = box_ops.elementwise_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )
        pos_ious = iou_targets.clone().detach()
        prob = src_logits.sigmoid()
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = prob**gamma

        # Expand each matched query across every positive class column.
        match_indices, class_indices = matched_label_sets.nonzero(as_tuple=True)
        if match_indices.numel() > 0:
            batch_idx = idx[0][match_indices]
            query_idx = idx[1][match_indices]
            iou_for_pos = pos_ious[match_indices]
            t = prob[batch_idx, query_idx, class_indices].pow(alpha) * iou_for_pos.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()
            pos_weights[batch_idx, query_idx, class_indices] = t.to(pos_weights.dtype)
            neg_weights[batch_idx, query_idx, class_indices] = (1 - t).to(neg_weights.dtype)

        loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
        loss_ce = loss_ce.sum() / num_boxes
        losses: dict[str, Tensor] = {"loss_ce": loss_ce}

        if log:
            # Top-1 is correct when the argmax class is anywhere in the matched label set.
            if matched_label_sets.shape[0] == 0:
                losses["class_error"] = src_logits.new_tensor(0.0)
            else:
                pred_classes = src_logits[idx].argmax(dim=-1)
                correct = matched_label_sets[
                    torch.arange(matched_label_sets.shape[0], device=matched_label_sets.device),
                    pred_classes,
                ]
                losses["class_error"] = 100 - 100 * correct.float().mean()
        return losses
