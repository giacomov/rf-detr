# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``Backbone`` subclass that swaps DINOv2 for the vendored C-RADIOv3-B encoder.

Overrides ``_build_encoder`` so the inherited ``MultiScaleProjector``, padding-mask, ``forward``, and
``forward_export`` paths stay those of :class:`~rfdetr.models.backbone.backbone.Backbone`. Also overrides
``get_named_param_lr_pairs`` because C-RADIO uses timm-style ``.blocks.N`` names (the base class keys on
HuggingFace DINOv2's ``.layer.``).

Registered encoder name: ``c_radio_v3_base``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.c_radio_v3 import CRadioV3


def _c_radio_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Per-block LR decay for timm-style ``.blocks.`` names.

    NOTE: near-duplicate of ``get_vit_lr_decay_rate`` in ``training/param_groups.py``. Kept local so importing
    this backbone module does not pull ``rfdetr.training`` (and its optional Lightning dependency) at package
    import time. If updating the formula, update the sibling too.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name or ".prefix_tokens" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _c_radio_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Zero weight decay on norms, biases, LayerScale, and position / prefix tokens."""
    if ("gamma" in name) or ("pos_embed" in name) or ("prefix_tokens" in name) or ("bias" in name) or ("norm" in name):
        return 0.0
    return weight_decay_rate


class CRadioV3EncoderBackbone(Backbone):
    """RF-DETR backbone that builds a :class:`~rfdetr.models.backbone.c_radio_v3.CRadioV3` encoder."""

    def _build_encoder(
        self,
        name: str,
        *,
        out_feature_indexes: list[int] | None,
        target_shape: tuple[int, int],
        gradient_checkpointing: bool,
        load_pretrained_weights: bool,
        drop_path: float,
        **kwargs: Any,
    ) -> nn.Module:
        """Build the C-RADIOv3-B encoder.

        Args:
            name: Must be ``"c_radio_v3_base"``.
            out_feature_indexes: 1-based block indices to tap (e.g. ``[3, 6, 9, 12]``).
            target_shape: Native ``(height, width)`` used to bake CPE position embeddings.
            gradient_checkpointing: Checkpoint each ViT block during training.
            load_pretrained_weights: Download and load the pinned Hub checkpoint.
            drop_path: Stochastic-depth rate (linear schedule across the 12 blocks).
            **kwargs: Ignored; accepted so the hook signature can grow.

        Returns:
            The :class:`CRadioV3` module.
        """
        if name != "c_radio_v3_base":
            raise ValueError(f"CRadioV3EncoderBackbone only supports 'c_radio_v3_base', got {name!r}.")
        if out_feature_indexes is None:
            raise ValueError("CRadioV3EncoderBackbone requires out_feature_indexes.")
        return CRadioV3(
            out_feature_indexes=out_feature_indexes,
            target_shape=target_shape,
            load_pretrained_weights=load_pretrained_weights,
            drop_path_rate=drop_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    def get_named_param_lr_pairs(self, args: Any, prefix: str = "backbone.0") -> dict[str, dict[str, Any]]:
        """Per-block LR decay using timm-style ``.blocks.`` names."""
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs: dict[str, dict[str, Any]] = {}
        for name, param in self.named_parameters():
            full_name = prefix + "." + name
            if backbone_key not in full_name or not param.requires_grad:
                continue
            lr = (
                args.lr_encoder
                * _c_radio_lr_decay_rate(full_name, lr_decay_rate=args.lr_vit_layer_decay, num_layers=num_layers)
                * args.lr_component_decay**2
            )
            wd = args.weight_decay * _c_radio_weight_decay_rate(full_name)
            named_param_lr_pairs[full_name] = {"params": param, "lr": lr, "weight_decay": wd}
        return named_param_lr_pairs


def _register_names() -> None:
    """Register ``c_radio_v3_base`` on the in-core backbone registry (bypasses the guarded public helper)."""
    from rfdetr.models.backbone import _BACKBONE_REGISTRY

    _BACKBONE_REGISTRY["c_radio_v3_base"] = CRadioV3EncoderBackbone


_register_names()
