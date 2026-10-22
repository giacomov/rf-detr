# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``Backbone`` subclass wiring the DINOv3-ConvNeXt encoder to a selectable neck.

``Backbone.__init__`` unconditionally builds a ``MultiScaleProjector`` (``rfdetr.models.backbone.projector``),
which is the wrong shape for a true CNN pyramid (see ``dinov3_convnext_neck``'s module docstring). This class
overrides ``__init__`` to swap in one of the necks defined there instead; everything else -- ``export()``,
``_level_mask``, ``forward``, ``forward_export``, ``get_named_param_lr_pairs`` -- is inherited unchanged from
``Backbone``, since all of it is already generic with respect to the encoder/projector pair.

Registered encoder names are ``dinov3_convnext_{size}_{neck}`` for ``size`` in ``tiny``/``small``/``base`` and
``neck`` in ``lateral``/``fpn`` -- one string encoding both choices, the same way DINOv2's own encoder names
encode ``[registers]_[windowed]_{size}``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.dinov3_convnext import DINOv3ConvNext
from rfdetr.models.backbone.dinov3_convnext_neck import FPNNeck, LateralNeck

#: Sizes this backbone supports. Kept in one place so `_parse_name`'s validation and the name-registration loop
#: at the bottom of this module can't drift apart.
_SIZES = ("tiny", "small", "base")
#: Neck kinds this backbone supports, mapped to their implementing class (see
#: rfdetr.models.backbone.dinov3_convnext_neck).
_NECKS: dict[str, type[nn.Module]] = {"lateral": LateralNeck, "fpn": FPNNeck}


def _parse_name(name: str) -> tuple[str, str]:
    """Split ``"dinov3_convnext_{size}_{neck}"`` into ``(size, neck)``, validating both against what's supported."""
    parts = name.split("_")
    if parts[:2] != ["dinov3", "convnext"] or len(parts) != 4:
        raise ValueError(f"Expected an encoder name of the form 'dinov3_convnext_{{size}}_{{neck}}', got {name!r}.")
    size, neck = parts[2], parts[3]
    if size not in _SIZES:
        raise ValueError(f"Unknown DINOv3-ConvNeXt size {size!r} in encoder name {name!r}; expected one of {_SIZES}.")
    if neck not in _NECKS:
        raise ValueError(
            f"Unknown DINOv3-ConvNeXt neck {neck!r} in encoder name {name!r}; expected one of {tuple(_NECKS)}."
        )
    return size, neck


class DINOv3ConvNextEncoderBackbone(Backbone):
    """DINOv3-ConvNeXt encoder paired with a selectable multi-scale neck (see module docstring)."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str | None = None,
        *,
        drop_path: float = 0.0,
        out_channels: int = 256,
        out_feature_indexes: list[int] | None = None,
        projector_scale: list[str] | None = None,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        rms_norm: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        dual_projector: bool = False,
        **kwargs: Any,
    ) -> None:
        BackboneBase.__init__(self)  # Backbone.__init__ unconditionally builds MultiScaleProjector; skip it.
        size, neck_kind = _parse_name(name)

        if out_feature_indexes is None:
            raise ValueError("DINOv3ConvNextEncoderBackbone requires out_feature_indexes (ConvNeXt stage numbers).")
        self.encoder = DINOv3ConvNext(
            size=size,
            out_feature_indexes=out_feature_indexes,
            load_pretrained_weights=load_dinov2_weights,
            drop_path_rate=drop_path,
            gradient_checkpointing=gradient_checkpointing,
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.projector_scale = projector_scale
        if not self.projector_scale or sorted(self.projector_scale) != self.projector_scale:
            raise ValueError(
                f"projector_scale must be a non-empty, ascending-order subset of ['P3','P4','P5'], "
                f"got {projector_scale!r}."
            )
        encoder_channels = self.encoder._out_feature_channels
        if len(encoder_channels) != len(self.projector_scale):
            raise ValueError(
                "DINOv3-ConvNeXt necks need one ConvNeXt stage per projector_scale level (stage2/3/4 == "
                f"P3/P4/P5 natively); got {len(encoder_channels)} stages for projector_scale={self.projector_scale}."
            )

        neck_cls = _NECKS[neck_kind]
        # Backbone.projector/.cross_attn_projector are only ever assigned a MultiScaleProjector in Backbone's own
        # __init__ (which this subclass never calls), so mypy infers that narrower type for the base class; a
        # neck here is a different nn.Module shape that satisfies the same list[Tensor] -> list[Tensor] contract.
        self.projector = neck_cls(  # type: ignore[assignment]
            encoder_channels, out_channels, layer_norm=layer_norm, rms_norm=rms_norm
        )
        # The decoder's cross-attention path assumes the cross-attn projector's output shares `spatial_shapes_hw`
        # with the primary projector's output (see rfdetr.models.transformer); true here only because every neck
        # in `_NECKS` preserves each input level's spatial shape (channel projection only).
        self.cross_attn_projector = (
            neck_cls(encoder_channels, out_channels, layer_norm=layer_norm, rms_norm=rms_norm)  # type: ignore[assignment]
            if dual_projector
            else None
        )
        self._export = False

    # get_named_param_lr_pairs is inherited unchanged from Backbone. Its per-ViT-layer decay formula keys on a
    # ".layer." substring that ConvNeXt's own parameter names never contain, so it degrades gracefully to a flat
    # (non-decayed) backbone LR rather than erroring -- an intentional v1 limitation, not a bug (see the plan's
    # "explicitly out of scope for v1" notes).


def _register_names() -> None:
    """Populate ``_BACKBONE_REGISTRY`` directly for every supported ``size``/``neck`` combination.

    Deliberately bypasses the public, guarded ``register_backbone()``: that function rejects any name already in
    ``EncoderName``'s ``Literal``, which these names must be (for ``ModelConfig.encoder`` to validate them) --
    calling it here would make it self-reject. Its guard exists to stop *external* packages from colliding with
    built-in names; core's own additions to its own registry don't need that guard.
    """
    from rfdetr.models.backbone import _BACKBONE_REGISTRY

    for size in _SIZES:
        for neck in _NECKS:
            _BACKBONE_REGISTRY[f"dinov3_convnext_{size}_{neck}"] = DINOv3ConvNextEncoderBackbone


_register_names()
