# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""DINOv3-ConvNeXt encoder wrapper.

Wraps ``transformers``' native DINOv3-ConvNeXt backbone (no vendoring needed, unlike DINOv2's windowed-attention
variant, which RF-DETR vendors-and-modifies specifically for an efficiency trick that doesn't apply to a CNN) so it
satisfies the ``Backbone._build_encoder`` contract: map ``(B, 3, H, W)`` to a list of per-level
``(B, C, H', W')`` feature maps and expose their channel counts as ``_out_feature_channels``.

Unlike DINOv2, this backbone needs no position-embedding interpolation (no ``export()``/``set_export_shape``
override) and already returns NCHW feature maps with no token-grid reshape required. ConvNeXt's stage2/stage3/
stage4 outputs land at strides 8/16/32, which already match RF-DETR's P3/P4/P5 convention with zero resampling
needed -- see :mod:`rfdetr.models.backbone.dinov3_convnext_neck` for the necks that consume them.
"""

from __future__ import annotations

from torch import Tensor, nn
from transformers import AutoBackbone, DINOv3ConvNextBackbone, DINOv3ConvNextConfig

from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Per-size ``hidden_sizes``/``depths`` presets, matching the standard ConvNeXt family schedule.
#:
#: NOTE: confirmed against this repo's pinned ``transformers`` version for the Tiny/default preset only
#: (``DINOv3ConvNextConfig()`` with no args returns exactly these Tiny values). Small/Base are the canonical
#: ConvNeXt-paper numbers and very likely what ``DINOv3ConvNextConfig``'s own per-size presets use, but could not
#: be independently verified for this project: the Hub repos are gated even for reading ``config.json`` (a
#: logged-out ``AutoConfig.from_pretrained`` returns a 401 for every size, confirmed this session), so there is
#: no way to check them without access. Confirm against the real ``config.json`` once access is granted, before
#: relying on these for a real pretrained-weight load -- if they differ, only this one dict needs updating.
SIZE_TO_HIDDEN_SIZES: dict[str, list[int]] = {
    "tiny": [96, 192, 384, 768],
    "small": [96, 192, 384, 768],
    "base": [128, 256, 512, 1024],
}
SIZE_TO_DEPTHS: dict[str, list[int]] = {
    "tiny": [3, 3, 9, 3],
    "small": [3, 3, 27, 3],
    "base": [3, 3, 27, 3],
}

#: Hugging Face Hub repo id for each size's DINOv3-ConvNeXt pretrained checkpoint. All three are confirmed gated
#: (a logged-out ``AutoConfig.from_pretrained`` against each repo returns a 401 "You are trying to access a gated
#: repo" error, even for the config alone, not just the weights): downloading requires requesting/accepting access
#: on the model page, then authenticating with ``huggingface-cli login`` or the ``HF_TOKEN`` environment variable.
SIZE_TO_HUB_ID: dict[str, str] = {
    "tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
}


def _load_pretrained_convnext_backbone(hub_id: str, config: DINOv3ConvNextConfig) -> DINOv3ConvNextBackbone:
    """Download *hub_id*'s pretrained weights into a `DINOv3ConvNextBackbone` shaped by *config*.

    Args:
        hub_id: Hugging Face Hub repo id, e.g. ``"facebook/dinov3-convnext-tiny-pretrain-lvd1689m"``.
        config: Backbone config controlling which stages are returned (``out_features``) and the stochastic-depth
            rate (``drop_path_rate``). Its ``hidden_sizes``/``depths`` must match *hub_id*'s actual architecture or
            the state dict will fail to load -- see the module-level note on :data:`SIZE_TO_HIDDEN_SIZES`.

    Returns:
        The pretrained backbone.

    Raises:
        RuntimeError: If the download fails, most likely because the (gated) Hub repo hasn't been granted to the
            authenticated user yet, or there is no network access.
    """
    try:
        return AutoBackbone.from_pretrained(hub_id, config=config)  # type: ignore[no-any-return,no-untyped-call]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download pretrained weights for {hub_id!r}. DINOv3-ConvNeXt checkpoints on the "
            f"Hugging Face Hub are gated: request access at https://huggingface.co/{hub_id}, authenticate with "
            "`huggingface-cli login` or the HF_TOKEN environment variable, and retry."
        ) from exc


class DINOv3ConvNext(nn.Module):
    """Thin wrapper around `transformers`' DINOv3-ConvNeXt backbone.

    Returns one feature map per entry of ``out_feature_indexes``, already at its native resolution and channel
    count -- no resampling or token-grid reshape needed, unlike DINOv2's output.
    """

    def __init__(
        self,
        size: str,
        out_feature_indexes: list[int],
        load_pretrained_weights: bool,
        drop_path_rate: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if size not in SIZE_TO_HIDDEN_SIZES:
            raise ValueError(
                f"Unknown DINOv3-ConvNeXt size {size!r}; expected one of {sorted(SIZE_TO_HIDDEN_SIZES)}."
            )
        config = DINOv3ConvNextConfig(
            hidden_sizes=SIZE_TO_HIDDEN_SIZES[size],
            depths=SIZE_TO_DEPTHS[size],
            drop_path_rate=drop_path_rate,
            # The constructor's typed signature only declares the private `_out_features`; the public
            # `out_features` name is accepted (and normalized) via the `BackboneConfigMixin` kwarg-passthrough
            # every HF backbone config uses -- confirmed empirically (`DINOv3ConvNextConfig(out_features=...)`
            # round-trips into `.out_features`/`.out_indices` correctly in this repo's pinned transformers).
            out_features=[f"stage{i}" for i in out_feature_indexes],  # type: ignore[call-arg]
        )
        self.encoder: DINOv3ConvNextBackbone = (
            _load_pretrained_convnext_backbone(SIZE_TO_HUB_ID[size], config)
            if load_pretrained_weights
            else DINOv3ConvNextBackbone(config)
        )
        #: Per-level channel counts, consumed by ``Backbone``/the registered subclass to size the neck.
        self._out_feature_channels: list[int] = list(self.encoder.channels)

        if gradient_checkpointing:
            # `hasattr(encoder, "gradient_checkpointing_enable")` is not a sufficient check: HF's
            # `PreTrainedModel` defines that method generically on every model, but it raises `ValueError`
            # itself when the concrete model's `supports_gradient_checkpointing` is False -- confirmed true
            # for `DINOv3ConvNextBackbone` in this repo's pinned transformers version. Check that flag first
            # so this degrades to a warning instead of propagating that error.
            if getattr(self.encoder, "supports_gradient_checkpointing", False):
                self.encoder.gradient_checkpointing_enable()
            else:
                logger.warning(
                    "DINOv3-ConvNeXt does not support gradient checkpointing in this transformers version; "
                    "continuing without it."
                )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return one ``(B, C, H', W')`` feature map per requested stage, finest resolution first."""
        return list(self.encoder(x).feature_maps)
