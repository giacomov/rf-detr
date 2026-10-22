# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Offline, network-free tests for the DINOv3-ConvNeXt encoder wrapper.

All tests use ``load_pretrained_weights=False`` (random init), so they run with no network access and no Hugging
Face Hub gating concerns.
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.backbone.dinov3_convnext import (
    SIZE_TO_HIDDEN_SIZES,
    DINOv3ConvNext,
    _load_pretrained_convnext_backbone,
)


class TestDINOv3ConvNext:
    """Shape/channel contract of the encoder wrapper, random init."""

    @pytest.mark.parametrize(
        ("out_feature_indexes", "expected_channels", "expected_strides"),
        [
            pytest.param([2, 3, 4], [192, 384, 768], [8, 16, 32], id="p3-p4-p5"),
            pytest.param([4], [768], [32], id="single-stage"),
            pytest.param([3, 4], [384, 768], [16, 32], id="two-stages"),
        ],
    )
    def test_forward_shapes_match_requested_stages(
        self, out_feature_indexes: list[int], expected_channels: list[int], expected_strides: list[int]
    ) -> None:
        encoder = DINOv3ConvNext(
            size="tiny", out_feature_indexes=out_feature_indexes, load_pretrained_weights=False
        )

        assert encoder._out_feature_channels == expected_channels

        images = torch.randn(2, 3, 256, 256)
        feats = encoder(images)

        assert len(feats) == len(out_feature_indexes)
        for feat, channels, stride in zip(feats, expected_channels, expected_strides):
            assert tuple(feat.shape) == (2, channels, 256 // stride, 256 // stride)

    def test_unknown_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown DINOv3-ConvNeXt size"):
            DINOv3ConvNext(size="huge", out_feature_indexes=[4], load_pretrained_weights=False)

    def test_drop_path_rate_is_forwarded_to_the_hf_config(self) -> None:
        encoder = DINOv3ConvNext(
            size="tiny", out_feature_indexes=[4], load_pretrained_weights=False, drop_path_rate=0.2
        )

        assert encoder.encoder.config.drop_path_rate == 0.2

    def test_gradient_checkpointing_warns_when_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`DINOv3ConvNextBackbone.supports_gradient_checkpointing` is False in this transformers version; this
        must degrade to a warning rather than propagate HF's own `ValueError`.

        rfdetr's logger sets `propagate = False` (see `rfdetr.utilities.logger.get_logger`), so `caplog`'s
        root-logger handler never sees anything it emits; spying on the module's `logger.warning` directly
        avoids depending on that propagation at all.
        """
        import rfdetr.models.backbone.dinov3_convnext as module

        calls: list[str] = []
        monkeypatch.setattr(module.logger, "warning", lambda msg, *a, **kw: calls.append(msg))

        module.DINOv3ConvNext(
            size="tiny",
            out_feature_indexes=[4],
            load_pretrained_weights=False,
            gradient_checkpointing=True,
        )

        assert any("gradient checkpointing" in msg for msg in calls)

    def test_gradient_checkpointing_enables_when_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If a future transformers version adds support, the wrapper must call through rather than warn."""
        import rfdetr.models.backbone.dinov3_convnext as module

        real_init = module.DINOv3ConvNextBackbone.__init__
        enabled: dict[str, bool] = {"called": False}

        def _patched_init(self, config):  # type: ignore[no-untyped-def]
            real_init(self, config)
            self.supports_gradient_checkpointing = True
            self.gradient_checkpointing_enable = lambda: enabled.update(called=True)

        monkeypatch.setattr(module.DINOv3ConvNextBackbone, "__init__", _patched_init)

        module.DINOv3ConvNext(
            size="tiny", out_feature_indexes=[4], load_pretrained_weights=False, gradient_checkpointing=True
        )

        assert enabled["called"]


class TestLoadPretrainedConvNextBackbone:
    """`_load_pretrained_convnext_backbone`'s error-wrapping, exercised without any real network call."""

    def test_wraps_download_failures_with_a_gating_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rfdetr.models.backbone.dinov3_convnext as module

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise OSError("401 Client Error")

        monkeypatch.setattr(module.AutoBackbone, "from_pretrained", _raise)

        with pytest.raises(RuntimeError, match="gated") as excinfo:
            _load_pretrained_convnext_backbone(
                "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                module.DINOv3ConvNextConfig(hidden_sizes=SIZE_TO_HIDDEN_SIZES["tiny"], out_features=["stage4"]),
            )

        assert "huggingface-cli login" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)
