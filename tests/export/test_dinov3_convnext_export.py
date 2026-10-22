# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Export smoke test for the DINOv3-ConvNeXt backbone.

Confirms the generic export path -- ``prepare_export_graph``, ``Backbone.export()``/``forward_export``, and a
real ``torch.onnx.export`` trace -- needs no ConvNeXt-specific code, as expected (this backbone has no
resolution-dependent position embedding to freeze, so neither the ``isinstance(..., DinoV2)`` branch nor the
``set_export_shape`` protocol in ``rfdetr.export.prepare`` ever fires for it; see that module and
``dinov3_convnext.py``'s module docstring).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.jit import TracerWarning

from rfdetr.config import RFDETRDINOv3ConvNextTinyFPNConfig, RFDETRDINOv3ConvNextTinyLateralConfig
from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export.prepare import prepare_export_graph
from rfdetr.models import MODEL_DEFAULTS, build_model_from_config
from tests._markers import onnx_and_onnxruntime_only

#: Random init: no trained RF-DETR checkpoint exists for this backbone yet, and the upstream DINOv3-ConvNeXt
#: weights are gated -- this test only needs the architecture to trace, not real weights.
_NO_PRETRAIN = replace(MODEL_DEFAULTS, force_no_pretrain=True)


@onnx_and_onnxruntime_only
@pytest.mark.parametrize(
    "config_class",
    [
        pytest.param(RFDETRDINOv3ConvNextTinyLateralConfig, id="lateral"),
        pytest.param(RFDETRDINOv3ConvNextTinyFPNConfig, id="fpn"),
    ],
)
def test_onnx_export_traces_and_runs(config_class: type, tmp_path: Path) -> None:
    import onnxruntime

    mc = config_class(resolution=128, num_classes=3)
    model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
    model.eval()

    graph = prepare_export_graph(model, mc, shape=(128, 128), device="cpu")
    # prepare_export_graph hands back the model in its training forward; every real exporter (Exporter.__call__)
    # switches it into export mode (LWDETR.forward -> forward_export) exactly once before tracing.
    _switch_to_export_mode(graph.model)

    export_kwargs: dict = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    output_file = str(tmp_path / "model.onnx")
    input_tensors = graph.input_tensors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        torch.onnx.export(
            graph.model,
            (input_tensors,) if isinstance(input_tensors, torch.Tensor) else tuple(input_tensors),
            output_file,
            input_names=list(graph.input_names),
            output_names=list(graph.output_names),
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            opset_version=17,
            dynamic_axes=graph.dynamic_axes,
            **export_kwargs,
        )

    session = onnxruntime.InferenceSession(output_file, providers=["CPUExecutionProvider"])
    feed = {
        name: tensor.numpy()
        for name, tensor in zip(
            graph.input_names, (input_tensors,) if isinstance(input_tensors, torch.Tensor) else input_tensors
        )
    }
    onnx_outputs = session.run(list(graph.output_names), feed)

    assert len(onnx_outputs) == len(graph.output_names)
    for output in onnx_outputs:
        assert (output == output).all(), "ONNX output contains NaN"
