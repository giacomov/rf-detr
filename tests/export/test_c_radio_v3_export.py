# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Export smoke test for the C-RADIOv3-B backbone (``set_export_shape`` + ONNX trace)."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.jit import TracerWarning

from rfdetr.config import RFDETRCRadioV3BaseConfig
from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export.prepare import prepare_export_graph
from rfdetr.models import MODEL_DEFAULTS, build_model_from_config
from rfdetr.models.backbone.c_radio_v3 import CRadioV3
from tests._markers import onnx_and_onnxruntime_only

_NO_PRETRAIN = replace(MODEL_DEFAULTS, force_no_pretrain=True)


@onnx_and_onnxruntime_only
@pytest.mark.parametrize(
    "export_shape",
    [
        pytest.param((128, 128), id="native-ish"),
        pytest.param((160, 192), id="non-square"),
    ],
)
def test_onnx_export_traces_and_runs(export_shape: tuple[int, int], tmp_path: Path) -> None:
    import onnxruntime

    mc = RFDETRCRadioV3BaseConfig(resolution=128, num_classes=3)
    model = build_model_from_config(mc, defaults=_NO_PRETRAIN)
    model.eval()

    # Confirm the encoder opted into the set_export_shape protocol used by prepare_export_graph.
    encoder = model.backbone[0].encoder
    assert isinstance(encoder, CRadioV3)
    assert callable(getattr(type(encoder), "set_export_shape", None))

    graph = prepare_export_graph(model, mc, shape=export_shape, device="cpu")
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
