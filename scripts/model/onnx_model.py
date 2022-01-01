#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""Script for creating onnx models."""

import onnx
import tf2onnx
from onnxsim import simplify
from tf_model import load_model, convert_model_to_nchw


def create_onnx_model(config_path: str, weight_path: str,
                      opset: int = 12) -> onnx.ModelProto:
    """Create ONNX model."""
    model = load_model(
        config_path=config_path,
        weight_path=weight_path
    )
    model = convert_model_to_nchw(model)
    model_proto, _ = tf2onnx.convert.from_keras(
        model=model,
        opset=opset
    )
    model_proto, check = simplify(model_proto, input_shapes={
        "cur_frame": [1] + list(model.input[0].shape)[1:],
        "last_frame": [1] + list(model.input[1].shape)[1:],
        "pre_gen": [1] + list(model.input[2].shape)[1:],
    })
    assert check
    return model_proto
