# -*- encoding: utf-8 -*-

"""Script for creating onnx models."""

import numpy as np
import onnx
import tf2onnx
from onnxsim import simplify
import onnxconverter_common
from onnxruntime import quantization
from tf_model_utils import load_model, convert_model_to_nchw


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
    }, check_n=3)
    assert check
    return model_proto


def quantize_model_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32 model to FP16."""
    return onnxconverter_common.convert_float_to_float16(
        model,
        keep_io_types=True
    )


class DataReader(quantization.CalibrationDataReader):
    """Data reader for ONNX model calibration."""

    def __init__(self, hr_data: np.array, lr_data: np.array, **kwargs) -> None:
        """Construct object."""
        super().__init__(**kwargs)
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.iter = iter(range(hr_data.shape[0] - 1))

    def get_next(self) -> dict:
        """Iterate over data."""
        i = next(self.iter, None)
        if i is None:
            return None
        return {
            "cur_frame": self.lr_data[[i + 1]],
            "last_frame": self.lr_data[[i]],
            "pre_gen": self.hr_data[[i]]
        }


def create_calibration_table(
    model: onnx.ModelProto,
    hr_data: np.array,
    lr_data: np.array
) -> bytes:
    """Calibrate model."""
    calibrator = quantization.create_calibrator(
        model,
        calibrate_method=quantization.CalibrationMethod.Entropy
    )
    # Collecting data for one sample at a time due to possible OOM
    for i in range(hr_data.shape[0] - 1):
        calibrator.collect_data(DataReader(
            hr_data[i:i+2], lr_data[i:i+2]))
    quantization.write_calibration_table(calibrator.compute_range())


def quantize_model_int8(
        input_model_path: str,
        output_model_path: str,
        hr_data: np.array,
        lr_data: np.array) -> None:
    """Quantize model to int8."""
    quantization.quantize_static(
        input_model_path,
        output_model_path,
        DataReader(hr_data, lr_data),
        quant_format=quantization.QuantFormat.QOperator,
        weight_type=quantization.QuantType.QInt8
    )
