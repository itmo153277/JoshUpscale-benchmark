// Copyright 2021 Ivanov Viktor

#include "benchmark/backend.h"

#include <memory>
#include <stdexcept>

#include "benchmark/backends/onnxruntime.h"
#include "benchmark/backends/tensorflow.h"
#include "benchmark/backends/tensorrt.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

std::unique_ptr<Backend> createBackend(const config::BackendConfig &config,
    const Tensor<float> &examples, const Tensor<float> &exampleOut,
    const path_type *profilePath, const path_type *cachePath) {
	TensorShape inputShape = examples.shape;
	TensorShape outputShape = exampleOut.shape;
	inputShape[0] = 1;
	outputShape[0] = 1;
	switch (config.backendType) {
	case config::BackendType::TENSORFLOW:
		return std::make_unique<TensorflowBackend>(
		    config.tensorflowConfig.value(), profilePath, inputShape,
		    outputShape);
	case config::BackendType::ONNXRUNTIME:
		return std::make_unique<OnnxruntimeBackend>(
		    config.onnxruntimeConfig.value(), cachePath, profilePath,
		    inputShape, outputShape);
	case config::BackendType::TENSORRT: {
		return std::make_unique<TensorRTBackend>(config.tensorRTConfig.value(),
		    profilePath, inputShape, outputShape, examples, exampleOut);
	}
	default:
		throw std::invalid_argument("Unsupported backend type");
	}
}

}  // namespace backend

}  // namespace benchmark
