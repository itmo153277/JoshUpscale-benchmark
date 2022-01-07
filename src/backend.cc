// Copyright 2021 Ivanov Viktor

#include "benchmark/backend.h"

#include <memory>
#include <stdexcept>

#include "benchmark/backends/onnxruntime.h"
#include "benchmark/backends/tensorflow.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"

namespace benchmark {

namespace backend {

std::unique_ptr<Backend> createBackend(const config::BackendConfig &config,
    const Tensor<float> &examples, const Tensor<float> &exampleOut) {
	switch (config.backendType) {
	case config::BackendType::TENSORFLOW: {
		TensorShape inputShape = examples.shape;
		TensorShape outputShape = exampleOut.shape;
		inputShape[0] = 1;
		outputShape[0] = 1;
		return std::make_unique<TensorflowBackend>(
		    config.tensorflowConfig.value(), inputShape, outputShape);
	}
	case config::BackendType::ONNXRUNTIME: {
		TensorShape inputShape = examples.shape;
		TensorShape outputShape = exampleOut.shape;
		inputShape[0] = 1;
		outputShape[0] = 1;
		return std::make_unique<OnnxruntimeBackend>(
		    config.onnxruntimeConfig.value(), inputShape, outputShape);
	}
	default:
		throw std::invalid_argument("Unsupported backend type");
	}
}

}  // namespace backend

}  // namespace benchmark
