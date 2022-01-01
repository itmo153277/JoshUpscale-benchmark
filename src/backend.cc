// Copyright 2021 Ivanov Viktor

#include "benchmark/backend.h"

#include <memory>
#include <stdexcept>

#include "benchmark/backends/tensorflow.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"

namespace benchmark {

namespace backend {

std::unique_ptr<Backend> createBackend(const config::BackendConfig &config,
    const Tensor<float> &examples, const TensorShape &outShape) {
	switch (config.backendType) {
	case config::BackendType::TENSORFLOW: {
		TensorShape inputShape(examples.shape.begin(), examples.shape.end());
		TensorShape outputShape(outShape.begin(), outShape.end());
		inputShape[0] = 1;
		outputShape.insert(outputShape.begin(), 1);
		return std::make_unique<TensorflowBackend>(
		    config.tensorflowConfig.value(), inputShape, outputShape);
	}
	default:
		throw std::invalid_argument("Unsupported backend type");
	}
}

}  // namespace backend

}  // namespace benchmark
