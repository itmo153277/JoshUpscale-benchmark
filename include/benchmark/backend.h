// Copyright 2021 Ivanov Viktor

#pragma once

#include <memory>

#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

class Backend {
public:
	virtual ~Backend() {
	}

	virtual Tensor<float> forwardPass(const Tensor<float> &input) = 0;
	virtual Tensor<float> profile(
	    const Tensor<float> &input, const path_type *savePath) = 0;
};

std::unique_ptr<Backend> createBackend(const config::BackendConfig &config,
    const Tensor<float> &examples, const TensorShape &outShape);

}  // namespace backend

}  // namespace benchmark
