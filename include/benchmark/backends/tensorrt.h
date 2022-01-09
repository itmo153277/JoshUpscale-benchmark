// Copyright 2021 Ivanov Viktor

#pragma once

#include <string>

#include "benchmark/backend.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/tensorrt/logger.h"
#include "benchmark/tensorrt/errorRecorder.h"

namespace benchmark {

namespace backend {

class TensorRTBackend : public Backend {
public:
	explicit TensorRTBackend(const config::TensorRTConfig &config);

	Tensor<float> forwardPass(const Tensor<float> &input) override;

private:
	trt::ErrorRecorder m_ErrorRecorder;
	trt::Logger m_Logger;
};

}  // namespace backend

}  // namespace benchmark
