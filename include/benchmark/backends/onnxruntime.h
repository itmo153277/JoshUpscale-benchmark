// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/backend.h"
#include "benchmark/config.h"
#include "benchmark/onnxruntime/api.h"
#include "benchmark/tensor.h"

namespace benchmark {

namespace backend {

class OnnxruntimeBackend : public Backend {
public:
	OnnxruntimeBackend(const config::OnnxruntimeConfig &config,
	    const TensorShape &inputShape, const TensorShape &outputSHape);

	Tensor<float> forwardPass(const Tensor<float> &input) override;

private:
	::Ort::Env m_Env;
	::Ort::Session m_Session{nullptr};
	std::vector<std::int64_t> m_InputShape;
	TensorShape m_OutputShape;
	std::vector<::Ort::Value> m_Inputs;
	std::vector<const char *> m_NodeNames;
	std::string m_TensorRTCachePath;
	std::string m_TensorRTCalibrationTable;
};

}  // namespace backend

}  // namespace benchmark
