// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "benchmark/backend.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/tensorrt/errorRecorder.h"
#include "benchmark/tensorrt/logger.h"

namespace benchmark {

namespace backend {

class TensorRTBackend : public Backend {
public:
	TensorRTBackend(const config::TensorRTConfig &config,
	    const TensorShape &inputShape, const TensorShape &outputShape,
	    const Tensor<float> &examples, const Tensor<float> &exampleOut);

	Tensor<float> forwardPass(const Tensor<float> &input) override;

private:
	std::uint_fast32_t m_RotIndex = 0;
	trt::ErrorRecorder m_ErrorRecorder;
	trt::Logger m_Logger;
	TensorShape m_OutputShape;
	std::size_t m_InputSize;
	std::size_t m_OutputSize;
	trt::CudaDeviceBuffer<float> m_LowResTensors[2];
	trt::CudaDeviceBuffer<float> m_HiResTensor;
	std::size_t m_BindingIdx[4] = {};
	trt::TrtPtr<nvinfer1::ICudaEngine> m_Engine;
	trt::TrtPtr<nvinfer1::IExecutionContext> m_Context;
	trt::CudaStream m_CudaStream;
};

}  // namespace backend

}  // namespace benchmark
