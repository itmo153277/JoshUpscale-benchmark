// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
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
	trt::ErrorRecorder m_ErrorRecorder;
	trt::Logger m_Logger;
	TensorShape m_OutputShape;
	std::size_t m_InputSize;
	std::size_t m_OutputSize;
	trt::CudaDeviceBuffer<float> m_CurFrameTensor;
	trt::CudaDeviceBuffer<float> m_LastFrameTensor;
	trt::CudaDeviceBuffer<float> m_PreGenTensor;
	trt::CudaDeviceBuffer<float> m_OutputTensor;
	void *m_Bindings[4];
	trt::TrtPtr<nvinfer1::ICudaEngine> m_Engine;
	trt::TrtPtr<nvinfer1::IExecutionContext> m_Context;
	trt::CudaStream m_CudaStream;
};

}  // namespace backend

}  // namespace benchmark
