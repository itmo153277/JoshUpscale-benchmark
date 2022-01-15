// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "benchmark/backend.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/tensorrt/errorRecorder.h"
#include "benchmark/tensorrt/logger.h"
#include "benchmark/tensorrt/profiler.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

class TensorRTSession : public BackendSession {
public:
	TensorRTSession(bool profile, const path_type *profilePath,
	    trt::ErrorRecorder *errorRecorder, trt::Logger *logger,
	    nvinfer1::IHostMemory *engine, const TensorShape &inputShape,
	    const TensorShape &outputShape, const char *names[]);

	Tensor<float> forwardPass(const Tensor<float> &input) override;
	Tensor<float> forwardPassAsync(const Tensor<float> &input);
	Tensor<float> forwardPassSync(const Tensor<float> &input);

private:
	bool m_Profile;
	std::uint_fast32_t m_RotIndex = 0;
	trt::ErrorRecorder *m_ErrorRecorder;
	TensorShape m_OutputShape;
	std::size_t m_InputSize;
	std::size_t m_OutputSize;
	trt::CudaDeviceBuffer<float> m_LowResTensors[2];
	trt::CudaDeviceBuffer<float> m_HiResTensor;
	std::size_t m_BindingIdx[4] = {};
	trt::TrtPtr<nvinfer1::ICudaEngine> m_Engine;
	trt::TrtPtr<nvinfer1::IExecutionContext> m_Context;
	trt::CudaStream m_CudaStream;
	std::unique_ptr<trt::Profiler> m_Profiler;
};

class TensorRTBackend : public Backend {
public:
	TensorRTBackend(const config::TensorRTConfig &config,
	    const path_type *profilePath, const TensorShape &inputShape,
	    const TensorShape &outputShape, const Tensor<float> &examples,
	    const Tensor<float> &exampleOut);

	std::unique_ptr<BackendSession> createSession(bool profile) override;

private:
	std::filesystem::path m_ProfilePath;
	TensorShape m_InputShape;
	TensorShape m_OutputShape;
	const char *m_Names[4];
	trt::ErrorRecorder m_ErrorRecorder;
	trt::Logger m_Logger;
	trt::TrtPtr<nvinfer1::IHostMemory> m_Engine;
};

}  // namespace backend

}  // namespace benchmark
