// Copyright 2021 Ivanov Viktor

#include "benchmark/backends/onnxruntime.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "benchmark/backend.h"
#include "benchmark/logging.h"
#include "benchmark/onnxruntime/api.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

OnnxruntimeSession::OnnxruntimeSession(const config::OnnxruntimeConfig &config,
    bool profile, const char *cachePath, const path_type *profilePath,
    const TensorShape &inputShape, const TensorShape &outputShape,
    ::Ort::Env *env)
    : m_Profile(profile)
    , m_Session{nullptr}
    , m_InputShape(inputShape.begin(), inputShape.end())
    , m_OutputShape(outputShape.begin() + 1, outputShape.end())
    , m_NodeNames{config.inputOps.at(0).c_str(), config.inputOps.at(1).c_str(),
          config.inputOps.at(2).c_str(), config.outputOp.c_str()} {
	::Ort::SessionOptions options;
	if (config.enableTensorRT) {
		::OrtTensorRTProviderOptions tensorRTOptions{};
		tensorRTOptions.trt_max_workspace_size = 2UL << 30;
		tensorRTOptions.trt_max_partition_iterations = 1000;
		tensorRTOptions.trt_min_subgraph_size = 1;
		tensorRTOptions.trt_fp16_enable =
		    static_cast<int>(config.enableTensorRTFP16);
		tensorRTOptions.trt_int8_enable =
		    static_cast<int>(config.enableTensorRTINT8);
		tensorRTOptions.trt_engine_cache_path = cachePath;
		tensorRTOptions.trt_engine_cache_enable = 1;
		tensorRTOptions.trt_int8_calibration_table_name =
		    "calibration.flatbuffers";
		options.AppendExecutionProvider_TensorRT(tensorRTOptions);
	}
	{
		::OrtCUDAProviderOptions cudaOptions{};
		options.AppendExecutionProvider_CUDA(cudaOptions);
	}
	if (m_Profile) {
		options.EnableProfiling(profilePath);
	}
	m_Session = ::Ort::Session(*env, config.modelFileName.c_str(), options);
	::Ort::AllocatorWithDefaultOptions alloc;
	m_Inputs.emplace_back(nullptr);
	m_Inputs.push_back(::Ort::Value::CreateTensor<float>(
	    alloc, m_InputShape.data(), m_InputShape.size()));
	std::vector<std::int64_t> preGenShape(
	    outputShape.begin(), outputShape.end());
	m_Inputs.push_back(::Ort::Value::CreateTensor<float>(
	    alloc, preGenShape.data(), preGenShape.size()));
}

OnnxruntimeSession::~OnnxruntimeSession() {
	try {
		if (m_Profile) {
			Ort::AllocatorWithDefaultOptions alloc;
			char *profilePath = nullptr;
			profilePath = m_Session.EndProfiling(alloc);
			LOG_INFO << "Profile path: " << profilePath;
			alloc.Free(profilePath);
		}
	} catch (...) {
		LOG_EXCEPTION;
	}
}

Tensor<float> OnnxruntimeSession::forwardPass(const Tensor<float> &input) {
	::Ort::RunOptions options;
	m_Inputs[0] = ::Ort::Value::CreateTensor(
	    ::Ort::MemoryInfo::CreateCpu(::OrtDeviceAllocator, ::OrtMemTypeCPU),
	    const_cast<float *>(input.data.data()), input.data.size(),
	    m_InputShape.data(), m_InputShape.size());
	m_Session.Run(::Ort::RunOptions{}, m_NodeNames.data(), m_Inputs.data(), 3,
	    m_NodeNames.data() + 3, m_Inputs.data() + 2, 1);
	auto outputInfo = m_Inputs[2].GetTensorTypeAndShapeInfo();
	const float *outputData = m_Inputs[2].GetTensorData<float>();
	std::memcpy(m_Inputs[1].GetTensorMutableData<float>(), input.data.data(),
	    input.data.size() * sizeof(float));
	return {m_OutputShape, std::vector<float>(outputData,
	                           outputData + outputInfo.GetElementCount())};
}

OnnxruntimeBackend::OnnxruntimeBackend(const config::OnnxruntimeConfig &config,
    const path_type *cachePath, const path_type *profilePath,
    const TensorShape &inputShape, const TensorShape &outputShape)
    : m_InputShape(inputShape)
    , m_OutputShape(outputShape)
    , m_Env(::ORT_LOGGING_LEVEL_WARNING)
    , m_Config{config}
    , m_CachePath{std::filesystem::path(cachePath).string()}
    , m_ProfilePath{profilePath} {
	m_Env.DisableTelemetryEvents();
	if (config.enableTensorRT && config.enableTensorRTINT8) {
		std::filesystem::copy_file(config.tensorRTCalibrationTable,
		    std::filesystem::path(cachePath) / "calibration.flatbuffers");
	}
	auto inputSize = std::accumulate(m_InputShape.begin(), m_InputShape.end(),
	    std::size_t(1), std::multiplies<std::size_t>());
	m_ProfilePath /= "profile";
	OnnxruntimeBackend::createSession(false)->forwardPass(
	    {m_InputShape, std::vector<float>(inputSize)});
}

std::unique_ptr<BackendSession> OnnxruntimeBackend::createSession(
    bool profile) {
	return std::make_unique<OnnxruntimeSession>(m_Config, profile,
	    m_CachePath.c_str(), m_ProfilePath.c_str(), m_InputShape, m_OutputShape,
	    &m_Env);
}

}  // namespace backend

}  // namespace benchmark
