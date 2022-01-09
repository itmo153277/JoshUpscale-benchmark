// Copyright 2021 Ivanov Viktor

#include "benchmark/backends/onnxruntime.h"

#include <cstring>

#include "benchmark/onnxruntime/api.h"
#include "benchmark/tensor.h"

namespace benchmark {

namespace backend {

OnnxruntimeBackend::OnnxruntimeBackend(const config::OnnxruntimeConfig &config,
    const TensorShape &inputShape, const TensorShape &outputShape)
    : m_Env(::ORT_LOGGING_LEVEL_WARNING)
    , m_InputShape(inputShape.begin(), inputShape.end())
    , m_OutputShape(outputShape)
    , m_NodeNames{config.inputOps.at(0).c_str(), config.inputOps.at(1).c_str(),
          config.inputOps.at(2).c_str(), config.outputOp.c_str()} {
	m_Env.DisableTelemetryEvents();
	::Ort::SessionOptions options;
	if (config.enableTensorRT) {
		OrtTensorRTProviderOptions tensorRTOptions{};
		tensorRTOptions.trt_max_workspace_size = 2UL << 30;
		tensorRTOptions.trt_max_partition_iterations = 1000;
		tensorRTOptions.trt_min_subgraph_size = 1;
		tensorRTOptions.trt_fp16_enable =
		    static_cast<int>(config.enableTensorRTFP16);
		tensorRTOptions.trt_int8_enable =
		    static_cast<int>(config.enableTensorRTINT8);
		tensorRTOptions.trt_engine_cache_path =
		    config.tensorRTCachePath.c_str();
		tensorRTOptions.trt_int8_calibration_table_name =
		    config.tensorRTCalibrationTable.c_str();
		options.AppendExecutionProvider_TensorRT(tensorRTOptions);
	}
	{
		OrtCUDAProviderOptions cudaOptions{};
		options.AppendExecutionProvider_CUDA(cudaOptions);
	}
	m_Session = ::Ort::Session(m_Env, config.modelFileName.c_str(), options);
	::Ort::AllocatorWithDefaultOptions alloc;
	m_Inputs.emplace_back(nullptr);
	m_Inputs.push_back(
	    ::Ort::Value::CreateTensor<float>(::Ort::AllocatorWithDefaultOptions{},
	        m_InputShape.data(), m_InputShape.size()));
	std::vector<std::int64_t> preGenShape(
	    m_OutputShape.begin(), m_OutputShape.end());
	m_Inputs.push_back(
	    ::Ort::Value::CreateTensor<float>(::Ort::AllocatorWithDefaultOptions{},
	        preGenShape.data(), preGenShape.size()));
	m_OutputShape.erase(m_OutputShape.begin());
	OnnxruntimeBackend::forwardPass({inputShape,
	    std::vector<float>(
	        m_Inputs[1].GetTensorTypeAndShapeInfo().GetElementCount())});
}

Tensor<float> OnnxruntimeBackend::forwardPass(const Tensor<float> &input) {
	::Ort::RunOptions options;
	m_Inputs[0] = ::Ort::Value::CreateTensor(
	    ::Ort::MemoryInfo::CreateCpu(::OrtDeviceAllocator, ::OrtMemTypeCPU),
	    const_cast<float *>(input.data.data()), input.data.size(),
	    m_InputShape.data(), m_InputShape.size());
	auto output = m_Session.Run(::Ort::RunOptions{}, m_NodeNames.data(),
	    m_Inputs.data(), 3, m_NodeNames.data() + 3, 1);
	auto outputInfo = output[0].GetTensorTypeAndShapeInfo();
	const float *outputData = output[0].GetTensorData<float>();
	std::memcpy(m_Inputs[1].GetTensorMutableData<float>(), input.data.data(),
	    input.data.size() * sizeof(float));
	std::memcpy(m_Inputs[2].GetTensorMutableData<float>(), outputData,
	    outputInfo.GetElementCount() * sizeof(float));
	return {m_OutputShape, std::vector<float>(outputData,
	                           outputData + outputInfo.GetElementCount())};
}

}  // namespace backend

}  // namespace benchmark
