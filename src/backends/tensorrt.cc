// Copyright 2021 Ivanov Viktor

#include "benchmark/backends/tensorrt.h"

#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>
#include <numeric>
#include <utility>

#include "benchmark/logging.h"
#include "benchmark/tensorrt/api.h"
#include "benchmark/tensorrt/calibrator.h"
#include "benchmark/tensorrt/logger.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

TensorRTSession::TensorRTSession(bool profile, const path_type *profilePath,
    trt::ErrorRecorder *errorRecorder, trt::Logger *logger,
    nvinfer1::IHostMemory *engine, const TensorShape &inputShape,
    const TensorShape &outputShape, const char *names[])
    : m_Profile{profile}
    , m_ErrorRecorder{errorRecorder}
    , m_OutputShape(outputShape.begin() + 1, outputShape.end())
    , m_InputSize{std::accumulate(inputShape.begin(), inputShape.end(),
          std::size_t(1), std::multiplies<std::size_t>())}
    , m_OutputSize{std::accumulate(outputShape.begin(), outputShape.end(),
          std::size_t(1), std::multiplies<std::size_t>())}
    , m_LowResTensors{trt::CudaDeviceBuffer<float>(m_InputSize),
          trt::CudaDeviceBuffer<float>(m_InputSize)}
    , m_HiResTensor(m_OutputSize)
    , m_Engine{nullptr}
    , m_Context{nullptr} {
	try {
		auto runtime = trt::TrtPtr(nvinfer1::createInferRuntime(*logger));
		runtime->setErrorRecorder(m_ErrorRecorder);
		m_Engine = trt::TrtPtr(
		    runtime->deserializeCudaEngine(engine->data(), engine->size()));
		for (std::size_t i = 0; i < 4; ++i) {
			auto &name = names[i];
			auto idx = m_Engine->getBindingIndex(name);
			if (idx < 0) {
				throw std::invalid_argument("Invalid node name");
			}
			m_BindingIdx[i] = static_cast<std::size_t>(idx);
		}
		m_Context = trt::TrtPtr(m_Engine->createExecutionContext());
		if (m_Profile) {
			m_Profiler = std::make_unique<trt::Profiler>(profilePath);
			m_Context->setProfiler(m_Profiler.get());
		}
	} catch (...) {
		m_ErrorRecorder->rethrowException();
	}
}

Tensor<float> TensorRTSession::forwardPass(const Tensor<float> &input) {
	if (m_Profile) {
		return forwardPassSync(input);
	}
	return forwardPassAsync(input);
}

Tensor<float> TensorRTSession::forwardPassAsync(const Tensor<float> &input) {
	try {
		auto idx0 = m_RotIndex;
		auto idx1 = idx0 ^ 1;
		m_RotIndex = idx1;
		std::vector<float> result(m_OutputSize);
		trt::cudaCheck(::cudaMemcpyAsync(m_LowResTensors[idx0].get(),
		    input.data.data(), m_InputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, m_CudaStream));
		void *bindings[4];
		bindings[m_BindingIdx[0]] = m_LowResTensors[idx0].get();
		bindings[m_BindingIdx[1]] = m_LowResTensors[idx1].get();
		bindings[m_BindingIdx[2]] = m_HiResTensor.get();
		bindings[m_BindingIdx[3]] = m_HiResTensor.get();
		if (!m_Context->enqueueV2(bindings, m_CudaStream, nullptr)) {
			throw trt::TrtException();
		}
		trt::cudaCheck(::cudaMemcpyAsync(result.data(), m_HiResTensor.get(),
		    m_OutputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToHost, m_CudaStream));
		trt::cudaCheck(::cudaStreamSynchronize(m_CudaStream));
		return {m_OutputShape, std::move(result)};
	} catch (...) {
		m_ErrorRecorder->rethrowException();
	}
}

Tensor<float> TensorRTSession::forwardPassSync(const Tensor<float> &input) {
	try {
		auto idx0 = m_RotIndex;
		auto idx1 = idx0 ^ 1;
		m_RotIndex = idx1;
		std::vector<float> result(m_OutputSize);
		trt::cudaCheck(::cudaMemcpy(m_LowResTensors[idx0].get(),
		    input.data.data(), m_InputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice));
		void *bindings[4];
		bindings[m_BindingIdx[0]] = m_LowResTensors[idx0].get();
		bindings[m_BindingIdx[1]] = m_LowResTensors[idx1].get();
		bindings[m_BindingIdx[2]] = m_HiResTensor.get();
		bindings[m_BindingIdx[3]] = m_HiResTensor.get();
		if (!m_Context->executeV2(bindings)) {
			throw trt::TrtException();
		}
		trt::cudaCheck(::cudaMemcpy(result.data(), m_HiResTensor.get(),
		    m_OutputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToHost));
		return {m_OutputShape, std::move(result)};
	} catch (...) {
		m_ErrorRecorder->rethrowException();
	}
}

TensorRTBackend::TensorRTBackend(const config::TensorRTConfig &config,
    const path_type *profilePath, const TensorShape &inputShape,
    const TensorShape &outputShape, const Tensor<float> &examples,
    const Tensor<float> &exampleOut)
    : m_ProfilePath(profilePath)
    , m_InputShape(inputShape)
    , m_OutputShape(outputShape)
    , m_Names{config.inputOps.at(0).c_str(), config.inputOps.at(1).c_str(),
          config.inputOps.at(2).c_str(), config.outputOp.c_str()}
    , m_Engine{nullptr} {
	try {
		auto builder = trt::TrtPtr(nvinfer1::createInferBuilder(m_Logger));
		builder->setErrorRecorder(&m_ErrorRecorder);
		auto builderConfig = trt::TrtPtr(builder->createBuilderConfig());
		auto network = trt::TrtPtr(builder->createNetworkV2(1));
		auto parser =
		    trt::TrtPtr(nvonnxparser::createParser(*network, m_Logger));
		if (builder->getNbDLACores() > 0) {
			LOG_INFO << "Using DLA";
			builderConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
			builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
		}
		builderConfig->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
		auto workspaceSize = config.maxWorkspaceSizeBytes;
		if (workspaceSize == 0) {
			workspaceSize = 2UL << 30;
		}
		builderConfig->setMaxWorkspaceSize(workspaceSize);
		if (config.enableFP16) {
			if (!builder->platformHasFastFp16()) {
				LOG_WARN << "FP16 is not fast on this platform!";
			}
			builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		std::unique_ptr<trt::Calibrator> calibrator;
		if (config.enableINT8) {
			if (!builder->platformHasFastInt8()) {
				LOG_WARN << "INT8 is not fast on this platform!";
			}
			builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
			calibrator = std::make_unique<trt::Calibrator>(
			    examples, exampleOut, config.inputOps);
			builderConfig->setInt8Calibrator(calibrator.get());
		}
		{
			std::ifstream modelFile(config.modelFileName.c_str(),
			    std::ios::in | std::ios::binary | std::ios::ate);
			modelFile.exceptions(std::ios::failbit | std::ios::badbit);
			auto size = static_cast<std::size_t>(modelFile.tellg());
			modelFile.seekg(0, std::ios::beg);
			auto buf = std::make_unique<std::byte[]>(size);
			modelFile.read(reinterpret_cast<char *>(buf.get()),
			    static_cast<std::streamsize>(size));
			if (!parser->parse(buf.get(), size)) {
				throw trt::TrtParserException(*parser);
			}
		}
		m_Engine = trt::TrtPtr(
		    builder->buildSerializedNetwork(*network, *builderConfig));
		m_ProfilePath /= "profile.log";
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

std::unique_ptr<BackendSession> TensorRTBackend::createSession(bool profile) {
	return std::make_unique<TensorRTSession>(profile, m_ProfilePath.c_str(),
	    &m_ErrorRecorder, &m_Logger, m_Engine.get(), m_InputShape,
	    m_OutputShape, m_Names);
}

}  // namespace backend

}  // namespace benchmark
