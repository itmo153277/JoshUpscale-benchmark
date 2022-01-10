// Copyright 2021 Ivanov Viktor

#include "benchmark/backends/tensorrt.h"

#include <array>
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

namespace benchmark {

namespace backend {

TensorRTBackend::TensorRTBackend(const config::TensorRTConfig &config,
    const TensorShape &inputShape, const TensorShape &outputShape,
    const Tensor<float> &examples, const Tensor<float> &exampleOut)
    : m_OutputShape(outputShape.begin() + 1, outputShape.end())
    , m_InputSize{std::accumulate(inputShape.begin(), inputShape.end(),
          std::size_t(1), std::multiplies<std::size_t>())}
    , m_OutputSize{std::accumulate(outputShape.begin(), outputShape.end(),
          std::size_t(1), std::multiplies<std::size_t>())}
    , m_CurFrameTensor(m_InputSize)
    , m_LastFrameTensor(m_InputSize)
    , m_PreGenTensor(m_OutputSize)
    , m_OutputTensor{m_OutputSize}
    , m_Bindings{}
    , m_Engine{nullptr}
    , m_Context{nullptr} {
	try {
		trt::TrtPtr<nvinfer1::IHostMemory> serializedEngine{nullptr};
		{
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
			serializedEngine = trt::TrtPtr(
			    builder->buildSerializedNetwork(*network, *builderConfig));
		}
		{
			auto runtime = trt::TrtPtr(nvinfer1::createInferRuntime(m_Logger));
			runtime->setErrorRecorder(&m_ErrorRecorder);
			m_Engine = trt::TrtPtr(runtime->deserializeCudaEngine(
			    serializedEngine->data(), serializedEngine->size()));
			std::array<std::pair<const char *, void *>, 4> names = {
			    {{config.inputOps.at(0).c_str(), m_CurFrameTensor.get()},
			        {config.inputOps.at(1).c_str(), m_LastFrameTensor.get()},
			        {config.inputOps.at(2).c_str(), m_PreGenTensor.get()},
			        {config.outputOp.c_str(), m_OutputTensor.get()}}};
			for (auto &[name, ptr] : names) {
				auto idx = m_Engine->getBindingIndex(name);
				if (idx < 0) {
					throw std::invalid_argument("Invalid node name");
				}
				m_Bindings[idx] = ptr;
			}
			m_Context = trt::TrtPtr(m_Engine->createExecutionContext());
		}
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

Tensor<float> TensorRTBackend::forwardPass(const Tensor<float> &input) {
	try {
		std::vector<float> result(m_OutputSize);
		trt::cudaCheck(::cudaMemcpyAsync(m_CurFrameTensor.get(),
		    input.data.data(), m_InputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, m_CudaStream));
		if (!m_Context->enqueueV2(m_Bindings, m_CudaStream, nullptr)) {
			throw trt::TrtException();
		}
		trt::cudaCheck(::cudaMemcpyAsync(m_LastFrameTensor.get(),
		    m_CurFrameTensor.get(), m_InputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_CudaStream));
		trt::cudaCheck(::cudaMemcpyAsync(m_PreGenTensor.get(),
		    m_OutputTensor.get(), m_OutputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_CudaStream));
		trt::cudaCheck(::cudaMemcpyAsync(result.data(), m_OutputTensor.get(),
		    m_OutputSize * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToHost, m_CudaStream));
		trt::cudaCheck(::cudaStreamSynchronize(m_CudaStream));
		return {m_OutputShape, std::move(result)};
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

}  // namespace backend

}  // namespace benchmark
