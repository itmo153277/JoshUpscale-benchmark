// Copyright 2021 Ivanov Viktor

#include "benchmark/backends/tensorrt.h"

#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>

#include "benchmark/logging.h"
#include "benchmark/tensorrt/api.h"
#include "benchmark/tensorrt/logger.h"

namespace benchmark {

namespace backend {

TensorRTBackend::TensorRTBackend(const config::TensorRTConfig &config) {
	try {
		auto builder = trt::TrtPtr<nvinfer1::IBuilder>(
		    nvinfer1::createInferBuilder(m_Logger));
		builder->setErrorRecorder(&m_ErrorRecorder);
		auto builderConfig = trt::TrtPtr<nvinfer1::IBuilderConfig>(
		    builder->createBuilderConfig());
		auto network = trt::TrtPtr<nvinfer1::INetworkDefinition>(
		    builder->createNetworkV2(1));
		auto parser = trt::TrtPtr<nvonnxparser::IParser>(
		    nvonnxparser::createParser(*network, m_Logger));
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
		if (config.enableINT8) {
			if (!builder->platformHasFastInt8()) {
				LOG_WARN << "INT8 is not fast on this platform!";
			}
			builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
			// builderConfig->setInt8Calibrator(nullptr);
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
		auto serializedEngine = trt::TrtPtr<nvinfer1::IHostMemory>(
		    builder->buildSerializedNetwork(*network, *builderConfig));
	} catch (trt::TrtException &e) {
		m_ErrorRecorder.rethrowException(&e);
	}
}

Tensor<float> TensorRTBackend::forwardPass(
    [[maybe_unused]] const Tensor<float> &input) {
	return {};
}

}  // namespace backend

}  // namespace benchmark
