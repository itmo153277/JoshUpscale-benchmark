// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "benchmark/data.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace config {

struct ConfigParseException : std::exception {
	ConfigParseException() : std::exception("Failed to parse config") {
	}
	explicit ConfigParseException(const char *msg) : std::exception(msg) {
	}
};

struct DataConfig {
	std::string lowResPath;
	TensorShape lowResShape;
	std::string hiResPath;
	TensorShape hiResShape;
	data::DataFormat dataFormat;
};

enum class BackendType { ONNXRUNTIME, TENSORFLOW, TENSORRT };

struct TensorflowConfig {
	std::filesystem::path graphFileName;
	std::vector<std::string> inputOps;
	std::string outputOp;
	bool enableXLA;
};

struct OnnxruntimeConfig {
	std::filesystem::path modelFileName;
	std::vector<std::string> inputOps;
	std::string outputOp;
	bool enableTensorRT;
	bool enableTensorRTFP16;
	bool enableTensorRTINT8;
	std::filesystem::path tensorRTCalibrationTable;
};

struct TensorRTConfig {
	std::filesystem::path modelFileName;
	std::vector<std::string> inputOps;
	std::string outputOp;
	bool enableFP16;
	bool enableINT8;
	std::size_t maxWorkspaceSizeBytes;
};

struct BackendConfig {
	BackendType backendType;
	std::optional<OnnxruntimeConfig> onnxruntimeConfig;
	std::optional<TensorflowConfig> tensorflowConfig;
	std::optional<TensorRTConfig> tensorRTConfig;
};

struct BenchmarkConfig {
	DataConfig dataConfig;
	BackendConfig backendConfig;
	std::string profileTag;
	std::size_t numIterations;
};

BenchmarkConfig readConfig(const path_type *configPath);

}  // namespace config

}  // namespace benchmark
