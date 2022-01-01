// Copyright 2021 Ivanov Viktor

#pragma once

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

class ConfigParseException : public std::exception {
public:
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
	bool enableTensorRT;
};

struct TensorRTConfig {
	std::filesystem::path modelFileName;
	bool enableFP16;
	bool enableINT8;
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
};

BenchmarkConfig readConfig(const path_type *configPath);

}  // namespace config

}  // namespace benchmark
