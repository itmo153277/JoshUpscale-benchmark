// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "benchmark/utils.h"
#include "benchmark/tensor.h"

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
	std::string hiResPath;
	TensorDim width;
	TensorDim height;
	TensorDim upscaleFactor;
};

enum class BackendType { ONNXRUNTIME, TENSORFLOW, TENSORRT };

struct BackendConfig {
	BackendType backendType;
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
