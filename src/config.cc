// Copyright 2021 Ivanov Viktor

#include "benchmark/config.h"

#ifdef _MSC_VER
// Prefer 'enum class' over 'enum' (Enum.3)
#pragma warning(disable : 26812)
#endif

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "benchmark/data.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace config {

template <typename T>
T deserialize(const YAML::Node &node);

template <>
data::DataFormat deserialize<data::DataFormat>(const YAML::Node &node) {
	auto value = node.as<std::string>();
	if (value == "NHWC") {
		return benchmark::data::DataFormat::NHWC;
	}
	if (value == "NCHW") {
		return benchmark::data::DataFormat::NCHW;
	}
	throw ConfigParseException("Invalid data format value");
}

namespace {
void validatePathMask(const std::string &s) {
	auto maskStart = s.find('%');
	if (maskStart == std::string::npos) {
		goto bad;
	}
	for (auto i = maskStart + 1, l = s.size(); i < l; ++i) {
		if (s[i] == 'd') {
			return;
		}
		if (s[i] < '0' || s[i] > '9') {
			break;
		}
	}
bad:
	throw ConfigParseException("Incorrect path format");
}
}  // namespace

template <>
DataConfig deserialize<DataConfig>(const YAML::Node &node) {
	if (!node || !node.IsMap()) {
		throw ConfigParseException("Missing data config");
	}
	try {
		auto lowResPath = node["LowResPath"].as<std::string>();
		auto lowResShape = node["LowResShape"].as<TensorShape>();
		auto hiResPath = node["HiResPath"].as<std::string>();
		auto hiResShape = node["HiResShape"].as<TensorShape>();
		auto dataFormat = deserialize<data::DataFormat>(node["DataFormat"]);
		validatePathMask(lowResPath);
		validatePathMask(hiResPath);
		return {lowResPath, lowResShape, hiResPath, hiResShape, dataFormat};
	} catch (...) {
		throw_with_nested_id(
		    ConfigParseException("Failed to parse data config"));
	}
}

template <>
OnnxruntimeConfig deserialize<OnnxruntimeConfig>(const YAML::Node &node) {
	if (!node || !node.IsMap()) {
		throw ConfigParseException("Missing onnxruntime backend config");
	}
	try {
		return {node["ModelFileName"].as<std::string>(),
		    node["EnableTensorRT"].as<bool>(false)};
	} catch (...) {
		throw_with_nested_id(
		    ConfigParseException("Failed to parse onnxruntime backend config"));
	}
}

template <>
TensorflowConfig deserialize<TensorflowConfig>(const YAML::Node &node) {
	if (!node || !node.IsMap()) {
		throw ConfigParseException("Missing tensorflow backend config");
	}
	try {
		return {node["GraphFileName"].as<std::string>(),
		    node["InputOps"].as<std::vector<std::string>>(),
		    node["OutputOp"].as<std::string>(),
		    node["EnableXLA"].as<bool>(false)};
	} catch (...) {
		throw_with_nested_id(
		    ConfigParseException("Failed to parse tensorflow backend config"));
	}
}

template <>
TensorRTConfig deserialize<TensorRTConfig>(const YAML::Node &node) {
	if (!node || !node.IsMap()) {
		throw ConfigParseException("Missing TensorRT backend config");
	}
	try {
		return {node["ModelFileName"].as<std::string>(),
		    node["EnableFP16"].as<bool>(false),
		    node["EnableINT8"].as<bool>(false)};
	} catch (...) {
		throw_with_nested_id(
		    ConfigParseException("Failed to parse TensorRT backend config"));
	}
}

template <>
BackendType deserialize<BackendType>(const YAML::Node &node) {
	auto value = node.as<std::string>();
	if (value == "onnxruntime") {
		return BackendType::ONNXRUNTIME;
	}
	if (value == "tensorflow") {
		return BackendType::TENSORFLOW;
	}
	if (value == "TensorRT") {
		return BackendType::TENSORRT;
	}
	throw ConfigParseException("Invalid backend type");
}

template <>
BackendConfig deserialize<BackendConfig>(const YAML::Node &node) {
	if (!node || !node.IsMap()) {
		throw ConfigParseException("Missing backend config");
	}
	try {
		auto backendType = deserialize<BackendType>(node["BackendType"]);
		switch (backendType) {
		case BackendType::ONNXRUNTIME:
			return {backendType,
			    deserialize<OnnxruntimeConfig>(node["OnnxruntimeConfig"]),
			    std::nullopt, std::nullopt};
		case BackendType::TENSORFLOW:
			return {backendType, std::nullopt,
			    deserialize<TensorflowConfig>(node["TensorflowConfig"]),
			    std::nullopt};
		case BackendType::TENSORRT:
			return {backendType, std::nullopt, std::nullopt,
			    deserialize<TensorRTConfig>(node["TensorRTConfig"])};
		default:
			return {backendType, std::nullopt, std::nullopt, std::nullopt};
		}
	} catch (...) {
		throw_with_nested_id(
		    ConfigParseException("Failed to parse backend config"));
	}
}

template <>
BenchmarkConfig deserialize<BenchmarkConfig>(const YAML::Node &node) {
	if (!node || !node.IsMap()) {
		throw ConfigParseException();
	}
	try {
		return {deserialize<DataConfig>(node["DataConfig"]),
		    deserialize<BackendConfig>(node["BackendConfig"]),
		    node["ProfileTag"].as<std::string>("")};
	} catch (...) {
		throw_with_nested_id(ConfigParseException());
	}
}

BenchmarkConfig readConfig(const path_type *configPath) {
	std::ostringstream ss;
	try {
		std::ifstream configFile(configPath);
		configFile.exceptions(std::ios::failbit | std::ios::badbit);
		ss << configFile.rdbuf();
	} catch (...) {
		throw_with_nested_id(ConfigParseException("Failed to read config"));
	}
	return deserialize<BenchmarkConfig>(YAML::Load(ss.str().c_str()));
}

}  // namespace config

}  // namespace benchmark
