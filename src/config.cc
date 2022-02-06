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
#include <ios>
#include <sstream>
#include <string>

#include "benchmark/utils.h"

namespace benchmark {

namespace config {

template <typename T>
T deserialize(const YAML::Node &node);

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
		auto hiResPath = node["HiResPath"].as<std::string>();
		validatePathMask(lowResPath);
		validatePathMask(hiResPath);
		return {lowResPath, hiResPath};
	} catch (...) {
		throw_with_nested_id(
		    ConfigParseException("Failed to parse data config"));
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
		return {backendType};
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
		    node["ProfileTag"].as<std::string>(""),
		    node["NumIterations"].as<std::size_t>(10)};
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
