// Copyright 2021 Ivanov Viktor

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "benchmark/tensorrt/api.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace trt {

class Profiler : public nvinfer1::IProfiler {
public:
	explicit Profiler(const path_type *filePath) : m_FilePath(filePath) {
	}
	~Profiler();

	void reportLayerTime(const char *layerName, float ms) noexcept override {
		m_Trace.push_back({layerName, ms});
	}

private:
	const path_type *m_FilePath;
	std::vector<std::pair<std::string, float>> m_Trace;
};

}  // namespace trt

}  // namespace benchmark
