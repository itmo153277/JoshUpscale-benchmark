// Copyright 2021 Ivanov Viktor

#include "benchmark/tensorrt/profiler.h"

#include <fstream>
#include <ios>

#include "benchmark/logging.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace trt {

Profiler::~Profiler() {
	try {
		std::ofstream profileFile(m_FilePath);
		profileFile.exceptions(std::ios::badbit | std::ios::failbit);
		for (auto &[layerName, ms] : m_Trace) {
			profileFile << layerName << ' ' << ms << std::endl;
		}
	} catch (...) {
		LOG_EXCEPTION;
	}
}

}  // namespace trt

}  // namespace benchmark
