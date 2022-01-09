// Copyright 2021 Ivanov Viktor

#pragma once

#include "benchmark/logging.h"
#include "benchmark/tensorrt/api.h"

namespace benchmark {

namespace trt {

class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char *msg) noexcept override {
		static const char kTrtTag[] = "TensorRT";
		switch (severity) {
		case Severity::kERROR:
			[[fallthrough]];
		case Severity::kINTERNAL_ERROR:
			logError(kTrtTag) << msg;
			break;
		case Severity::kINFO:
			logInfo(kTrtTag) << msg;
			break;
		case Severity::kWARNING:
			logWarn(kTrtTag) << msg;
			break;
		default:
			break;
		}
	}
};

}  // namespace trt

}  // namespace benchmark
