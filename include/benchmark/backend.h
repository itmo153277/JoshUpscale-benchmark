// Copyright 2021 Ivanov Viktor

#pragma once

#include <memory>
#include <string>

#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

class BackendSession {
public:
	virtual ~BackendSession() {
	}

	virtual Tensor<float> forwardPass(const Tensor<float> &input) = 0;
};

class Backend {
public:
	virtual ~Backend() {
	}

	virtual std::unique_ptr<BackendSession> createSession(bool profile) = 0;
};

std::unique_ptr<Backend> createBackend(const config::BackendConfig &config,
    const Tensor<float> &examples, const Tensor<float> &exampleOut,
    const path_type *profilePath, const path_type *cachePath);

}  // namespace backend

}  // namespace benchmark
