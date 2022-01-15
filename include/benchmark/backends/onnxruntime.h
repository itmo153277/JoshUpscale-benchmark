// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/backend.h"
#include "benchmark/config.h"
#include "benchmark/onnxruntime/api.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

class OnnxruntimeSession : public BackendSession {
public:
	OnnxruntimeSession(const config::OnnxruntimeConfig &config, bool profile,
	    const char *cachePath, const path_type *profilePath,
	    const TensorShape &inputShape, const TensorShape &outputShape,
	    ::Ort::Env *env);
	~OnnxruntimeSession();

	Tensor<float> forwardPass(const Tensor<float> &input) override;

private:
	bool m_Profile;
	::Ort::Session m_Session;
	std::vector<std::int64_t> m_InputShape;
	TensorShape m_OutputShape;
	std::vector<::Ort::Value> m_Inputs;
	std::vector<const char *> m_NodeNames;
};

class OnnxruntimeBackend : public Backend {
public:
	OnnxruntimeBackend(const config::OnnxruntimeConfig &config,
	    const path_type *cachePath, const path_type *profilePath,
	    const TensorShape &inputShape, const TensorShape &outputShape);

	std::unique_ptr<BackendSession> createSession(bool profile) override;

private:
	TensorShape m_InputShape;
	TensorShape m_OutputShape;
	::Ort::Env m_Env;
	const config::OnnxruntimeConfig &m_Config;
	std::string m_CachePath;
	std::filesystem::path m_ProfilePath;
};

}  // namespace backend

}  // namespace benchmark
