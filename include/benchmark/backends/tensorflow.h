// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/backend.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/tensorflow/api.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

namespace tf = ::benchmark::tensorflow;

class TensorflowSession : public BackendSession {
public:
	TensorflowSession(const TensorShape &inputShape,
	    const TensorShape &outputShape, tf::TF_Session *session,
	    const std::vector<::TF_Output> &inputOps, const ::TF_Output &outputOp,
	    bool profile, const path_type *profilePath);

	Tensor<float> forwardPass(const Tensor<float> &input) override;

private:
	bool m_Profile;
	const path_type *m_ProfilePath;
	std::uint_fast32_t m_RotIndex = 0;
	TensorShape m_OutputShape;
	tf::TF_Tensor<float> m_LowResTensors[2];
	tf::TF_Tensor<float> m_PreGenTensor;
	const std::vector<::TF_Output> &m_InputOps;
	const ::TF_Output &m_OutputOp;
	tf::TF_Session *m_Session;
};

class TensorflowBackend : public Backend {
public:
	TensorflowBackend(const config::TensorflowConfig &config,
	    const path_type *profilePath, const TensorShape &inputShape,
	    const TensorShape &outputShape);

	std::unique_ptr<BackendSession> createSession(bool profile) override;

private:
	std::filesystem::path m_ProfilePath;
	tf::TF_Graph m_Graph;
	std::vector<::TF_Output> m_InputOps;
	::TF_Output m_OutputOp;
	TensorShape m_InputShape;
	TensorShape m_OutputShape;
	tf::TF_Session m_Session;
};

}  // namespace backend

}  // namespace benchmark
