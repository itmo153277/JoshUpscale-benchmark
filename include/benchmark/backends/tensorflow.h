// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstdint>
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

class TensorflowBackend : public Backend {
public:
	TensorflowBackend(const config::TensorflowConfig &config,
	    const TensorShape &inputShape, const TensorShape &outputShape);

	Tensor<float> forwardPass(const Tensor<float> &input) override;

private:
	std::uint_fast32_t m_RotIndex = 0;
	tf::TF_Graph m_Graph;
	std::vector<::TF_Output> m_InputOps;
	::TF_Output m_OutputOp;
	TensorShape m_OutputShape;
	tf::TF_Tensor<float> m_LowResTensors[2];
	tf::TF_Tensor<float> m_PreGenTensor;
	tf::TF_Session m_Session;
};

}  // namespace backend

}  // namespace benchmark
