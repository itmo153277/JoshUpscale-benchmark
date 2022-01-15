// Copyright 2021 Ivanov Viktor

#include "benchmark/backends/tensorflow.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

static const unsigned char sessionOptionsEncoded[] = {50, 2, 32, 1};
static const tf::TF_SessionOptionsProto sessionOptions = {sessionOptionsEncoded,
    sizeof(sessionOptionsEncoded) / sizeof(*sessionOptionsEncoded)};

TensorflowSession::TensorflowSession(const TensorShape &inputShape,
    const TensorShape &outputShape, tf::TF_Session *session,
    const std::vector<::TF_Output> &inputOps, const ::TF_Output &outputOp,
    bool profile, const path_type *profilePath)
    : m_Profile{profile}
    , m_ProfilePath{profilePath}
    , m_OutputShape(outputShape.begin() + 1, outputShape.end())
    , m_LowResTensors{tf::TF_Tensor<float>(inputShape),
          tf::TF_Tensor<float>(inputShape)}
    , m_PreGenTensor(outputShape)
    , m_InputOps{inputOps}
    , m_OutputOp{outputOp}
    , m_Session{session} {
}

Tensor<float> TensorflowSession::forwardPass(const Tensor<float> &input) {
	auto idx0 = m_RotIndex;
	auto idx1 = idx0 ^ 1;
	m_RotIndex = idx1;
	m_LowResTensors[idx0].copyFromTensor(input);
	tf::TF_BufferUnmanaged runMetadata{nullptr};
	if (m_Profile) {
		runMetadata = tf::TF_BufferUnmanaged();
	}
	m_PreGenTensor = m_Session->run(m_InputOps,
	    {m_LowResTensors[idx0], m_LowResTensors[idx1], m_PreGenTensor},
	    m_OutputOp, m_Profile, runMetadata);
	if (m_Profile) {
		std::ofstream profile(
		    m_ProfilePath, std::ios::binary | std::ios::out | std::ios::app);
		profile.exceptions(std::ios::failbit | std::ios::badbit);
		profile.write(reinterpret_cast<char *>(&runMetadata->length),
		    sizeof(std::size_t));
		profile.write(reinterpret_cast<const char *>(runMetadata->data),
		    static_cast<std::streamsize>(runMetadata->length));
	}
	return {m_OutputShape,
	    std::vector<float>(m_PreGenTensor.begin(), m_PreGenTensor.end())};
}

TensorflowBackend::TensorflowBackend(const config::TensorflowConfig &config,
    const path_type *profilePath, const TensorShape &inputShape,
    const TensorShape &outputShape)
    : m_ProfilePath(profilePath)
    , m_Graph{tf::readGraph(config.graphFileName.c_str())}
    , m_InputOps{{::TF_GraphOperationByName(
                      m_Graph, config.inputOps.at(0).c_str()),
                     0},
          {::TF_GraphOperationByName(m_Graph, config.inputOps.at(1).c_str()),
              0},
          {::TF_GraphOperationByName(m_Graph, config.inputOps.at(2).c_str()),
              0}}
    , m_OutputOp{::TF_GraphOperationByName(m_Graph, config.outputOp.c_str()), 0}
    , m_InputShape{inputShape}
    , m_OutputShape{outputShape}
    , m_Session(m_Graph, &sessionOptions, config.enableXLA) {
	if (m_OutputOp.oper == nullptr || m_InputOps[0].oper == nullptr ||
	    m_InputOps[1].oper == nullptr || m_InputOps[2].oper == nullptr) {
		throw std::invalid_argument("Invalid op name");
	}
	m_ProfilePath /= "profile.pb";
	auto inputSize = std::accumulate(m_InputShape.begin(), m_InputShape.end(),
	    std::size_t(1), std::multiplies<std::size_t>());
	TensorflowBackend::createSession(false)->forwardPass(
	    {m_InputShape, std::vector<float>(inputSize)});
}

std::unique_ptr<BackendSession> TensorflowBackend::createSession(bool profile) {
	return std::make_unique<TensorflowSession>(m_InputShape, m_OutputShape,
	    &m_Session, m_InputOps, m_OutputOp, profile, m_ProfilePath.c_str());
}

}  // namespace backend

}  // namespace benchmark
