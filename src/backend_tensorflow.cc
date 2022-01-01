// Copyright 2021 Ivanov Viktor

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "benchmark/backends/tensorflow.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace backend {

static const unsigned char sessionOptionsEncoded[] = {50, 2, 32, 1};
static const tf::TF_SessionOptionsProto sessionOptions = {sessionOptionsEncoded,
    sizeof(sessionOptionsEncoded) / sizeof(*sessionOptionsEncoded)};

TensorflowBackend::TensorflowBackend(const config::TensorflowConfig &config,
    const TensorShape &inputShape, const TensorShape &outputShape)
    : m_Graph{tf::readGraph(config.graphFileName.c_str())}
    , m_InputOps{{::TF_GraphOperationByName(
                      m_Graph, config.inputOps.at(0).c_str()),
                     0},
          {::TF_GraphOperationByName(m_Graph, config.inputOps.at(1).c_str()),
              0},
          {::TF_GraphOperationByName(m_Graph, config.inputOps.at(2).c_str()),
              0}}
    , m_OutputOp{::TF_GraphOperationByName(m_Graph, config.outputOp.c_str()), 0}
    , m_OutputShape(outputShape)
    , m_InputTensor(inputShape)
    , m_LastFrameTensor(inputShape)
    , m_PreGenTensor(outputShape)
    , m_Session(m_Graph, &sessionOptions, config.enableXLA) {
	if (m_OutputOp.oper == nullptr || m_InputOps[0].oper == nullptr ||
	    m_InputOps[1].oper == nullptr || m_InputOps[2].oper == nullptr) {
		throw std::invalid_argument("Invalid op name");
	}
}

Tensor<float> TensorflowBackend::forwardPass(const Tensor<float> &input) {
	m_InputTensor.copyFromTensor(input);
	m_PreGenTensor = m_Session.run(m_InputOps,
	    {m_InputTensor, m_LastFrameTensor, m_PreGenTensor}, m_OutputOp, false,
	    nullptr);
	m_LastFrameTensor.copyFromTensor(m_InputTensor);
	return {m_OutputShape,
	    std::vector<float>(m_PreGenTensor.begin(), m_PreGenTensor.end())};
}
Tensor<float> TensorflowBackend::profile(
    const Tensor<float> &input, const path_type *savePath) {
	auto profileFileName = std::filesystem::path(savePath) / "profile.proto";
	std::ofstream profileFile(
	    profileFileName.c_str(), std::ofstream::binary | std::ofstream::out);
	profileFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
	tf::TF_BufferUnmanaged runMetadata;
	m_InputTensor.copyFromTensor(input);
	m_PreGenTensor = m_Session.run(m_InputOps,
	    {m_InputTensor, m_LastFrameTensor, m_PreGenTensor}, m_OutputOp, true,
	    runMetadata);
	profileFile.write(reinterpret_cast<const char *>(runMetadata->data),
	    static_cast<std::streamsize>(runMetadata->length));
	m_LastFrameTensor.copyFromTensor(m_InputTensor);
	return {m_OutputShape,
	    std::vector<float>(m_PreGenTensor.begin(), m_PreGenTensor.end())};
}

}  // namespace backend

}  // namespace benchmark
