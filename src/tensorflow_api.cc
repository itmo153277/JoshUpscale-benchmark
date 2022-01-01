// Copyright 2021 Ivanov Viktor

#include <cstddef>
#include <fstream>
#include <iostream>

#include "benchmark/tensorflow/api.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace tensorflow {

TF_Graph readGraph(const path_type *fileName) {
	std::ifstream file(fileName, std::ios::binary | std::ios::ate);
	file.exceptions(std::ios::failbit | std::ios::badbit);
	auto size = static_cast<std::streamsize>(file.tellg());
	file.seekg(0, std::ios::beg);
	TF_Buffer buffer(static_cast<std::size_t>(size));
	file.read(buffer.getData(), size);
	TF_Graph graph;
	TF_Status status;
	TF_ImportGraphDefOptions importOptions;
	::TF_ImportGraphDefOptionsSetPrefix(importOptions, "");
	::TF_GraphImportGraphDef(graph, buffer, importOptions, status);
	if (::TF_GetCode(status) != ::TF_OK) {
		throw TF_Exception(status);
	}
	return graph;
}

TF_Session::TF_Session(
    const TF_Graph &graph, const TF_SessionOptionsProto *options, bool xla) {
	TF_SessionOptions sessionOptions;
	if (options != nullptr) {
		TF_Status status;
		::TF_SetConfig(sessionOptions, options->proto, options->size, status);
		if (::TF_GetCode(status) != ::TF_OK) {
			throw TF_Exception(status);
		}
	}
	::TF_EnableXLACompilation(sessionOptions, static_cast<char>(xla));
	TF_Status status;
	m_Session = ::TF_NewSession(graph, sessionOptions, status);
	if (::TF_GetCode(status) != ::TF_OK) {
		throw TF_Exception(status);
	}
}

TF_Tensor<float> TF_Session::run(const std::vector<::TF_Output> &inputOp,
    const std::vector<::TF_Tensor *> &inputValue, const TF_Output &outputOp,
    bool profile, ::TF_Buffer *runMetadata) {
	TF_Status status;
	::TF_Tensor *outputValue = nullptr;
	TF_BufferUnmanaged runOptions;
	if (profile) {
		runOptions = TF_BufferUnmanaged::own(
		    ::TF_CreateRunOptions(static_cast<char>(true)));
	}
	::TF_SessionRun(m_Session, runOptions, inputOp.data(), inputValue.data(),
	    static_cast<int>(inputValue.size()), &outputOp, &outputValue, 1,
	    nullptr, 0, runMetadata, status);
	if (::TF_GetCode(status) != ::TF_OK) {
		throw TF_Exception(status);
	}
	return TF_Tensor<float>::own(outputValue);
}

}  // namespace tensorflow

}  // namespace benchmark
