// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/tensor.h"
#include "benchmark/tensorrt/api.h"

namespace benchmark {

namespace trt {

class Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
	Calibrator(const Tensor<float> &examples, const Tensor<float> &exampleOut,
	    const std::vector<std::string> &names)
	    : m_Size{examples.shape.at(0) - 1}
	    , m_Stride{1, 1}
	    , m_Names{names}
	    , m_Examples(examples.data.size())
	    , m_ExampleOut(exampleOut.data.size()) {
		for (auto &dim : examples.shape) {
			m_Stride[0] *= dim;
		}
		for (auto &dim : exampleOut.shape) {
			m_Stride[1] *= dim;
		}
		m_Stride[0] /= examples.shape.at(0);
		m_Stride[1] /= exampleOut.shape.at(0);
		cudaCheck(::cudaMemcpy(m_Examples.get(), examples.data.data(),
		    examples.data.size() * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice));
		cudaCheck(::cudaMemcpy(m_ExampleOut.get(), exampleOut.data.data(),
		    exampleOut.data.size() * sizeof(float),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	std::int32_t getBatchSize() const noexcept override {
		return 1;
	}

	bool getBatch(void *bindings[], const char *names[],
	    std::int32_t nbBindings) noexcept override {
		if (m_Iter >= m_Size) {
			return false;
		}
		std::size_t offset[3] = {(m_Iter + 1) * m_Stride[0],
		    m_Iter * m_Stride[0], m_Iter * m_Stride[1]};
		for (auto i = nbBindings - 1; i >= 0; --i) {
			if (names[i] == m_Names.at(0)) {
				bindings[i] = m_Examples.get() + offset[0];
			} else if (names[i] == m_Names.at(1)) {
				bindings[i] = m_Examples.get() + offset[1];
			} else if (names[i] == m_Names.at(2)) {
				bindings[i] = m_ExampleOut.get() + offset[2];
			} else {
				bindings[i] = nullptr;
			}
		}
        ++m_Iter;
		return true;
	}

	const void *readCalibrationCache(std::size_t &) noexcept override {
		return nullptr;
	}

	void writeCalibrationCache(const void *, std::size_t) noexcept override {
	}

private:
	std::size_t m_Size;
	std::size_t m_Iter = 0;
	std::size_t m_Stride[2];
	std::vector<std::string> m_Names;
	CudaDeviceBuffer<float> m_Examples;
	CudaDeviceBuffer<float> m_ExampleOut;
};

}  // namespace trt

}  // namespace benchmark
