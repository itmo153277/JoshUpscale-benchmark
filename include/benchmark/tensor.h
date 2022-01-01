// Copyright 2021 Ivanov Viktor

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace benchmark {

using TensorDim = unsigned int;

using TensorShape = std::vector<TensorDim>;

template <typename T>
struct Tensor {
	TensorShape shape;
	std::vector<T> data;

	bool operator==(const Tensor &t) {
		return std::equal(shape.begin(), shape.end(), t.shape.begin()) &&
		       std::equal(data.begin(), data.end(), t.data.begin());
	}
};

template <typename T>
std::vector<Tensor<T>> unbatch(const Tensor<T> &tensor) {
	auto batch = static_cast<TensorDim>(tensor.shape[0]);
	TensorShape newShape(tensor.shape.begin() + 1, tensor.shape.end());
	std::vector<Tensor<T>> newTensors;
	newTensors.reserve(batch);
	std::size_t tensorSize =
	    tensor.data.size() / static_cast<std::size_t>(batch);
	auto dataIter = tensor.data.begin();
	for (TensorDim i = 0; i < batch; ++i, dataIter += tensorSize) {
		newTensors.push_back(
		    {newShape, std::vector<T>(dataIter, dataIter + tensorSize)});
	}
	return newTensors;
}

template <typename T>
Tensor<T> batch(const std::vector<Tensor<T>> &tensors) {
	assert(tensors.size() > 0);
	TensorShape shape = tensors[0].shape;
	for ([[maybe_unused]] auto &tensor : tensors) {
		assert(std::equal(shape.begin(), shape.end(), tensor.shape.begin()));
	}
	shape.insert(shape.begin(), static_cast<TensorDim>(tensors.size()));
	std::size_t tensorSize = tensors[0].data.size();
	std::vector<T> data(tensorSize * tensors.size());
	auto *dataIter = data.data();
	for (auto tensorIter = tensors.begin(), tensorEnd = tensors.end();
	     tensorIter != tensorEnd; dataIter += tensorSize, tensorIter++) {
		std::memcpy(dataIter, tensorIter->data.data(), tensorSize * sizeof(T));
	}
	return {shape, std::move(data)};
}

}  // namespace benchmark
