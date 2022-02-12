// Copyright 2022 Ivanov Viktor

#include "benchmark/data.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "benchmark/png.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace data {

Tensor<std::uint8_t> loadData(const char *mask, const TensorShape &shape) {
	std::vector<Tensor<std::uint8_t>> images;
	if (shape.getBatchSize() != 1) {
		throw std::invalid_argument("Invalid shape");
	}

	for (TensorDim batch = 0;; ++batch) {
		std::filesystem::path filePath = formatString(mask, batch + 1);
		if (!std::filesystem::exists(filePath)) {
			break;
		}
		auto image = readPng(filePath.c_str());
		if (image.getShape() != shape) {
			throw std::runtime_error(
			    "Invalid image size: " + filePath.string());
		}
		images.push_back(std::move(image));
	}
	return batch(images);
}

float diff(
    const Tensor<std::uint8_t> &left, const Tensor<std::uint8_t> &right) {
	assert(left.getShape().getSize() == right.getShape().getSize());
	float result = 0;

	for (auto leftIter = left.begin(), rightIter = right.begin(),
	          leftEnd = left.end(), rightEnd = right.end();
	     leftIter != leftEnd && rightIter != rightEnd;
	     ++leftIter, ++rightIter) {
		result += std::fabs(static_cast<float>(*leftIter - *rightIter));
	}
	result /= 255.F;
	result /= static_cast<float>(left.getShape().getSize());
	return result;
}

}  // namespace data

}  // namespace benchmark
