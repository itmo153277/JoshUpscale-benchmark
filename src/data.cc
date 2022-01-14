// Copyright 2021 Ivanov Viktor

#include "benchmark/data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include "benchmark/png.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace data {

Tensor<float> loadData(
    const char *mask, DataFormat format, const TensorShape &shape) {
	if (shape.size() != 3) {
		throw std::invalid_argument("Invalid shape");
	}
	std::size_t width;
	std::size_t height;
	std::size_t channels;
	TensorShape outputShape;
	switch (format) {
	case DataFormat::NCHW:
		width = shape[2];
		height = shape[1];
		channels = shape[0];
		outputShape = {0, static_cast<TensorDim>(channels),
		    static_cast<TensorDim>(height), static_cast<TensorDim>(width)};
		break;
	case DataFormat::NHWC:
		width = shape[1];
		height = shape[0];
		channels = shape[2];
		outputShape = {0, static_cast<TensorDim>(height),
		    static_cast<TensorDim>(width), static_cast<TensorDim>(channels)};
		break;
	default:
		throw std::invalid_argument("Invalid format");
	}
	if (channels != 3) {
		throw std::invalid_argument("Invalid shape");
	}
	auto pixelCount = width * height;
	auto imageSize = pixelCount * 3;
	std::vector<std::vector<std::uint8_t>> images;
	std::vector<std::uint8_t> imageTransposed(imageSize);
	for (int batch = 0;; ++batch) {
		auto filePath = formatString(mask, batch + 1);
		if (!std::filesystem::exists(filePath)) {
			break;
		}
		auto image = readPng(filePath.c_str());
		if (image.shape[0] != height || image.shape[1] != width) {
			throw std::runtime_error("Invalid image size: " + filePath);
		}
		if (format == DataFormat::NCHW) {
			for (std::size_t i = 0; i < pixelCount; ++i) {
				for (std::size_t c = 0; c < 3; ++c) {
					imageTransposed[c * pixelCount + i] = image.data[i * 3 + c];
				}
			}
			images.emplace_back(std::move(imageTransposed));
			imageTransposed = std::move(image.data);
		} else {
			images.emplace_back(std::move(image.data));
		}
	}
	outputShape[0] = static_cast<TensorDim>(images.size());
	std::size_t totalSize = images.size() * imageSize;
	std::vector<float> data(totalSize);
	for (std::size_t batch = 0; batch < images.size(); ++batch) {
		auto &image = images[batch];
		float *dataPtr = data.data() + batch * imageSize;
		for (std::size_t i = 0; i < imageSize; ++i) {
			dataPtr[i] = static_cast<float>(image[i]) / 255.0F;
		}
	}
	return {outputShape, std::move(data)};
}

float diff(const Tensor<float> &left, const Tensor<float> &right) {
	if (left.shape.size() != right.shape.size() ||
	    !std::equal(
	        left.shape.begin(), left.shape.end(), right.shape.begin())) {
		throw std::invalid_argument("Shape mismatch");
	}
	float diff = 0;
	for (auto leftIter = left.data.begin(), rightIter = right.data.begin(),
	          leftEnd = left.data.end(), rightEnd = right.data.end();
	     leftIter != leftEnd && rightIter != rightEnd;
	     ++leftIter, ++rightIter) {
		diff += std::fabs(*leftIter - *rightIter);
	}
	return diff / static_cast<float>(left.data.size());
}

}  // namespace data

}  // namespace benchmark
