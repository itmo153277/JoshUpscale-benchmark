// Copyright 2021 Ivanov Viktor

#include "benchmark/benchmark.h"

#include <iostream>

#include "benchmark/backend.h"
#include "benchmark/data.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

void benchmark(backend::Backend *backend, const Tensor<float> &lowResImgs,
    const Tensor<float> &hiResImgs, const path_type *profilePath) {
	auto lowResImgArray = unbatch(lowResImgs);

	{
		std::clog << "Begin warmup" << std::endl;
		for (auto &img : lowResImgArray) {
			backend->forwardPass(img);
		}
		std::clog << "End warmup" << std::endl;
	}

	{
		std::clog << "Calculating loss" << std::endl;
		std::vector<Tensor<float>> outImgArray;
		outImgArray.reserve(lowResImgArray.size());
		for (auto &img : lowResImgArray) {
			outImgArray.emplace_back(backend->forwardPass(img));
		}
		auto diff = data::diff(hiResImgs, batch(outImgArray));
		std::clog << "Difference: " << diff << std::endl;
	}

	{
		std::clog << "Profiling" << std::endl;
		backend->profile(lowResImgArray[0], profilePath);
	}
}

}  // namespace benchmark
