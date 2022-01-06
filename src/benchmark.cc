// Copyright 2021 Ivanov Viktor

#include "benchmark/benchmark.h"

#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "benchmark/backend.h"
#include "benchmark/data.h"
#include "benchmark/tensor.h"
#include "benchmark/timer.h"
#include "benchmark/utils.h"

namespace benchmark {

void benchmark(backend::Backend *backend, const Tensor<float> &lowResImgs,
    const Tensor<float> &hiResImgs, std::size_t numIterations,
    const path_type *profilePath) {
	TIMED;
	auto lowResImgArray = unbatch(lowResImgs);

	{
		std::clog << "Begin warmup" << std::endl;
		for (auto &img : lowResImgArray) {
			backend->forwardPass(img);
		}
		std::clog << "End warmup" << std::endl;
	}

	{
		TIMED_W_TAG("benchmark");
		std::clog << "Begin benchmark" << std::endl;
		for (std::size_t i = 0; i < numIterations; ++i) {
			std::ostringstream ss;
			ss << "Iteration " << (i + 1);
			TIMED_W_TAG(ss.str());
			std::vector<timer::timestamp> timestmaps;
			timestmaps.reserve(lowResImgArray.size());
			auto start = timer::clock::now();
			for (auto &img : lowResImgArray) {
				backend->forwardPass(img);
				timestmaps.emplace_back(timer::clock::now());
			}
			for (auto &ts : timestmaps) {
				std::clog << timer::TimestampPrinter(start, ts) << ' ';
				start = ts;
			}
			std::clog << std::endl;
		}
		std::clog << "End benchmark" << std::endl;
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
