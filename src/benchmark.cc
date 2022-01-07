// Copyright 2021 Ivanov Viktor

#include "benchmark/benchmark.h"

#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark/backend.h"
#include "benchmark/data.h"
#include "benchmark/logging.h"
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
		LOG_INFO << "Begin warmup";
		for (auto &img : lowResImgArray) {
			backend->forwardPass(img);
		}
		LOG_INFO << "End warmup";
	}

	{
		TIMED_W_TAG("benchmark");
		LOG_INFO << "Begin benchmark";
		for (std::size_t i = 0; i < numIterations; ++i) {
			TIMED_W_TAG(formatString("Iteration %zu", i + 1));
			std::vector<timer::timestamp> timestmaps;
			timestmaps.reserve(lowResImgArray.size());
			auto start = timer::clock::now();
			for (auto &img : lowResImgArray) {
				backend->forwardPass(img);
				timestmaps.push_back(timer::clock::now());
			}
			for (auto &ts : timestmaps) {
				std::clog << timer::TimestampPrinter(start, ts) << ' ';
				start = ts;
			}
			std::clog << std::endl;
		}
		LOG_INFO << "End benchmark";
	}

	{
		LOG_INFO << "Calculating loss";
		std::vector<Tensor<float>> outImgArray;
		outImgArray.reserve(lowResImgArray.size());
		for (auto &img : lowResImgArray) {
			outImgArray.push_back(backend->forwardPass(img));
		}
		auto diff = data::diff(hiResImgs, batch(outImgArray));
		LOG_INFO << "Difference: " << diff;
	}

	{
		LOG_INFO << "Profiling";
		for (int i = 0; i < 5; ++i) {
			backend->profile(
			    lowResImgArray[i], profilePath, formatString("run-%02d", i));
		}
	}
}

}  // namespace benchmark
