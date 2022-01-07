// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>

#include "benchmark/backend.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

void benchmark(backend::Backend *backend, const Tensor<float> &lowResImgs,
    const Tensor<float> &hiResImgs, std::size_t numIterations);

}
