// Copyright 2021 Ivanov Viktor

#pragma once

#include "benchmark/backend.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

void benchmark(backend::Backend *backend, const Tensor<float> &lowResImgs,
    const Tensor<float> &hiResImgs, const path_type *profilePath);

}
