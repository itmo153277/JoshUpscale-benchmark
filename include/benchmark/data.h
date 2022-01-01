// Copyright 2021 Ivanov Viktor

#pragma once

#include "benchmark/tensor.h"

namespace benchmark {

namespace data {

enum class DataFormat { NHWC, NCHW };

Tensor<float> loadData(
    const char *mask, DataFormat format, const TensorShape &shape);
float diff(const Tensor<float> &left, const Tensor<float> &right);

}  // namespace data

}  // namespace benchmark
