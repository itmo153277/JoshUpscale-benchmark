// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstdint>

#include "benchmark/tensor.h"

namespace benchmark {

namespace data {

Tensor<std::uint8_t> loadData(const char *mask, const TensorShape &shape);

float diff(const Tensor<std::uint8_t> &left, const Tensor<std::uint8_t> &right);

}  // namespace data

}  // namespace benchmark
