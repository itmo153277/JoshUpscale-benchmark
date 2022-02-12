// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstdint>
#include <vector>

#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

Tensor<std::uint8_t> readPng(const path_type *fileName);

}  // namespace benchmark
