// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstdint>

#include "benchmark/tensor.h"

namespace benchmark {

Tensor<std::uint8_t> readPng(const char *fileName);

}  // namespace benchmark
