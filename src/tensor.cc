// Copyright 2022 Ivanov Viktor

#include "benchmark/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace benchmark {

namespace detail {

constexpr int defaultDataAlign = 16;

namespace {

void *aligned_alloc(std::size_t offset, std::size_t size) {
#if defined(_MSC_VER)
	return _aligned_malloc(size, offset);
#else
	return std::aligned_alloc(offset, size);
#endif
}

void aligned_free(void *ptr) {
#if defined(_MSC_VER)
	return _aligned_free(ptr);
#else
	return std::free(ptr);
#endif
}

}  // namespace

struct AlignedStorage : TensorStorage {
	explicit AlignedStorage(std::size_t size) {
		m_Ptr = aligned_alloc(defaultDataAlign, size);
		if (m_Ptr == nullptr) {
			throw std::bad_alloc();
		}
	}
	~AlignedStorage() {
		if (m_Ptr != nullptr) {
			aligned_free(m_Ptr);
		}
	}

	void *getPtr() const override {
		return m_Ptr;
	}

private:
	void *m_Ptr = nullptr;
};

struct PlainStorage : TensorStorage {
	explicit PlainStorage(std::size_t size) {
		m_Ptr = std::malloc(size);
		if (m_Ptr == nullptr) {
			throw std::bad_alloc();
		}
	}
	~PlainStorage() {
		if (m_Ptr != nullptr) {
			std::free(m_Ptr);
		}
	}

	void *getPtr() const override {
		return m_Ptr;
	}

private:
	void *m_Ptr = nullptr;
};

std::tuple<std::unique_ptr<TensorStorage>, TensorStrides> allocOpt(
    const TensorShape &shape, std::size_t elementSize) {
	auto strides = shape.getPlainStrides();
	strides[1] += defaultDataAlign - 1;
	strides[1] -= strides[1] % defaultDataAlign;
	strides[0] = static_cast<TensorStride>(strides[1] * shape.getHeight());
	TensorDim size = strides[0] * shape.getBatchSize();
	return {std::make_unique<AlignedStorage>(size * elementSize), strides};
}

std::unique_ptr<TensorStorage> allocPlain(
    const TensorShape &shape, std::size_t elementSize) {
	return std::make_unique<AlignedStorage>(shape.getSize() * elementSize);
}

}  // namespace detail

}  // namespace benchmark
