// Copyright 2021 Ivanov Viktor

#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 26812)
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace benchmark {

namespace trt {

struct CudaException : std::exception {
	CudaException() : CudaException("CUDA general failure") {
	}
	explicit CudaException(::cudaError_t error)
	    : CudaException(::cudaGetErrorString(error)) {
	}
	explicit CudaException(const char *msg) : std::exception(msg) {
	}
};

struct TrtException : std::exception {
	TrtException() : TrtException("TensorRT general failure") {
	}
	explicit TrtException(const std::string &msg) : TrtException(msg.c_str()) {
	}
	explicit TrtException(const char *msg) : std::exception(msg) {
	}
};

struct TrtParserException : TrtException {
	explicit TrtParserException(const nvonnxparser::IParser &parser)
	    : TrtException(parser.getNbErrors() > 0
	                       ? parser.getError(0)->desc()
	                       : "TensorRT parser general failure") {
	}
};

template <typename T>
T *throwIfNull(T *val) {
	if (val == nullptr) {
		throw TrtException();
	}
	return val;
}

inline void cudaCheck(::cudaError_t error) {
	if (error != ::cudaSuccess) {
		throw CudaException(error);
	}
}

template <typename T>
struct TrtPtr : std::unique_ptr<T> {
	using unique_ptr = std::unique_ptr<T>;
	explicit TrtPtr(nullptr_t) : unique_ptr(nullptr) {
	}
	explicit TrtPtr(T *obj) : unique_ptr(throwIfNull(obj)) {
	}

	TrtPtr(const TrtPtr &) = delete;
	TrtPtr(TrtPtr &&) noexcept = default;
	TrtPtr &operator=(const TrtPtr &) = delete;
	TrtPtr &operator=(TrtPtr &&) noexcept = default;

	operator T *() const {
		return get();
	}
};

template <typename T>
struct CudaDeviceBuffer : std::unique_ptr<float, decltype(&::cudaFree)> {
	using unique_ptr = std::unique_ptr<float, decltype(&::cudaFree)>;

	explicit CudaDeviceBuffer(std::size_t size)
	    : unique_ptr(alloc(size), &::cudaFree) {
	}
	CudaDeviceBuffer(const CudaDeviceBuffer &) = delete;
	CudaDeviceBuffer(CudaDeviceBuffer &&) noexcept = default;

	CudaDeviceBuffer &operator=(const CudaDeviceBuffer &) = delete;
	CudaDeviceBuffer &operator=(CudaDeviceBuffer &&) noexcept = default;

	T &operator*() = delete;
	T *operator->() = delete;

private:
	static T *alloc(std::size_t size) {
		void *result;
		cudaCheck(::cudaMalloc(&result, size * sizeof(T)));
		return reinterpret_cast<T *>(result);
	}
};

class CudaStream {
public:
	CudaStream() {
		cudaCheck(::cudaStreamCreate(&m_Stream));
	}
	explicit CudaStream(nullptr_t) {
	}
	CudaStream(const CudaStream &) = delete;
	CudaStream(CudaStream &&s) noexcept {
		m_Stream = s.m_Stream;
		s.m_Stream = nullptr;
	}
	~CudaStream() {
		if (m_Stream != nullptr) {
			::cudaStreamDestroy(m_Stream);
		}
	}

	CudaStream &operator=(const CudaStream &) = delete;
	CudaStream &operator=(CudaStream &&s) noexcept {
		if (this != &s) {
			this->~CudaStream();
			new (this) CudaStream(std::move(s));
		}
	}

	::cudaStream_t get() const {
		return get();
	}

	operator ::cudaStream_t() const {
		return m_Stream;
	}

private:
	::cudaStream_t m_Stream = nullptr;
};

}  // namespace trt

}  // namespace benchmark
