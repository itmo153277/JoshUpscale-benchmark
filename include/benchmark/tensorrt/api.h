// Copyright 2021 Ivanov Viktor

#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include <NvInfer.h>
#include <NvOnnxParser.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace benchmark {

namespace trt {

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

}  // namespace trt

}  // namespace benchmark
