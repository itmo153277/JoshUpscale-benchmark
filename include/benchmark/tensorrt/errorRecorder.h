// Copyright 2021 Ivanov Viktor

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>

#include "benchmark/logging.h"
#include "benchmark/tensorrt/api.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace trt {

class ErrorRecorder : public nvinfer1::IErrorRecorder {
public:
	using ErrorCode = nvinfer1::ErrorCode;

	std::int32_t getNbErrors() const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			if (m_HasValue) {
				return 1;
			}
		} catch (...) {
			printException();
		}
		return 0;
	}

	ErrorCode getErrorCode(std::int32_t errorIdx) const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			if (m_HasValue && errorIdx == 0) {
				return m_ErrorCode;
			}
		} catch (...) {
			printException();
		}
		return ErrorCode::kSUCCESS;
	}

	ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			if (m_HasValue && errorIdx == 0) {
				return m_ErrorDesc.c_str();
			}
		} catch (...) {
			printException();
		}
		return "";
	}

	bool hasOverflowed() const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			return m_HasOverflowed;
		} catch (...) {
			printException();
		}
		return false;
	}

	void clear() noexcept override {
		try {
			std::unique_lock guard{m_Mutex};
			m_HasValue = false;
			m_HasOverflowed = false;
		} catch (...) {
			printException();
		}
	}

	bool reportError(ErrorCode val, ErrorDesc desc) noexcept override {
		try {
			std::unique_lock guard{m_Mutex};
			if (m_HasValue) {
				m_HasOverflowed = true;
			} else {
				m_HasValue = true;
				m_ErrorCode = val;
				m_ErrorDesc = desc;
			}
		} catch (...) {
			printException();
		}
		return true;
	}

	RefCount incRefCount() noexcept override {
		return ++m_RefCount;
	}
	RefCount decRefCount() noexcept override {
		return --m_RefCount;
	}

	[[noreturn]] void rethrowException(TrtException *e) {
		try {
			std::unique_lock guard{m_Mutex};
			if (m_HasValue) {
				throw TrtException(m_ErrorDesc);
			}
		} catch (TrtException &) {
			throw;
		} catch (...) {
			printException();
		}
		throw *e;
	}

private:
	std::atomic<RefCount> m_RefCount{0};
	mutable std::shared_mutex m_Mutex;
	bool m_HasValue = false;
	bool m_HasOverflowed = false;
	ErrorCode m_ErrorCode{ErrorCode::kSUCCESS};
	std::string m_ErrorDesc;

	void printException() const noexcept {
		logError("ErrorRecorder")
		    << "Unhandled exception during error tracking: "
		    << getExceptionString();
	}
};

}  // namespace trt

}  // namespace benchmark
