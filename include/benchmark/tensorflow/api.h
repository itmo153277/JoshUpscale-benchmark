// Copyright 2021 Ivanov Viktor

#pragma once

#ifdef _MSC_VER
// Prefer 'enum class' over 'enum' (Enum.3)
#pragma warning(disable : 26812)
#pragma warning(push)
// 'identifier1' has C-linkage specified, but returns UDT 'identifier2' which is
// incompatible with C
#pragma warning(disable : 4190)
// dereferencing NULL pointer <name>
#pragma warning(disable : 6011)
// 'realloc' may return null pointer: assigning a null pointer to <variable>,
// which is passed as an argument to 'realloc', will cause the original memory
// block to be leaked
#pragma warning(disable : 6308)
// buffer overrun: accessing <buffer name>, the writable size is <size1> bytes,
// but <size2> bytes may be written: Lines: x, y
#pragma warning(disable : 6386)
// <argument> may be <value>: this does not adhere to the specification for the
// function <function name>: Lines: x, y
#pragma warning(disable : 6387)
#endif
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cassert>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace benchmark {

namespace tensorflow {

class TF_Exception : public std::exception {
public:
	explicit TF_Exception(const ::TF_Status *status)
	    : m_Msg(::TF_Message(status)) {
	}

	const char *what() const noexcept override {
		return m_Msg.c_str();
	}

private:
	std::string m_Msg;
};

#define DEFINE_SMART_TF_CLASS(TF_Class, TF_Constructor, TF_Destructor)  \
	struct TF_Class                                                     \
	    : std::unique_ptr<::TF_Class, decltype(&::TF_Destructor)> {     \
		using unique_ptr =                                              \
		    std::unique_ptr<::TF_Class, decltype(&::TF_Destructor)>;    \
                                                                        \
		TF_Class() : unique_ptr(::TF_Constructor(), &::TF_Destructor) { \
		}                                                               \
		TF_Class(const TF_Class &) = delete;                            \
		TF_Class(TF_Class &&) noexcept = default;                       \
                                                                        \
		TF_Class &operator=(const TF_Class &) = delete;                 \
		TF_Class &operator=(TF_Class &&) = default;                     \
                                                                        \
		operator ::TF_Class *() const {                                 \
			return get();                                               \
		}                                                               \
	};

DEFINE_SMART_TF_CLASS(TF_Status, TF_NewStatus, TF_DeleteStatus);

DEFINE_SMART_TF_CLASS(TF_Graph, TF_NewGraph, TF_DeleteGraph);

DEFINE_SMART_TF_CLASS(TF_ImportGraphDefOptions, TF_NewImportGraphDefOptions,
    TF_DeleteImportGraphDefOptions);

DEFINE_SMART_TF_CLASS(
    TF_SessionOptions, TF_NewSessionOptions, TF_DeleteSessionOptions);

#undef DEFINE_SMART_TF_CLASS

class TF_Buffer {
public:
	explicit TF_Buffer(std::size_t size)
	    : m_Data(std::make_unique<char[]>(size)) {
		m_Buffer = ::TF_NewBuffer();
		m_Buffer->data = m_Data.get();
		m_Buffer->length = size;
		m_Buffer->data_deallocator = nullptr;
	}
	TF_Buffer(const TF_Buffer &) = delete;
	TF_Buffer(TF_Buffer &&s) noexcept : m_Data(std::move(s.m_Data)) {
		if (this != &s) {
			m_Buffer = s.m_Buffer;
			s.m_Buffer = nullptr;
		}
	}
	~TF_Buffer() {
		if (m_Buffer) {
			TF_DeleteBuffer(m_Buffer);
		}
	}

	TF_Buffer &operator=(const TF_Buffer &) = delete;
	TF_Buffer &operator=(TF_Buffer &&s) noexcept {
		if (this != &s) {
			this->~TF_Buffer();
			new (this) TF_Buffer(std::move(s));
		}
		return *this;
	}
	operator ::TF_Buffer *() const {
		return m_Buffer;
	}
	operator bool() const {
		return m_Buffer != nullptr;
	}
	::TF_Buffer &operator*() const {
		return *m_Buffer;
	}
	::TF_Buffer *operator->() const {
		return m_Buffer;
	}
	::TF_Buffer *get() const {
		return m_Buffer;
	}
	char *getData() const {
		return m_Data.get();
	}

private:
	::TF_Buffer *m_Buffer = nullptr;
	std::unique_ptr<char[]> m_Data;
};

class TF_BufferUnmanaged {
public:
	TF_BufferUnmanaged() {
		m_Buffer = ::TF_NewBuffer();
		m_Buffer->data = nullptr;
		m_Buffer->length = 0;
		m_Buffer->data_deallocator = nullptr;
	}
	TF_BufferUnmanaged(const void *p, std::size_t size) {
		m_Buffer = ::TF_NewBufferFromString(p, size);
	}
	explicit TF_BufferUnmanaged(nullptr_t) {
	}
	TF_BufferUnmanaged(const TF_BufferUnmanaged &) = delete;
	TF_BufferUnmanaged(TF_BufferUnmanaged &&s) noexcept {
		if (this != &s) {
			m_Buffer = s.m_Buffer;
			s.m_Buffer = nullptr;
		}
	}
	~TF_BufferUnmanaged() {
		if (m_Buffer) {
			TF_DeleteBuffer(m_Buffer);
		}
	}

	TF_BufferUnmanaged &operator=(const TF_BufferUnmanaged &) = delete;
	TF_BufferUnmanaged &operator=(TF_BufferUnmanaged &&s) noexcept {
		if (this != &s) {
			this->~TF_BufferUnmanaged();
			new (this) TF_BufferUnmanaged(std::move(s));
		}
		return *this;
	}
	operator ::TF_Buffer *() const {
		return m_Buffer;
	}
	operator bool() const {
		return m_Buffer != nullptr;
	}
	::TF_Buffer &operator*() const {
		return *m_Buffer;
	}
	::TF_Buffer *operator->() const {
		return m_Buffer;
	}
	::TF_Buffer *get() const {
		return m_Buffer;
	}

	static TF_BufferUnmanaged own(::TF_Buffer *buffer) {
		return TF_BufferUnmanaged{buffer};
	}

private:
	explicit TF_BufferUnmanaged(::TF_Buffer *buffer) : m_Buffer(buffer) {
	}

	::TF_Buffer *m_Buffer = nullptr;
};

template <typename T>
class TF_Tensor;

template <>
class TF_Tensor<float> {
public:
	explicit TF_Tensor(const benchmark::TensorShape &shape) {
		std::size_t len = sizeof(float);
		std::vector<std::int64_t> dims(shape.begin(), shape.end());
		for (auto &dim : shape) {
			len *= dim;
		}
		m_Tensor = ::TF_AllocateTensor(
		    ::TF_FLOAT, dims.data(), static_cast<int>(dims.size()), len);
	}
	~TF_Tensor() {
		if (m_Tensor != nullptr) {
			::TF_DeleteTensor(m_Tensor);
		}
	}
	TF_Tensor(const TF_Tensor &) = delete;
	TF_Tensor(TF_Tensor &&s) noexcept {
		if (this != &s) {
			m_Tensor = s.m_Tensor;
			s.m_Tensor = nullptr;
		}
	}

	TF_Tensor &operator=(const TF_Tensor &) = delete;
	TF_Tensor &operator=(TF_Tensor &&s) noexcept {
		if (this != &s) {
			this->~TF_Tensor();
			new (this) TF_Tensor(std::move(s));
		}
		return *this;
	}
	operator ::TF_Tensor *() const {
		return m_Tensor;
	}
	operator bool() const {
		return m_Tensor != nullptr;
	}
	::TF_Tensor *get() const {
		return m_Tensor;
	}
	float *begin() const {
		return reinterpret_cast<float *>(::TF_TensorData(m_Tensor));
	}
	float *end() const {
		return reinterpret_cast<float *>(::TF_TensorData(m_Tensor)) +
		       ::TF_TensorElementCount(m_Tensor);
	}
	std::size_t size() const {
		return static_cast<std::size_t>(::TF_TensorElementCount(m_Tensor));
	}

	static TF_Tensor own(::TF_Tensor *tensor) {
		return TF_Tensor(tensor);
	}
	static TF_Tensor createFromTensor(const benchmark::Tensor<float> &tensor) {
		TF_Tensor newTensor(tensor.shape);
		std::memcpy(::TF_TensorData(newTensor), tensor.data.data(),
		    ::TF_TensorByteSize(newTensor));
		return newTensor;
	}
	TF_Tensor &copyFromTensor(const benchmark::Tensor<float> &tensor) {
		assert(m_Tensor != nullptr);
		assert(tensor.data.size() ==
		       static_cast<std::size_t>(::TF_TensorElementCount(m_Tensor)));
		std::memcpy(::TF_TensorData(m_Tensor), tensor.data.data(),
		    ::TF_TensorByteSize(m_Tensor));
		return *this;
	}
	TF_Tensor &copyFromTensor(::TF_Tensor *tensor) {
		assert(m_Tensor != nullptr);
		assert(tensor != nullptr);
		assert(::TF_TensorByteSize(tensor) == ::TF_TensorByteSize(m_Tensor));
		assert(::TF_TensorType(tensor) == ::TF_TensorType(m_Tensor));
		std::memcpy(::TF_TensorData(m_Tensor), ::TF_TensorData(tensor),
		    ::TF_TensorByteSize(m_Tensor));
		return *this;
	}

private:
	explicit TF_Tensor(::TF_Tensor *tensor) : m_Tensor(tensor) {
	}
	::TF_Tensor *m_Tensor = nullptr;
};

struct TF_SessionOptionsProto {
	const void *proto;
	std::size_t size;
};

struct TF_Session {
	TF_Session(const TF_Graph &graph, const TF_SessionOptionsProto *options,
	    bool xla = false);
	TF_Session(const TF_Session &) = delete;
	TF_Session(TF_Session &&s) noexcept {
		if (this != &s) {
			m_Session = s.m_Session;
			s.m_Session = nullptr;
		}
	}
	~TF_Session() {
		if (m_Session) {
			TF_Status status;
			::TF_CloseSession(m_Session, status);
			::TF_DeleteSession(m_Session, status);
		}
	}

	TF_Session &operator=(const TF_Session &) = delete;
	TF_Session &operator=(TF_Session &&s) noexcept {
		if (this != &s) {
			this->~TF_Session();
			new (this) TF_Session(std::move(s));
		}
		return *this;
	}
	operator ::TF_Session *() const {
		return m_Session;
	}
	operator bool() const {
		return m_Session != nullptr;
	}
	::TF_Session *get() {
		return m_Session;
	}

	TF_Tensor<float> run(const std::vector<::TF_Output> &inputOp,
	    const std::vector<::TF_Tensor *> &inputValue, const TF_Output &outputOp,
	    bool profile, ::TF_Buffer *runMetadata);

private:
	::TF_Session *m_Session = nullptr;
};

TF_Graph readGraph(const path_type *fileName);

};  // namespace tensorflow

}  // namespace benchmark
