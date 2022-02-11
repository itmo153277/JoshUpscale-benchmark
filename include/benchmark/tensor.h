// Copyright 2022 Ivanov Viktor

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace benchmark {

using TensorDim = std::uint_fast32_t;
using TensorStride = std::make_signed_t<TensorDim>;
using TensorStrides = std::array<TensorStride, 2>;

struct TensorStorage {
	virtual void *getPtr() const = 0;
	virtual ~TensorStorage() {
	}
};

using TensorStoragePtr = std::shared_ptr<TensorStorage>;

class TensorShape {
public:
	TensorShape() = default;
	TensorShape(TensorDim batchSize, TensorDim height, TensorDim width)
	    : m_BatchSize(batchSize), m_Height(height), m_Width(width) {
	}
	TensorShape(const TensorShape &) = default;
	TensorShape(TensorShape &&) noexcept = default;

	TensorShape &operator=(const TensorShape &) = default;
	TensorShape &operator=(TensorShape &&) noexcept = default;

	TensorDim getBatchSize() const {
		return m_BatchSize;
	}
	TensorDim getHeight() const {
		return m_Height;
	}
	TensorDim getWidth() const {
		return m_Width;
	}
	TensorDim getSize() const {
		return m_BatchSize * m_Height * m_Width * 3;
	}
	TensorStrides getPlainStrides() const {
		return {static_cast<TensorStride>(m_Height * m_Width * 3),
		    static_cast<TensorStride>(m_Width * 3)};
	}

private:
	TensorDim m_BatchSize;
	TensorDim m_Height;
	TensorDim m_Width;
};

namespace detail {

template <typename T>
struct FastImpl;

std::tuple<std::unique_ptr<TensorStorage>, TensorStrides> allocOpt(
    const TensorShape &shape, std::size_t elementSize);
std::unique_ptr<TensorStorage> allocPlain(
    const TensorShape &shape, std::size_t elementSize);

}  // namespace detail

template <typename T>
class Tensor {
	static_assert(std::is_trivial_v<T> && std::is_copy_assignable_v<T>,
	    "T has to be trivial");

private:
	template <typename Val = T>
	struct BaseIter {
		using iterator_category = std::forward_iterator_tag;
		using value_type = Val;
		using reference = void;
		using pointer = void;
		using difference_type = int;

		BaseIter(const BaseIter &) = default;
		BaseIter(BaseIter &&) noexcept = default;

		BaseIter &operator=(const BaseIter &) = default;
		BaseIter &operator=(BaseIter &&) noexcept = default;

		bool operator!=(const BaseIter &s) const {
			return m_BatchNumber != s.m_BatchNumber || m_Height != s.m_Height ||
			       m_Width != s.m_Width || m_Channel != s.m_Channel;
		}
		BaseIter &operator++() {
			++m_Ptr;
			++m_Channel;
			if (m_Channel >= 3) {
				m_Channel = 0;
				++m_Width;
			}
			if (m_Width >= m_Tensor->getShape().getWidth()) {
				m_Width = 0;
				++m_Height;
				m_Ptr += m_Tensor->m_IncrementalStrides[1];
			}
			if (m_Height >= m_Tensor->getShape().getHeight()) {
				m_Height = 0;
				++m_BatchNumber;
				m_Ptr += m_Tensor->m_IncrementalStrides[0];
			}
			return *this;
		}
		BaseIter operator++(int) {
			auto copy = *this;
			++(*this);
			return copy;
		}
		value_type &operator*() {
			return *m_Ptr;
		}
		difference_type operator-(const BaseIter &s) {
			assert(s.m_Tensor == m_Tensor);
			return static_cast<difference_type>(getOffset() - s.getOffset());
		}

	private:
		BaseIter(Tensor *tensor, TensorDim batchNumber)
		    : m_Tensor(tensor)
		    , m_Ptr(tensor->m_Ptr + batchNumber * static_cast<std::intptr_t>(
		                                              tensor->m_Strides[0]))
		    , m_BatchNumber(batchNumber)
		    , m_Height(0)
		    , m_Width(0)
		    , m_Channel(0) {
		}

		Tensor *m_Tensor;
		Val *m_Ptr;
		TensorDim m_BatchNumber;
		TensorDim m_Height;
		TensorDim m_Width;
		TensorDim m_Channel;

		TensorDim getOffset() const {
			return m_Channel +
			       (m_Width +
			           (m_Height +
			               m_BatchNumber * m_Tensor->getShape().getHeight()) *
			               m_Tensor->getShape().getWidth()) *
			           3;
		}

		friend class Tensor;
	};

public:
	using iterator = BaseIter<T>;
	using const_iterator = BaseIter<const T>;

	Tensor(const TensorShape &shape, T *ptr, TensorStrides strides,
	    const TensorStoragePtr &storage)
	    : m_Storage(storage), m_Ptr(ptr), m_Strides(strides), m_Shape(shape) {
		updateIncrementalStrides();
	}
	explicit Tensor(const TensorShape &shape) : m_Shape(shape) {
		std::tie(m_Storage, m_Strides) = detail::allocOpt(m_Shape, sizeof(T));
		m_Ptr = reinterpret_cast<T *>(m_Storage->getPtr());
		updateIncrementalStrides();
		// TODO(viktprog): Storage left uninitialized. Reading from the tensor
		// is UB until assign() is called
	}
	Tensor(const Tensor &) = delete;
	Tensor(Tensor &&) noexcept = default;

	Tensor &operator=(const Tensor &) = delete;
	Tensor &operator=(Tensor &&) noexcept = default;

	T *data() {
		return m_Ptr;
	}
	const T *data() const {
		return m_Ptr;
	}
	std::size_t size() const {
		return m_Shape.getSize() * sizeof(T);
	}

	const TensorShape &getShape() const {
		return m_Shape;
	}
	const TensorStrides &getStrides() const {
		return m_Strides;
	}
	const TensorStoragePtr &getStorage() const {
		return m_Storage;
	}

	template <typename Val = T>
	struct PixelIndexer;
	template <typename Val = T>
	struct RowIndexer;
	template <typename Val = T>
	struct ImageIndexer;

	template <typename Val>
	struct PixelIndexer {
		Val &operator[](TensorDim i) const {
			return m_Ptr[i];
		}

	private:
		explicit PixelIndexer(Val *ptr) : m_Ptr(ptr) {
		}
		Val *m_Ptr;

		friend struct Tensor::template RowIndexer<Val>;
	};

	template <typename Val>
	struct RowIndexer {
		PixelIndexer<Val> operator[](TensorDim i) const {
			TensorStride offset = i * 3;
			return PixelIndexer<Val>(m_Ptr + offset);
		}
		Val *data() const {
			return m_Ptr;
		}

	private:
		explicit RowIndexer(Val *ptr) : m_Ptr(ptr) {
		}
		Val *m_Ptr;

		friend struct Tensor::template ImageIndexer<Val>;
	};

	template <typename Val>
	struct ImageIndexer {
		RowIndexer<Val> operator[](TensorDim i) const {
			TensorStride offset = m_Tensor->m_Strides[1] * i;
			return RowIndexer<Val>(m_Ptr + offset);
		}
		Val *data() const {
			return m_Ptr;
		}

		Tensor getTensor() {
			static_assert(
			    !std::is_const_v<Val>, "Cannot create tensor for const value");
			return {{1, m_Tensor->getShape().getHeight(),
			            m_Tensor->getShape().getWidth()},
			    m_Ptr,
			    {static_cast<TensorStride>(
			         m_Tensor->m_Strides[1] * m_Tensor->getShape().getHeight()),
			        m_Tensor->m_Strides[1]},
			    m_Tensor->m_Storage};
		}

	private:
		ImageIndexer(Val *ptr, Tensor *tensor) : m_Ptr(ptr), m_Tensor(tensor) {
		}
		Val *m_Ptr;
		Tensor *m_Tensor;

		friend class Tensor;
	};

	ImageIndexer<T> operator[](TensorDim i) {
		TensorStride offset = m_Strides[0] * i;
		return ImageIndexer<T>(m_Ptr + offset, this);
	}
	ImageIndexer<const T> operator[](TensorDim i) const {
		TensorStride offset = m_Strides[0] * i;
		return ImageIndexer<const T>(
		    m_Ptr + offset, const_cast<Tensor *>(this));
	}

	iterator begin() {
		return {this, 0};
	}
	iterator end() {
		return {this, m_Shape.getBatchSize()};
	}
	const_iterator begin() const {
		return {const_cast<Tensor *>(this), 0};
	}
	const_iterator end() const {
		return {const_cast<Tensor *>(this), m_Shape.getBatchSize()};
	}

	template <typename U>
	void assign(const Tensor<U> &s) {
		static_assert(std::is_assignable_v<T &, U>, "Cannot assign values");
		assert(m_Shape.getSize() == s.m_Shape.getSize());
		detail::FastImpl<T>::copy(this, s);
	}

	bool isPlain() const {
		return m_IncrementalStrides[0] == 0 && m_IncrementalStrides[1] == 0;
	}

	Tensor copyOpt() const {
		Tensor result(getShape());
		result.assign(*this);
		return result;
	}
	Tensor copyPlain() const {
		TensorStoragePtr storage = detail::allocPlain(getShape(), sizeof(T));
		Tensor result(getShape(), reinterpret_cast<T *>(storage->getPtr()),
		    getShape().getPlainStrides(), storage);
		result.assign(*this);
		return result;
	}
	void convertToPlain() {
		if (isPlain()) {
			return;
		}
		*this = copyPlain();
	}
	Tensor duplicate() {
		return {m_Shape, m_Ptr, m_Strides, m_Storage};
	}

private:
	TensorStoragePtr m_Storage = nullptr;
	T *m_Ptr = nullptr;
	TensorStrides m_Strides = {};
	TensorStrides m_IncrementalStrides = {};
	TensorShape m_Shape;

	void updateIncrementalStrides() {
		m_IncrementalStrides[0] =
		    m_Strides[0] - m_Strides[1] * m_Shape.getHeight();
		m_IncrementalStrides[1] = m_Strides[1] - m_Shape.getWidth() * 3;
	}
};

template <typename T>
std::vector<Tensor<T>> unbatch(Tensor<T> *tensor) {
	std::vector<Tensor<T>> result;
	TensorDim batchSize = tensor->getShape().getBatchSize();
	result.reserve(batchSize);
	for (TensorDim i = 0; i < batchSize; ++i) {
		result.push_back((*tensor)[i].getTensor());
	}
	return result;
}

template <typename T>
Tensor<T> batch(const std::vector<Tensor<T>> &tensors) {
	assert(tensors.size() > 0);
	auto shape = tensors[0].getShape();
	assert(shape.getBatchSize() == 1);
	Tensor<T> result({static_cast<TensorDim>(tensors.size()), shape.getHeight(),
	    shape.getWidth()});
	for (TensorDim batch = 0, batchSize = result.getShape().getBatchSize();
	     batch < batchSize; ++batch) {
		auto &tensor = tensors[batch];
		assert(tensor.getShape() == shape);
		result[batch].getTensor().assign(tensor);
	}
	return result;
}

namespace detail {

template <typename T>
struct SimpleImpl {
	template <typename U>
	static void copy(Tensor<T> *to, const Tensor<U> &from) {
		if constexpr (std::is_same_v<U, T>) {
			if (to->isPlain() && from.isPlain()) {
				std::memcpy(to->data(), from.data(), from.size());
				return;
			}
		}
		std::copy(from.begin(), from.end(), to->begin());
	}
};

template <typename T>
struct FastImpl {
	template <typename U>
	static void copy(Tensor<T> *to, const Tensor<U> &from) {
		SimpleImpl<T>::copy(to, from);
	}
};

}  // namespace detail

}  // namespace benchmark
