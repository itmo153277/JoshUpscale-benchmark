// Copyright 2022 Ivanov Viktor

#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 26451)
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
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
	virtual void *getPtr() = 0;
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
	TensorShape(const TensorDim (&shape)[3])  // NOLINT(runtime/explicit)
	    : TensorShape(shape[0], shape[1], shape[2]) {
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

std::tuple<TensorStoragePtr, TensorStrides> allocateOptimal(
    const TensorShape &shape, std::size_t elementSize);
TensorStoragePtr allocatePlain(
    const TensorShape &shape, std::size_t elementSize);

}  // namespace detail

template <typename T>
class Tensor {
private:
	template <typename Val = T, typename TensorType = Tensor>
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
			return m_BatchNUmber != s.m_BatchNUmber || m_Height != s.m_Height ||
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
				++m_BatchNUmber;
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
		BaseIter(TensorType *tensor, TensorDim batchNumber)
		    : m_Tensor(tensor)
		    , m_Ptr(tensor->data() + batchNumber * tensor->m_Strides[0])
		    , m_BatchNUmber(batchNumber)
		    , m_Height(0)
		    , m_Width(0)
		    , m_Channel(0) {
		}

		TensorType *m_Tensor;
		Val *m_Ptr;
		TensorDim m_BatchNUmber;
		TensorDim m_Height;
		TensorDim m_Width;
		TensorDim m_Channel;

		TensorDim getOffset() const {
			return m_Channel +
			       (m_Width +
			           (m_Height +
			               m_BatchNUmber * m_Tensor->getShape().getHeight()) *
			               m_Tensor->getShape().getWidth()) *
			           3;
		}

		friend class Tensor;
	};

public:
	using iterator = BaseIter<T, Tensor>;
	using const_iterator = BaseIter<const T, const Tensor>;

	Tensor(const TensorShape &shape, T *ptr, TensorStrides strides,
	    const TensorStoragePtr &storage)
	    : m_Storage(storage), m_Ptr(ptr), m_Strides(strides), m_Shape(shape) {
		updateIncrementalStrides();
	}
	explicit Tensor(const TensorShape &shape) : m_Shape(shape) {
		std::tie(m_Storage, m_Strides) =
		    detail::allocateOptimal(m_Shape, sizeof(T));
		updateIncrementalStrides();
	}
	Tensor(const Tensor &) = default;
	Tensor(Tensor &&) noexcept = default;

	Tensor &operator=(const Tensor &) = default;
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
		Val &operator[](TensorDim i) {
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
		PixelIndexer<Val> operator[](TensorDim i) {
			return PixelIndexer<Val>(m_Ptr + i * 3);
		}
		template <int size>
		auto operator[](const TensorDim (&i)[size]) {
			static_assert(size == 2);
			return (*this)[i[0]][i[1]];
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
		RowIndexer<Val> operator[](TensorDim i) {
			return RowIndexer<Val>(m_Ptr + m_Tensor->m_Strides[1] * i);
		}
		template <int size>
		auto operator[](const TensorDim (&i)[size]) {
			static_assert(size >= 2 && size <= 3);
			if constexpr (size == 2) {
				return (*this)[i[0]][i[1]];
			} else {
				return (*this)[i[0]][i[1]][i[2]];
			}
		}
		Val *data() const {
			return m_Ptr;
		}

		Tensor<Val> getTensor() {
			return Tensor{{1, m_Tensor->getShape().getHeight(),
			                  m_Tensor->getShape().getWidth()},
			    const_cast<T *>(m_Ptr),
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
		return ImageIndexer<T>(m_Ptr + m_Strides[0] * i, this);
	}
	template <int size>
	auto operator[](const TensorDim (&i)[size]) {
		static_assert(size >= 2 && size <= 4);
		if constexpr (size == 2) {
			return (*this)[i[0]][i[1]];
		} else if constexpr (size == 3) {
			return (*this)[i[0]][i[1]][i[2]];
		} else {
			return (*this)[i[0]][i[1]][i[2]][i[3]];
		}
	}
	ImageIndexer<const T> operator[](TensorDim i) const {
		return ImageIndexer<const T>(
		    m_Ptr + m_Strides[0] * i, const_cast<Tensor *>(this));
	}
	template <int size>
	auto operator[](const TensorDim (&i)[size]) const {
		static_assert(size >= 2 && size <= 4);
		if constexpr (size == 2) {
			return (*this)[i[0]][i[1]];
		} else if constexpr (size == 3) {
			return (*this)[i[0]][i[1]][i[2]];
		} else {
			return (*this)[i[0]][i[1]][i[2]][i[3]];
		}
	}

	iterator begin() {
		return {this, 0};
	}
	iterator end() {
		return {this, m_Shape.getBatchSize()};
	}
	const_iterator begin() const {
		return {this, 0};
	}
	const_iterator end() const {
		return {this, m_Shape.getBatchSize()};
	}

	void copyFrom(const Tensor &s) {
		assert(m_Shape.getSize() == s.m_Shape.getSize());
		detail::FastImpl<T>::copy(this, s);
	}

	bool isPlain() const {
		return m_IncrementalStrides[0] == 0 && m_IncrementalStrides[1] == 0;
	}

	static Tensor copyOptimal(const Tensor &s) {
		Tensor result(s.getShape());
		result.copyFrom(s);
		return result;
	}
	static Tensor copyPlain(const Tensor &s) {
		auto storage = detail::allocatePlain(s.getShape(), sizeof(T));
		Tensor result(s.getShape(), reinterpret_cast<T *>(storage->getPtr()),
		    s.getShape().getPlainStrides(), storage);
		result.copyFrom(s);
		return result;
	}
	void convertToPlain() {
		if (isPlain()) {
			return;
		}
		*this = copyPlain(*this);
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
Tensor<T> convertRgbBgr(const Tensor<T> &tensor) {
	Tensor<T> result(tensor.getShape());
	detail::FastImpl<T>::convertRgbBgr(&result, tensor);
	return result;
}

namespace detail {

template <typename T>
struct SimpleImpl {
	static void copy(Tensor<T> *to, const Tensor<T> &from) {
		std::copy(from.begin(), from.end(), to->begin());
	}

	static void convertRgbBgr(Tensor<T> *to, const Tensor<T> &from) {
		for (TensorDim batch = 0; batch < from.getShape().getBatchSize();
		     ++batch) {
			for (TensorDim y = 0; y < from.getShape().getHeight(); ++y) {
				T *toPtr = (*to)[batch][y].data();
				const T *fromPtr = from[batch][y].data();
				for (TensorDim x = 0; x < from.getShape().getWidth();
				     ++x, toPtr += 3, fromPtr += 3) {
					toPtr[0] = fromPtr[2];
					toPtr[1] = fromPtr[1];
					toPtr[0] = fromPtr[0];
				}
			}
		}
	}
};

template <typename T>
struct FastImpl {
	static void copy(Tensor<T> *to, const Tensor<T> &from) {
		SimpleImpl<T>::copy(to, from);
	}

	static void convertRgbBgr(Tensor<T> *to, const Tensor<T> &from) {
		SimpleImpl<T>::convertRgbBgr(to, from);
	}
};

}  // namespace detail

}  // namespace benchmark
