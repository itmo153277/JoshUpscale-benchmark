// Copyright 2021 Ivanov Viktor

#include "benchmark/png.h"

#include <png.h>

#include <cassert>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace benchmark {

constexpr int kPngSignatureSize = 8;

struct PngStruct {
	::png_structp pngPtr;
	::png_infop pngInfoPtr;

	PngStruct() {
		pngPtr = ::png_create_read_struct(
		    PNG_LIBPNG_VER_STRING, nullptr,
		    []([[maybe_unused]] ::png_structp pngPtr,
		        ::png_const_charp errorStr) {
			    throw std::runtime_error(errorStr);
		    },
		    nullptr);
		if (pngPtr == nullptr) {
			throw std::bad_alloc();
		}
		pngInfoPtr = ::png_create_info_struct(pngPtr);
		if (pngInfoPtr == nullptr) {
			::png_destroy_read_struct(&pngPtr, nullptr, nullptr);
			throw std::bad_alloc();
		}
	}
	PngStruct(const PngStruct &) = delete;
	PngStruct(PngStruct &&s) noexcept
	    : pngPtr(s.pngPtr), pngInfoPtr(s.pngInfoPtr) {
		s.pngPtr = nullptr;
		s.pngInfoPtr = nullptr;
	}
	~PngStruct() {
		::png_destroy_read_struct(&pngPtr, &pngInfoPtr, nullptr);
	}
	PngStruct &operator=(const PngStruct &) = delete;
	PngStruct &operator=(PngStruct &&s) noexcept {
		if (this != &s) {
			this->~PngStruct();
			new (this) PngStruct(std::move(s));
		}
		return *this;
	}
};

Tensor<std::uint8_t> readPng(const char *fileName) {
	std::ifstream pngFile(fileName, std::ios::in | std::ios::binary);
	pngFile.exceptions(std::ios::failbit | std::ios::badbit);
	char header[kPngSignatureSize];
	pngFile.read(header, sizeof(header));
	if (::png_sig_cmp(reinterpret_cast<::png_const_bytep>(header), 0,
	        kPngSignatureSize) != 0) {
		throw std::runtime_error("Invalid PNG file");
	}
	PngStruct pngStruct;
	::png_set_read_fn(pngStruct.pngPtr, &pngFile,
	    [](png_structp pngPtr, png_bytep data, png_size_t length) {
		    std::ifstream *pngFile =
		        reinterpret_cast<std::ifstream *>(::png_get_io_ptr(pngPtr));
		    pngFile->read(reinterpret_cast<char *>(data),
		        static_cast<std::streamsize>(length));
	    });
	::png_set_sig_bytes(pngStruct.pngPtr, 8);
	::png_read_info(pngStruct.pngPtr, pngStruct.pngInfoPtr);
	::png_uint_32 imgWidth =
	    ::png_get_image_width(pngStruct.pngPtr, pngStruct.pngInfoPtr);
	::png_uint_32 imgHeight =
	    ::png_get_image_height(pngStruct.pngPtr, pngStruct.pngInfoPtr);
	::png_uint_32 bitdepth =
	    ::png_get_bit_depth(pngStruct.pngPtr, pngStruct.pngInfoPtr);
	::png_uint_32 channels =
	    ::png_get_channels(pngStruct.pngPtr, pngStruct.pngInfoPtr);
	::png_uint_32 colorType =
	    ::png_get_color_type(pngStruct.pngPtr, pngStruct.pngInfoPtr);

	if ((colorType & PNG_COLOR_MASK_PALETTE) == PNG_COLOR_TYPE_PALETTE) {
		::png_set_palette_to_rgb(pngStruct.pngPtr);
		channels = 3;
	}
	if ((colorType & PNG_COLOR_MASK_COLOR) == PNG_COLOR_TYPE_GRAY) {
		if (bitdepth < 8) {
			::png_set_expand_gray_1_2_4_to_8(pngStruct.pngPtr);
			bitdepth = 8;
		}
		::png_set_gray_to_rgb(pngStruct.pngPtr);
		channels = 3;
	}
	if (::png_get_valid(
	        pngStruct.pngPtr, pngStruct.pngInfoPtr, PNG_INFO_tRNS) != 0) {
		::png_set_tRNS_to_alpha(pngStruct.pngPtr);
		channels += 1;
		colorType |= PNG_COLOR_MASK_ALPHA;
	}
	if (bitdepth == 16) {
		::png_set_strip_16(pngStruct.pngPtr);
		bitdepth = 8;
	}
	if ((colorType & PNG_COLOR_MASK_ALPHA) != 0) {
		::png_set_strip_alpha(pngStruct.pngPtr);
		channels -= 1;
	}
	::png_read_update_info(pngStruct.pngPtr, pngStruct.pngInfoPtr);
	assert(bitdepth ==
	       ::png_get_bit_depth(pngStruct.pngPtr, pngStruct.pngInfoPtr));
	assert(
	    channels == ::png_get_channels(pngStruct.pngPtr, pngStruct.pngInfoPtr));
	if (bitdepth != 8 || channels != 3) {
		throw std::runtime_error("Unsupported PNG");
	}
	Tensor<std::uint8_t> pngImage({1, imgHeight, imgWidth});
	std::vector<::png_bytep> rowPointers(imgHeight);
	for (::png_uint_32 i = 0; i < imgHeight; ++i) {
		rowPointers[i] = reinterpret_cast<::png_bytep>(pngImage[0][i].data());
	}
	::png_read_image(pngStruct.pngPtr, rowPointers.data());
	return pngImage;
}

}  // namespace benchmark
