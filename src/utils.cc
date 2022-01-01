// Copyright 2021 Ivanov Viktor

#include "benchmark/utils.h"

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#include <memory>
#endif

namespace benchmark {

std::string formatString(const char *format, ...) {
	va_list args;
	va_start(args, format);
	std::size_t size = std::vsnprintf(nullptr, 0, format, args);
	std::string result(size, 0);
	std::vsnprintf(result.data(), size + 1, format, args);
	va_end(args);
	return result;
}

namespace {
std::string demangle(const char *name) {
#if defined(__GNUG__)
	int status = EXIT_FAILURE;
	std::unique_ptr<char, void (*)(void *)> res{
	    abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

	return (status == EXIT_SUCCESS) ? res.get() : name;
#elif defined(_MSC_VER)
	static const char *prefixes[] = {"class ", "struct ", "union ", ""};
	static const std::size_t prefixLens[] = {6, 7, 6, 0};
	const char **prefixPtr = prefixes;
	const std::size_t *prefixLenPtr = prefixLens;
	for (; **prefixPtr != 0; prefixPtr++, prefixLenPtr++) {
		if (std::strncmp(name, *prefixPtr, *prefixLenPtr) == 0) {
			return name + *prefixLenPtr;
		}
	}
	return name;
#else
	return name;
#endif
}
}  // namespace

void printException(
    std::ostream &os, const std::exception &e, const type_info &info) {
	os << demangle(info.name()) << ": " << e.what();
	try {
		std::rethrow_if_nested(e);
	} catch (...) {
		os << "\n  ";
		printException(os);
	}
}

void printException(std::ostream &os) {
	try {
		throw;
	} catch (std::exception &e) {
		try {
			throw;
		} catch (ExceptionWithIdBase &id) {
			printException(os, e, id.type_info());
		} catch (...) {
			printException(os, e);
		}
	} catch (...) {
		os << "Unknown error";
	}
}

std::string getExceptionString() {
	std::ostringstream ss;
	printException(ss);
	return ss.str();
}

}  // namespace benchmark
