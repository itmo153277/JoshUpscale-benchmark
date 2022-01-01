// Copyright 2021 Ivanov Viktor

#pragma once

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace benchmark {

using path_type = std::filesystem::path::value_type;

std::string formatString(const char *format, ...);

struct ExceptionWithIdBase {
	virtual const std::type_info &type_info() const noexcept = 0;
};

template <typename T>
struct ExceptionWithId : T, ExceptionWithIdBase {
	explicit ExceptionWithId(T base) : T(std::move(base)) {
	}

	const std::type_info &type_info() const noexcept override {
		return typeid(T);
	}
};

template <typename T>
ExceptionWithId<std::decay_t<T>> make_exception_with_id(T &&exception) {
	return ExceptionWithId<std::decay_t<T>>{std::forward<T>(exception)};
}

template <typename T>
[[noreturn]] void throw_with_nested_id(T &&exception) {
	std::throw_with_nested(make_exception_with_id(std::forward<T>(exception)));
}

void printException(
    std::ostream &os, const std::exception &e, const std::type_info &info);

template <typename T>
void printException(std::ostream &os, const T &e) {
	printException(os, e, typeid(e));
}

void printException(std::ostream &os);

std::string getExceptionString();

}  // namespace benchmark
