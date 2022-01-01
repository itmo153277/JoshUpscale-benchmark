// Copyright 2021 Ivanov Viktor

#pragma once

#include <exception>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace benchmark {

namespace cmdline {

class ArgParseException : public std::exception {
public:
	ArgParseException() : ArgParseException("Error during argument parsing") {
	}
	explicit ArgParseException(const std::string &msg) : m_msg(msg) {
	}
	explicit ArgParseException(const char *msg) : m_msg(msg) {
	}
	const char *what() const noexcept override {
		return m_msg.c_str();
	}

private:
	std::string m_msg;
};

enum class ArgType { BOOL, STRING };

using ArgDataPtr =
    std::variant<nullptr_t, bool *, std::optional<std::string> *>;

struct ArgDef {
	ArgType type;
	ArgDataPtr data = nullptr;
};

struct ArgOption {
	std::string option;
	ArgDef definition;
	bool abortIfPresent = false;
};

void parseArguments(const std::vector<ArgOption> &options,
    const std::vector<ArgDef> &positional, int argc, char *argv[]);

}  // namespace cmdline

}  // namespace benchmark
