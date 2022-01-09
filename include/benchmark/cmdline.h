// Copyright 2021 Ivanov Viktor

#pragma once

#include <exception>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace benchmark {

namespace cmdline {

struct ArgParseException : std::exception {
	ArgParseException() : std::exception("Error during argument parsing") {
	}
	explicit ArgParseException(const std::string &msg)
	    : std::exception(msg.c_str()) {
	}
	explicit ArgParseException(const char *msg) : std::exception(msg) {
	}
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
