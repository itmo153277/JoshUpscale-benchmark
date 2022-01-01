// Copyright 2021 Ivanov Viktor

#include "benchmark/cmdline.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace benchmark {

namespace cmdline {

void parseArguments(const std::vector<ArgOption> &options,
    const std::vector<ArgDef> &positional, int argc, char *argv[]) {
	for (const auto &posArg : positional) {
		if (posArg.type != ArgType::STRING) {
			throw std::invalid_argument("Wrong positional argument type");
		}
	}
	auto positionalArg = positional.begin();
	for (char **argp = argv + 1, **argEnd = argv + argc; argp != argEnd;
	     argp++) {
		bool optionFound = false;
		if (**argp == '-') {
			for (const auto &option : options) {
				if (option.option == *argp) {
					if (option.definition.type == ArgType::BOOL) {
						*std::get<bool *>(option.definition.data) = true;
					} else if (option.definition.type == ArgType::STRING) {
						argp++;
						if (argp == argEnd) {
							throw ArgParseException(
							    "Missing argument for " + option.option);
						}
						*std::get<std::optional<std::string> *>(
						    option.definition.data) = *argp;
					}
					if (option.abortIfPresent) {
						return;
					}
					optionFound = true;
					break;
				}
			}
			if (!optionFound) {
				throw ArgParseException(
				    std::string() + "Unrecognized option: " + *argp);
			}
			continue;
		}
		if (positionalArg == positional.end()) {
			throw ArgParseException(
			    std::string() + "Unrecognized positional argument: " + *argp);
		}
		*std::get<std::optional<std::string> *>(positionalArg->data) = *argp;
		positionalArg++;
	}
}

}  // namespace cmdline

}  // namespace benchmark
