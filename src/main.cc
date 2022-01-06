// Copyright 2021 Ivanov Viktor

#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "benchmark/backend.h"
#include "benchmark/benchmark.h"
#include "benchmark/cmdline.h"
#include "benchmark/config.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace cmdline = benchmark::cmdline;
namespace config = benchmark::config;
namespace data = benchmark::data;
namespace backend = benchmark::backend;

using Tensor = benchmark::Tensor<float>;

struct CmdArguments {
	std::string profilePath = ".";
	std::string configPath;
	int returnValue = 0;
};

bool parseArguments(CmdArguments *argState, int argc, char *argv[]) {
	const char usage[] =
	    "Usage:\n"
	    "benchmark [options] config\n"
	    "\n"
	    "Positional arguments:\n"
	    "  config  Benchmark configuration file\n"
	    "\n"
	    "Options:\n"
	    "  -h,--help            Show this help\n"
	    "  --profile-path path  Set output directory for benchmark "
	    "profiles\n";
	bool showHelp = false;
	std::optional<std::string> profilePath = std::move(argState->profilePath);
	std::optional<std::string> configPath = std::nullopt;

	std::vector<cmdline::ArgOption> options = {
	    {
	        "--help",  // name
	        {
	            cmdline::ArgType::BOOL,  // type,
	            &showHelp                // data
	        },
	        true  // abortIfPresent
	    },
	    {
	        "-h",  // name
	        {
	            cmdline::ArgType::BOOL,  // type,
	            &showHelp                // data
	        },
	        true  // abortIfPresent
	    },
	    {
	        "--profile-path",  // name
	        {
	            cmdline::ArgType::STRING,  // type
	            &profilePath               // data
	        },
	        false  // abortIfPresent
	    }};
	std::vector<cmdline::ArgDef> positional = {{
	    cmdline::ArgType::STRING,  // type
	    &configPath                // data
	}};
	try {
		cmdline::parseArguments(options, positional, argc, argv);
		if (showHelp) {
			std::clog << usage << std::endl;
			argState->returnValue = 0;
			return false;
		}
		if (!configPath.has_value()) {
			throw cmdline::ArgParseException("Benchmark config is not set");
		}
	} catch (cmdline::ArgParseException &e) {
		std::clog << e.what() << "\n\n" << usage << std::endl;
		argState->returnValue = 1;
		return false;
	}
	argState->profilePath = std::move(profilePath.value());
	argState->configPath = std::move(configPath.value());
	return true;
}

std::pair<Tensor, Tensor> readData(const config::DataConfig &dataConfig) {
	auto lowResImgs = data::loadData(dataConfig.lowResPath.c_str(),
	    dataConfig.dataFormat, dataConfig.lowResShape);
	auto hiResImgs = data::loadData(dataConfig.hiResPath.c_str(),
	    dataConfig.dataFormat, dataConfig.hiResShape);
	if (hiResImgs.shape[0] != lowResImgs.shape[0]) {
		throw std::invalid_argument(
		    "Number of images for hi and low res must be equal");
	}
	return {lowResImgs, hiResImgs};
}

int main(int argc, char *argv[]) {
	try {
		CmdArguments argState;
		if (!parseArguments(&argState, argc, argv)) {
			return argState.returnValue;
		}
		std::filesystem::path configPath = argState.configPath;
		config::BenchmarkConfig benchmarkConfig =
		    config::readConfig(configPath.c_str());
		auto [lowResImgs, hiResImgs] = readData(benchmarkConfig.dataConfig);
		auto backend = backend::createBackend(
		    benchmarkConfig.backendConfig, lowResImgs, hiResImgs);
		std::filesystem::path profilePath = argState.profilePath;
		if (!benchmarkConfig.profileTag.empty()) {
			profilePath /= benchmarkConfig.profileTag;
		}
		std::filesystem::create_directories(profilePath);
		benchmark::benchmark(backend.get(), lowResImgs, hiResImgs,
		    benchmarkConfig.numIterations, profilePath.c_str());
	} catch (...) {
		std::clog << "Exception: " << benchmark::getExceptionString()
		          << std::endl;
		return 1;
	}
	return 0;
}
