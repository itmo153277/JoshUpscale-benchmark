// Copyright 2021 Ivanov Viktor

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "benchmark/cmdline.h"
#include "benchmark/config.h"
#include "benchmark/data.h"
#include "benchmark/logging.h"
#include "benchmark/tensor.h"
#include "benchmark/utils.h"

namespace cmdline = benchmark::cmdline;
namespace config = benchmark::config;
namespace data = benchmark::data;

using TensorShape = benchmark::TensorShape;
using Tensor = benchmark::Tensor<std::uint8_t>;

struct CmdArguments {
	std::string profilePath = ".";
	std::string cachePath = ".";
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
	    "profiles\n"
	    "  --cache-path path    Set cache directory\n";
	bool showHelp = false;
	std::optional<std::string> profilePath = std::move(argState->profilePath);
	std::optional<std::string> cachePath = std::move(argState->cachePath);
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
	    },
	    {
	        "--cache-path",  // name
	        {
	            cmdline::ArgType::STRING,  // type
	            &cachePath                 // data
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
	argState->cachePath = std::move(cachePath.value());
	argState->configPath = std::move(configPath.value());
	return true;
}

std::tuple<Tensor, Tensor> readData(const config::DataConfig &dataConfig) {
	TensorShape lowResShape = {1, dataConfig.height, dataConfig.width};
	TensorShape hiResShape = {1, dataConfig.height * dataConfig.upscaleFactor,
	    dataConfig.width * dataConfig.upscaleFactor};
	auto lowResImgs =
	    data::loadData(dataConfig.lowResPath.c_str(), lowResShape);
	auto hiResImgs = data::loadData(dataConfig.hiResPath.c_str(), hiResShape);
	if (hiResImgs.getShape().getBatchSize() !=
	    lowResImgs.getShape().getBatchSize()) {
		throw std::invalid_argument(
		    "Number of images for hi and low res must be equal");
	}
	return {std::move(lowResImgs), std::move(hiResImgs)};
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
		std::filesystem::path profilePath = argState.profilePath;
		std::filesystem::path cachePath = argState.cachePath;
		LOG_INFO << "Loading data";
		readData(benchmarkConfig.dataConfig);
	} catch (...) {
		LOG_EXCEPTION;
		return 1;
	}
	return 0;
}
