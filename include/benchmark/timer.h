// Copyright 2021 Ivanov Viktor

#pragma once

#include <chrono>
#include <iostream>
#include <string>

#include "benchmark/utils.h"

namespace benchmark {

namespace timer {

using clock = std::chrono::high_resolution_clock;
using timestamp = clock::time_point;

struct TimestampPrinter {
	timestamp begin;
	timestamp end;

	TimestampPrinter(timestamp begin, timestamp end) : begin(begin), end(end) {
	}
};

inline std::ostream &operator<<(std::ostream &os, const TimestampPrinter &ts) {
	os << std::chrono::duration_cast<std::chrono::microseconds>(
	          ts.end - ts.begin)
	              .count() /
	          1e6
	   << 's';
	return os;
}

class Timer {
public:
	explicit Timer(const char *tag) : m_Start{clock::now()}, m_Tag(tag) {
	}
	explicit Timer(const std::string &tag) : Timer(tag.c_str()) {
	}
	Timer(const Timer &) = delete;
	Timer(Timer &&) = delete;
	Timer &operator=(const Timer &) = delete;
	Timer &operator=(Timer &&) = delete;

	~Timer() {
		timestamp end = clock::now();
		std::clog << m_Tag << ": " << TimestampPrinter(m_Start, end)
		          << std::endl;
	}

private:
	timestamp m_Start;
	std::string m_Tag;
};

}  // namespace timer

#define TIMER_NAME_IMPL(X, Y) X##Y
#define TIMER_NAME(X) TIMER_NAME_IMPL(_benchmark__timer_, X)
#define TIMED_W_TAG(tag)                             \
	::benchmark::timer::Timer TIMER_NAME(__LINE__) { \
		tag                                          \
	}
#define TIMED TIMED_W_TAG(FUNCTION_NAME)

}  // namespace benchmark
