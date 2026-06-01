#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <vector>

namespace ctdp::spmv_dsl {

struct timing_summary {
    double median_ns = 0.0;
    double p99_ns = 0.0;
};

template<class Fn>
timing_summary measure(Fn&& fn, std::size_t iterations = 31, std::size_t warmup = 3) {
    using clock = std::chrono::steady_clock;

    for (std::size_t i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> samples;
    samples.reserve(iterations);

    for (std::size_t i = 0; i < iterations; ++i) {
        const auto t0 = clock::now();
        fn();
        const auto t1 = clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }

    std::sort(samples.begin(), samples.end());
    const auto mid = samples.size() / 2;
    const auto p99_index = std::min(samples.size() - 1, static_cast<std::size_t>(samples.size() * 99 / 100));
    return {.median_ns = samples[mid], .p99_ns = samples[p99_index]};
}

} // namespace ctdp::spmv_dsl
