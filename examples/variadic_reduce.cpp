// examples/variadic_reduce.cpp
//
// Variadic Reduction Example: Single-Pass Statistics
//
// Demonstrates the tuple algebra by computing 7 statistics
// (count, sum, sum_x2, sum_x3, sum_x4, min, max) in a single
// pass over the data, compared with the naive multi-pass approach.
//
// This is the foundational example from P3666R0. In production,
// the addition chain optimisation would further reduce the cost
// of computing x^2, x^3, x^4 by sharing intermediates (x^2 is
// computed once, reused for x^4 = (x^2)^2). That optimisation
// sits at the plan level and is not applied here — this example
// shows the algebra primitives without fusion.
//
// Usage:
//   ./variadic_reduce
//   ./variadic_reduce <N>         # N data points (default 1000000)
//
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#include <ct_dp/algebra/algebra.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

using namespace ct_dp::algebra;

// ============================================================================
// 1. Compile-time verification
// ============================================================================

// Define the 7-lane reduction.
constexpr auto seven_lane_stats = make_tuple_reduction(
    reduction_lane{constant_t<1>{}, plus_fn{},  0},                                    // count
    reduction_lane{identity_t{},    plus_fn{},  0.0},                                  // sum(x)
    reduction_lane{power_t<2>{},    plus_fn{},  0.0},                                  // sum(x^2)
    reduction_lane{power_t<3>{},    plus_fn{},  0.0},                                  // sum(x^3)
    reduction_lane{power_t<4>{},    plus_fn{},  0.0},                                  // sum(x^4)
    reduction_lane{identity_t{},    min_fn{},   std::numeric_limits<double>::max()},    // min
    reduction_lane{identity_t{},    max_fn{},   std::numeric_limits<double>::lowest()}  // max
);

// Verify at compile time with known data.
constexpr std::array<double, 5> compile_time_data = {1.0, 2.0, 3.0, 4.0, 5.0};
constexpr auto compile_time_result =
    seven_lane_stats.reduce(compile_time_data.begin(), compile_time_data.end());

static_assert(std::get<0>(compile_time_result) == 5,           "count");
static_assert(std::get<1>(compile_time_result) == 15.0,        "sum");
static_assert(std::get<2>(compile_time_result) == 55.0,        "sum_x2");   // 1+4+9+16+25
static_assert(std::get<3>(compile_time_result) == 225.0,       "sum_x3");   // 1+8+27+64+125
static_assert(std::get<4>(compile_time_result) == 979.0,       "sum_x4");   // 1+16+81+256+625
static_assert(std::get<5>(compile_time_result) == 1.0,         "min");
static_assert(std::get<6>(compile_time_result) == 5.0,         "max");

// ============================================================================
// 2. Multi-pass baseline (naive: one loop per statistic)
// ============================================================================

struct multi_pass_result {
    int    count;
    double sum;
    double sum_x2;
    double sum_x3;
    double sum_x4;
    double min_val;
    double max_val;
};

multi_pass_result compute_multi_pass(const std::vector<double>& data) {
    multi_pass_result r{};
    r.count = static_cast<int>(data.size());

    r.sum = 0.0;
    for (auto x : data) r.sum += x;

    r.sum_x2 = 0.0;
    for (auto x : data) r.sum_x2 += x * x;

    r.sum_x3 = 0.0;
    for (auto x : data) r.sum_x3 += x * x * x;

    r.sum_x4 = 0.0;
    for (auto x : data) { auto x2 = x * x; r.sum_x4 += x2 * x2; }

    r.min_val = *std::min_element(data.begin(), data.end());
    r.max_val = *std::max_element(data.begin(), data.end());

    return r;
}

// ============================================================================
// 3. Single-pass variadic reduction (tuple algebra)
// ============================================================================

// Uses the seven_lane_stats defined above.
// Just calls: seven_lane_stats.reduce(data)

// ============================================================================
// 4. Derived statistics from raw moments
// ============================================================================

struct descriptive_stats {
    int    count;
    double mean;
    double variance;
    double skewness;
    double kurtosis;
    double min_val;
    double max_val;
};

template<typename Result>
descriptive_stats compute_derived(const Result& raw) {
    auto n       = static_cast<double>(std::get<0>(raw));
    auto sum     = std::get<1>(raw);
    auto sum_x2  = std::get<2>(raw);
    auto sum_x3  = std::get<3>(raw);
    auto sum_x4  = std::get<4>(raw);
    auto min_val = std::get<5>(raw);
    auto max_val = std::get<6>(raw);

    double mean = sum / n;

    // Central moments from raw moments
    double m2 = sum_x2 / n - mean * mean;                              // variance
    double m3 = sum_x3 / n - 3.0 * mean * sum_x2 / n + 2.0 * mean * mean * mean;
    double m4 = sum_x4 / n - 4.0 * mean * sum_x3 / n
                + 6.0 * mean * mean * sum_x2 / n - 3.0 * mean * mean * mean * mean;

    double std_dev = std::sqrt(m2);
    double skewness = (std_dev > 0.0) ? m3 / (std_dev * std_dev * std_dev) : 0.0;
    double kurtosis = (m2 > 0.0) ? m4 / (m2 * m2) - 3.0 : 0.0;  // excess kurtosis

    return {static_cast<int>(n), mean, m2, skewness, kurtosis, min_val, max_val};
}

// ============================================================================
// 5. Timing utility
// ============================================================================

template<typename F>
double time_ns(F&& fn, int reps = 10) {
    // Warmup
    for (int i = 0; i < 3; ++i) fn();

    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best = std::min(best, ns);
    }
    return best;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::size_t N = 1'000'000;
    if (argc > 1) N = std::stoull(argv[1]);

    std::cout << "ct_dp::algebra — Variadic Reduction Example\n";
    std::cout << "=============================================\n\n";

    // -- Compile-time verification --
    std::cout << "Compile-time verification (static_assert passed):\n";
    std::cout << "  data     = {1, 2, 3, 4, 5}\n";
    std::cout << "  count    = " << std::get<0>(compile_time_result) << "\n";
    std::cout << "  sum      = " << std::get<1>(compile_time_result) << "\n";
    std::cout << "  sum(x^2) = " << std::get<2>(compile_time_result) << "\n";
    std::cout << "  sum(x^3) = " << std::get<3>(compile_time_result) << "\n";
    std::cout << "  sum(x^4) = " << std::get<4>(compile_time_result) << "\n";
    std::cout << "  min      = " << std::get<5>(compile_time_result) << "\n";
    std::cout << "  max      = " << std::get<6>(compile_time_result) << "\n";
    std::cout << "\n";

    // -- Generate runtime data --
    std::vector<double> data(N);
    // Deterministic pseudo-random: linear congruential
    double seed = 0.5;
    for (auto& x : data) {
        seed = std::fmod(seed * 6364136223846793005.0 + 1442695040888963407.0, 1e18);
        x = std::fmod(seed, 1000.0) / 100.0;  // values in [0, 10)
    }

    std::cout << "Runtime benchmark: N = " << N << " doubles\n\n";

    // -- Multi-pass --
    multi_pass_result mp_result{};
    double mp_ns = time_ns([&]() {
        auto r = compute_multi_pass(data);
        mp_result = r;
#if defined(_MSC_VER)
        volatile auto sink = &mp_result;  // prevent DCE
        (void)sink;
#else
        asm volatile("" : : "g"(&mp_result) : "memory");  // prevent DCE
#endif
    });

    // -- Single-pass variadic --
    using result_type = decltype(seven_lane_stats.reduce(data));
    result_type sp_result;
    double sp_ns = time_ns([&]() {
        auto r = seven_lane_stats.reduce(data);
        sp_result = r;
#if defined(_MSC_VER)
        volatile auto sink = &sp_result;  // prevent DCE
        (void)sink;
#else
        asm volatile("" : : "g"(&sp_result) : "memory");  // prevent DCE
#endif
    });

    // -- Results --
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Multi-pass (6 loops):  " << std::setw(10) << mp_ns / 1e6 << " ms\n";
    std::cout << "  Single-pass (1 loop):  " << std::setw(10) << sp_ns / 1e6 << " ms\n";
    std::cout << "  Speedup:               " << std::setw(10) << mp_ns / sp_ns << "x\n";
    std::cout << "\n";

    // -- Verify agreement --
    auto mp = compute_multi_pass(data);
    auto sp = seven_lane_stats.reduce(data);

    std::cout << std::setprecision(6);
    std::cout << "  Verification (single-pass vs multi-pass):\n";
    std::cout << "                  Multi-pass       Single-pass      Match\n";
    std::cout << "  count    " << std::setw(16) << mp.count
              << std::setw(16) << std::get<0>(sp)
              << std::setw(10) << (mp.count == std::get<0>(sp) ? "OK" : "FAIL") << "\n";
    std::cout << "  sum      " << std::setw(16) << mp.sum
              << std::setw(16) << std::get<1>(sp)
              << std::setw(10) << (std::abs(mp.sum - std::get<1>(sp)) < 1e-6 ? "OK" : "FAIL") << "\n";
    std::cout << "  sum(x^2) " << std::setw(16) << mp.sum_x2
              << std::setw(16) << std::get<2>(sp)
              << std::setw(10) << (std::abs(mp.sum_x2 - std::get<2>(sp)) < 1e-3 ? "OK" : "FAIL") << "\n";
    std::cout << "  min      " << std::setw(16) << mp.min_val
              << std::setw(16) << std::get<5>(sp)
              << std::setw(10) << (mp.min_val == std::get<5>(sp) ? "OK" : "FAIL") << "\n";
    std::cout << "  max      " << std::setw(16) << mp.max_val
              << std::setw(16) << std::get<6>(sp)
              << std::setw(10) << (mp.max_val == std::get<6>(sp) ? "OK" : "FAIL") << "\n";
    std::cout << "\n";

    // -- Derived statistics --
    auto derived = compute_derived(sp);

    std::cout << std::setprecision(6);
    std::cout << "  Derived statistics (from single-pass raw moments):\n";
    std::cout << "    Count:           " << derived.count << "\n";
    std::cout << "    Mean:            " << derived.mean << "\n";
    std::cout << "    Variance:        " << derived.variance << "\n";
    std::cout << "    Skewness:        " << derived.skewness << "\n";
    std::cout << "    Excess kurtosis: " << derived.kurtosis << "\n";
    std::cout << "    Min:             " << derived.min_val << "\n";
    std::cout << "    Max:             " << derived.max_val << "\n";
    std::cout << "\n";

    std::cout << "Done. All compile-time static_asserts passed.\n";

    return 0;
}
