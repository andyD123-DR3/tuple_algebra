// examples/standalone/dep_break_opt.cpp
// CT-DP standalone example: compile-time loop dependency breaking
//
// 3 dimensions: strategy × unroll_factor × num_accumulators
// 105 total points, 3 instances with different optimal strategies
//
// Teaches: same space, different instance → different winner.
// Conditional dimension relevance (unroll meaningless for REDUCTION_TREE).

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>

// ============================================================================
// 1. Dimensions
// ============================================================================

enum class dep_strategy : int {
    NONE, LOOP_UNROLLING, LOOP_FISSION, SCALAR_EXPANSION,
    REDUCTION_TREE, LOOP_INTERCHANGE, SOFTWARE_PIPELINING
};

constexpr std::array<dep_strategy, 7> all_strategies = {
    dep_strategy::NONE, dep_strategy::LOOP_UNROLLING, dep_strategy::LOOP_FISSION,
    dep_strategy::SCALAR_EXPANSION, dep_strategy::REDUCTION_TREE,
    dep_strategy::LOOP_INTERCHANGE, dep_strategy::SOFTWARE_PIPELINING
};
constexpr std::array<int, 5> all_unroll = {1, 2, 3, 4, 8};
constexpr std::array<int, 3> all_accum = {2, 4, 8};

struct dep_config {
    dep_strategy strategy;
    int unroll_factor;
    int num_accumulators;
    constexpr bool operator==(dep_config const&) const = default;
};

// ============================================================================
// 2. Instance
// ============================================================================

struct dep_instance {
    bool is_associative;
    bool is_commutative;
    int  dep_distance;       // 0 = no dep, 1 = RAW adjacent, >1 = distant
    std::size_t loop_size;
};

constexpr dep_instance accumulation    = {true,  true,  1, 1000000};
constexpr dep_instance recurrence_d3   = {false, false, 3, 1000000};
constexpr dep_instance prefix_sum      = {false, false, 1, 1000000};

// ============================================================================
// 3. Validity
// ============================================================================

constexpr bool is_valid(dep_config const& pt, dep_instance const& inst) {
    if (pt.strategy == dep_strategy::REDUCTION_TREE && !inst.is_associative) return false;
    if (pt.strategy == dep_strategy::SOFTWARE_PIPELINING && inst.dep_distance <= 1) return false;
    if (pt.strategy == dep_strategy::SOFTWARE_PIPELINING && pt.unroll_factor > inst.dep_distance) return false;
    if (pt.strategy == dep_strategy::LOOP_INTERCHANGE && inst.dep_distance == 0) return false;
    return true;
}

// ============================================================================
// 4. Cost
// ============================================================================

constexpr double evaluate_cost(dep_config const& pt, dep_instance const& inst) {
    double cost = 0.0;
    switch (pt.strategy) {
        case dep_strategy::NONE:               cost = 0.0; break;
        case dep_strategy::LOOP_UNROLLING:     cost = -1000.0 - pt.unroll_factor * 100.0; break;
        case dep_strategy::LOOP_FISSION:       cost = -1500.0; break;
        case dep_strategy::SCALAR_EXPANSION:   cost = -2000.0 - pt.num_accumulators * 200.0;
                                                       if (inst.dep_distance > 1) cost += 2000.0;
                                                       break;
        case dep_strategy::REDUCTION_TREE:     cost = -3000.0 - pt.num_accumulators * 200.0; break;
        case dep_strategy::LOOP_INTERCHANGE:   cost = -1800.0; break;
        case dep_strategy::SOFTWARE_PIPELINING:cost = -2500.0 - pt.unroll_factor * 150.0; break;
    }
    if (inst.is_commutative) cost -= 500.0;
    return cost;
}

// ============================================================================
// 5. Solve
// ============================================================================

struct search_result {
    dep_config best{};
    double best_cost = std::numeric_limits<double>::max();
    std::size_t total = 0;
    std::size_t feasible = 0;
};

constexpr search_result solve(dep_instance const& inst) {
    search_result result{};
    for (auto strat : all_strategies)
      for (auto unr : all_unroll)
        for (auto acc : all_accum) {
            ++result.total;
            dep_config pt{strat, unr, acc};
            if (!is_valid(pt, inst)) continue;
            ++result.feasible;
            double c = evaluate_cost(pt, inst);
            if (c < result.best_cost) {
                result.best_cost = c;
                result.best = pt;
            }
        }
    return result;
}

// Compile-time solve for all three instances
static constexpr auto r_accum  = solve(accumulation);
static constexpr auto r_recur  = solve(recurrence_d3);
static constexpr auto r_prefix = solve(prefix_sum);

static_assert(r_accum.total == 105);
static_assert(r_accum.best.strategy == dep_strategy::REDUCTION_TREE);
static_assert(r_accum.best.num_accumulators == 8);

static_assert(r_recur.best.strategy == dep_strategy::SOFTWARE_PIPELINING);
static_assert(r_recur.best.unroll_factor == 3);

static_assert(r_prefix.best.strategy == dep_strategy::SCALAR_EXPANSION);
static_assert(r_prefix.best.num_accumulators == 8);

// ============================================================================
// 6. Executors
// ============================================================================

template<dep_config Cfg>
struct dep_executor {
    static double execute(double const* data, std::size_t N) {
        if constexpr (Cfg.strategy == dep_strategy::REDUCTION_TREE) {
            // Multi-accumulator reduction
            constexpr int A = Cfg.num_accumulators;
            double acc[A] = {};
            std::size_t i = 0;
            for (; i + A <= N; i += A)
                for (int a = 0; a < A; ++a)
                    acc[a] += data[i + a];
            double result = 0.0;
            for (int a = 0; a < A; ++a) result += acc[a];
            for (; i < N; ++i) result += data[i];
            return result;
        } else if constexpr (Cfg.strategy == dep_strategy::SOFTWARE_PIPELINING) {
            constexpr int U = Cfg.unroll_factor;
            double result = data[0];
            std::size_t i = 1;
            for (; i + U <= N; i += U)
                for (int u = 0; u < U; ++u)
                    result += data[i + u];
            for (; i < N; ++i) result += data[i];
            return result;
        } else if constexpr (Cfg.strategy == dep_strategy::SCALAR_EXPANSION) {
            constexpr int A = Cfg.num_accumulators;
            double acc[A] = {};
            std::size_t i = 0;
            for (; i + A <= N; i += A)
                for (int a = 0; a < A; ++a)
                    acc[a] += data[i + a];
            double result = 0.0;
            for (int a = 0; a < A; ++a) result += acc[a];
            for (; i < N; ++i) result += data[i];
            return result;
        } else {
            // NONE / fallback
            double result = 0.0;
            for (std::size_t i = 0; i < N; ++i) result += data[i];
            return result;
        }
    }
};

// Dispatch
using accum_executor  = dep_executor<r_accum.best>;
using recur_executor  = dep_executor<r_recur.best>;
using prefix_executor = dep_executor<r_prefix.best>;

// ============================================================================
// 7. Benchmark
// ============================================================================

int main() {
    constexpr std::size_t N = 1'000'000;
    std::vector<double> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = static_cast<double>((7*i + 3) % 100) * 0.001;

    double ref = 0.0;
    for (auto x : data) ref += x;

    auto bench = [&](auto fn, const char* label) {
        double result = fn(data.data(), N);  // warm up
        constexpr int REPS = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; ++r)
            result = fn(data.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REPS;
        double err = std::abs(result - ref);
        std::cout << "  " << std::setw(25) << label << ": "
                  << std::fixed << std::setprecision(0) << us << " us"
                  << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
        return us;
    };

    auto baseline = [](double const* d, std::size_t n) -> double {
        double r = 0.0; for (std::size_t i = 0; i < n; ++i) r += d[i]; return r;
    };

    auto print_instance = [](const char* name, search_result const& r) {
        const char* strat_names[] = {
            "NONE", "LOOP_UNROLLING", "LOOP_FISSION", "SCALAR_EXPANSION",
            "REDUCTION_TREE", "LOOP_INTERCHANGE", "SOFTWARE_PIPELINING"
        };
        std::cout << "\n  " << name << ": " << r.feasible << "/" << r.total
                  << " feasible → " << strat_names[static_cast<int>(r.best.strategy)]
                  << " (unroll=" << r.best.unroll_factor
                  << ", accum=" << r.best.num_accumulators << ")\n";
    };

    std::cout << "Dependency Breaking Demo (N=" << N << ")\n";

    print_instance("Accumulation", r_accum);
    double t_base = bench(baseline, "Scalar baseline");
    double t_opt = bench([](auto const* d, auto n) { return accum_executor::execute(d, n); },
                         "REDUCTION_TREE(8)");
    std::cout << "  Speedup: " << std::fixed << std::setprecision(1) << t_base / t_opt << "x\n";

    print_instance("Recurrence d=3", r_recur);
    bench(baseline, "Scalar baseline");
    bench([](auto const* d, auto n) { return recur_executor::execute(d, n); },
          "SOFTWARE_PIPELINING(3)");

    print_instance("Prefix sum", r_prefix);
    bench(baseline, "Scalar baseline");
    bench([](auto const* d, auto n) { return prefix_executor::execute(d, n); },
          "SCALAR_EXPANSION(8)");

    return 0;
}
