#ifndef CTDP_CALIBRATOR_BENCHMARK_EXPLORER_H
#define CTDP_CALIBRATOR_BENCHMARK_EXPLORER_H

// ctdp::calibrator::benchmark_explorer — Google Benchmark adapter
//
// Design v2.2 §5.6:
//   Optional GBench adapter for interactive exploration.
//   Not part of the production pipeline.  Requires linking libbenchmark.
//
// Bridges the Scenario concept to Google Benchmark's registration API,
// letting users explore calibration points interactively in GBench's
// tooling (console output, JSON export, comparison runs, etc.) without
// duplicating measurement logic.
//
// Usage:
//   #include <ctdp/calibrator/benchmark_explorer.h>
//   #include "my_scenarios.h"
//
//   int main(int argc, char** argv) {
//       my_scenario scen;
//       ctdp::calibrator::register_scenario_benchmarks(scen, "MyParser");
//       benchmark::Initialize(&argc, argv);
//       benchmark::RunSpecifiedBenchmarks();
//       return 0;
//   }
//
// Build: requires -lbenchmark -lpthread
//
// Each space point becomes a separate GBench benchmark named:
//   {prefix}/{point_index}
// The point index is the ordinal position in scenario.points().
//
// For human-readable names, provide a PointLabeller:
//   register_scenario_benchmarks(scen, "Parser", [](auto const& pt) {
//       return std::to_string(pt.digits) + "_" + strategy_name(pt.strategy);
//   });
// Produces benchmarks named: Parser/4_swar, Parser/8_unrolled, etc.

#if __has_include(<benchmark/benchmark.h>)
#  include <benchmark/benchmark.h>
#  define CTDP_HAS_GBENCH 1
#else
#  define CTDP_HAS_GBENCH 0
#endif

#include "scenario.h"
#include <ctdp/bench/compiler_barrier.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ctdp::calibrator {

#if CTDP_HAS_GBENCH

// ─── Internal: type-erased benchmark wrapper ─────────────────────

namespace detail {

/// Holds a scenario + point index, callable as a GBench function.
template <Scenario S>
struct scenario_benchmark {
    S*          scenario;
    std::size_t point_index;

    void run(benchmark::State& state) {
        auto const& pts = scenario->points();
        auto const& pt  = pts[point_index];
        scenario->prepare(pt);

        for (auto _ : state) {
            auto tok = scenario->execute(pt);
            benchmark::DoNotOptimize(tok.value);
        }
    }
};

/// Static dispatch target for GBench registration.
/// GBench requires a function pointer; we trampoline through a static
/// that indexes into a shared vector of wrappers.
template <Scenario S>
struct benchmark_registry {
    static inline std::vector<scenario_benchmark<S>> entries;

    static void run_entry(benchmark::State& state) {
        auto idx = static_cast<std::size_t>(state.range(0));
        if (idx < entries.size()) {
            entries[idx].run(state);
        }
    }
};

} // namespace detail

// ─── Public API ──────────────────────────────────────────────────

/// Default point labeller: uses ordinal index.
struct index_labeller {
    template <typename PointType>
    std::string operator()(std::size_t idx, PointType const&) const {
        return std::to_string(idx);
    }
};

/// Register all points in a Scenario as individual Google Benchmarks.
///
/// @param scenario   The scenario to explore (must outlive benchmarks)
/// @param prefix     Benchmark name prefix (e.g. "FIX_SWAR")
/// @param labeller   Callable(size_t index, point_type const& pt) → string
///                   Produces the suffix for each benchmark name.
///
/// The scenario must remain alive until benchmarks have run (typically
/// it should be a local variable in main() before RunSpecifiedBenchmarks).
///
template <Scenario S, typename Labeller = index_labeller>
void register_scenario_benchmarks(
    S& scenario,
    std::string_view prefix,
    Labeller labeller = Labeller{})
{
    auto const& pts = scenario.points();
    auto& registry = detail::benchmark_registry<S>::entries;
    auto base_index = registry.size();

    for (std::size_t i = 0; i < pts.size(); ++i) {
        registry.push_back({&scenario, i});

        std::string name = std::string(prefix) + "/"
                         + labeller(i, pts[i]);

        benchmark::RegisterBenchmark(
            name.c_str(),
            [base_idx = base_index + i](benchmark::State& state) {
                detail::benchmark_registry<S>::entries[base_idx].run(state);
            });
    }
}

/// Convenience: register with a lambda labeller.
/// Example:
///   register_labelled(scen, "Parser", [](auto i, auto const& pt) {
///       return std::to_string(pt.digits) + "_swar";
///   });
template <Scenario S, typename Fn>
void register_labelled(S& scenario, std::string_view prefix, Fn&& fn) {
    register_scenario_benchmarks(scenario, prefix, std::forward<Fn>(fn));
}

#else // !CTDP_HAS_GBENCH

// ─── Stub API when Google Benchmark is not available ─────────────

/// Stub: prints a message and does nothing.
template <Scenario S, typename Labeller = int>
void register_scenario_benchmarks(S&, std::string_view, Labeller = {}) {
    // Google Benchmark not available — no-op.
    // To enable, install libbenchmark and ensure <benchmark/benchmark.h>
    // is on the include path.
}

template <Scenario S, typename Fn>
void register_labelled(S&, std::string_view, Fn&&) {}

#endif // CTDP_HAS_GBENCH

// ─── Feature-detection macro for user code ───────────────────────

/// Users can test: #if CTDP_HAS_GBENCH to conditionally compile
/// benchmark explorer code.

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_BENCHMARK_EXPLORER_H
