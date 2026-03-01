#ifndef CTDP_CALIBRATOR_CALIBRATION_HARNESS_H
#define CTDP_CALIBRATOR_CALIBRATION_HARNESS_H

// ctdp::calibrator::calibration_harness — The composition point
//
// Composes a Scenario with a bench::Metric to produce data_points.
// This is the central class of the calibrator library.
//
// Architecture (from design v2.2):
//
//   calibration_harness::measure_point(pt)
//       │
//       ├── scenario_.prepare(pt);                  // calibrator: space-aware
//       │
//       ├── auto fn = [&]() -> result_token {       // lambda bridge
//       │       return scenario_.execute(pt);
//       │   };
//       │
//       ├── bench::measure_repeated(fn, setup, ...)  // bench: opaque callable
//       │
//       └── return data_point{pt, result.*};         // weld point + observables
//
// Part A (bench) never sees point_type, Space, or feature vectors.
// Part B (calibrator) never manages timing, counter setup, or anti-elision.
//
// Restructured from calibrator.h:193–376.

#include "data_point.h"
#include "scenario.h"

#include <ctdp/bench/cache_thrasher.h>
#include <ctdp/bench/environment.h>
#include <ctdp/bench/measurement_kernel.h>
#include <ctdp/bench/metric.h>

#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace ctdp::calibrator {

/// Configuration for calibration runs.
/// Pin CPU policy lives HERE (not in bench layer).
struct harness_config {
    std::size_t reps            = 10;    ///< Repetitions per space point
    std::size_t warmup_iters    = 200;   ///< Warmup iterations per rep
    std::size_t measure_iters   = 1;     ///< Measurement iterations per rep
    bool        pin_cpu         = true;  ///< Pin thread to a specific CPU
    int         pin_cpu_id      = -1;    ///< CPU to pin to (-1 = auto-select)
    bool        boost_priority  = true;  ///< Elevate process priority
    bool        flush_cache     = true;  ///< Flush LLC between measurements
    std::size_t llc_bytes       = 0;     ///< LLC size for cache thrasher (0 = auto)
    bool        verbose         = true;  ///< Print progress to stderr
};

/// The calibration harness: composes a Scenario with a Metric to
/// produce data_points.
///
/// @tparam S  Scenario type (must satisfy the Scenario concept)
/// @tparam M  Metric type (defaults to counter_metric)
template <Scenario S, bench::Metric M = bench::counter_metric>
class calibration_harness {
public:
    // ─── Type aliases ───────────────────────────────────────────
    using point_type      = typename S::point_type;
    using callable_type   = typename S::callable_type;
    using snapshot_type   = typename M::snapshot_type;
    using data_point_type = data_point<point_type, snapshot_type>;

    // ─── Construction ───────────────────────────────────────────

    /// Construct with a scenario and optional config
    explicit calibration_harness(S scenario, harness_config cfg = {})
        : scenario_{std::move(scenario)}
        , config_{cfg}
        , thrasher_{cfg.llc_bytes == 0 ? bench::detect_llc_bytes() : cfg.llc_bytes}
    {}

    /// Access the config (for inspection or late modification)
    [[nodiscard]] harness_config const& config() const noexcept {
        return config_;
    }
    harness_config& config() noexcept { return config_; }

    /// Access the scenario
    [[nodiscard]] S const& scenario() const noexcept { return scenario_; }

    // ─── Run ────────────────────────────────────────────────────

    /// Run the full calibration: iterate over all space points,
    /// measure each, and return the complete dataset.
    [[nodiscard]] auto run() -> std::vector<data_point_type> {
        // Resolve auto-select CPU pin
        int pin_id = resolve_pin_cpu();

        // Set up environment guard (RAII: pin + priority)
        bench::environment_guard guard(
            config_.pin_cpu ? pin_id : -1,
            config_.boost_priority
        );

        // Capture environment context
        auto env_ctx = bench::capture_environment();
        env_ctx.pinned_cpu = config_.pin_cpu ? pin_id : -1;
        env_ctx.llc_bytes  = thrasher_.buffer_bytes() * 2 / 3; // recover LLC from 1.5×
        if constexpr (requires { metric_.tier1_available(); }) {
            env_ctx.tier1_counters = metric_.tier1_available();
        }

        // Collect all points
        auto point_range = scenario_.points();
        std::vector<point_type> points(
            std::ranges::begin(point_range),
            std::ranges::end(point_range));

        if (config_.verbose) {
            std::cerr << "[calibrator] Scenario: " << scenario_.name() << "\n"
                      << "[calibrator] Points: " << points.size()
                      << ", Reps: " << config_.reps
                      << ", Warmup: " << config_.warmup_iters
                      << ", Measure iters: " << config_.measure_iters << "\n";
            if (config_.pin_cpu) {
                std::cerr << "[calibrator] Pinned to CPU " << pin_id << "\n";
            }
            if (config_.flush_cache) {
                std::cerr << "[calibrator] Cache flush: "
                          << (thrasher_.buffer_bytes() / 1024) << " KiB buffer\n";
            }
            std::cerr << "[calibrator] Tier 1 counters: "
                      << (env_ctx.tier1_counters ? "available" : "unavailable")
                      << "\n";
        }

        // Measure each point
        std::vector<data_point_type> results;
        results.reserve(points.size());

        for (std::size_t i = 0; i < points.size(); ++i) {
            auto dp = measure_point(points[i], env_ctx);
            assert(dp.invariant());

            if (config_.verbose) {
                std::cerr << "[calibrator] [" << (i + 1) << "/"
                          << points.size() << "] median="
                          << dp.median_ns << " ns, MAD="
                          << dp.mad_ns << " ns ("
                          << (dp.relative_mad() * 100.0) << "% rel)\n";
            }

            results.push_back(std::move(dp));
        }

        if (config_.verbose) {
            std::cerr << "[calibrator] Done. " << results.size()
                      << " data points collected.\n";
        }

        return results;
    }

    /// Measure a single space point (public for testing/debugging)
    [[nodiscard]] auto measure_point(point_type const& pt,
                                      bench::environment_context const& env)
        -> data_point_type
    {
        // 1. Prepare the scenario for this point
        scenario_.prepare(pt);

        // 2. Lambda bridge: scenario → opaque callable
        auto fn = [this, &pt]() -> bench::result_token {
            return scenario_.execute(pt);
        };

        // 3. Setup callable: cache thrash if configured
        auto setup = [this]() {
            if (config_.flush_cache) {
                thrasher_.thrash();
            }
        };

        // 4. Delegate to bench::measure_repeated
        auto result = bench::measure_repeated(
            fn, setup, metric_,
            config_.reps,
            config_.warmup_iters,
            config_.measure_iters
        );

        // 5. Weld point + observables into a data_point
        data_point_type dp;
        dp.space_point    = pt;
        dp.median_ns      = result.median_ns;
        dp.mad_ns         = result.mad_ns;
        dp.raw_timings    = std::move(result.all_ns);
        dp.raw_snapshots  = std::move(result.all_snapshots);
        dp.env            = env;

        return dp;
    }

private:
    S             scenario_;
    harness_config config_;
    M             metric_;
    bench::cache_thrasher thrasher_;

    /// Resolve pin_cpu_id = -1 to an actual CPU.
    /// Policy: choose the highest-numbered CPU that isn't CPU 0
    /// (CPU 0 typically handles interrupts on Linux).
    [[nodiscard]] int resolve_pin_cpu() const noexcept {
        if (config_.pin_cpu_id >= 0) return config_.pin_cpu_id;

        int n = bench::cpu_count();
        if (n <= 1) return 0;
        return n - 1; // Highest-numbered CPU
    }
};

/// Factory function for type deduction
template <Scenario S, bench::Metric M = bench::counter_metric>
[[nodiscard]] auto make_harness(S scenario, harness_config cfg = {}) {
    return calibration_harness<S, M>{std::move(scenario), cfg};
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_CALIBRATION_HARNESS_H
