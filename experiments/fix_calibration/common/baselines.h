#ifndef CTDP_FIX_EXPERIMENT_BASELINES_H
#define CTDP_FIX_EXPERIMENT_BASELINES_H

// experiments/fix_calibration/common/baselines.h
//
// Named baseline configurations and the measurement_result type shared
// by mock_measurer (PR2, testing) and compiled_measurer (PR5, RDTSC).
//
// Baselines are the reference points that every experiment program
// reports alongside its ML predictions.  They answer "how does the
// model's best compare to the obvious choices?"
//
// Dependencies: fix_et_parser.h (fix_config, Strategy, config_to_string)
// C++20

#include <ctdp/calibrator/fix_et_parser.h>

#include <array>
#include <cstdint>
#include <string_view>

namespace ctdp::fix_experiment {

namespace fix = ctdp::calibrator::fix;

// ─────────────────────────────────────────────────────────────────────
//  measurement_result — what any measurer returns
//
//  Both mock_measurer and compiled_measurer produce this type.
//  They are not polymorphic; the experiment_runner is templated on the
//  measurer type.  This struct is the shared vocabulary.
// ─────────────────────────────────────────────────────────────────────

struct measurement_result {
    double p50_ns = 0.0;   ///< Median latency in nanoseconds
    double p99_ns = 0.0;   ///< 99th-percentile latency in nanoseconds

    /// Tail ratio: how much worse the tail is vs median.
    [[nodiscard]] constexpr double tail_ratio() const noexcept {
        return (p50_ns > 0.0) ? (p99_ns / p50_ns) : 0.0;
    }

    /// Is this a valid measurement (non-zero)?
    [[nodiscard]] constexpr bool valid() const noexcept {
        return p50_ns > 0.0 && p99_ns > 0.0;
    }
};

// ─────────────────────────────────────────────────────────────────────
//  Named baselines
// ─────────────────────────────────────────────────────────────────────

/// Phase 10g DP-optimal from prior investigation (UUSLSUUSUUUU).
/// Achieves ~27.8 ns/msg p99 on the reference hardware.
inline constexpr fix::fix_config phase10g_optimal = {
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
    fix::Strategy::SWAR,     fix::Strategy::Loop,
    fix::Strategy::SWAR,     fix::Strategy::Unrolled,
    fix::Strategy::Unrolled, fix::Strategy::SWAR,
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
};

/// A baseline entry: name + config, for iteration.
struct baseline_entry {
    std::string_view   name;
    fix::fix_config    config;
};

/// All named baselines.  Programs iterate this to produce the
/// reference rows in every output table.
inline constexpr std::array<baseline_entry, 5> baselines = {{
    {"all_unrolled",   fix::all_unrolled},
    {"all_swar",       fix::all_swar},
    {"all_loop",       fix::all_loop},
    {"all_generic",    fix::all_generic},
    {"phase10g_opt",   phase10g_optimal},
}};

/// Number of named baselines.
inline constexpr std::size_t num_baselines = baselines.size();

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_BASELINES_H
