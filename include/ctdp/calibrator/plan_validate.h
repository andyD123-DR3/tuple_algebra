#ifndef CTDP_CALIBRATOR_PLAN_VALIDATE_H
#define CTDP_CALIBRATOR_PLAN_VALIDATE_H

// ctdp::calibrator::plan_validate — Validate plans via re-measurement
//
// Design v2.2 §5.6:
//   Re-measures individual plan steps using bench::measure_once,
//   compares predicted vs actual cost.  Templated on <Space, Callable>
//   to ensure the validator uses the same executor as the calibrator.
//
// The validator answers: "Does the calibrated cost model still agree
// with reality?"  It catches stale profiles, regime mismatches, and
// cost model inaccuracies before they reach production.
//
// Usage:
//   auto result = validate_profile(profile, scenario, env, cfg);
//   if (!result.passed(0.10)) { /* re-calibrate */ }

#include "calibration_profile.h"
#include "scenario.h"

#include <ctdp/bench/cache_thrasher.h>
#include <ctdp/bench/compiler_barrier.h>
#include <ctdp/bench/environment.h>
#include <ctdp/bench/measurement_kernel.h>
#include <ctdp/bench/metric.h>
#include <ctdp/bench/statistics.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace ctdp::calibrator {

// ─── Validation result for a single point ────────────────────────

struct point_validation {
    double predicted_ns  = 0.0;    ///< Cost model prediction
    double measured_ns   = 0.0;    ///< Fresh measurement (median)
    double measured_mad  = 0.0;    ///< Measurement MAD
    double absolute_error= 0.0;    ///< |predicted - measured|
    double relative_error= 0.0;    ///< |predicted - measured| / measured

    [[nodiscard]] bool within_tolerance(double tol) const noexcept {
        return relative_error <= tol;
    }
};

// ─── Aggregate validation result ─────────────────────────────────

struct validation_result {
    std::vector<point_validation> points;

    /// Overall: fraction of points within tolerance
    std::size_t total_points      = 0;
    std::size_t points_within_tol = 0;
    double max_relative_error     = 0.0;
    double mean_relative_error    = 0.0;
    double median_relative_error  = 0.0;

    /// Overall pass: all points within tolerance
    [[nodiscard]] bool passed(double tol) const noexcept {
        if (total_points == 0) return false;
        for (auto const& pv : points) {
            if (!pv.within_tolerance(tol)) return false;
        }
        return true;
    }

    /// Fraction within tolerance
    [[nodiscard]] double pass_rate() const noexcept {
        return (total_points > 0)
            ? static_cast<double>(points_within_tol) /
              static_cast<double>(total_points)
            : 0.0;
    }

    /// Summary string for reporting
    [[nodiscard]] auto summary(double tol) const -> std::string {
        std::string s;
        s += "validation: ";
        s += std::to_string(points_within_tol) + "/" +
             std::to_string(total_points) + " within " +
             std::to_string(static_cast<int>(tol * 100)) + "% tolerance\n";

        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "  max_rel_error:    %.4f\n"
            "  mean_rel_error:   %.4f\n"
            "  median_rel_error: %.4f\n",
            max_relative_error, mean_relative_error,
            median_relative_error);
        s += buf;

        s += passed(tol) ? "  result: PASS\n" : "  result: FAIL\n";
        return s;
    }
};

// ─── Validation configuration ────────────────────────────────────

struct validation_config {
    std::size_t reps          = 20;   ///< More reps for validation accuracy
    std::size_t warmup_iters  = 200;
    std::size_t measure_iters = 1;
    bool        flush_cache   = true;
    double      tolerance     = 0.10; ///< 10% default tolerance
};

// ─── Validate a lookup profile ───────────────────────────────────

/// Re-measure every point in a lookup profile and compare predicted
/// vs actual cost.
///
/// The Scenario must have a matching callable_type — the template
/// system enforces this since both the profile and the scenario
/// are parameterised on the same Callable.
///
/// @param profile   The lookup profile to validate
/// @param scenario  The scenario (same kernel as calibration)
/// @param cfg       Validation configuration
/// @return          Aggregate validation result
///
template <typename Space, typename Callable,
          Scenario S, bench::Metric M = bench::null_metric>
    requires std::same_as<typename S::callable_type, Callable>
auto validate_profile(
    calibration_profile<Space, Callable> const& profile,
    S& scenario,
    validation_config const& cfg = {},
    M metric = M{})
    -> validation_result
{
    validation_result result;
    auto const& entries = profile.lookup.entries;
    result.total_points = entries.size();
    result.points.reserve(entries.size());

    // Set up cache thrasher if needed
    auto llc = bench::detect_llc_bytes();
    bench::cache_thrasher thrasher(llc > 0 ? llc : (6u << 20));

    for (auto const& [pt, predicted] : entries) {
        // Prepare the scenario for this point
        scenario.prepare(pt);

        // Create opaque callable (same lambda bridge as harness)
        auto fn = [&]() -> bench::result_token {
            return scenario.execute(pt);
        };

        // Create setup hook
        auto setup = cfg.flush_cache
            ? std::function<void()>([&]() { thrasher.thrash(); })
            : std::function<void()>([]() {});

        // Measure
        auto meas = bench::measure_repeated(
            fn, setup, metric,
            cfg.reps, cfg.warmup_iters, cfg.measure_iters);

        // Compute errors
        point_validation pv;
        pv.predicted_ns   = predicted;
        pv.measured_ns    = meas.median_ns;
        pv.measured_mad   = meas.mad_ns;
        pv.absolute_error = std::abs(predicted - meas.median_ns);
        pv.relative_error = (meas.median_ns > 1e-9)
            ? pv.absolute_error / meas.median_ns
            : 0.0;

        result.points.push_back(pv);
    }

    // Aggregate statistics
    if (!result.points.empty()) {
        result.points_within_tol = 0;
        double sum_rel = 0.0;
        result.max_relative_error = 0.0;

        std::vector<double> rel_errors;
        rel_errors.reserve(result.points.size());

        for (auto const& pv : result.points) {
            rel_errors.push_back(pv.relative_error);
            sum_rel += pv.relative_error;
            result.max_relative_error = std::max(
                result.max_relative_error, pv.relative_error);
            if (pv.within_tolerance(cfg.tolerance)) {
                ++result.points_within_tol;
            }
        }

        result.mean_relative_error = sum_rel /
            static_cast<double>(result.points.size());

        // Median relative error
        std::sort(rel_errors.begin(), rel_errors.end());
        auto n = rel_errors.size();
        result.median_relative_error = (n % 2 == 1)
            ? rel_errors[n / 2]
            : (rel_errors[n / 2 - 1] + rel_errors[n / 2]) / 2.0;
    }

    return result;
}

/// Convenience: validate with explicit tolerance, returns bool
template <typename Space, typename Callable, Scenario S>
    requires std::same_as<typename S::callable_type, Callable>
[[nodiscard]] bool validate_or_fail(
    calibration_profile<Space, Callable> const& profile,
    S& scenario,
    double tolerance = 0.10)
{
    validation_config cfg;
    cfg.tolerance = tolerance;
    auto result = validate_profile<Space, Callable>(profile, scenario, cfg);
    return result.passed(tolerance);
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_PLAN_VALIDATE_H
