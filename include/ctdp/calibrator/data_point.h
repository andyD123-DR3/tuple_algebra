#ifndef CTDP_CALIBRATOR_DATA_POINT_H
#define CTDP_CALIBRATOR_DATA_POINT_H

// ctdp::calibrator::data_point — Atomic output of calibration
//
// A data_point welds a space point to its observables. It is the
// fundamental unit of calibration output: every measurement run
// produces a vector of data_points.
//
// Invariant: raw_timings.size() == raw_snapshots.size()
//   — never split. Space point and observables travel together.
//
// Design note (from v2.2 §2):
//   data_point is parameterised on SpacePoint and MetricSnapshot,
//   making it independent of any specific Space or Metric type.
//   The calibration_harness binds these via its Scenario and Metric
//   template parameters.

#include <ctdp/bench/environment.h>
#include <ctdp/bench/perf_counter.h>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace ctdp::calibrator {

/// The atomic output of calibration: a space point welded to its
/// measurement observables.
///
/// @tparam SpacePoint     The point type from the Scenario's space
/// @tparam MetricSnapshot The snapshot type from the Metric
template <typename SpacePoint, typename MetricSnapshot>
struct data_point {
    SpacePoint                    space_point;     ///< Where in the space
    double                        median_ns = 0.0; ///< Median wall-clock ns
    double                        mad_ns    = 0.0; ///< MAD of wall-clock ns
    std::vector<double>           raw_timings;     ///< Per-rep wall_ns
    std::vector<MetricSnapshot>   raw_snapshots;   ///< Per-rep metric samples
    bench::environment_context    env;             ///< Platform + regime

    /// Check the structural invariant
    [[nodiscard]] bool invariant() const noexcept {
        return raw_timings.size() == raw_snapshots.size();
    }

    /// Number of repetitions
    [[nodiscard]] std::size_t reps() const noexcept {
        return raw_timings.size();
    }

    /// Coefficient of variation (MAD / median), a relative noise measure
    [[nodiscard]] double relative_mad() const noexcept {
        return (median_ns > 0.0) ? mad_ns / median_ns : 0.0;
    }
};

/// Convenience type alias for data_points using counter_snapshot
template <typename SpacePoint>
using counter_data_point = data_point<SpacePoint, bench::counter_snapshot>;

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_DATA_POINT_H
