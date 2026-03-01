#ifndef CTDP_CALIBRATOR_CALIBRATION_DATASET_H
#define CTDP_CALIBRATOR_CALIBRATION_DATASET_H

// ctdp::calibrator::calibration_dataset — Type-safe dataset container
//
// Design v2.2 §5.6:
//   Typed dataset container wrapping vector<data_point> with schema
//   metadata, provenance, and callable identity.
//   Templated on <Space, Callable, MetricSnapshot>.
//
//   The Callable type parameter is the compile-time key that prevents
//   datasets from being fed to the wrong cost model.
//
// A dataset is produced by one Scenario, and a Scenario has one
// Callable, so the identity is unambiguous at the dataset level.
// data_points inherit callable identity from their dataset — they
// do not carry it as a field.
//
// Usage:
//   auto results = harness.run();
//   auto dataset = make_dataset<parser_space, fix_swar_parser>(
//       std::move(results), provenance);

#include "data_point.h"
#include "provenance.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

/// A calibration dataset: a typed collection of data_points with
/// provenance and callable identity.
///
/// @tparam Space           The search space type (provides point_type)
/// @tparam Callable        The kernel identity (compile-time type key)
/// @tparam MetricSnapshot  The metric's snapshot type
///
/// The Space and Callable type parameters serve as compile-time keys:
///   - A profile fitter for fix_swar_parser won't accept a dataset
///     keyed to fix_lookup_parser — that's a type error.
///   - A solver consuming profile<S, fix_swar_parser> can only be
///     fed profiles fitted from dataset<S, fix_swar_parser, _>.
///
template <typename Space, typename Callable, typename MetricSnapshot>
struct calibration_dataset {
    using space_type     = Space;
    using callable_type  = Callable;
    using snapshot_type  = MetricSnapshot;
    using point_type     = typename Space::point_type;
    using data_point_type = data_point<point_type, MetricSnapshot>;

    /// The measured data points
    std::vector<data_point_type> points;

    /// Provenance: when, where, and how this data was collected
    dataset_provenance provenance;

    // ─── Queries ─────────────────────────────────────────────────

    /// Number of space points in the dataset
    [[nodiscard]] std::size_t size() const noexcept {
        return points.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return points.empty();
    }

    /// Read-only span over the data points
    [[nodiscard]] auto data() const noexcept
        -> std::span<const data_point_type>
    {
        return {points.data(), points.size()};
    }

    /// Total number of raw measurements (sum of reps across all points)
    [[nodiscard]] std::size_t total_measurements() const noexcept {
        std::size_t total = 0;
        for (auto const& dp : points) total += dp.reps();
        return total;
    }

    /// Check that all data_points satisfy the structural invariant
    [[nodiscard]] bool invariant() const noexcept {
        return std::all_of(points.begin(), points.end(),
            [](auto const& dp) { return dp.invariant(); });
    }

    // ─── Statistics ──────────────────────────────────────────────

    /// Fastest point (by median_ns)
    [[nodiscard]] auto fastest() const -> data_point_type const* {
        if (points.empty()) return nullptr;
        return &*std::min_element(points.begin(), points.end(),
            [](auto const& a, auto const& b) {
                return a.median_ns < b.median_ns;
            });
    }

    /// Slowest point (by median_ns)
    [[nodiscard]] auto slowest() const -> data_point_type const* {
        if (points.empty()) return nullptr;
        return &*std::max_element(points.begin(), points.end(),
            [](auto const& a, auto const& b) {
                return a.median_ns < b.median_ns;
            });
    }

    /// Mean median_ns across all points
    [[nodiscard]] double mean_median_ns() const noexcept {
        if (points.empty()) return 0.0;
        double sum = 0.0;
        for (auto const& dp : points) sum += dp.median_ns;
        return sum / static_cast<double>(points.size());
    }

    /// Maximum relative MAD — a data quality indicator
    [[nodiscard]] double max_relative_mad() const noexcept {
        double worst = 0.0;
        for (auto const& dp : points) {
            worst = std::max(worst, dp.relative_mad());
        }
        return worst;
    }

    // ─── Filtering ───────────────────────────────────────────────

    /// Return a new dataset with only the points matching a predicate
    template <typename Pred>
    [[nodiscard]] auto filter(Pred&& pred) const
        -> calibration_dataset
    {
        calibration_dataset result;
        result.provenance = provenance;
        for (auto const& dp : points) {
            if (pred(dp)) {
                result.points.push_back(dp);
            }
        }
        result.provenance.total_points = result.size();
        return result;
    }

    /// Remove points with relative MAD exceeding a threshold
    [[nodiscard]] auto filter_by_quality(double max_rel_mad) const
        -> calibration_dataset
    {
        return filter([max_rel_mad](auto const& dp) {
            return dp.relative_mad() <= max_rel_mad;
        });
    }
};

// ─── Factory function ────────────────────────────────────────────

/// Construct a dataset from harness results and provenance.
///
/// The Space and Callable template parameters must be provided
/// explicitly — they are the type-level keys that connect the
/// dataset to the correct profile fitter and plan consumer.
///
template <typename Space, typename Callable, typename MetricSnapshot>
[[nodiscard]] auto make_dataset(
    std::vector<data_point<typename Space::point_type, MetricSnapshot>> pts,
    dataset_provenance prov = {})
    -> calibration_dataset<Space, Callable, MetricSnapshot>
{
    prov.total_points = pts.size();

    calibration_dataset<Space, Callable, MetricSnapshot> ds;
    ds.points = std::move(pts);
    ds.provenance = std::move(prov);

    assert(ds.invariant());
    return ds;
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_CALIBRATION_DATASET_H
