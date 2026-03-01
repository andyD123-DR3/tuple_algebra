#ifndef CTDP_CALIBRATOR_PLAN_H
#define CTDP_CALIBRATOR_PLAN_H

// ctdp::calibrator::plan — Optimal dispatch plan
//
// Design v2.2 §2, §6, §8.6:
//   plan<Space, Callable> is the terminal output of the calibration →
//   optimisation pipeline.  It maps space points to predicted costs
//   and identifies the Pareto-optimal configurations.
//
//   The Callable type parameter flows from Scenario → Harness →
//   Dataset → Profile → Plan, ensuring that a plan fitted from
//   fix_swar_parser measurements cannot be consumed by a
//   fix_lookup_parser dispatcher.
//
// A plan can represent:
//   - Single-objective: one optimal point (minimum cost)
//   - Multi-objective:  a Pareto frontier of non-dominated points
//   - Constrained:      optimal within user-defined feasibility bounds
//
// Usage:
//   auto plan = exhaustive_search(profile, scenario);
//   auto best = plan.optimal_point();        // lowest cost
//   auto top5 = plan.top_n(5);               // 5 cheapest points
//   auto cost = plan.predict(some_point);     // predicted cost

#include "provenance.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

/// A single entry in a plan: a space point and its predicted cost.
template <typename PointType>
struct plan_entry {
    PointType point;
    double    cost_ns = 0.0;  ///< Predicted wall-clock cost (ns)

    /// Additional objective values for multi-objective plans.
    /// Index 0 is always cost_ns (duplicated for uniformity).
    /// Further indices map to user-defined objectives.
    std::vector<double> objectives;

    /// Whether this entry is Pareto-optimal in the multi-objective sense.
    bool pareto_optimal = false;
};

/// An optimisation plan: the output of the solver.
///
/// @tparam Space     The search space type (provides point_type)
/// @tparam Callable  The kernel identity (compile-time type key)
///
/// The plan is immutable after construction.  It is the "wisdom file"
/// at the type level — the frozen result of calibration + optimisation
/// that can be:
///   - Queried for optimal points
///   - Emitted as a constexpr header
///   - Validated against fresh measurements
///   - Consumed by a code instantiation layer
///
template <typename Space, typename Callable>
struct plan {
    using space_type    = Space;
    using callable_type = Callable;
    using point_type    = typename Space::point_type;
    using entry_type    = plan_entry<point_type>;

    // ─── Data ────────────────────────────────────────────────────

    /// All evaluated entries, sorted by cost_ns ascending.
    std::vector<entry_type> entries;

    /// Provenance: where the profile came from.
    dataset_provenance provenance;

    /// Number of points evaluated by the solver.
    std::size_t evaluated_points = 0;

    /// Name of the solver algorithm used.
    std::string solver_name;

    // ─── Single-objective queries ────────────────────────────────

    /// The optimal (lowest cost) point.
    /// Precondition: !entries.empty()
    [[nodiscard]] point_type const& optimal_point() const {
        assert(!entries.empty());
        return entries.front().point;
    }

    /// The optimal cost (ns).
    [[nodiscard]] double optimal_cost() const {
        assert(!entries.empty());
        return entries.front().cost_ns;
    }

    /// The N cheapest entries.
    [[nodiscard]] std::vector<entry_type> top_n(std::size_t n) const {
        auto count = std::min(n, entries.size());
        return {entries.begin(), entries.begin()
                + static_cast<std::ptrdiff_t>(count)};
    }

    /// Predict cost for a specific point.  Returns -1.0 if not in plan.
    [[nodiscard]] double predict(point_type const& pt) const {
        for (auto const& e : entries) {
            if (e.point == pt) return e.cost_ns;
        }
        return -1.0;
    }

    // ─── Multi-objective queries ─────────────────────────────────

    /// All Pareto-optimal entries (non-dominated in the multi-objective sense).
    [[nodiscard]] std::vector<entry_type> pareto_frontier() const {
        std::vector<entry_type> result;
        for (auto const& e : entries) {
            if (e.pareto_optimal) result.push_back(e);
        }
        return result;
    }

    /// Number of Pareto-optimal points.
    [[nodiscard]] std::size_t pareto_size() const {
        std::size_t count = 0;
        for (auto const& e : entries) {
            if (e.pareto_optimal) ++count;
        }
        return count;
    }

    // ─── Utilities ───────────────────────────────────────────────

    /// Total number of entries in the plan.
    [[nodiscard]] std::size_t size() const noexcept {
        return entries.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return entries.empty();
    }

    /// Sort entries by cost ascending (primary ordering).
    void sort_by_cost() {
        std::sort(entries.begin(), entries.end(),
            [](entry_type const& a, entry_type const& b) {
                return a.cost_ns < b.cost_ns;
            });
    }

    /// Filter entries by a predicate on the point.
    template <typename Pred>
    [[nodiscard]] plan filtered(Pred&& pred) const {
        plan result;
        result.provenance   = provenance;
        result.solver_name  = solver_name;
        for (auto const& e : entries) {
            if (pred(e.point)) {
                result.entries.push_back(e);
            }
        }
        result.evaluated_points = result.entries.size();
        return result;
    }
};

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_PLAN_H
