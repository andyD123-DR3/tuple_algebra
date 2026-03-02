#ifndef CTDP_CALIBRATOR_DISPATCH_TABLE_H
#define CTDP_CALIBRATOR_DISPATCH_TABLE_H

// ctdp::calibrator::dispatch_table — Runtime dispatch from plan
//
// The dispatch table is the "manufacturing die" — the frozen result
// of calibration + optimisation that executes at zero cost.
//
// Given a plan<Space, Callable>, a dispatch_table maps space points
// to their ranked entries (cost, Pareto status, objectives).
// For the common case (single-objective, "just give me the best"),
// it provides O(1) lookup of the optimal point.
//
// Usage:
//
//   auto plan = build_plan(profile, scenario);
//   auto table = make_dispatch_table(plan);
//
//   // O(1) optimal point
//   auto best = table.optimal();
//
//   // Lookup cost for a specific point
//   if (auto* e = table.lookup(point)) {
//       use(e->cost_ns);
//   }
//
//   // Ranked iteration (cheapest first)
//   for (auto const& entry : table.ranked()) {
//       process(entry);
//   }
//
//   // Dispatch: call a visitor with the optimal point
//   table.dispatch([](auto const& pt, double cost) {
//       execute_kernel(pt);
//   });

#include "plan.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

/// A dispatch table entry: point + cost + metadata.
template <typename PointType>
struct dispatch_entry {
    PointType   point;
    double      cost_ns        = 0.0;
    std::size_t rank           = 0;      ///< 0 = optimal
    bool        pareto_optimal = false;
    std::vector<double> objectives;      ///< Multi-objective values
};

/// Runtime dispatch table built from a plan.
///
/// @tparam Space     Search space type
/// @tparam Callable  Kernel identity
///
/// The table stores entries sorted by cost (ascending).  It provides:
///   - O(1) optimal point access
///   - O(N) lookup by point equality
///   - Ranked iteration
///   - Visitor-based dispatch
///   - Filtering by predicate
///   - Pareto frontier access
///
template <typename Space, typename Callable>
class dispatch_table {
public:
    using space_type    = Space;
    using callable_type = Callable;
    using point_type    = typename Space::point_type;
    using entry_type    = dispatch_entry<point_type>;

    // ─── Construction ────────────────────────────────────────────

    dispatch_table() = default;

    /// Build from a plan (entries are already sorted by cost).
    explicit dispatch_table(plan<Space, Callable> const& p)
        : provenance_info_(p.provenance.scenario_name
                           + " [" + p.solver_name + "]")
    {
        entries_.reserve(p.entries.size());
        for (std::size_t i = 0; i < p.entries.size(); ++i) {
            auto const& pe = p.entries[i];
            entries_.push_back(entry_type{
                .point          = pe.point,
                .cost_ns        = pe.cost_ns,
                .rank           = i,
                .pareto_optimal = pe.pareto_optimal,
                .objectives     = pe.objectives
            });
        }
    }

    // ─── Optimal point access ────────────────────────────────────

    /// The optimal (cheapest) entry.  O(1).
    /// Precondition: !empty()
    [[nodiscard]] entry_type const& optimal() const {
        assert(!entries_.empty());
        return entries_.front();
    }

    /// The optimal point.
    [[nodiscard]] point_type const& optimal_point() const {
        return optimal().point;
    }

    /// The optimal cost.
    [[nodiscard]] double optimal_cost() const {
        return optimal().cost_ns;
    }

    // ─── Lookup ──────────────────────────────────────────────────

    /// Look up an entry by point equality.  O(N).
    /// Returns nullptr if not found.
    [[nodiscard]] entry_type const* lookup(point_type const& pt) const {
        for (auto const& e : entries_) {
            if (e.point == pt) return &e;
        }
        return nullptr;
    }

    /// Predict cost for a point.  Returns -1.0 if not found.
    [[nodiscard]] double predict(point_type const& pt) const {
        auto* e = lookup(pt);
        return e ? e->cost_ns : -1.0;
    }

    // ─── Ranked access ───────────────────────────────────────────

    /// All entries, ranked by cost (cheapest first).
    [[nodiscard]] std::vector<entry_type> const& ranked() const {
        return entries_;
    }

    /// Top N entries by cost.
    [[nodiscard]] std::vector<entry_type> top_n(std::size_t n) const {
        auto count = std::min(n, entries_.size());
        return {entries_.begin(), entries_.begin()
                + static_cast<std::ptrdiff_t>(count)};
    }

    /// The Nth-ranked entry (0 = optimal).
    [[nodiscard]] entry_type const& at_rank(std::size_t rank) const {
        assert(rank < entries_.size());
        return entries_[rank];
    }

    // ─── Pareto frontier ─────────────────────────────────────────

    /// All Pareto-optimal entries.
    [[nodiscard]] std::vector<entry_type> pareto_frontier() const {
        std::vector<entry_type> result;
        for (auto const& e : entries_) {
            if (e.pareto_optimal) result.push_back(e);
        }
        return result;
    }

    // ─── Dispatch ────────────────────────────────────────────────

    /// Call a visitor with the optimal point and its cost.
    template <typename Visitor>
    void dispatch(Visitor&& visitor) const {
        assert(!entries_.empty());
        auto const& best = entries_.front();
        visitor(best.point, best.cost_ns);
    }

    /// Call a visitor with the Nth-ranked point.
    template <typename Visitor>
    void dispatch_rank(std::size_t rank, Visitor&& visitor) const {
        assert(rank < entries_.size());
        auto const& e = entries_[rank];
        visitor(e.point, e.cost_ns);
    }

    /// Call a visitor with every entry matching a predicate.
    template <typename Pred, typename Visitor>
    void dispatch_if(Pred&& pred, Visitor&& visitor) const {
        for (auto const& e : entries_) {
            if (pred(e.point)) {
                visitor(e.point, e.cost_ns);
            }
        }
    }

    // ─── Filtering ───────────────────────────────────────────────

    /// Return a new table with only entries matching the predicate.
    template <typename Pred>
    [[nodiscard]] dispatch_table filtered(Pred&& pred) const {
        dispatch_table result;
        result.provenance_info_ = provenance_info_;
        std::size_t rank = 0;
        for (auto const& e : entries_) {
            if (pred(e.point)) {
                auto copy = e;
                copy.rank = rank++;
                result.entries_.push_back(std::move(copy));
            }
        }
        return result;
    }

    // ─── Size / empty ────────────────────────────────────────────

    [[nodiscard]] std::size_t size() const noexcept {
        return entries_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }

    // ─── Metadata ────────────────────────────────────────────────

    [[nodiscard]] std::string const& provenance_info() const {
        return provenance_info_;
    }

    // ─── Summary ─────────────────────────────────────────────────

    /// Human-readable summary string.
    [[nodiscard]] std::string summary() const {
        std::string s = "dispatch_table [" + provenance_info_ + "]: ";
        s += std::to_string(entries_.size()) + " entries";
        if (!entries_.empty()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), ", optimal=%.2f ns",
                          entries_.front().cost_ns);
            s += buf;
        }
        auto pareto_count = std::count_if(entries_.begin(), entries_.end(),
            [](auto const& e) { return e.pareto_optimal; });
        if (pareto_count > 0) {
            s += ", " + std::to_string(pareto_count) + " Pareto-optimal";
        }
        return s;
    }

private:
    std::vector<entry_type> entries_;
    std::string provenance_info_;
};

// ─── Factory ─────────────────────────────────────────────────────

/// Build a dispatch table from a plan.
template <typename Space, typename Callable>
[[nodiscard]] auto make_dispatch_table(plan<Space, Callable> const& p)
    -> dispatch_table<Space, Callable>
{
    return dispatch_table<Space, Callable>{p};
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_DISPATCH_TABLE_H
