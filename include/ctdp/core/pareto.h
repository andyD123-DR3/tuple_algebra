// core/pareto.h — Pareto filter and selection policies
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Multi-objective search produces a set of evaluated points, each with a
// cost_vector<D>.  This header provides two operations:
//
//   1. pareto_filter: remove dominated points, yielding the Pareto frontier.
//   2. Selection policies: pick one point from the frontier.
//
// The filter is O(n²) which is fine for the frontier sizes we encounter
// (typically 10–50 points out of hundreds or low thousands of feasible
// candidates).  All operations are constexpr.
//
// The evaluated_point<C, D> type bundles a candidate with its cost vector.
// It is deliberately NOT plan<C> — plan<C> carries scalar predicted_cost.
// After frontier selection, the chosen point is converted to a plan<C>
// using a scalarisation of the cost vector (or one chosen objective).
//
// Dependencies: cost_vector.h, constexpr_vector.h.
// C++ standard: C++20.

#ifndef CTDP_CORE_PARETO_H
#define CTDP_CORE_PARETO_H

#include "cost_vector.h"
#include "constexpr_vector.h"

#include <array>
#include <cstddef>
#include <limits>

namespace ctdp {

// =============================================================================
// evaluated_point — candidate + cost vector bundle
// =============================================================================

/// A search-space point bundled with its multi-objective cost.
template<typename Candidate, std::size_t D>
struct evaluated_point {
    Candidate candidate{};
    cost_vector<D> cost{};

    friend constexpr bool operator==(evaluated_point const&,
                                     evaluated_point const&) = default;
};

// =============================================================================
// pareto_frontier — non-dominated subset extraction
// =============================================================================

/// Extract the Pareto frontier from a set of evaluated points.
///
/// Returns a constexpr_vector containing only the non-dominated points.
/// A point is dominated iff there exists another point whose cost_vector
/// Pareto-dominates it.
///
/// Complexity: O(n²) where n = input size.  Practical for n < 1000.
///
/// Example:
/// ```cpp
/// constexpr_vector<evaluated_point<Config, 3>, 100> points;
/// // ... fill points ...
/// auto frontier = pareto_frontier(points);
/// static_assert(frontier.size() <= points.size());
/// ```
template<typename Candidate, std::size_t D, std::size_t MaxN>
[[nodiscard]] constexpr auto
pareto_frontier(constexpr_vector<evaluated_point<Candidate, D>, MaxN> const& points)
    -> constexpr_vector<evaluated_point<Candidate, D>, MaxN>
{
    constexpr_vector<evaluated_point<Candidate, D>, MaxN> frontier;
    auto const n = points.size();

    for (std::size_t i = 0; i < n; ++i) {
        bool dominated = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && points[j].cost.dominates(points[i].cost)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            frontier.push_back(points[i]);
        }
    }
    return frontier;
}

// =============================================================================
// Selection policies — pick one point from the frontier
// =============================================================================

/// Lexicographic selection: compare objectives in the given priority order.
///
/// Template parameter Priority is a pack of objective indices, most
/// important first.  Ties on objective Priority[0] are broken by
/// Priority[1], and so on.
///
/// Example:
/// ```cpp
/// // Priority: determinism (obj 4), then error (obj 2), then throughput (obj 3)
/// auto winner = lex_select<4, 2, 3>(frontier);
/// ```
template<std::size_t... Priority, typename Candidate, std::size_t D, std::size_t MaxN>
[[nodiscard]] constexpr auto
lex_select(constexpr_vector<evaluated_point<Candidate, D>, MaxN> const& frontier)
    -> evaluated_point<Candidate, D>
{
    static_assert(sizeof...(Priority) > 0, "At least one priority index required");
    static_assert(((Priority < D) && ...), "Priority indices must be < D");

    if (frontier.empty()) {
        return {};  // default-constructed (infeasible sentinel)
    }

    constexpr std::size_t prio[] = {Priority...};
    std::size_t best = 0;

    for (std::size_t i = 1; i < frontier.size(); ++i) {
        bool i_is_better = false;
        for (std::size_t p = 0; p < sizeof...(Priority); ++p) {
            auto const dim = prio[p];
            if (frontier[i].cost[dim] < frontier[best].cost[dim]) {
                i_is_better = true;
                break;
            }
            if (frontier[i].cost[dim] > frontier[best].cost[dim]) {
                break;  // best is better on this priority
            }
            // tie: continue to next priority
        }
        if (i_is_better) {
            best = i;
        }
    }
    return frontier[best];
}

/// Weighted scalarisation: pick the point minimising dot(weights, cost).
///
/// Example:
/// ```cpp
/// auto winner = weighted_select(frontier, cost_vector<6>{{0.1, 0.2, 0.3, 0.2, 0.1, 0.1}});
/// ```
template<typename Candidate, std::size_t D, std::size_t MaxN>
[[nodiscard]] constexpr auto
weighted_select(constexpr_vector<evaluated_point<Candidate, D>, MaxN> const& frontier,
                cost_vector<D> const& weights)
    -> evaluated_point<Candidate, D>
{
    if (frontier.empty()) {
        return {};
    }

    auto scalarise = [&](cost_vector<D> const& cv) -> double {
        double sum = 0.0;
        for (std::size_t i = 0; i < D; ++i) {
            sum += weights[i] * cv[i];
        }
        return sum;
    };

    std::size_t best = 0;
    double best_val = scalarise(frontier[0].cost);

    for (std::size_t i = 1; i < frontier.size(); ++i) {
        double val = scalarise(frontier[i].cost);
        if (val < best_val) {
            best = i;
            best_val = val;
        }
    }
    return frontier[best];
}

// =============================================================================
// pareto_search — convenience: enumerate + filter + return frontier
// =============================================================================

/// Enumerate a space, evaluate multi-objective cost, return Pareto frontier.
///
/// This is the multi-objective analogue of exhaustive_search_with_cost.
/// It evaluates every point in the space, computes a cost_vector<D>,
/// and returns the non-dominated subset.
///
/// MaxEval: upper bound on feasible points (for constexpr_vector capacity).
///
/// Example:
/// ```cpp
/// auto frontier = pareto_search<6, 500>(feasible_space, multi_cost_fn);
/// auto winner = lex_select<4, 2, 3>(frontier);
/// ```
template<std::size_t D, std::size_t MaxEval, typename Space, typename MultiCostFn>
[[nodiscard]] constexpr auto
pareto_search(Space const& space, MultiCostFn const& cost)
{
    using point_type = typename Space::point_type;
    constexpr_vector<evaluated_point<point_type, D>, MaxEval> all_points;

    space.enumerate([&](point_type const& pt) {
        all_points.push_back(evaluated_point<point_type, D>{pt, cost(pt)});
    });

    return pareto_frontier(all_points);
}

} // namespace ctdp

#endif // CTDP_CORE_PARETO_H
