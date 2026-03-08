// core/cost_vector.h — Multi-objective cost vector with Pareto dominance
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Single-objective optimisation returns one cost (double).  Multi-objective
// optimisation returns a vector of D objectives, each "lower is better."
// cost_vector<D> is the vocabulary type for this: a fixed-size array of
// doubles with a constexpr dominance test.
//
// This complements plan<C> (which carries scalar predicted_cost) and
// plan_set<C, MaxN, Dom> (which accepts any dominance predicate).
// For multi-objective search, the workflow is:
//
//   1. Enumerate space, evaluate cost_vector<D> per point.
//   2. Pareto-filter to the non-dominated frontier.
//   3. Apply a selection policy (lexicographic, weighted, minimax regret)
//      to pick one point from the frontier.
//
// cost_vector lives in core/ because it is consumed by both the solver
// layer (pareto_search) and the plan layer (plan_set with Pareto dominance).
//
// Dependencies: none (std only).
// C++ standard: C++20.

#ifndef CTDP_CORE_COST_VECTOR_H
#define CTDP_CORE_COST_VECTOR_H

#include <array>
#include <cstddef>

namespace ctdp {

// =============================================================================
// cost_vector<D> — fixed-size multi-objective cost
// =============================================================================

/// A vector of D objective values, each "lower is better."
///
/// Provides Pareto dominance testing and element access.
/// All operations are constexpr.
///
/// Example:
/// ```cpp
/// cost_vector<3> a{{1.0, 2.0, 3.0}};
/// cost_vector<3> b{{1.0, 3.0, 3.0}};
/// static_assert(a.dominates(b));   // a ≤ b in all, strict in obj[1]
/// static_assert(!b.dominates(a));
/// ```
template<std::size_t D>
struct cost_vector {
    std::array<double, D> obj{};

    // ─── Element access ──────────────────────────────────────────────

    [[nodiscard]] constexpr double operator[](std::size_t i) const {
        return obj[i];
    }

    [[nodiscard]] constexpr double& operator[](std::size_t i) {
        return obj[i];
    }

    [[nodiscard]] static constexpr std::size_t dimensions() { return D; }

    // ─── Pareto dominance ────────────────────────────────────────────

    /// Returns true iff this vector Pareto-dominates other:
    ///   - this[i] <= other[i] for ALL i, AND
    ///   - this[j] <  other[j] for at least one j.
    [[nodiscard]] constexpr bool dominates(cost_vector const& other) const {
        bool at_least_one_strict = false;
        for (std::size_t i = 0; i < D; ++i) {
            if (obj[i] > other.obj[i]) return false;
            if (obj[i] < other.obj[i]) at_least_one_strict = true;
        }
        return at_least_one_strict;
    }

    /// Returns true iff neither vector dominates the other.
    [[nodiscard]] constexpr bool incomparable(cost_vector const& other) const {
        return !dominates(other) && !other.dominates(*this);
    }

    // ─── Comparison ──────────────────────────────────────────────────

    friend constexpr bool operator==(cost_vector const&,
                                     cost_vector const&) = default;
};

// =============================================================================
// Compile-time verification
// =============================================================================

namespace detail {
    // dominance: a <= b in all dimensions, strictly less in at least one
    static_assert( cost_vector<3>{{1.0, 2.0, 3.0}}.dominates(
                   cost_vector<3>{{1.0, 3.0, 3.0}}));
    // not dominated: worse in dimension 0
    static_assert(!cost_vector<3>{{2.0, 1.0, 1.0}}.dominates(
                   cost_vector<3>{{1.0, 3.0, 3.0}}));
    // equal: not dominated (requires strict improvement)
    static_assert(!cost_vector<3>{{1.0, 2.0, 3.0}}.dominates(
                   cost_vector<3>{{1.0, 2.0, 3.0}}));
    // incomparable: each better in different objectives
    static_assert( cost_vector<2>{{1.0, 3.0}}.incomparable(
                   cost_vector<2>{{2.0, 1.0}}));
} // namespace detail

} // namespace ctdp

#endif // CTDP_CORE_COST_VECTOR_H
