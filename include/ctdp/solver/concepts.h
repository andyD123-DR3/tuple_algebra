// ctdp/solver/concepts.h
// Compile-time dynamic programming framework — Analytics: Solver
// Solver-specific concepts beyond core/concepts.h.
//
// Seven concepts:
//   factored_space          — per-position structural access (num_choices/choice)
//   has_neighbours          — one-parameter: space supports neighbour enumeration
//   neighbourhood_space     — two-parameter: neighbours with specific callable F
//   cost_function_for       — (candidate) → floating-point
//   element_cost_for        — (descriptor, choice_index) → floating-point
//   element_predicate_for   — (descriptor, choice_index) → bool
//   interval_cost           — combine(i,mid,j) + leaf(i)
//   dynamic_constraint_for  — (candidate) → bool, bound to concrete Candidate

#ifndef CTDP_SOLVER_CONCEPTS_H
#define CTDP_SOLVER_CONCEPTS_H

#include "../core/concepts.h"
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ctdp {

// ---------------------------------------------------------------------------
// factored_space: per-position structural access for algorithms that exploit
// independence (per_element_argmin).
//
// Requires num_choices(pos) → size_t and choice(pos, s) → strategy_type.
// These are the methods per_element_argmin actually calls.  This supports
// both uniform spaces (every position has the same choices) and
// heterogeneous spaces (per-position choice sets).
//
// v0.5.5: replaced strategy(idx) with num_choices/choice (Issues A+B).
// ---------------------------------------------------------------------------
template<typename S>
concept factored_space =
    search_space<S> &&
    requires(S const& s, std::size_t idx) {
        typename S::strategy_type;
        { S::dimension } -> std::convertible_to<std::size_t>;
        { S::branching } -> std::convertible_to<std::size_t>;
        { s.num_choices(idx) } -> std::convertible_to<std::size_t>;
        { s.choice(idx, idx) } -> std::convertible_to<typename S::strategy_type>;
    };

// ---------------------------------------------------------------------------
// has_neighbours: one-parameter check — does the space support neighbour
// enumeration at all?  Used by select_and_run for dispatch without
// knowing the callback type F.
// ---------------------------------------------------------------------------
template<typename S>
concept has_neighbours =
    search_space<S> &&
    requires(S const& s, typename S::candidate_type const& c) {
        s.neighbours(c, [](typename S::candidate_type const&){});
    };

// ---------------------------------------------------------------------------
// neighbourhood_space: two-parameter check with specific callable F.
// Used by algorithm templates to constrain the exact callback type.
// Contract: neighbours() invokes f with each neighbouring candidate.
// The original candidate is NOT included.  Order is deterministic.
// ---------------------------------------------------------------------------
template<typename S, typename F>
concept neighbourhood_space =
    has_neighbours<S> &&
    requires(S const& s, typename S::candidate_type const& c, F f) {
        { s.neighbours(c, f) } -> std::same_as<void>;
    };

// ---------------------------------------------------------------------------
// cost_function_for: general cost callable (candidate) → floating-point.
// Descriptors are captured in the callable at construction, not passed
// by the algorithm.
// ---------------------------------------------------------------------------
template<typename F, typename Candidate>
concept cost_function_for =
    std::is_invocable_v<F const&, Candidate const&> &&
    std::floating_point<std::invoke_result_t<F const&, Candidate const&>>;

// ---------------------------------------------------------------------------
// element_cost_for: per-position cost callable (descriptor, choice_index) → double.
//
// The cost function operates in the descriptor domain.  It doesn't know
// about strategy types — just "what does choice j cost at position described
// by descriptor d?"  The space maps choice indices to strategy values.
//
// This is the natural interface for additive cost models: the algorithm
// calls element_cost(descriptor, choice) per position and sums.
// ---------------------------------------------------------------------------
template<typename F, typename Descriptor>
concept element_cost_for =
    std::is_invocable_v<F const&, Descriptor const&, std::size_t> &&
    std::floating_point<std::invoke_result_t<F const&, Descriptor const&, std::size_t>>;

// ---------------------------------------------------------------------------
// element_predicate_for: per-position constraint (descriptor, choice_index) → bool.
//
// Same principle: the constraint operates in the descriptor domain.
// "Is choice j allowed at position described by d?"
// ---------------------------------------------------------------------------
template<typename F, typename Descriptor>
concept element_predicate_for =
    std::is_invocable_r_v<bool, F const&, Descriptor const&, std::size_t>;

// ---------------------------------------------------------------------------
// interval_cost: cost protocol for interval_dp.
//
// Contract (right-start convention):
//   combine(i, mid, j) → cost of merging subproblems [i,mid-1] and [mid,j]
//     where mid is the START INDEX OF THE RIGHT HALF.  Range: i+1 ≤ mid ≤ j.
//   leaf(i) → base cost of a single-element interval [i,i].
//
// The DP algorithm iterates k ∈ [i, j) as the last index of the left half,
// then calls combine(i, k+1, j).  The candidate stores k (last-of-left),
// recoverable as mid = split(i,j) + 1.
//
// Example (matrix chain):
//   combine(i, mid, j) = dims[i] * dims[mid] * dims[j+1]
// ---------------------------------------------------------------------------
template<typename F>
concept interval_cost =
    requires(F const& f, std::size_t i, std::size_t mid, std::size_t j) {
        { f.combine(i, mid, j) } -> std::floating_point;
        { f.leaf(i) }            -> std::floating_point;
    };

// ---------------------------------------------------------------------------
// dynamic_constraint_for: (candidate) → bool, parameterised on Candidate.
// This concept lives here (not in core/) because it binds to a concrete
// candidate type — a solver-level concern.
// ---------------------------------------------------------------------------
template<typename F, typename Candidate>
concept dynamic_constraint_for =
    std::is_invocable_r_v<bool, F const&, Candidate const&>;

} // namespace ctdp

#endif // CTDP_SOLVER_CONCEPTS_H
