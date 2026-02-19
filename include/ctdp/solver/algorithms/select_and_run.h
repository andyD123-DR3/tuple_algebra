// ctdp/solver/algorithms/select_and_run.h
// Compile-time dynamic programming framework — Analytics: Solver
// Default heuristic dispatch via concept-constrained overload set.
//
// Dispatch is on space structure (concept overloads), not on cost type.
// The cost concept determines which overload is viable.
// Runtime size thresholds are the domain's responsibility.
//
// Overload map:
//   factored + whole-candidate cost           → per_element_argmin (no constraints)
//   described+factored + element cost         → per_element_argmin (with element preds)
//   factored + whole-candidate cost + dyn.cs  → beam_search (v0.5.5)
//   interval-compatible space + interval cost → interval_dp
//   has_neighbours + cost + dyn. constraints  → local_search (v0.5.5)
//   general enumerable + whole-candidate cost → exhaustive_search (with constraints)

#ifndef CTDP_SOLVER_ALGORITHMS_SELECT_AND_RUN_H
#define CTDP_SOLVER_ALGORITHMS_SELECT_AND_RUN_H

#include "../concepts.h"
#include "per_element_argmin.h"
#include "interval_dp.h"
#include "exhaustive_search.h"
#include "beam_search.h"
#include "local_search.h"

namespace ctdp {

// ---------------------------------------------------------------------------
// interval_dp_compatible: structural check for spaces that interval_dp needs.
// Requires max_size (compile-time capacity) and n (runtime problem size).
// ---------------------------------------------------------------------------
template<typename S>
concept interval_dp_compatible =
    search_space<S> &&
    requires(S const& s) {
        { S::max_size } -> std::convertible_to<std::size_t>;
        { s.n }          -> std::convertible_to<std::size_t>;
    };

// Overload 1a: factored + whole-candidate cost → per_element_argmin (no constraints)
template<factored_space Space, typename Cost>
    requires cost_function_for<Cost, typename Space::candidate_type>
         && (!interval_cost<Cost>)
         && (!described_space<Space>
             || !element_cost_for<Cost, typename Space::descriptor_type>)
[[nodiscard]] constexpr auto select_and_run(
    Space const& space,
    Cost const& cost)
    -> plan<typename Space::candidate_type>
{
    return per_element_argmin(space, cost);
}

// Overload 1b: described+factored + element cost → per_element_argmin (with element preds)
template<typename Space, typename ElementCost, typename... ElementConstraints>
    requires described_space<Space>
         && factored_space<Space>
         && element_cost_for<ElementCost, typename Space::descriptor_type>
         && (element_predicate_for<ElementConstraints, typename Space::descriptor_type> && ...)
[[nodiscard]] constexpr auto select_and_run(
    Space const& space,
    ElementCost const& cost,
    ElementConstraints const&... cs)
    -> plan<typename Space::candidate_type>
{
    return per_element_argmin(space, cost, cs...);
}

// Overload 1c: factored + whole-candidate cost + dynamic constraints → beam_search
// This overload handles the case that per_element_argmin cannot: global constraints
// that inspect multiple positions simultaneously (budget, mutual exclusion, etc.).
template<typename Space, typename Cost, typename FirstConstraint, typename... MoreConstraints>
    requires factored_space<Space>
         && cost_function_for<Cost, typename Space::candidate_type>
         && (!interval_cost<Cost>)
         && dynamic_constraint_for<FirstConstraint, typename Space::candidate_type>
         && (dynamic_constraint_for<MoreConstraints, typename Space::candidate_type> && ...)
[[nodiscard]] constexpr auto select_and_run(
    Space const& space,
    Cost const& cost,
    FirstConstraint const& c0,
    MoreConstraints const&... cs)
    -> plan<typename Space::candidate_type>
{
    return beam_search(space, cost, c0, cs...);
}

// Overload 2: interval-compatible spaces with interval cost → interval_dp
template<interval_dp_compatible Space, interval_cost Cost>
[[nodiscard]] constexpr auto select_and_run(
    Space const& space,
    Cost const& cost)
    -> plan<typename Space::candidate_type>
{
    return interval_dp(space, cost);
}

// Overload 3: general enumerable spaces → exhaustive_search (with constraints)
// Positive requirement: must be enumerable.
// Negative: not already handled by overloads 1 or 2.
template<typename Space, typename Cost, typename... Constraints>
    requires (!factored_space<Space>)
         && (!interval_dp_compatible<Space>)
         && cost_function_for<Cost, typename Space::candidate_type>
[[nodiscard]] constexpr auto select_and_run(
    Space const& space,
    Cost const& cost,
    Constraints const&... cs)
    -> plan<typename Space::candidate_type>
{
    return exhaustive_search(space, cost, cs...);
}

} // namespace ctdp

#endif // CTDP_SOLVER_ALGORITHMS_SELECT_AND_RUN_H
