// ctdp/solver/algorithms/per_element_argmin.h
// Compile-time dynamic programming framework — Analytics: Solver
// Independent per-position optimisation for factored spaces.
//
// NOT dynamic programming — when cost is additive and positions are
// independent, each position is a standalone argmin.
//
// Two overloads:
//
//   1. Whole-candidate cost: cost(candidate) → double
//      NO constraints.  If the cost is truly decomposable (no cross-
//      position dependencies), unconstrained argmin is correct.
//      If you need constraints with a whole-candidate cost, use
//      exhaustive_search.
//
//   2. Element cost: cost(descriptor, choice_index) → double
//      With optional element predicates: (descriptor, choice_index) → bool.
//      Element predicates are per-position by construction — they cannot
//      see other positions, so they are always safe here.  This is the
//      preferred API for described spaces.
//
// DESIGN NOTE: The whole-candidate overload deliberately does not accept
// constraints.  Evaluating whole-candidate constraints on partial candidates
// (where undecided positions hold default values) produces wrong results
// for any constraint that inspects more than one position.  See review
// item A in solver_review.md.

#ifndef CTDP_SOLVER_ALGORITHMS_PER_ELEMENT_ARGMIN_H
#define CTDP_SOLVER_ALGORITHMS_PER_ELEMENT_ARGMIN_H

#include "../concepts.h"
#include "../../core/ct_limits.h"
#include "../../core/plan.h"
#include "../../core/solve_stats.h"
#include <cstddef>
#include <limits>

namespace ctdp {

// =============================================================================
// Overload 1: whole-candidate cost, no constraints
// =============================================================================

template<factored_space Space, typename Cost>
    requires cost_function_for<Cost, typename Space::candidate_type>
         && (!described_space<Space>
             || !element_cost_for<Cost, typename Space::descriptor_type>)
[[nodiscard]] constexpr auto per_element_argmin(
    Space const& space,
    Cost const& cost
) -> plan<typename Space::candidate_type>
{
    using candidate_type = typename Space::candidate_type;

    static_assert(Space::dimension * Space::branching <= ct_limits::exhaustive_max,
        "per_element_argmin state space (N*S) exceeds compile-time limit. "
        "Use build-step mode or increase ct_limits.");

    candidate_type best{};
    solve_stats stats{};
    stats.subproblems_total = Space::dimension;

    for (std::size_t pos = 0; pos < Space::dimension; ++pos) {
        double best_pos_cost = std::numeric_limits<double>::infinity();
        bool any_selected = false;

        auto const nc = space.num_choices(pos);
        for (std::size_t s = 0; s < nc; ++s) {
            auto strategy = space.choice(pos, s);
            stats.candidates_total++;

            candidate_type trial = best;
            trial[pos] = strategy;

            double c = cost(trial);
            stats.candidates_evaluated++;

            if (c < best_pos_cost) {
                best_pos_cost = c;
                best[pos] = strategy;
                any_selected = true;
            }
        }

        // All strategies at this position yielded infinity cost
        if (!any_selected) {
            return plan<candidate_type>{
                candidate_type{},
                std::numeric_limits<double>::infinity(),
                stats
            };
        }

        stats.subproblems_evaluated++;
    }

    double total_cost = cost(best);
    return plan<candidate_type>{best, total_cost, stats};
}

// =============================================================================
// Overload 2: element cost + element predicates (descriptor-aware)
// =============================================================================

template<typename Space, typename ElementCost, typename... ElementConstraints>
    requires described_space<Space>
         && factored_space<Space>
         && element_cost_for<ElementCost, typename Space::descriptor_type>
         && (element_predicate_for<ElementConstraints, typename Space::descriptor_type> && ...)
[[nodiscard]] constexpr auto per_element_argmin(
    Space const& space,
    ElementCost const& cost,
    ElementConstraints const&... constraints
) -> plan<typename Space::candidate_type>
{
    using candidate_type = typename Space::candidate_type;

    static_assert(Space::dimension * Space::branching <= ct_limits::exhaustive_max,
        "per_element_argmin state space (N*S) exceeds compile-time limit.");

    candidate_type best{};
    double total_cost = 0.0;
    solve_stats stats{};
    stats.subproblems_total = Space::dimension;

    for (std::size_t pos = 0; pos < Space::dimension; ++pos) {
        auto const desc = space.descriptor(pos);
        double best_pos_cost = std::numeric_limits<double>::infinity();
        bool any_selected = false;

        auto const nc = space.num_choices(pos);
        for (std::size_t s = 0; s < nc; ++s) {
            stats.candidates_total++;

            bool feasible = true;
            if constexpr (sizeof...(ElementConstraints) > 0) {
                feasible = (constraints(desc, s) && ...);
            }
            if (!feasible) {
                stats.candidates_pruned++;
                continue;
            }

            double c = cost(desc, s);
            stats.candidates_evaluated++;

            if (c < best_pos_cost) {
                best_pos_cost = c;
                best[pos] = space.choice(pos, s);
                any_selected = true;
            }
        }

        // All strategies at this position were infeasible
        if (!any_selected) {
            return plan<candidate_type>{
                candidate_type{},
                std::numeric_limits<double>::infinity(),
                stats
            };
        }

        total_cost += best_pos_cost;
        stats.subproblems_evaluated++;
    }

    return plan<candidate_type>{best, total_cost, stats};
}

} // namespace ctdp

#endif // CTDP_SOLVER_ALGORITHMS_PER_ELEMENT_ARGMIN_H
