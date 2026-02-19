// ctdp/solver/algorithms/local_search.h
// Compile-time dynamic programming framework — Analytics: Solver
// Steepest-descent local search for neighbourhood spaces with constraints.
//
// Starting from an initial candidate, repeatedly moves to the best
// improving feasible neighbour.  Terminates when no improving neighbour
// exists (local optimum) or when the iteration limit is reached.
//
// Requires: has_neighbours<Space> — the space must define a
//   neighbours(candidate, callback) method.  This is satisfied by
//   permutation_space, cartesian_space, per_element_space (v0.5.5+),
//   and heterogeneous_per_element_space (v0.5.5+).
//
// Template parameters:
//   MaxIter — maximum descent iterations (default: 1000)
//
// Stats field semantics:
//   candidates_total      — total neighbours examined across all iterations
//   candidates_evaluated  — cost evaluations (feasible neighbours)
//   candidates_pruned     — constraint failures
//   local_search_moves    — iterations actually performed

#ifndef CTDP_SOLVER_ALGORITHMS_LOCAL_SEARCH_H
#define CTDP_SOLVER_ALGORITHMS_LOCAL_SEARCH_H

#include "../concepts.h"
#include "../../core/ct_limits.h"
#include "../../core/plan.h"
#include "../../core/solve_stats.h"
#include <cstddef>
#include <limits>

namespace ctdp {

// =============================================================================
// local_search: steepest-descent with constraint support
// =============================================================================

template<std::size_t MaxIter = 1000,
         typename Space, typename Cost, typename... Constraints>
    requires has_neighbours<Space>
         && cost_function_for<Cost, typename Space::candidate_type>
         && (dynamic_constraint_for<Constraints, typename Space::candidate_type> && ...)
[[nodiscard]] constexpr auto local_search(
    Space const& space,
    Cost const& cost,
    typename Space::candidate_type const& initial,
    Constraints const&... constraints
) -> plan<typename Space::candidate_type>
{
    using candidate_type = typename Space::candidate_type;

    static_assert(MaxIter >= 1, "MaxIter must be at least 1.");

    solve_stats stats{};

    // --- Evaluate initial candidate ---

    candidate_type current = initial;

    // Check initial feasibility.
    bool feasible = true;
    if constexpr (sizeof...(Constraints) > 0) {
        feasible = (constraints(current) && ...);
    }

    if (!feasible) {
        // Initial candidate is infeasible — cannot proceed.
        stats.candidates_total = 1;
        stats.candidates_pruned = 1;
        return plan<candidate_type>{
            candidate_type{},
            std::numeric_limits<double>::infinity(),
            stats
        };
    }

    double current_cost = cost(current);
    stats.candidates_evaluated = 1;
    stats.candidates_total = 1;

    // --- Steepest-descent loop ---

    for (std::size_t iter = 0; iter < MaxIter; ++iter) {
        candidate_type best_neighbour{};
        double best_neighbour_cost = std::numeric_limits<double>::infinity();
        bool found_improvement = false;

        space.neighbours(current,
            [&](candidate_type const& neighbour) constexpr {
                stats.candidates_total++;

                // Constraint check.
                bool nbr_feasible = true;
                if constexpr (sizeof...(Constraints) > 0) {
                    nbr_feasible = (constraints(neighbour) && ...);
                }
                if (!nbr_feasible) {
                    stats.candidates_pruned++;
                    return;
                }

                double c = cost(neighbour);
                stats.candidates_evaluated++;

                if (c < current_cost && c < best_neighbour_cost) {
                    best_neighbour = neighbour;
                    best_neighbour_cost = c;
                    found_improvement = true;
                }
            });

        stats.local_search_moves++;

        if (!found_improvement) {
            break;  // Local optimum reached.
        }

        current = best_neighbour;
        current_cost = best_neighbour_cost;
    }

    return plan<candidate_type>{current, current_cost, stats};
}

// =============================================================================
// Convenience: local_search from default candidate
// =============================================================================
//
// For spaces that provide a natural "identity" or "default" starting point,
// this overload constructs the initial candidate automatically.

template<std::size_t MaxIter = 1000,
         typename Space, typename Cost, typename... Constraints>
    requires has_neighbours<Space>
         && cost_function_for<Cost, typename Space::candidate_type>
         && (dynamic_constraint_for<Constraints, typename Space::candidate_type> && ...)
[[nodiscard]] constexpr auto local_search(
    Space const& space,
    Cost const& cost,
    Constraints const&... constraints
) -> plan<typename Space::candidate_type>
{
    typename Space::candidate_type initial{};
    return local_search<MaxIter>(space, cost, initial, constraints...);
}

} // namespace ctdp

#endif // CTDP_SOLVER_ALGORITHMS_LOCAL_SEARCH_H
