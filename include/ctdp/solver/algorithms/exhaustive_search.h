// ctdp/solver/algorithms/exhaustive_search.h
// Compile-time dynamic programming framework — Analytics: Solver
// Enumerate all candidates in the space.  Guarded by ct_limits.
//
// Compile-time limit: static_assert on Space::static_size if available.
// For runtime-sized spaces, the domain selects the regime (§8 of design doc).
//
// Stats field semantics (solve_stats canonical names):
//   candidates_total     — total candidates visited
//   candidates_evaluated — cost evaluations (feasible candidates only)
//   candidates_pruned    — constraint failures
//   memo_*               — not used (each candidate evaluated once)

#ifndef CTDP_SOLVER_ALGORITHMS_EXHAUSTIVE_SEARCH_H
#define CTDP_SOLVER_ALGORITHMS_EXHAUSTIVE_SEARCH_H

#include "../concepts.h"
#include "../../core/ct_limits.h"
#include "../../core/plan.h"
#include "../../core/solve_stats.h"
#include <cstddef>
#include <limits>

namespace ctdp {

template<typename Space, typename Cost, typename... Constraints>
    requires cost_function_for<Cost, typename Space::candidate_type>
[[nodiscard]] constexpr auto exhaustive_search(
    Space const& space,
    Cost const& cost,
    Constraints const&... constraints
) -> plan<typename Space::candidate_type>
{
    using candidate_type = typename Space::candidate_type;

    // Compile-time guard for spaces with static size.
    if constexpr (requires { { Space::static_size } -> std::convertible_to<std::size_t>; }) {
        static_assert(Space::static_size <= ct_limits::exhaustive_max,
            "Space too large for compile-time exhaustive search. "
            "Use beam_search, greedy_search, or build-step mode.");
    }

    candidate_type best{};
    double best_cost = std::numeric_limits<double>::infinity();
    solve_stats stats{};

    space.enumerate([&](candidate_type const& candidate) constexpr {
        stats.candidates_total++;

        // Dynamic constraints — short-circuit on first failure
        bool feasible = true;
        if constexpr (sizeof...(Constraints) > 0) {
            feasible = (constraints(candidate) && ...);
        }
        if (!feasible) {
            stats.candidates_pruned++;
            return;
        }

        double c = cost(candidate);
        stats.candidates_evaluated++;

        if (c < best_cost) {
            best_cost = c;
            best = candidate;
        }
    });

    return plan<candidate_type>{best, best_cost, stats};
}

} // namespace ctdp

#endif // CTDP_SOLVER_ALGORITHMS_EXHAUSTIVE_SEARCH_H
