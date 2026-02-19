// ctdp/solver/algorithms/beam_search.h
// Compile-time dynamic programming framework — Analytics: Solver
// Constructive beam search for factored spaces with global constraints.
//
// per_element_argmin exploits independence: O(N×S).  But it cannot
// handle whole-candidate constraints — predicates that inspect multiple
// positions simultaneously (budget caps, mutual exclusion, etc.).
//
// exhaustive_search handles constraints but enumerates S^N candidates.
//
// beam_search bridges this gap:
//   - Builds candidates level-by-level (one position per level)
//   - At each level, expands each beam entry by all choices at that position
//   - Evaluates cost and constraints on the partially-built candidate
//   - Keeps only the top BeamWidth feasible candidates (by cost)
//   - After N levels, returns the best complete candidate
//
// Complexity: O(N × BeamWidth × branching) cost evaluations.
// This is exact for BeamWidth ≥ S^(N-1), and heuristic otherwise.
//
// Template parameters:
//   BeamWidth — number of candidates retained at each level (default: 32)
//
// Stats field semantics:
//   candidates_total     — total expansions attempted
//   candidates_evaluated — cost evaluations (feasible expansions)
//   candidates_pruned    — constraint failures
//   subproblems_total    — Space::dimension (levels)
//   subproblems_evaluated — levels actually expanded
//   beam_width_used      — BeamWidth

#ifndef CTDP_SOLVER_ALGORITHMS_BEAM_SEARCH_H
#define CTDP_SOLVER_ALGORITHMS_BEAM_SEARCH_H

#include "../concepts.h"
#include "../../core/ct_limits.h"
#include "../../core/plan.h"
#include "../../core/solve_stats.h"
#include <array>
#include <cstddef>
#include <limits>

namespace ctdp {

namespace detail {

// Beam entry: candidate + its cost.
template<typename Candidate>
struct beam_entry {
    Candidate candidate{};
    double cost = std::numeric_limits<double>::infinity();
};

// Constexpr insertion sort for beam entries by cost (ascending).
// Used instead of constexpr_sort to avoid header coupling; beam
// sizes are small enough that O(n²) is faster than the overhead
// of heap sort.
template<typename Candidate, std::size_t N>
constexpr void beam_sort(std::array<beam_entry<Candidate>, N>& buf,
                         std::size_t count) {
    for (std::size_t i = 1; i < count; ++i) {
        auto key = buf[i];
        std::size_t j = i;
        while (j > 0 && buf[j - 1].cost > key.cost) {
            buf[j] = buf[j - 1];
            --j;
        }
        buf[j] = key;
    }
}

} // namespace detail

// =============================================================================
// beam_search: constrained search for factored spaces
// =============================================================================
//
// CONSTRAINT APPLICATION STRATEGY:
// Constraints are applied only after all N dimensions have been filled.
// During level-by-level expansion, candidates are ranked by cost alone.
//
// Rationale: a general constraint lambda inspects the entire candidate
// including positions not yet filled (which hold default values).  This
// causes spurious rejections — e.g., mutual_exclusion(c[0], c[1]) would
// reject c[0]=Fast at level 0 because c[1] is still default=Fast.
//
// Deferring constraint evaluation to the final beam guarantees correctness
// for all constraint types.  Users who want early pruning can encode
// position-aware logic in their constraint lambdas.

template<std::size_t BeamWidth = 32,
         typename Space, typename Cost, typename... Constraints>
    requires factored_space<Space>
         && cost_function_for<Cost, typename Space::candidate_type>
         && (dynamic_constraint_for<Constraints, typename Space::candidate_type> && ...)
[[nodiscard]] constexpr auto beam_search(
    Space const& space,
    Cost const& cost,
    Constraints const&... constraints
) -> plan<typename Space::candidate_type>
{
    using candidate_type = typename Space::candidate_type;
    using entry_type = detail::beam_entry<candidate_type>;

    static_assert(BeamWidth >= 1, "BeamWidth must be at least 1.");
    static_assert(BeamWidth <= ct_limits::beam_width_max,
        "BeamWidth exceeds ct_limits::beam_width_max. "
        "Increase ct_limits or reduce beam width.");

    // Expansion buffer: each beam entry can produce up to branching children.
    constexpr std::size_t MaxExpanded = BeamWidth * Space::branching;

    // Compile-time sanity: expansion buffer must be representable.
    static_assert(MaxExpanded / Space::branching == BeamWidth,
        "BeamWidth * branching overflows. Reduce BeamWidth.");

    solve_stats stats{};
    stats.subproblems_total = Space::dimension;
    stats.beam_width_used = BeamWidth;

    // Double-buffered beam: current and expanded.
    std::array<entry_type, BeamWidth>    beam{};
    std::array<entry_type, MaxExpanded>  expanded{};
    std::size_t beam_size = 0;

    // Seed beam with default candidate.
    {
        candidate_type seed{};
        stats.candidates_total++;
        double c = cost(seed);
        stats.candidates_evaluated++;
        beam[0] = entry_type{seed, c};
        beam_size = 1;
    }

    // Level-by-level expansion: one dimension per level.
    for (std::size_t d = 0; d < Space::dimension; ++d) {
        std::size_t exp_size = 0;
        auto const nc = space.num_choices(d);

        for (std::size_t b = 0; b < beam_size; ++b) {
            for (std::size_t s = 0; s < nc; ++s) {
                stats.candidates_total++;

                candidate_type trial = beam[b].candidate;
                trial[d] = space.choice(d, s);

                double c = cost(trial);
                stats.candidates_evaluated++;

                if (exp_size < MaxExpanded) {
                    expanded[exp_size] = entry_type{trial, c};
                    ++exp_size;
                } else {
                    // Buffer full — replace worst if this is better.
                    std::size_t worst_idx = 0;
                    double worst_cost = expanded[0].cost;
                    for (std::size_t k = 1; k < MaxExpanded; ++k) {
                        if (expanded[k].cost > worst_cost) {
                            worst_cost = expanded[k].cost;
                            worst_idx = k;
                        }
                    }
                    if (c < worst_cost) {
                        expanded[worst_idx] = entry_type{trial, c};
                    }
                }
            }
        }

        if (exp_size == 0) {
            // No expansions at this level (should not happen for valid spaces).
            return plan<candidate_type>{
                candidate_type{},
                std::numeric_limits<double>::infinity(),
                stats
            };
        }

        // Sort expanded buffer by cost (ascending) and truncate to beam width.
        detail::beam_sort(expanded, exp_size);

        beam_size = exp_size < BeamWidth ? exp_size : BeamWidth;
        for (std::size_t i = 0; i < beam_size; ++i) {
            beam[i] = expanded[i];
        }

        stats.subproblems_evaluated++;
    }

    // --- Post-construction: apply constraints on complete candidates ---

    if constexpr (sizeof...(Constraints) > 0) {
        // Scan beam for best feasible candidate.
        // Beam is sorted by cost, so the first feasible entry is optimal.
        for (std::size_t i = 0; i < beam_size; ++i) {
            stats.candidates_total++;
            bool feasible = (constraints(beam[i].candidate) && ...);
            if (feasible) {
                return plan<candidate_type>{
                    beam[i].candidate, beam[i].cost, stats};
            }
            stats.candidates_pruned++;
        }

        // No feasible candidate in beam.
        return plan<candidate_type>{
            candidate_type{},
            std::numeric_limits<double>::infinity(),
            stats
        };
    }

    // No constraints — best is first in sorted beam.
    return plan<candidate_type>{beam[0].candidate, beam[0].cost, stats};
}

} // namespace ctdp

#endif // CTDP_SOLVER_ALGORITHMS_BEAM_SEARCH_H
