// core/solve_stats.h - Statistics from dynamic programming solve process
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// solve_stats captures metrics from the DP solving process. It lives in core/
// because plan<C> (also in core/) contains a solve_stats member.
//
// The struct is designed for:
// - Constexpr compatibility (trivially copyable aggregate)
// - Zero overhead when not used (all fields optional to populate)
// - Aggregation across multiple solves
// - Human-readable reporting
//
// Different solvers populate different subsets of fields based on what they track.

#ifndef CTDP_CORE_SOLVE_STATS_H
#define CTDP_CORE_SOLVE_STATS_H

#include <cstddef>
#include <limits>

namespace ctdp {

/// Statistics collected during dynamic programming solve process.
///
/// This struct captures performance and behavior metrics from solvers.
/// Different solver algorithms populate different subsets of fields.
///
/// All counters default to 0. Unused fields can be left at default.
/// For values that are "not applicable", use 0 rather than special sentinels.
///
/// Example usage:
/// ```cpp
/// template<typename Space, typename Cost>
/// constexpr auto sequence_dp(Space const& space, Cost const& cost_fn) -> plan<...> {
///     solve_stats stats;
///     stats.subproblems_total = space.size();
///     
///     constexpr_map<subproblem_key, double, 1000> memo;
///     
///     // During solve:
///     ++stats.subproblems_evaluated;
///     if (memo.find(key) != memo.end()) {
///         ++stats.memo_hits;
///     } else {
///         ++stats.memo_misses;
///     }
///     
///     stats.memo_table_size = memo.size();
///     return plan{best_candidate, best_cost, stats};
/// }
/// ```
struct solve_stats {
    // =============================================================================
    // Subproblem Metrics
    // =============================================================================
    
    /// Total number of unique subproblems in the search space.
    /// For interval DP: O(n²), for sequence DP: O(n), etc.
    size_t subproblems_total = 0;
    
    /// Number of subproblems actually evaluated (not pruned).
    /// May be less than subproblems_total due to:
    /// - Early termination (found optimal)
    /// - Pruning (bounds, constraints)
    /// - Beam search (limited exploration)
    size_t subproblems_evaluated = 0;
    
    /// Number of subproblems stored in memoization table.
    /// Typically equals subproblems_evaluated for DP solvers.
    /// May differ for solvers that don't memoize every evaluation.
    size_t subproblems_cached = 0;
    
    // =============================================================================
    // Candidate Metrics
    // =============================================================================
    
    /// Total number of candidate solutions considered across all subproblems.
    /// For exhaustive search: product of all choice counts.
    /// For greedy: equals subproblems_total (one candidate per subproblem).
    size_t candidates_total = 0;
    
    /// Number of candidates actually evaluated (cost function called).
    /// May be less than candidates_total due to:
    /// - Constraint violations (static checks)
    /// - Dominance pruning (Pareto)
    /// - Branch bounds (A* style pruning)
    size_t candidates_evaluated = 0;
    
    /// Number of candidates skipped without evaluation.
    /// candidates_pruned = candidates_total - candidates_evaluated
    size_t candidates_pruned = 0;
    
    // =============================================================================
    // Memoization Metrics
    // =============================================================================
    
    /// Peak size of memoization table (number of cached results).
    /// Useful for understanding memory requirements.
    size_t memo_table_size = 0;
    
    /// Number of times a cached result was reused (cache hits).
    /// High hit rate indicates effective memoization.
    size_t memo_hits = 0;
    
    /// Number of times a cache lookup failed (cache misses).
    /// memo_misses ≈ subproblems_evaluated for first-time solves.
    size_t memo_misses = 0;
    
    // =============================================================================
    // Search-Specific Metrics
    // =============================================================================
    
    /// Maximum beam width used (for beam search).
    /// 0 indicates not a beam search.
    size_t beam_width_used = 0;
    
    /// Number of local search moves attempted (for local search).
    /// 0 indicates not a local search.
    size_t local_search_moves = 0;
    
    /// Size of Pareto frontier (for multi-objective search).
    /// 0 indicates not a Pareto search.
    size_t pareto_front_size = 0;
    
    // =============================================================================
    // Recursion Metrics
    // =============================================================================
    
    /// Maximum recursion depth reached during solve.
    /// For iterative solvers, this is 0.
    /// For recursive DP, indicates stack depth.
    /// Useful for detecting potential stack overflow.
    size_t max_recursion_depth = 0;
    
    // =============================================================================
    // Aggregation Operations
    // =============================================================================
    
    /// Combine statistics from two solve processes.
    /// Used when composing multiple plans or running multiple solves.
    ///
    /// Addition semantics:
    /// - Counters (subproblems_*, candidates_*, memo_*): sum
    /// - Maximums (max_recursion_depth, beam_width_used, memo_table_size): max
    /// - Pareto front size: sum (combining fronts)
    ///
    /// Example:
    /// ```cpp
    /// auto plan1 = solve_subproblem_A();
    /// auto plan2 = solve_subproblem_B();
    /// auto combined = plan1.stats + plan2.stats;
    /// ```
    constexpr solve_stats operator+(solve_stats const& other) const {
        return solve_stats{
            .subproblems_total = subproblems_total + other.subproblems_total,
            .subproblems_evaluated = subproblems_evaluated + other.subproblems_evaluated,
            .subproblems_cached = subproblems_cached + other.subproblems_cached,
            
            .candidates_total = candidates_total + other.candidates_total,
            .candidates_evaluated = candidates_evaluated + other.candidates_evaluated,
            .candidates_pruned = candidates_pruned + other.candidates_pruned,
            
            .memo_table_size = (memo_table_size > other.memo_table_size) 
                ? memo_table_size : other.memo_table_size,  // max
            .memo_hits = memo_hits + other.memo_hits,
            .memo_misses = memo_misses + other.memo_misses,
            
            .beam_width_used = (beam_width_used > other.beam_width_used)
                ? beam_width_used : other.beam_width_used,  // max
            .local_search_moves = local_search_moves + other.local_search_moves,
            .pareto_front_size = pareto_front_size + other.pareto_front_size,
            
            .max_recursion_depth = (max_recursion_depth > other.max_recursion_depth)
                ? max_recursion_depth : other.max_recursion_depth,  // max
        };
    }
    
    /// In-place addition.
    constexpr solve_stats& operator+=(solve_stats const& other) {
        *this = *this + other;
        return *this;
    }
    
    // =============================================================================
    // Derived Metrics (Computed Properties)
    // =============================================================================
    
    /// Cache hit rate (0.0 to 1.0).
    /// Returns 0.0 if no cache accesses occurred.
    [[nodiscard]] constexpr double cache_hit_rate() const {
        size_t total_accesses = memo_hits + memo_misses;
        if (total_accesses == 0) return 0.0;
        return static_cast<double>(memo_hits) / static_cast<double>(total_accesses);
    }
    
    /// Pruning effectiveness (0.0 to 1.0).
    /// Fraction of candidates pruned without evaluation.
    /// Returns 0.0 if no candidates were considered.
    [[nodiscard]] constexpr double pruning_rate() const {
        if (candidates_total == 0) return 0.0;
        return static_cast<double>(candidates_pruned) / static_cast<double>(candidates_total);
    }
    
    /// Subproblem coverage (0.0 to 1.0).
    /// Fraction of total subproblems actually evaluated.
    /// Returns 1.0 for exhaustive search, < 1.0 for pruned/beam search.
    [[nodiscard]] constexpr double subproblem_coverage() const {
        if (subproblems_total == 0) return 0.0;
        return static_cast<double>(subproblems_evaluated) / static_cast<double>(subproblems_total);
    }
    
    /// Average candidates per subproblem.
    /// Useful for understanding search space size.
    [[nodiscard]] constexpr double avg_candidates_per_subproblem() const {
        if (subproblems_evaluated == 0) return 0.0;
        return static_cast<double>(candidates_evaluated) / static_cast<double>(subproblems_evaluated);
    }
    
    // =============================================================================
    // Comparison (for testing)
    // =============================================================================
    
    constexpr bool operator==(solve_stats const& other) const = default;
};

} // namespace ctdp

#endif // CTDP_CORE_SOLVE_STATS_H
