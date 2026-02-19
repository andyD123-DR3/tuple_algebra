// ctdp/solver/algorithms/interval_dp.h
// Compile-time dynamic programming framework — Analytics: Solver
// Classic O(N³) optimal parenthesisation.
//
// Internal 2D memo table.  Reconstructs split tree into candidate.
// No dynamic constraints — works on subproblems (i,j), not full candidates.
// If constraints needed: encode in cost (return infinity) or post-filter.
//
// Stats field semantics (solve_stats canonical names):
//   subproblems_total     — O(N²) interval subproblems
//   subproblems_evaluated — (i,j) pairs completed
//   candidates_total      — split-point evaluations (innermost loop), O(N³)
//   candidates_evaluated  — same (no pruning in interval DP)
//   candidates_pruned     — always 0 (no constraint pruning)
//   memo_*                — always 0 (uses own 2D array, not candidate_cache)

#ifndef CTDP_SOLVER_ALGORITHMS_INTERVAL_DP_H
#define CTDP_SOLVER_ALGORITHMS_INTERVAL_DP_H

#include "../concepts.h"
#include "../spaces/interval_split_space.h"
#include "../../core/ct_limits.h"
#include "../../core/plan.h"
#include "../../core/solve_stats.h"
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ctdp {

template<typename Space, interval_cost Cost>
    requires requires {
        { Space::max_size } -> std::convertible_to<std::size_t>;
    }
[[nodiscard]] constexpr auto interval_dp(
    Space const& space,
    Cost const& cost
) -> plan<typename Space::candidate_type>
{
    using candidate_type = typename Space::candidate_type;
    constexpr std::size_t MaxN = Space::max_size;

    static_assert(MaxN * MaxN <= ct_limits::exhaustive_max,
        "interval_dp memo table (MaxN^2) exceeds compile-time limit. "
        "Use build-step mode or increase ct_limits.");

    auto const n = space.n;

    // Precondition: n ≤ MaxN.  Same contract as constexpr_vector overflow.
    // In constexpr context: throw is ill-formed → compile error with diagnostic.
    // In runtime context: clean exception, not silent out-of-bounds corruption.
    if (n > MaxN) {
        throw std::logic_error(
            "interval_dp: space.n exceeds MaxN capacity. "
            "Increase the template parameter or reduce problem size.");
    }

    solve_stats stats{};
    stats.subproblems_total = n * (n + 1) / 2;  // all (i,j) intervals

    if (n == 0) {
        return plan<candidate_type>{candidate_type{}, 0.0, stats};
    }
    if (n == 1) {
        candidate_type result{};
        result.n = 1;
        return plan<candidate_type>{result, cost.leaf(0), stats};
    }

    // DP tables — linearised 2D
    std::array<double, MaxN * MaxN> dp{};
    std::array<std::size_t, MaxN * MaxN> split_table{};

    // Linearisation: dp[i][j] stored at i * MaxN + j
    // MaxN is constexpr — no capture needed.
    auto idx = [](std::size_t i, std::size_t j) constexpr -> std::size_t {
        return i * MaxN + j;
    };

    // Initialise diagonals
    for (std::size_t i = 0; i < n; ++i) {
        dp[idx(i, i)] = cost.leaf(i);
    }

    // Fill by increasing chain length
    //
    // Split convention: k = last index of left half.
    // Left: [i, k], Right: [k+1, j].
    // combine receives mid = k+1 (right-start convention).
    // split_table stores k (last-of-left).
    for (std::size_t len = 2; len <= n; ++len) {
        for (std::size_t i = 0; i + len <= n; ++i) {
            auto const j = i + len - 1;
            dp[idx(i, j)] = std::numeric_limits<double>::infinity();

            for (std::size_t k = i; k < j; ++k) {
                auto const mid = k + 1;  // right-start
                double q = dp[idx(i, k)] + dp[idx(mid, j)]
                         + cost.combine(i, mid, j);
                stats.candidates_evaluated++;
                stats.candidates_total++;

                if (q < dp[idx(i, j)]) {
                    dp[idx(i, j)] = q;
                    split_table[idx(i, j)] = k;
                }
            }
            stats.subproblems_evaluated++;
        }
    }

    // Reconstruct candidate from split table
    candidate_type result{};
    result.n = n;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j)
            result.optimal_split[i * MaxN + j] = split_table[idx(i, j)];

    return plan<candidate_type>{result, dp[idx(0, n - 1)], stats};
}

} // namespace ctdp

#endif // CTDP_SOLVER_ALGORITHMS_INTERVAL_DP_H
