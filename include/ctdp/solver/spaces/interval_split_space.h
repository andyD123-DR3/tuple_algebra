// ctdp/solver/spaces/interval_split_space.h
// Compile-time dynamic programming framework — Analytics: Solver
// Optimal parenthesisation space.  Classic interval DP domain.
//
// candidate_type = interval_split_candidate<MaxN>
// Not enumerable — interval_dp exploits recursive subproblem structure.

#ifndef CTDP_SOLVER_SPACES_INTERVAL_SPLIT_SPACE_H
#define CTDP_SOLVER_SPACES_INTERVAL_SPLIT_SPACE_H

#include "../../core/concepts.h"
#include <array>
#include <cstddef>

namespace ctdp {

// ---------------------------------------------------------------------------
// interval_split_candidate: records the split decision for each subproblem.
// Linearised 2D table: optimal_split[i * MaxN + j] = k.
// ---------------------------------------------------------------------------
template<std::size_t MaxN>
struct interval_split_candidate {
    std::size_t n{};
    std::array<std::size_t, MaxN * MaxN> optimal_split{};

    [[nodiscard]] constexpr auto split(std::size_t i, std::size_t j) const
        -> std::size_t
    {
        return optimal_split[i * MaxN + j];
    }

    // Equality over active region [0, n) only.
    // No operator<=> — candidate ordering is via plan cost.
    // C++20 auto-generates operator!= from operator==.
    [[nodiscard]] constexpr auto operator==(
        interval_split_candidate const& o) const -> bool
    {
        if (n != o.n) return false;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                if (optimal_split[i * MaxN + j] != o.optimal_split[i * MaxN + j])
                    return false;
        return true;
    }
};

// ---------------------------------------------------------------------------
// interval_split_space: search space for optimal parenthesisation of n items.
// MaxN is the compile-time capacity.  n is the actual problem size (≤ MaxN).
// ---------------------------------------------------------------------------
template<std::size_t MaxN>
struct interval_split_space {
    using candidate_type = interval_split_candidate<MaxN>;

    static constexpr std::size_t max_size = MaxN;

    std::size_t n{};  // actual problem size ≤ MaxN

    // Catalan number — informational only.  No algorithm enumerates this.
    [[nodiscard]] constexpr auto size() const -> std::size_t {
        // C(n) = (2n)! / ((n+1)! * n!)
        // For small n, compute directly.  Good enough for informational use.
        if (n <= 1) return 1;
        std::size_t catalan = 1;
        for (std::size_t i = 0; i < n - 1; ++i) {
            catalan = catalan * 2 * (2 * i + 1) / (i + 2);
        }
        return catalan;
    }
};

// Verify search_space concept satisfaction
namespace detail { static_assert(search_space<interval_split_space<8>>); }

} // namespace ctdp

#endif // CTDP_SOLVER_SPACES_INTERVAL_SPLIT_SPACE_H
