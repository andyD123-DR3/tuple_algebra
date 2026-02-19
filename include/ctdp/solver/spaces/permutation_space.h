// ctdp/solver/spaces/permutation_space.h
// Compile-time dynamic programming framework — Analytics: Solver
// All orderings of N elements.  N! grows fast — practical only for small N.
//
// Satisfies: search_space, enumerable_space, neighbourhood_space.
// Neighbours: all (N choose 2) swap pairs.

#ifndef CTDP_SOLVER_SPACES_PERMUTATION_SPACE_H
#define CTDP_SOLVER_SPACES_PERMUTATION_SPACE_H

#include "../concepts.h"
#include <array>
#include <cstddef>
#include <utility>

namespace ctdp {

template<std::size_t N>
struct permutation_space {
    using candidate_type = std::array<std::size_t, N>;

    // N! — compile-time constant, also used by exhaustive_search guard.
    static constexpr std::size_t static_size = []() constexpr {
        std::size_t result = 1;
        for (std::size_t i = 2; i <= N; ++i)
            result *= i;
        return result;
    }();

    [[nodiscard]] static constexpr auto size() -> std::size_t {
        return static_size;
    }

    // Identity permutation as starting point
    [[nodiscard]] static constexpr auto identity() -> candidate_type {
        candidate_type c{};
        for (std::size_t i = 0; i < N; ++i)
            c[i] = i;
        return c;
    }

    // Enumerate all N! permutations via Heap's algorithm
    template<typename F>
    constexpr void enumerate(F fn) const {
        candidate_type c = identity();
        heap_permute(fn, c, N);
    }

    // Swap neighbours: all (N choose 2) transpositions
    template<typename F>
    constexpr void neighbours(candidate_type const& c, F fn) const {
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                candidate_type neighbour = c;
                std::swap(neighbour[i], neighbour[j]);
                fn(static_cast<candidate_type const&>(neighbour));
            }
        }
    }

private:
    template<typename F>
    static constexpr void heap_permute(F& fn, candidate_type& c, std::size_t k) {
        if (k == 1) {
            fn(static_cast<candidate_type const&>(c));
            return;
        }
        for (std::size_t i = 0; i < k; ++i) {
            heap_permute(fn, c, k - 1);
            if (k % 2 == 0) {
                std::swap(c[i], c[k - 1]);
            } else {
                std::swap(c[0], c[k - 1]);
            }
        }
    }
};

// Verify concept satisfaction
namespace detail { static_assert(search_space<permutation_space<4>>); }

} // namespace ctdp

#endif // CTDP_SOLVER_SPACES_PERMUTATION_SPACE_H
