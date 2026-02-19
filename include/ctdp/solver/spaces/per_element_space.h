// ctdp/solver/spaces/per_element_space.h
// Compile-time dynamic programming framework — Analytics: Solver
// N positions, each independently picks from S strategies.
// The archetype for factored problems.
//
// Satisfies: search_space, described_space, factored_space.
// per_element_argmin exploits factored structure: O(N×S), not O(S^N).
//
// Template parameters:
//   StrategyEnum — what the solver assigns to each position
//   N            — number of positions (the dimension count)
//   S            — choices per position (the branching factor)
//   Descriptor   — typed identity for each position (default: size_t)
//
// The separation:
//   Space    knows: descriptors (problem), strategies (search)
//   Cost     knows: descriptors (problem), choice indices (search geometry)
//   Plan     knows: descriptors (problem), strategies (assigned choices)
//
// Cost functions never see Strategy.  They take (Descriptor, size_t) and
// return a cost.  The space maps choice indices to strategy values.

#ifndef CTDP_SOLVER_SPACES_PER_ELEMENT_SPACE_H
#define CTDP_SOLVER_SPACES_PER_ELEMENT_SPACE_H

#include "../concepts.h"
#include "../../core/per_element_candidate.h"
#include <array>
#include <cstddef>

namespace ctdp {

template<typename StrategyEnum,
         std::size_t N,
         std::size_t S,
         typename Descriptor = std::size_t>
struct per_element_space {
    using candidate_type  = per_element_candidate<StrategyEnum, N>;
    using strategy_type   = StrategyEnum;
    using descriptor_type = Descriptor;

    static constexpr std::size_t dimension = N;
    static constexpr std::size_t branching = S;

    // S^N — compile-time constant, also used by exhaustive_search guard.
    static constexpr std::size_t static_size = []() constexpr {
        std::size_t result = 1;
        for (std::size_t i = 0; i < N; ++i)
            result *= S;
        return result;
    }();

    std::array<Descriptor, N>    descriptors{};
    std::array<StrategyEnum, S>  strategies{};

    // --- described_space interface ---

    [[nodiscard]] constexpr auto descriptor(std::size_t i) const -> Descriptor {
        return descriptors[i];
    }

    [[nodiscard]] static constexpr auto num_descriptors() -> std::size_t {
        return N;
    }

    // --- factored_space interface ---

    [[nodiscard]] constexpr auto strategy(std::size_t idx) const -> StrategyEnum {
        return strategies[idx];
    }

    // --- per-position choice interface (used by per_element_argmin) ---
    // Uniform: every position has the same S choices.

    [[nodiscard]] constexpr auto num_choices(std::size_t /*pos*/) const
        -> std::size_t
    {
        return S;
    }

    [[nodiscard]] constexpr auto choice(std::size_t /*pos*/, std::size_t s) const
        -> StrategyEnum
    {
        return strategies[s];
    }

    // --- search_space interface ---

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        return static_size;
    }

    // --- enumerable: all S^N candidates via callback ---

    template<typename F>
    constexpr void enumerate(F fn) const {
        candidate_type c{};
        enumerate_impl(fn, c, std::size_t{0});
    }

    // --- neighbourhood: Hamming-1 (change one position) ---

    template<typename F>
    constexpr void neighbours(candidate_type const& c, F fn) const {
        for (std::size_t pos = 0; pos < N; ++pos) {
            for (std::size_t s = 0; s < S; ++s) {
                if (strategies[s] != c[pos]) {
                    candidate_type neighbour = c;
                    neighbour[pos] = strategies[s];
                    fn(static_cast<candidate_type const&>(neighbour));
                }
            }
        }
    }

private:
    template<typename F>
    constexpr void enumerate_impl(F& fn, candidate_type& c, std::size_t pos) const {
        if (pos == N) {
            fn(static_cast<candidate_type const&>(c));
            return;
        }
        for (std::size_t s = 0; s < S; ++s) {
            c[pos] = strategies[s];
            enumerate_impl(fn, c, pos + 1);
        }
    }
};

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/// Construct with explicit descriptors.
template<typename StrategyEnum, std::size_t S, typename Descriptor, std::size_t N>
[[nodiscard]] constexpr auto make_per_element_space(
    std::array<Descriptor, N> const& descriptors,
    std::array<StrategyEnum, S> const& strategies)
    -> per_element_space<StrategyEnum, N, S, Descriptor>
{
    return { descriptors, strategies };
}

/// Construct with anonymous descriptors [0..N).
template<typename StrategyEnum, std::size_t N, std::size_t S>
[[nodiscard]] constexpr auto make_anonymous_space(
    std::array<StrategyEnum, S> const& strategies)
    -> per_element_space<StrategyEnum, N, S>
{
    std::array<std::size_t, N> descs{};
    for (std::size_t i = 0; i < N; ++i) descs[i] = i;
    return { descs, strategies };
}

// ---------------------------------------------------------------------------
// Concept verification
// ---------------------------------------------------------------------------

namespace detail {
    enum class TestStrat_ { A, B };
    static_assert(search_space<per_element_space<TestStrat_, 3, 2>>);
    static_assert(described_space<per_element_space<TestStrat_, 3, 2>>);
    static_assert(factored_space<per_element_space<TestStrat_, 3, 2>>);

    struct TestDesc_ { int id; constexpr bool operator==(TestDesc_ const&) const = default; };
    static_assert(described_space<per_element_space<TestStrat_, 3, 2, TestDesc_>>);
}

} // namespace ctdp

#endif // CTDP_SOLVER_SPACES_PER_ELEMENT_SPACE_H
