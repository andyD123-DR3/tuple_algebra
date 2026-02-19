// ctdp/solver/spaces/heterogeneous_per_element_space.h
// Compile-time dynamic programming framework — Analytics: Solver
// N positions, each independently picks from its OWN set of strategies.
//
// Unlike per_element_space (uniform: every position picks from the same
// S choices), this space allows each position to have a different number
// of valid strategies.  Example: a FIX parser where Tag can choose
// {Inline, Lookup} but Value can choose {Copy, SIMD, Skip}.
//
// Template parameters:
//   Strategy   — the common strategy type (single enum for all positions)
//   N          — number of positions
//   MaxS       — maximum choices at any one position (compile-time capacity)
//   Descriptor — typed identity for each position (default: size_t)
//
// Satisfies: search_space, described_space, factored_space.
// branching = MaxS (compile-time upper bound).
// factored_space is satisfied via num_choices(pos) and choice(pos, s).
//
// Construction:
//   Use make_heterogeneous_space() or aggregate init with the helpers.

#ifndef CTDP_SOLVER_SPACES_HETEROGENEOUS_PER_ELEMENT_SPACE_H
#define CTDP_SOLVER_SPACES_HETEROGENEOUS_PER_ELEMENT_SPACE_H

#include "../concepts.h"
#include "../../core/per_element_candidate.h"
#include <array>
#include <cstddef>
#include <limits>

namespace ctdp {

template<typename Strategy,
         std::size_t N,
         std::size_t MaxS,
         typename Descriptor = std::size_t>
struct heterogeneous_per_element_space {
    using candidate_type  = per_element_candidate<Strategy, N>;
    using strategy_type   = Strategy;
    using descriptor_type = Descriptor;

    static constexpr std::size_t dimension = N;
    static constexpr std::size_t branching = MaxS;  // upper bound for concept

    // Worst-case search space size: MaxS^N.
    // Used by exhaustive_search's compile-time guard (Issue C fix, v0.5.5).
    // The actual size() may be smaller since positions can have < MaxS choices.
    static constexpr std::size_t static_size = []() constexpr {
        std::size_t result = 1;
        for (std::size_t i = 0; i < N; ++i)
            result *= MaxS;
        return result;
    }();

    // Per-position data.
    std::array<Descriptor, N>                 descriptors{};
    std::array<std::array<Strategy, MaxS>, N> choices{};
    std::array<std::size_t, N>                choice_counts{};

    // --- per-position choice interface (primary) ---

    [[nodiscard]] constexpr auto num_choices(std::size_t pos) const
        -> std::size_t
    {
        return choice_counts[pos];
    }

    [[nodiscard]] constexpr auto choice(std::size_t pos, std::size_t s) const
        -> Strategy
    {
        return choices[pos][s];
    }

    // --- described_space interface ---

    [[nodiscard]] constexpr auto descriptor(std::size_t i) const
        -> Descriptor
    {
        return descriptors[i];
    }

    [[nodiscard]] static constexpr auto num_descriptors() -> std::size_t {
        return N;
    }

    // --- search_space interface ---

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        std::size_t result = 1;
        for (std::size_t i = 0; i < N; ++i) {
            // Overflow guard (Issue E fix, v0.5.5): detect before multiplying.
            if (choice_counts[i] != 0 &&
                result > std::numeric_limits<std::size_t>::max() / choice_counts[i])
            {
                return std::numeric_limits<std::size_t>::max(); // saturate
            }
            result *= choice_counts[i];
        }
        return result;
    }

    // --- enumerable: all candidates via callback ---

    template<typename F>
    constexpr void enumerate(F fn) const {
        candidate_type c{};
        enumerate_impl(fn, c, std::size_t{0});
    }

    // --- neighbourhood: Hamming-1 (change one position) ---

    template<typename F>
    constexpr void neighbours(candidate_type const& c, F fn) const {
        for (std::size_t pos = 0; pos < N; ++pos) {
            for (std::size_t s = 0; s < choice_counts[pos]; ++s) {
                if (choices[pos][s] != c[pos]) {
                    candidate_type neighbour = c;
                    neighbour[pos] = choices[pos][s];
                    fn(static_cast<candidate_type const&>(neighbour));
                }
            }
        }
    }

private:
    template<typename F>
    constexpr void enumerate_impl(F& fn, candidate_type& c,
                                  std::size_t pos) const
    {
        if (pos == N) {
            fn(static_cast<candidate_type const&>(c));
            return;
        }
        for (std::size_t s = 0; s < choice_counts[pos]; ++s) {
            c[pos] = choices[pos][s];
            enumerate_impl(fn, c, pos + 1);
        }
    }
};

// =============================================================================
// Builder helper: position_choices
// =============================================================================

/// Describe the choices available at one position.
/// Used with make_heterogeneous_space().
template<typename Descriptor, typename Strategy, std::size_t Count>
struct position_choices {
    Descriptor descriptor;
    std::array<Strategy, Count> options;
    static constexpr std::size_t count = Count;
};

/// Deduction guide: position_choices from descriptor + brace-init array.
template<typename D, typename S, std::size_t C>
position_choices(D, std::array<S, C>) -> position_choices<D, S, C>;

// =============================================================================
// Factory: make_heterogeneous_space
// =============================================================================

// Compile-time max over a pack of size_t values.
// static_assert guards against empty pack (Issue D fix, v0.5.5):
// max_of<> would return 0, creating zero-sized arrays in the space.
template<std::size_t... Ns>
constexpr std::size_t max_of = []() constexpr {
    static_assert(sizeof...(Ns) > 0,
        "max_of requires at least one value. "
        "Empty pack would produce MaxS=0, creating zero-sized arrays.");
    std::size_t m = 0;
    for (auto n : {Ns...})
        if (n > m) m = n;
    return m;
}();

/// Public factory — deduces MaxS from the position_choices.
template<typename Descriptor, typename Strategy, std::size_t... Counts>
[[nodiscard]] constexpr auto make_heterogeneous_space(
    position_choices<Descriptor, Strategy, Counts> const&... positions)
{
    constexpr std::size_t MaxS = max_of<Counts...>;
    constexpr std::size_t N = sizeof...(positions);
    heterogeneous_per_element_space<Strategy, N, MaxS, Descriptor> space{};

    std::size_t idx = 0;
    auto fill = [&](auto const& pos) constexpr {
        space.descriptors[idx] = pos.descriptor;
        space.choice_counts[idx] = pos.count;
        for (std::size_t s = 0; s < pos.count; ++s)
            space.choices[idx][s] = pos.options[s];
        ++idx;
    };
    (fill(positions), ...);

    return space;
}

// =============================================================================
// Concept verification
// =============================================================================

namespace detail {
    enum class HetStrat_ { A, B, C };
    using HetSpace3_ = heterogeneous_per_element_space<HetStrat_, 2, 3>;
    static_assert(search_space<HetSpace3_>);
    static_assert(described_space<HetSpace3_>);
    static_assert(factored_space<HetSpace3_>);
}

} // namespace ctdp

#endif // CTDP_SOLVER_SPACES_HETEROGENEOUS_PER_ELEMENT_SPACE_H
