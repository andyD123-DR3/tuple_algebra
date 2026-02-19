// core/per_element_candidate.h
// Compile-time dynamic programming framework — Core
// Typed candidate for per-element (factored) search spaces.
//
// Wraps std::array<Strategy, N> with semantic identity and
// candidate_traits specialisation.  Enables plan_traversal and
// deployment without coupling to solver internals.
//
// operator[] is provided for backward compatibility — algorithms
// and tests can continue to use candidate[pos] = strategy.

#ifndef CTDP_CORE_PER_ELEMENT_CANDIDATE_H
#define CTDP_CORE_PER_ELEMENT_CANDIDATE_H

#include "candidate_traits.h"
#include <array>
#include <cstddef>

namespace ctdp {

// =============================================================================
// per_element_candidate: assignment of one strategy per position
// =============================================================================

template<typename Strategy, std::size_t N>
struct per_element_candidate {
    using strategy_type = Strategy;
    static constexpr std::size_t dimension = N;

    std::array<Strategy, N> assignments{};

    // --- Array-like access (backward compat) ---

    [[nodiscard]] constexpr Strategy& operator[](std::size_t i) {
        return assignments[i];
    }
    [[nodiscard]] constexpr Strategy const& operator[](std::size_t i) const {
        return assignments[i];
    }
    [[nodiscard]] static constexpr std::size_t size() { return N; }

    // --- Comparison ---

    constexpr bool operator==(per_element_candidate const&) const = default;
};

// =============================================================================
// candidate_traits specialisation
// =============================================================================

template<typename Strategy, std::size_t N>
struct candidate_traits<per_element_candidate<Strategy, N>> {

    /// Iterate (descriptor, strategy) pairs.
    /// Descriptors must be indexable with operator[] (Fix 1 issue, v0.5.5).
    template<typename Descriptors, typename F>
        requires requires(Descriptors const& d, std::size_t i) {
            { d[i] };
        }
    static constexpr void for_each_assignment(
        per_element_candidate<Strategy, N> const& candidate,
        Descriptors const& descriptors,
        F fn)
    {
        for (std::size_t i = 0; i < N; ++i) {
            fn(descriptors[i], candidate.assignments[i]);
        }
    }

    /// O(1) assignment count — dimension is known statically.
    template<typename Descriptors>
        requires requires(Descriptors const& d, std::size_t i) {
            { d[i] };
        }
    static constexpr std::size_t assignment_count(
        per_element_candidate<Strategy, N> const& /*candidate*/,
        Descriptors const& /*descriptors*/)
    {
        return N;
    }

    /// Indexed access to individual assignment.
    static constexpr Strategy const& get_assignment_at(
        per_element_candidate<Strategy, N> const& candidate,
        std::size_t index)
    {
        return candidate.assignments[index];
    }
};

// =============================================================================
// Verify concept satisfaction
// =============================================================================
namespace detail {
    enum class PECTestStrat_ { X, Y };
    static_assert(has_candidate_traits<per_element_candidate<PECTestStrat_, 3>>);
    static_assert(has_indexed_access<per_element_candidate<PECTestStrat_, 3>>);
}

} // namespace ctdp

#endif // CTDP_CORE_PER_ELEMENT_CANDIDATE_H
