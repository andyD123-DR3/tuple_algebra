// core/concepts.h - Core concepts for compile-time dynamic programming framework
// Part of the compile-time DP library (C++20)
//
// DESIGN NOTE: cost_function previously required std::floating_point which blocked:
// - Integer costs (int64_t for discrete problems)
// - Fixed-point arithmetic (custom rational types)
// - Multi-objective costs (std::tuple<double, double>)
// - Strong cost types (struct cost { double value; })
//
// Solution: Define cost_value concept with minimal requirements, then constrain
// cost_function to return cost_value instead of std::floating_point.

#ifndef CTDP_CORE_CONCEPTS_H
#define CTDP_CORE_CONCEPTS_H

#include <concepts>
#include <cstdint>
#include <ranges>
#include <type_traits>

namespace ctdp {

// =============================================================================
// Cost Value Concept
// =============================================================================

/// A cost_value must be totally ordered, support addition, and be default-constructible.
/// 
/// Requirements:
/// - std::totally_ordered<T>: Enables cost comparisons (a < b, a <= b, etc.)
/// - a + b: Enables cost accumulation (sum of subproblem costs)
/// - T{}: Default-constructible to represent "infinity" or "no cost"
///
/// Satisfied by:
/// - Scalar types: double, float, int64_t, int32_t
/// - Custom types: struct cost { double val; auto operator<=>(cost const&) = default; }
/// - Multi-objective: std::tuple<double, double> (lexicographic comparison)
/// - Fixed-point: custom rational<N, D> types with comparison operators
///
/// NOT satisfied by:
/// - Non-comparable types (no operator<=>)
/// - Types without addition (no operator+)
/// - Non-default-constructible types
template<typename T>
concept cost_value = 
    std::totally_ordered<T> && 
    requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { T{} } -> std::same_as<T>;
    };

// Verify common types satisfy cost_value
static_assert(cost_value<double>);
static_assert(cost_value<int64_t>);
static_assert(cost_value<int32_t>);
// Multi-objective would be: static_assert(cost_value<std::tuple<double, double>>);
// (requires custom operator+ and operator<=> on tuple)

// =============================================================================
// Cost Function Concept
// =============================================================================

/// A cost_function computes the cost of a candidate solution for a descriptor.
///
/// Signature: Cost operator()(Descriptor const& desc, Candidate const& cand) const
///
/// The returned Cost type must satisfy cost_value (totally ordered, additive).
///
/// Examples:
/// ```cpp
/// // Simple floating-point cost
/// auto latency_cost = [](auto const& desc, auto const& cand) -> double {
///     return estimate_latency(desc, cand);
/// };
///
/// // Integer cost for discrete problems
/// auto edit_distance = [](auto const& desc, auto const& cand) -> int64_t {
///     return compute_edit_distance(desc, cand);
/// };
///
/// // Multi-objective cost (requires custom tuple operators)
/// auto pareto_cost = [](auto const& desc, auto const& cand) -> std::tuple<double, int> {
///     return {latency(desc, cand), memory(desc, cand)};
/// };
/// ```
template<typename F, typename Descriptor, typename Candidate>
concept cost_function = requires(F f, Descriptor const& desc, Candidate const& cand) {
    { f(desc, cand) } -> cost_value;
};

// =============================================================================
// Descriptor and Candidate Concepts
// =============================================================================

/// A descriptor_range provides random access to problem descriptors.
///
/// Descriptors describe individual subproblems. For matrix chain multiplication,
/// each descriptor is a contiguous subchain [i, j). For FIX parsing, each
/// descriptor is a message field.
///
/// Requirements:
/// - Random access iteration (operator[], size())
/// - Contiguous storage preferred but not required
/// - Compile-time or runtime size (both supported)
///
/// Examples:
/// - std::array<matrix_descriptor, N>
/// - std::span<field_descriptor const>
/// - constexpr_vector<interval_descriptor, MaxN>
template<typename R>
concept descriptor_range = std::ranges::random_access_range<R>;

/// A candidate represents a potential solution to a subproblem.
///
/// Different search spaces have different candidate structures:
/// - per_element_space: std::array<Strategy, N> (one strategy per element)
/// - interval_split_space: std::array<size_t, K> (K-1 split points)
/// - permutation_space: std::array<size_t, N> (permutation of [0..N))
/// - segmentation_space: std::vector<boundary> (variable-length boundaries)
///
/// DESIGN NOTE: We do NOT constrain candidate structure here. Instead, we use
/// candidate_traits<C> for type-based customization. See core/candidate_traits.h.
///
/// This concept only requires:
/// - Regular type (copyable, equality comparable)
/// - Default-constructible (for memoization table initialization)
template<typename C>
concept candidate = 
    std::regular<C> &&
    std::default_initializable<C>;

// =============================================================================
// Search Space Concept
// =============================================================================

/// A search_space defines the structure of a DP optimization problem.
///
/// The space defines what candidate solutions look like and how large the
/// problem is. Individual solvers may require additional interfaces beyond
/// this minimal concept (e.g., enumerate(), split_points(), etc.).
///
/// Requirements:
/// - candidate_type: The solution representation
/// - size(): Problem dimension (number of elements, intervals, etc.)
///
/// Example:
/// ```cpp
/// struct per_element_space {
///     using candidate_type = per_element_candidate<StrategyEnum, N>;
///     constexpr size_t size() const { return N; }
///     
///     // Solver-specific interfaces:
///     template<typename F>
///     constexpr void enumerate(F fn) const;
/// };
/// ```
template<typename Space>
concept search_space = requires(Space const& space) {
    typename Space::candidate_type;
    { space.size() } -> std::convertible_to<size_t>;
    requires candidate<typename Space::candidate_type>;
};

// =============================================================================
// Described Space Concept
// =============================================================================

/// A described_space augments a search_space with typed dimension identifiers.
///
/// Each dimension in the search has a descriptor — a typed identity that
/// propagates through cost functions, constraints, and plan output.
/// This prevents cross-space confusion: a cost function built for
/// fix_field_id dimensions won't compile against a node_id space,
/// even if both spaces have the same candidate shape.
///
/// Spaces with inherently anonymous dimensions (permutation_space,
/// interval_split_space) default descriptor_type to std::size_t.
///
/// Requirements:
/// - Everything from search_space
/// - descriptor_type: The type identifying each dimension
/// - descriptor(i): Accessor for the i-th dimension's identity
/// - num_descriptors(): How many dimensions (may differ from size())
template<typename Space>
concept described_space = search_space<Space> && requires(Space const& s, std::size_t i) {
    typename Space::descriptor_type;
    { s.descriptor(i) } -> std::convertible_to<typename Space::descriptor_type>;
    { s.num_descriptors() } -> std::convertible_to<std::size_t>;
};

} // namespace ctdp

#endif // CTDP_CORE_CONCEPTS_H
