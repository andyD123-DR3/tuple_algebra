// core/candidate_traits.h - Customization point for iterating over candidates
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Different candidate types have different internal structures:
// - per_element_candidate: std::array<Strategy, N> assignment
// - interval_candidate: std::vector<size_t> split_points
// - permutation_candidate: std::array<size_t, N> permutation
//
// plan_traversal needs to iterate over (descriptor, assignment) pairs uniformly.
// candidate_traits provides this abstraction through explicit specialization.
//
// DESIGN DECISION: Traits Class (not ADL)
// - Explicit extension point - clear where to customize
// - Easy to find - grep candidate_traits shows all specializations
// - Compile-time verification - no specialization = compile error
// - Can add multiple functions to same trait
// - Standard library pattern (iterator_traits, char_traits)
//
// NAMING DECISION: for_each_assignment (not for_each or traverse)
// - Explicit about what's being iterated - assignments of choices to descriptors
// - Matches domain terminology - DP assigns strategies to subproblems
// - Self-documenting - no ambiguity
// - No collision with std::for_each

#ifndef CTDP_CORE_CANDIDATE_TRAITS_H
#define CTDP_CORE_CANDIDATE_TRAITS_H

#include <cstddef>

namespace ctdp {

/// Customization point for iterating over candidate assignments.
///
/// Each search space must specialize this trait for its candidate type
/// to enable plan_traversal and deployment.
///
/// Required interface:
/// ```cpp
/// template<>
/// struct candidate_traits<YourCandidate> {
///     template<typename F>
///     static constexpr void for_each_assignment(
///         YourCandidate const& candidate,
///         auto const& descriptors,
///         F fn
///     );
/// };
/// ```
///
/// The function fn is called for each (descriptor, assignment) pair:
/// fn(descriptor, assigned_value)
///
/// Example specialization for per-element space:
/// ```cpp
/// template<typename Strategy, size_t N>
/// struct candidate_traits<per_element_candidate<Strategy, N>> {
///     template<typename F>
///     static constexpr void for_each_assignment(
///         per_element_candidate<Strategy, N> const& candidate,
///         auto const& descriptors,
///         F fn
///     ) {
///         for (size_t i = 0; i < N; ++i) {
///             fn(descriptors[i], candidate.assignment[i]);
///         }
///     }
/// };
/// ```
///
/// Example usage in plan_traversal:
/// ```cpp
/// template<typename Candidate>
/// void execute_plan(plan<Candidate> const& p, auto const& descs) {
///     candidate_traits<Candidate>::for_each_assignment(
///         p.params,
///         descs,
///         [](auto const& desc, auto const& strategy) {
///             apply_strategy(desc, strategy);
///         }
///     );
/// }
/// ```
template<typename Candidate>
struct candidate_traits;
// No default implementation - must specialize for each candidate type

// =============================================================================
// Concept: Check if a type has candidate_traits specialization
// =============================================================================

/// Check if a candidate type has a valid candidate_traits specialization.
///
/// Verifies that candidate_traits<C>::for_each_assignment exists and is callable.
template<typename C>
concept has_candidate_traits = requires(
    C const& candidate,
    int const* descriptors  // Use concrete type for concept check
) {
    // Can call for_each_assignment with a lambda
    candidate_traits<C>::for_each_assignment(
        candidate,
        descriptors,
        [](auto const&, auto const&) {}
    );
};

// =============================================================================
// Helper: Count assignments in a candidate
// =============================================================================

/// Count how many assignments a candidate contains.
///
/// Default implementation uses for_each_assignment and counts invocations.
/// Spaces can specialize this for O(1) count if known statically.
///
/// Example:
/// ```cpp
/// template<typename Strategy, size_t N>
/// struct candidate_traits<per_element_candidate<Strategy, N>> {
///     // ... for_each_assignment ...
///     
///     static constexpr size_t count_assignments(auto const& candidate) {
///         return N;  // O(1) - size is known
///     }
/// };
/// ```
template<typename Candidate>
    requires has_candidate_traits<Candidate>
constexpr size_t count_assignments(Candidate const& candidate, auto const& descriptors) {
    size_t count = 0;
    candidate_traits<Candidate>::for_each_assignment(
        candidate,
        descriptors,
        [&count](auto const&, auto const&) { ++count; }
    );
    return count;
}

// =============================================================================
// Helper: Check if candidate is empty
// =============================================================================

/// Check if a candidate has zero assignments.
template<typename Candidate>
    requires has_candidate_traits<Candidate>
constexpr bool is_empty_candidate(Candidate const& candidate, auto const& descriptors) {
    return count_assignments(candidate, descriptors) == 0;
}

// =============================================================================
// Helper: Get assignment at index (if supported)
// =============================================================================

/// Get the assignment at a specific index.
///
/// This is optional - not all candidate types support indexed access.
/// Spaces that do support it should specialize get_assignment_at.
///
/// Example:
/// ```cpp
/// template<typename Strategy, size_t N>
/// struct candidate_traits<per_element_candidate<Strategy, N>> {
///     // ... for_each_assignment ...
///     
///     static constexpr Strategy const& get_assignment_at(
///         per_element_candidate<Strategy, N> const& candidate,
///         size_t index
///     ) {
///         return candidate.assignment[index];
///     }
/// };
/// ```
template<typename Candidate>
concept has_indexed_access = requires(Candidate const& candidate, size_t index) {
    { candidate_traits<Candidate>::get_assignment_at(candidate, index) };
};

} // namespace ctdp

#endif // CTDP_CORE_CANDIDATE_TRAITS_H
