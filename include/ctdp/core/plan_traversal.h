// core/plan_traversal.h - Iterating and inspecting plan contents
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// A plan<C> contains a candidate that encodes assignments of strategies
// to descriptors. plan_traversal provides the bridge between solver output
// and deployment: it uses candidate_traits<C> to iterate over
// (descriptor, assignment) pairs without coupling to specific candidate types.
//
// This enables generic deployment code that works with any search space.
//
// DEPENDENCY:
// Requires candidate_traits<C> specialisation for the candidate type.
// See core/candidate_traits.h for the extension point.

#ifndef CTDP_CORE_PLAN_TRAVERSAL_H
#define CTDP_CORE_PLAN_TRAVERSAL_H

#include "candidate_traits.h"
#include "constexpr_vector.h"
#include "plan.h"

#include <cstddef>

namespace ctdp {

// =========================================================================
// for_each_assignment: visit all (descriptor, assignment) pairs
// =========================================================================

/// Iterate over all assignments in a plan.
///
/// Calls fn(descriptor, assignment) for each (descriptor, strategy) pair
/// defined by the plan's candidate and the descriptor range.
///
/// Requires: candidate_traits<Candidate> is specialised.
///
/// Example:
/// ```cpp
/// auto result = sequence_dp(space, cost_fn);
/// for_each_assignment(result, descriptors, [](auto const& desc, auto const& strat) {
///     apply_strategy(desc, strat);
/// });
/// ```
template<typename Candidate, typename Descriptors, typename F>
    requires has_candidate_traits<Candidate>
constexpr void for_each_assignment(plan<Candidate> const& p,
                                   Descriptors const& descriptors,
                                   F fn) {
    candidate_traits<Candidate>::for_each_assignment(
        p.params, descriptors, fn);
}

// =========================================================================
// assignment_count: how many assignments in the plan
// =========================================================================

/// Count the number of assignments in a plan's candidate.
///
/// Uses candidate_traits::for_each_assignment to count invocations.
/// Spaces can specialise for O(1) count.
template<typename Candidate, typename Descriptors>
    requires has_candidate_traits<Candidate>
[[nodiscard]] constexpr std::size_t
assignment_count(plan<Candidate> const& p,
                 Descriptors const& descriptors) {
    return count_assignments(p.params, descriptors);
}

// =========================================================================
// extract_assignments: collect assignments into a container
// =========================================================================

/// Extract all assignments from a plan into a constexpr_vector.
///
/// The Assignment type must be deducible from the callback or specified
/// explicitly. Each assignment is the value passed to the second argument
/// of the for_each_assignment callback.
///
/// Example:
/// ```cpp
/// auto assignments = extract_assignments<Strategy>(result, descriptors);
/// // assignments is constexpr_vector<Strategy, MaxN>
/// ```
template<typename Assignment, std::size_t MaxN,
         typename Candidate, typename Descriptors>
    requires has_candidate_traits<Candidate>
[[nodiscard]] constexpr constexpr_vector<Assignment, MaxN>
extract_assignments(plan<Candidate> const& p,
                    Descriptors const& descriptors) {
    constexpr_vector<Assignment, MaxN> result{};
    candidate_traits<Candidate>::for_each_assignment(
        p.params, descriptors,
        [&result](auto const&, auto const& assignment) {
            result.push_back(static_cast<Assignment>(assignment));
        });
    return result;
}

// =========================================================================
// Composite plan traversal
// =========================================================================

/// Visit each sub-plan in a composite plan.
///
/// Calls fn(sub_plan) for each sub-plan in the composite.
///
/// Example:
/// ```cpp
/// auto combined = compose_additive(plan_a, plan_b);
/// for_each_sub_plan(combined, [](auto const& sub) {
///     std::cout << "Sub-plan cost: " << sub.predicted_cost << "\n";
/// });
/// ```
template<typename F, typename... Candidates>
constexpr void for_each_sub_plan(composite_plan<Candidates...> const& cp,
                                 F fn) {
    std::apply([&fn](auto const&... plans) {
        (fn(plans), ...);
    }, cp.sub_plans);
}

/// Get the number of sub-plans in a composite plan.
template<typename... Candidates>
[[nodiscard]] constexpr std::size_t
sub_plan_count(composite_plan<Candidates...> const&) {
    return sizeof...(Candidates);
}

} // namespace ctdp

#endif // CTDP_CORE_PLAN_TRAVERSAL_H
