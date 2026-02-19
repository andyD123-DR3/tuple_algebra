// ctdp/solver/constraints/constraint.h
// Compile-time dynamic programming framework — Analytics: Solver
// Formal constraint protocol (Architecture §5.2).
//
// CONSTRAINT PROTOCOL
// ===================
// A constraint is any callable that returns bool.  Two signatures:
//
//   Element predicate:  (Descriptor, size_t choice_index) → bool
//     Per-position.  Safe for per_element_argmin.  Concept: element_predicate_for.
//
//   Dynamic constraint: (Candidate) → bool
//     Whole-candidate.  Requires beam_search / exhaustive_search / local_search.
//     Concept: dynamic_constraint_for.
//
// Constraints are passed as bare callables to algorithm parameter packs.
// This module provides combinators and factories — it does NOT introduce
// wrapper types, traits, or new concepts.  All existing code continues
// to work unchanged.
//
// COMBINATORS
//   not_c(c)         → negation
//   all_of(cs...)    → conjunction (all must hold)
//   any_of(cs...)    → disjunction (at least one must hold)
//
// DYNAMIC CONSTRAINT FACTORIES (return (Candidate) → bool callables)
//   budget_cap(cost_fn, limit)     → total cost ≤ limit
//   mutual_exclusion(pos_a, pos_b) → candidate[a] ≠ candidate[b]
//   forbid_at(pos, value)          → candidate[pos] ≠ value
//   require_at(pos, value)         → candidate[pos] == value
//
// ELEMENT PREDICATE FACTORIES (return (Descriptor, size_t) → bool callables)
//   forbid_choice(blocked_index)   → choice_index ≠ blocked_index
//   allow_only(allowed_indices...) → choice_index ∈ {allowed...}

#ifndef CTDP_SOLVER_CONSTRAINTS_CONSTRAINT_H
#define CTDP_SOLVER_CONSTRAINTS_CONSTRAINT_H

#include <cstddef>
#include <initializer_list>

namespace ctdp {

// ============================================================================
// Combinators — work with any callable (element predicates or dynamic)
// ============================================================================

/// Negate a constraint.
template<typename C>
[[nodiscard]] constexpr auto not_c(C c) {
    return [c](auto const&... args) constexpr -> bool {
        return !c(args...);
    };
}

/// Conjunction: all constraints must hold.
template<typename... Cs>
[[nodiscard]] constexpr auto all_of(Cs... cs) {
    return [... cs = cs](auto const&... args) constexpr -> bool {
        return (cs(args...) && ...);
    };
}

/// Disjunction: at least one constraint must hold.
template<typename... Cs>
[[nodiscard]] constexpr auto any_of(Cs... cs) {
    return [... cs = cs](auto const&... args) constexpr -> bool {
        return (cs(args...) || ...);
    };
}

// ============================================================================
// Dynamic constraint factories — return (Candidate const&) → bool
// ============================================================================

/// Budget cap: total cost of candidate must not exceed limit.
/// CostFn signature: (Candidate const&) → arithmetic type.
template<typename CostFn>
[[nodiscard]] constexpr auto budget_cap(CostFn cost_fn, double limit) {
    return [cost_fn, limit](auto const& candidate) constexpr -> bool {
        return cost_fn(candidate) <= limit;
    };
}

/// Mutual exclusion: two positions must have different assignments.
[[nodiscard]] constexpr auto mutual_exclusion(std::size_t pos_a,
                                              std::size_t pos_b) {
    return [pos_a, pos_b](auto const& candidate) constexpr -> bool {
        return candidate[pos_a] != candidate[pos_b];
    };
}

/// Forbid a specific value at a specific position.
template<typename Value>
[[nodiscard]] constexpr auto forbid_at(std::size_t pos, Value v) {
    return [pos, v](auto const& candidate) constexpr -> bool {
        return candidate[pos] != v;
    };
}

/// Require a specific value at a specific position.
template<typename Value>
[[nodiscard]] constexpr auto require_at(std::size_t pos, Value v) {
    return [pos, v](auto const& candidate) constexpr -> bool {
        return candidate[pos] == v;
    };
}

// ============================================================================
// Element predicate factories — return (Descriptor const&, size_t) → bool
// ============================================================================

/// Forbid a single choice index at every position.
[[nodiscard]] constexpr auto forbid_choice(std::size_t blocked) {
    return [blocked](auto const& /*desc*/, std::size_t choice) constexpr -> bool {
        return choice != blocked;
    };
}

/// Allow only the listed choice indices; reject all others.
/// MaxN: compile-time capacity for the allowed set.
template<std::size_t MaxN = 16>
[[nodiscard]] constexpr auto allow_only(
    std::initializer_list<std::size_t> allowed)
{
    // Copy into fixed-size array for constexpr capture.
    struct allowed_set {
        std::size_t indices[MaxN]{};
        std::size_t count = 0;
    };
    allowed_set set{};
    for (auto idx : allowed) {
        if (set.count < MaxN) {
            set.indices[set.count++] = idx;
        }
    }
    return [set](auto const& /*desc*/, std::size_t choice) constexpr -> bool {
        for (std::size_t i = 0; i < set.count; ++i) {
            if (set.indices[i] == choice) return true;
        }
        return false;
    };
}

} // namespace ctdp

#endif // CTDP_SOLVER_CONSTRAINTS_CONSTRAINT_H
