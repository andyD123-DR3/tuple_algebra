// tests/test_space_parameterised.cpp
//
// Tests for descriptor-aware spaces with element-level cost and constraints.
//
// The design:
//   Space  defines: descriptors (problem) + strategies (search)
//   Cost   takes:   (descriptor, choice_index) → double
//   Constraint:     (descriptor, choice_index) → bool
//
// Cost and constraints never see Strategy.  They operate in the
// descriptor domain.  The space maps choice indices to strategy values.
// Type safety comes from the descriptor type — the compiler enforces it.

#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/cost_models/additive.h"
#include "ctdp/solver/algorithms/per_element_argmin.h"
#include <gtest/gtest.h>
#include <array>

using namespace ctdp;

// ============================================================================
// Domain types
// ============================================================================

enum class Strategy { Baseline, Fast, Aggressive };

// Two descriptor types that must never be confused
enum class FixField  { Tag, Length, Value, Checksum };
enum class NodeId    { Input, Conv, Pool, Output };

// ============================================================================
// Cost tables: (descriptor × choice_index) → double
// ============================================================================

constexpr double fix_cost[4][3] = {
    { 10.0,  5.0,  2.0 },   // Tag
    { 20.0, 12.0, 30.0 },   // Length
    {  8.0,  3.0,  1.0 },   // Value
    { 15.0,  7.0, 25.0 },   // Checksum
};

constexpr double node_cost[4][3] = {
    {  5.0,  3.0,  1.0 },   // Input
    { 30.0, 15.0, 10.0 },   // Conv
    { 12.0,  8.0,  6.0 },   // Pool
    { 20.0, 10.0,  5.0 },   // Output
};

// ============================================================================
// Test: element cost with typed descriptors
// ============================================================================

TEST(DescribedSpace, ElementCostTypedDescriptors) {
    constexpr auto space = per_element_space<Strategy, 4, 3, FixField>{
        .descriptors = { FixField::Tag, FixField::Length,
                         FixField::Value, FixField::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };

    // Cost takes (FixField, choice_index).  No Strategy in sight.
    constexpr auto cost = [](FixField field, std::size_t choice) -> double {
        return fix_cost[static_cast<int>(field)][choice];
    };

    constexpr auto result = per_element_argmin(space, cost);

    // Optimal: Aggressive(2) + Fast(12) + Aggressive(1) + Fast(7) = 22
    static_assert(result.predicted_cost == 22.0);
    static_assert(result.params[0] == Strategy::Aggressive);
    static_assert(result.params[1] == Strategy::Fast);
    static_assert(result.params[2] == Strategy::Aggressive);
    static_assert(result.params[3] == Strategy::Fast);
}

// ============================================================================
// Test: different descriptor types are incompatible at compile time
// ============================================================================

TEST(DescribedSpace, DescriptorTypeSafety) {
    using FixSpace  = per_element_space<Strategy, 4, 3, FixField>;
    using NodeSpace = per_element_space<Strategy, 4, 3, NodeId>;

    // Same candidate shape...
    static_assert(std::is_same_v<FixSpace::candidate_type,
                                 NodeSpace::candidate_type>);
    // ...different descriptor types
    static_assert(!std::is_same_v<FixSpace::descriptor_type,
                                  NodeSpace::descriptor_type>);

    // A cost function taking (FixField, size_t):
    //   auto fix_cost = [](FixField, size_t) -> double { ... };
    //
    // Cannot be passed to per_element_argmin(node_space, fix_cost)
    // because element_cost_for<decltype(fix_cost), NodeId> is false.
    // The compiler rejects it.  No runtime check needed.
    SUCCEED();
}

// ============================================================================
// Test: element constraint
// ============================================================================

TEST(DescribedSpace, ElementConstraint) {
    constexpr auto space = per_element_space<Strategy, 4, 3, FixField>{
        .descriptors = { FixField::Tag, FixField::Length,
                         FixField::Value, FixField::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };

    constexpr auto cost = [](FixField field, std::size_t choice) -> double {
        return fix_cost[static_cast<int>(field)][choice];
    };

    // No Aggressive (choice 2) on Checksum
    constexpr auto no_aggressive_checksum =
        [](FixField field, std::size_t choice) -> bool {
            return !(field == FixField::Checksum && choice == 2);
        };

    constexpr auto result = per_element_argmin(space, cost,
                                                no_aggressive_checksum);

    // Checksum can't pick Aggressive(25).  Best remaining: Fast(7).
    static_assert(result.params[3] == Strategy::Fast);
    // Others unchanged
    static_assert(result.params[0] == Strategy::Aggressive);
    static_assert(result.params[2] == Strategy::Aggressive);
    // Total: 2 + 12 + 1 + 7 = 22
    static_assert(result.predicted_cost == 22.0);
}

// ============================================================================
// Test: budget constraint as element predicate
// ============================================================================

TEST(DescribedSpace, BudgetViaElementPredicate) {
    constexpr auto space = per_element_space<Strategy, 4, 3, FixField>{
        .descriptors = { FixField::Tag, FixField::Length,
                         FixField::Value, FixField::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };

    // Memory per (field, choice)
    static constexpr double mem[4][3] = {
        { 8, 16, 64 },    // Tag
        { 8, 16, 64 },    // Length
        { 8, 32, 128 },   // Value
        { 8, 16, 64 },    // Checksum
    };

    constexpr auto cost = [](FixField field, std::size_t choice) -> double {
        return fix_cost[static_cast<int>(field)][choice];
    };

    // Per-element budget: each position limited to 32 bytes
    constexpr auto mem_limit = [](FixField field, std::size_t choice) -> bool {
        return mem[static_cast<int>(field)][choice] <= 32.0;
    };

    constexpr auto result = per_element_argmin(space, cost, mem_limit);

    // Aggressive uses 64 bytes everywhere except Value(128) — all blocked.
    // Tag: Fast(16) ok, cost 5
    // Length: Fast(16) ok, cost 12
    // Value: Fast(32) ok, cost 3     (Aggressive=128, blocked)
    // Checksum: Fast(16) ok, cost 7
    // Total: 5 + 12 + 3 + 7 = 27
    static_assert(result.predicted_cost == 27.0);
    static_assert(result.params[0] == Strategy::Fast);
    static_assert(result.params[2] == Strategy::Fast);
}

// ============================================================================
// Test: multiple constraints compose
// ============================================================================

TEST(DescribedSpace, MultipleConstraints) {
    constexpr auto space = per_element_space<Strategy, 4, 3, FixField>{
        .descriptors = { FixField::Tag, FixField::Length,
                         FixField::Value, FixField::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };

    constexpr auto cost = [](FixField field, std::size_t choice) -> double {
        return fix_cost[static_cast<int>(field)][choice];
    };

    // Constraint 1: no Aggressive on Checksum
    constexpr auto c1 = [](FixField field, std::size_t choice) -> bool {
        return !(field == FixField::Checksum && choice == 2);
    };
    // Constraint 2: no Baseline on Tag
    constexpr auto c2 = [](FixField field, std::size_t choice) -> bool {
        return !(field == FixField::Tag && choice == 0);
    };

    constexpr auto result = per_element_argmin(space, cost, c1, c2);

    // Tag: Baseline blocked → Fast(5) or Aggressive(2) → Aggressive(2)
    // Checksum: Aggressive blocked → Fast(7) or Baseline(15) → Fast(7)
    static_assert(result.params[0] == Strategy::Aggressive);
    static_assert(result.params[3] == Strategy::Fast);
    static_assert(result.predicted_cost == 22.0);
}

// ============================================================================
// Test: swap cost functions — the original motivation
// ============================================================================

TEST(DescribedSpace, SwapCostFunctions) {
    constexpr auto space = per_element_space<Strategy, 4, 3, FixField>{
        .descriptors = { FixField::Tag, FixField::Length,
                         FixField::Value, FixField::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };

    constexpr auto latency = [](FixField f, std::size_t c) -> double {
        return fix_cost[static_cast<int>(f)][c];
    };

    static constexpr double tput[4][3] = {
        { 1.0, 2.0, 5.0 },
        { 1.0, 3.0, 2.0 },
        { 1.0, 4.0, 8.0 },
        { 1.0, 2.0, 3.0 },
    };
    constexpr auto throughput = [](FixField f, std::size_t c) -> double {
        return -tput[static_cast<int>(f)][c];  // negate: maximise throughput
    };

    // Same space, different cost → different plan
    constexpr auto plan_a = per_element_argmin(space, latency);
    constexpr auto plan_b = per_element_argmin(space, throughput);

    static_assert(plan_a.predicted_cost == 22.0);
    static_assert(plan_b.predicted_cost == -19.0);
}

// ============================================================================
// Test: backward compatibility — old-style whole-candidate cost still works
// ============================================================================

TEST(DescribedSpace, BackwardCompatWholeCandidateCost) {
    constexpr auto space = per_element_space<Strategy, 4, 3>{
        .descriptors = { 0, 1, 2, 3 },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };

    // Old-style: additive_cost wraps (size_t pos, Strategy s)
    constexpr auto cost = additive_cost{
        [](std::size_t pos, Strategy s) constexpr -> double {
            return fix_cost[pos][static_cast<int>(s)];
        }
    };

    // Whole-candidate overload: no constraints (by design — see review item A)
    constexpr auto result = per_element_argmin(space, cost);
    static_assert(result.predicted_cost == 22.0);
}

// ============================================================================
// Test: described_space concept conformance
// ============================================================================

TEST(DescribedSpace, ConceptConformance) {
    static_assert(described_space<per_element_space<Strategy, 4, 3, FixField>>);
    static_assert(described_space<per_element_space<Strategy, 4, 3, NodeId>>);
    static_assert(described_space<per_element_space<Strategy, 4, 3>>);
    static_assert(factored_space<per_element_space<Strategy, 4, 3, FixField>>);
    static_assert(search_space<per_element_space<Strategy, 4, 3, FixField>>);
    SUCCEED();
}

// ============================================================================
// Test: anonymous descriptors default to size_t indices
// ============================================================================

TEST(DescribedSpace, AnonymousDescriptorsDefault) {
    constexpr auto space = make_anonymous_space<Strategy, 4, 3>(
        { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive }
    );

    static_assert(std::is_same_v<decltype(space)::descriptor_type, std::size_t>);

    // Element cost with size_t descriptors — just position indices
    constexpr auto cost = [](std::size_t pos, std::size_t choice) -> double {
        return fix_cost[pos][choice];
    };

    constexpr auto result = per_element_argmin(space, cost);
    static_assert(result.predicted_cost == 22.0);
}
