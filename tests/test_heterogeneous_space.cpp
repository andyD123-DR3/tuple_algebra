// tests/test_heterogeneous_space.cpp
// Tests for heterogeneous_per_element_space — per-position choice sets.
// Proves the "uniform choices" limitation is resolved.

#include "ctdp/solver/spaces/heterogeneous_per_element_space.h"
#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/algorithms/exhaustive_search.h"
#include "ctdp/core/plan_traversal.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =============================================================================
// Domain: FIX-like fields with different strategies per field
// =============================================================================

enum class Field : std::size_t {
    Tag = 0, Length = 1, Value = 2, Checksum = 3
};

enum class Strategy {
    Baseline,     // available everywhere
    Inline,       // Tag only
    Lookup,       // Tag only
    Copy,         // Value, Length
    SIMD,         // Value only
    FastChecksum, // Checksum only
};

// Cost function: (field, choice_index) → double.
// choice_index is local to each position.
constexpr auto make_space() {
    return make_heterogeneous_space(
        //  Field            available strategies
        position_choices{ Field::Tag,
            std::array{ Strategy::Baseline, Strategy::Inline, Strategy::Lookup } },
        position_choices{ Field::Length,
            std::array{ Strategy::Baseline, Strategy::Copy } },
        position_choices{ Field::Value,
            std::array{ Strategy::Baseline, Strategy::Copy, Strategy::SIMD } },
        position_choices{ Field::Checksum,
            std::array{ Strategy::Baseline, Strategy::FastChecksum } }
    );
}

// Cost per (field, local choice index).
constexpr double costs[4][3] = {
    // Tag:      Baseline=10, Inline=3,  Lookup=5
    { 10.0, 3.0, 5.0 },
    // Length:   Baseline=8,  Copy=4
    { 8.0, 4.0, 0.0 },
    // Value:    Baseline=12, Copy=6,    SIMD=2
    { 12.0, 6.0, 2.0 },
    // Checksum: Baseline=7,  FastCheck=3
    { 7.0, 3.0, 0.0 },
};

constexpr auto field_cost = [](Field f, std::size_t choice_idx) -> double {
    return costs[static_cast<int>(f)][choice_idx];
};

// =============================================================================
// Test: concept satisfaction
// =============================================================================

TEST(HeterogeneousSpace, ConceptSatisfaction) {
    using space_t = decltype(make_space());

    static_assert(search_space<space_t>);
    static_assert(described_space<space_t>);
    static_assert(factored_space<space_t>);
    static_assert(has_candidate_traits<space_t::candidate_type>);
}

// =============================================================================
// Test: per-position choice counts are correct
// =============================================================================

TEST(HeterogeneousSpace, PerPositionChoiceCounts) {
    constexpr auto space = make_space();

    static_assert(space.num_choices(0) == 3);  // Tag: 3 choices
    static_assert(space.num_choices(1) == 2);  // Length: 2 choices
    static_assert(space.num_choices(2) == 3);  // Value: 3 choices
    static_assert(space.num_choices(3) == 2);  // Checksum: 2 choices

    // branching = max = 3
    static_assert(decltype(space)::branching == 3);

    // Total search space = 3 * 2 * 3 * 2 = 36
    EXPECT_EQ(space.size(), 36u);
}

// =============================================================================
// Test: solve with element cost — each position picks its cheapest
// =============================================================================

constexpr auto solve() {
    return per_element_argmin(make_space(), field_cost);
}

// Optimal: Tag=Inline(3), Length=Copy(4), Value=SIMD(2), Checksum=FastChecksum(3)
// Total = 12
static_assert(solve().predicted_cost == 12.0);
static_assert(solve().is_feasible());

TEST(HeterogeneousSpace, SolveElementCost) {
    constexpr auto plan = solve();
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 12.0);

    // Verify assignments
    EXPECT_EQ(plan.params[0], Strategy::Inline);
    EXPECT_EQ(plan.params[1], Strategy::Copy);
    EXPECT_EQ(plan.params[2], Strategy::SIMD);
    EXPECT_EQ(plan.params[3], Strategy::FastChecksum);
}

// =============================================================================
// Test: traverse the plan — full round-trip
// =============================================================================

TEST(HeterogeneousSpace, TraversePlan) {
    constexpr auto plan = solve();
    constexpr auto space = make_space();

    std::size_t count = 0;
    for_each_assignment(plan, space.descriptors,
        [&count](Field field, Strategy strat) {
            switch (field) {
                case Field::Tag:      EXPECT_EQ(strat, Strategy::Inline);        break;
                case Field::Length:   EXPECT_EQ(strat, Strategy::Copy);          break;
                case Field::Value:    EXPECT_EQ(strat, Strategy::SIMD);          break;
                case Field::Checksum: EXPECT_EQ(strat, Strategy::FastChecksum);  break;
            }
            ++count;
        });
    EXPECT_EQ(count, 4u);
}

// =============================================================================
// Test: exhaustive_search agrees with per_element_argmin
// =============================================================================

TEST(HeterogeneousSpace, ExhaustiveAgreesWithArgmin) {
    constexpr auto space = make_space();

    // Wrap element cost as whole-candidate cost for exhaustive_search
    auto whole_cost = [&space](auto const& candidate) constexpr -> double {
        double total = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            // Find which choice index this strategy corresponds to
            for (std::size_t s = 0; s < space.num_choices(i); ++s) {
                if (space.choice(i, s) == candidate[i]) {
                    total += costs[i][s];
                    break;
                }
            }
        }
        return total;
    };

    auto exhaustive_result = exhaustive_search(space, whole_cost);
    auto argmin_result = solve();

    EXPECT_DOUBLE_EQ(exhaustive_result.predicted_cost,
                     argmin_result.predicted_cost);
}

// =============================================================================
// Test: element predicate with heterogeneous space
// =============================================================================

TEST(HeterogeneousSpace, WithElementPredicate) {
    constexpr auto space = make_space();

    // Forbid choice index 1 at Tag (that's Strategy::Inline)
    auto no_inline_at_tag = [](Field f, std::size_t choice_idx) -> bool {
        if (f == Field::Tag && choice_idx == 1) return false;
        return true;
    };

    constexpr auto plan = per_element_argmin(space, field_cost, no_inline_at_tag);

    // Tag can't use Inline(3), next best is Lookup(5)
    // Total = 5 + 4 + 2 + 3 = 14
    static_assert(plan.predicted_cost == 14.0);
    static_assert(plan.params[0] == Strategy::Lookup);

    EXPECT_DOUBLE_EQ(plan.predicted_cost, 14.0);
}

// =============================================================================
// Test: enumerate produces correct number of candidates
// =============================================================================

TEST(HeterogeneousSpace, EnumerateCount) {
    constexpr auto space = make_space();

    std::size_t count = 0;
    space.enumerate([&count](auto const&) { ++count; });
    EXPECT_EQ(count, 36u);  // 3 * 2 * 3 * 2
}

// =============================================================================
// Test: degenerate case — single choice at some positions
// =============================================================================

TEST(HeterogeneousSpace, SingleChoicePosition) {
    constexpr auto space = make_heterogeneous_space(
        position_choices{ Field::Tag,    std::array{ Strategy::Baseline } },
        position_choices{ Field::Length,  std::array{ Strategy::Copy, Strategy::SIMD } }
    );

    static_assert(space.num_choices(0) == 1);
    static_assert(space.num_choices(1) == 2);
    EXPECT_EQ(space.size(), 2u);

    auto cost = [](Field, std::size_t choice) -> double {
        return static_cast<double>(choice + 1);
    };
    auto plan = per_element_argmin(space, cost);

    // Tag forced to choice 0, Length picks choice 0 (cost 1)
    EXPECT_EQ(plan.params[0], Strategy::Baseline);
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 2.0);  // 1.0 + 1.0
}
