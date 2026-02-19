// tests/test_candidate_traversal.cpp
// End-to-end test: solve → traverse → deploy using candidate_traits.
// Proves the candidate_traits gap is closed for per_element spaces.

#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/cost_models/additive.h"
#include "ctdp/core/plan_traversal.h"
#include "ctdp/core/per_element_candidate.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =============================================================================
// Domain: FIX-like field optimisation
// =============================================================================

enum class Field    : std::size_t { Tag = 0, Length = 1, Value = 2, Checksum = 3 };
enum class Strategy { Baseline, Fast, Aggressive };

constexpr double cost_table[4][3] = {
    { 10.0,  5.0,  2.0 },   // Tag:      Aggressive cheapest
    { 20.0, 12.0, 30.0 },   // Length:   Fast cheapest
    {  8.0,  3.0,  1.0 },   // Value:    Aggressive cheapest
    { 15.0,  7.0, 25.0 },   // Checksum: Fast cheapest
};

constexpr auto make_space() {
    return per_element_space<Strategy, 4, 3, Field>{
        .descriptors = { Field::Tag, Field::Length, Field::Value, Field::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };
}

constexpr auto field_cost = [](Field f, std::size_t choice) -> double {
    return cost_table[static_cast<int>(f)][choice];
};

// =============================================================================
// Test: candidate_type is per_element_candidate, not raw std::array
// =============================================================================

TEST(CandidateTraversal, CandidateTypeIsProper) {
    using space_t = decltype(make_space());
    using cand_t  = space_t::candidate_type;

    // It's per_element_candidate, not std::array
    static_assert(std::is_same_v<cand_t, per_element_candidate<Strategy, 4>>);

    // It satisfies has_candidate_traits
    static_assert(has_candidate_traits<cand_t>);

    // It supports indexed access
    static_assert(has_indexed_access<cand_t>);
}

// =============================================================================
// Test: solve then traverse — full round-trip
// =============================================================================

constexpr auto solve() {
    return per_element_argmin(make_space(), field_cost);
}

// Optimal: Tag=Aggressive(2), Length=Fast(12), Value=Aggressive(1), Checksum=Fast(7)
// Total = 22
static_assert(solve().predicted_cost == 22.0);
static_assert(solve().is_feasible());

TEST(CandidateTraversal, SolveAndTraverse) {
    constexpr auto plan = solve();
    constexpr auto space = make_space();

    // Traverse: collect (field, strategy) pairs
    std::size_t count = 0;
    for_each_assignment(plan, space.descriptors,
        [&count](Field field, Strategy strat) {
            switch (field) {
                case Field::Tag:      EXPECT_EQ(strat, Strategy::Aggressive); break;
                case Field::Length:   EXPECT_EQ(strat, Strategy::Fast);       break;
                case Field::Value:    EXPECT_EQ(strat, Strategy::Aggressive); break;
                case Field::Checksum: EXPECT_EQ(strat, Strategy::Fast);       break;
            }
            ++count;
        });
    EXPECT_EQ(count, 4u);
}

// =============================================================================
// Test: assignment_count
// =============================================================================

TEST(CandidateTraversal, AssignmentCount) {
    constexpr auto plan = solve();
    constexpr auto space = make_space();

    auto n = assignment_count(plan, space.descriptors);
    EXPECT_EQ(n, 4u);
}

// =============================================================================
// Test: extract_assignments
// =============================================================================

TEST(CandidateTraversal, ExtractAssignments) {
    constexpr auto plan = solve();
    constexpr auto space = make_space();

    constexpr auto strats = extract_assignments<Strategy, 4>(plan, space.descriptors);
    static_assert(strats.size() == 4);
    static_assert(strats[0] == Strategy::Aggressive);  // Tag
    static_assert(strats[1] == Strategy::Fast);         // Length
    static_assert(strats[2] == Strategy::Aggressive);   // Value
    static_assert(strats[3] == Strategy::Fast);         // Checksum

    EXPECT_EQ(strats.size(), 4u);
}

// =============================================================================
// Test: backward compatibility — operator[] still works on candidates
// =============================================================================

TEST(CandidateTraversal, BackwardCompatOperatorBracket) {
    constexpr auto plan = solve();

    // plan.params[i] still works
    static_assert(plan.params[0] == Strategy::Aggressive);
    static_assert(plan.params[1] == Strategy::Fast);
    static_assert(plan.params[2] == Strategy::Aggressive);
    static_assert(plan.params[3] == Strategy::Fast);

    // .size() works
    static_assert(plan.params.size() == 4);
}

// =============================================================================
// Test: additive_cost still works with per_element_candidate
// =============================================================================

constexpr auto solve_whole_candidate() {
    auto space = per_element_space<Strategy, 4, 3>{
        .strategies = { Strategy::Baseline, Strategy::Fast, Strategy::Aggressive },
    };
    auto cost = additive_cost{
        [](std::size_t pos, Strategy s) constexpr -> double {
            return cost_table[pos][static_cast<std::size_t>(s)];
        }
    };
    return per_element_argmin(space, cost);
}

static_assert(solve_whole_candidate().predicted_cost == 22.0);

TEST(CandidateTraversal, AdditiveCostWithNewCandidate) {
    constexpr auto plan = solve_whole_candidate();
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 22.0);
}

// =============================================================================
// Test: constexpr is_feasible works (no std::isfinite dependency)
// =============================================================================

TEST(CandidateTraversal, IsFeasibleConstexpr) {
    constexpr auto plan = solve();
    static_assert(plan.is_feasible());      // This was the broken path
    static_assert(!plan.is_infeasible());
    EXPECT_TRUE(plan.is_feasible());
}
