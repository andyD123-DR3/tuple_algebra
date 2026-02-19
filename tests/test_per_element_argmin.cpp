// tests/test_per_element_argmin.cpp
// Tests for solver/algorithms/per_element_argmin.h

#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/cost_models/additive.h"
#include <gtest/gtest.h>

using namespace ctdp;

enum class Strat { Fast, Small, Safe };

// --- Basic 3-position, 3-strategy problem ---
// Cost table:
//   pos 0: Fast=10, Small=20, Safe=30
//   pos 1: Fast=50, Small=5,  Safe=15
//   pos 2: Fast=8,  Small=12, Safe=25

constexpr double cost_table[3][3] = {
    {10.0, 20.0, 30.0},  // pos 0
    {50.0,  5.0, 15.0},  // pos 1
    { 8.0, 12.0, 25.0},  // pos 2
};

// --- Overload 1: whole-candidate cost, no constraints ---

constexpr auto make_test() {
    auto space = per_element_space<Strat, 3, 3>{
        .strategies = {Strat::Fast, Strat::Small, Strat::Safe}
    };

    auto cost = additive_cost{
        [](std::size_t pos, Strat s) constexpr -> double {
            return cost_table[pos][static_cast<std::size_t>(s)];
        }
    };

    return per_element_argmin(space, cost);
}

// Optimal: pos0=Fast(10), pos1=Small(5), pos2=Fast(8) → total = 23
static_assert(make_test().predicted_cost == 23.0);
static_assert(make_test().params[0] == Strat::Fast);
static_assert(make_test().params[1] == Strat::Small);
static_assert(make_test().params[2] == Strat::Fast);
static_assert(make_test().is_feasible());

TEST(PerElementArgmin, BasicOptimal) {
    constexpr auto result = make_test();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 23.0);
    EXPECT_EQ(result.params[0], Strat::Fast);
    EXPECT_EQ(result.params[1], Strat::Small);
    EXPECT_EQ(result.params[2], Strat::Fast);
}

TEST(PerElementArgmin, Stats) {
    constexpr auto result = make_test();
    EXPECT_EQ(result.stats.candidates_total, 9u);   // 3 positions × 3 strategies
    EXPECT_EQ(result.stats.candidates_evaluated, 9u);
    EXPECT_EQ(result.stats.candidates_pruned, 0u);
}

// --- Overload 2: element cost + element predicate ---

enum class Pos : std::size_t { A = 0, B = 1, C = 2 };

constexpr auto make_constrained_test() {
    auto space = per_element_space<Strat, 3, 3, Pos>{
        .descriptors = { Pos::A, Pos::B, Pos::C },
        .strategies  = {Strat::Fast, Strat::Small, Strat::Safe}
    };

    auto cost = [](Pos pos, std::size_t choice) constexpr -> double {
        return cost_table[static_cast<std::size_t>(pos)][choice];
    };

    // Position A cannot use choice 0 (Fast)
    auto constraint = [](Pos pos, std::size_t choice) constexpr -> bool {
        return !(pos == Pos::A && choice == 0);
    };

    return per_element_argmin(space, cost, constraint);
}

// Without constraint: pos0=Fast(10). With constraint: pos0=Small(20).
// Total = 20 + 5 + 8 = 33
static_assert(make_constrained_test().predicted_cost == 33.0);
static_assert(make_constrained_test().params[0] == Strat::Small);

TEST(PerElementArgmin, WithElementConstraint) {
    constexpr auto result = make_constrained_test();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 33.0);
    EXPECT_EQ(result.params[0], Strat::Small);
}

TEST(PerElementArgmin, ConstraintStats) {
    constexpr auto result = make_constrained_test();
    EXPECT_GT(result.stats.candidates_pruned, 0u);
}

// --- Uniform cost — first strategy wins ---
constexpr auto make_uniform_test() {
    auto space = per_element_space<Strat, 2, 2>{
        .strategies = {Strat::Fast, Strat::Small}
    };

    auto cost = additive_cost{
        [](std::size_t, Strat) constexpr -> double { return 1.0; }
    };

    return per_element_argmin(space, cost);
}

static_assert(make_uniform_test().predicted_cost == 2.0);

TEST(PerElementArgmin, UniformCost) {
    constexpr auto result = make_uniform_test();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 2.0);
}
