// tests/test_cost_models.cpp
// Tests for solver/cost_models/ (additive, chain, weighted)

#include "ctdp/solver/cost_models/additive.h"
#include "ctdp/solver/cost_models/chain.h"
#include "ctdp/solver/cost_models/weighted.h"
#include "ctdp/solver/concepts.h"
#include <array>
#include <gtest/gtest.h>

using namespace ctdp;

// =====================================================================
// additive_cost
// =====================================================================

enum class S { X, Y };

constexpr auto make_additive() {
    return additive_cost{
        [](std::size_t pos, S s) constexpr -> double {
            constexpr double table[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
            return table[pos][static_cast<std::size_t>(s)];
        }
    };
}

static_assert(make_additive()(std::array<S, 3>{S::X, S::X, S::X}) == 9.0);   // 1+3+5
static_assert(make_additive()(std::array<S, 3>{S::Y, S::Y, S::Y}) == 12.0);  // 2+4+6
static_assert(make_additive()(std::array<S, 3>{S::X, S::Y, S::X}) == 10.0);  // 1+4+5

TEST(AdditiveCost, BasicEvaluation) {
    constexpr auto cost = make_additive();
    EXPECT_DOUBLE_EQ(cost(std::array<S, 3>{S::X, S::X, S::X}), 9.0);
    EXPECT_DOUBLE_EQ(cost(std::array<S, 3>{S::Y, S::Y, S::Y}), 12.0);
    EXPECT_DOUBLE_EQ(cost(std::array<S, 3>{S::X, S::Y, S::X}), 10.0);
}

// Concept satisfaction
static_assert(cost_function_for<decltype(make_additive()), std::array<S, 3>>);

TEST(AdditiveCost, ConceptSatisfaction) {
    EXPECT_TRUE((cost_function_for<decltype(make_additive()), std::array<S, 3>>));
}

// =====================================================================
// chain_cost
// =====================================================================

constexpr auto make_test_chain() {
    constexpr std::array<std::size_t, 4> dims{10, 30, 5, 60};
    return make_chain_cost(dims);
}

// leaf cost = 0
static_assert(make_test_chain().leaf(0) == 0.0);
static_assert(make_test_chain().leaf(1) == 0.0);

// combine(0, 1, 2): dims[0]*dims[1]*dims[3] = 10*30*60... 
// Actually combine(i, k, j) = dims[i] * dims[k] * dims[j+1]
// combine(0, 1, 1): split at 1 for subproblem [0,1]
// = dims[0]*dims[1]*dims[2] = 10*30*5 = 1500
static_assert(make_test_chain().combine(0, 1, 1) == 1500.0);

TEST(ChainCost, CombineValues) {
    constexpr auto cost = make_test_chain();
    EXPECT_DOUBLE_EQ(cost.leaf(0), 0.0);
    EXPECT_DOUBLE_EQ(cost.combine(0, 1, 1), 1500.0);
}

// interval_cost concept
static_assert(interval_cost<decltype(make_test_chain())>);

TEST(ChainCost, IntervalCostConcept) {
    EXPECT_TRUE(interval_cost<decltype(make_test_chain())>);
}

// =====================================================================
// weighted_cost
// =====================================================================

constexpr auto make_weighted_test() {
    auto c1 = [](std::array<int, 2> const& x) constexpr -> double {
        return static_cast<double>(x[0] + x[1]);
    };
    auto c2 = [](std::array<int, 2> const& x) constexpr -> double {
        return static_cast<double>(x[0] * x[1]);
    };

    return make_weighted_cost(std::array{0.5, 0.5}, c1, c2);
}

// x = {3, 4}: c1=7, c2=12, weighted = 0.5*7 + 0.5*12 = 9.5
static_assert(make_weighted_test()(std::array{3, 4}) == 9.5);

TEST(WeightedCost, BasicEvaluation) {
    constexpr auto cost = make_weighted_test();
    EXPECT_DOUBLE_EQ(cost(std::array{3, 4}), 9.5);
    EXPECT_DOUBLE_EQ(cost(std::array{0, 0}), 0.0);
}
