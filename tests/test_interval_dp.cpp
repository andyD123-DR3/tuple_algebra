// tests/test_interval_dp.cpp
// Tests for solver/algorithms/interval_dp.h
// Includes the Cormen matrix chain known test case.

#include "ctdp/solver/algorithms/interval_dp.h"
#include "ctdp/solver/cost_models/chain.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =====================================================================
// Matrix chain — Cormen et al. (CLRS)
// Dimensions: [30, 35, 15, 5, 10, 20, 25]  (6 matrices)
// Optimal cost: 15,125 scalar multiplications
// Optimal parenthesisation: ((A1(A2 A3))((A4 A5)A6))
// =====================================================================

constexpr auto cormen_result() {
    constexpr std::array<std::size_t, 7> dims{30, 35, 15, 5, 10, 20, 25};
    auto space = interval_split_space<7>{.n = 6};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

// Compile-time verification
static_assert(cormen_result().predicted_cost == 15125.0);
static_assert(cormen_result().is_feasible());
static_assert(cormen_result().stats.candidates_evaluated > 0);
static_assert(cormen_result().stats.candidates_pruned == 0);

TEST(IntervalDP, CormenMatrixChainCost) {
    constexpr auto result = cormen_result();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 15125.0);
}

TEST(IntervalDP, CormenMatrixChainFeasible) {
    constexpr auto result = cormen_result();
    EXPECT_TRUE(result.is_feasible());
}

TEST(IntervalDP, CormenMatrixChainStats) {
    constexpr auto result = cormen_result();
    EXPECT_GT(result.stats.candidates_evaluated, 0u);
    EXPECT_GT(result.stats.subproblems_evaluated, 0u);
    EXPECT_EQ(result.stats.candidates_pruned, 0u);
    EXPECT_EQ(result.stats.memo_hits, 0u);
    EXPECT_EQ(result.stats.memo_misses, 0u);
}

// =====================================================================
// Trivial cases
// =====================================================================

// Single matrix — no multiplications
constexpr auto single_matrix() {
    constexpr std::array<std::size_t, 2> dims{10, 20};
    auto space = interval_split_space<2>{.n = 1};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

static_assert(single_matrix().predicted_cost == 0.0);

TEST(IntervalDP, SingleMatrix) {
    constexpr auto result = single_matrix();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 0.0);
}

// Two matrices — only one way to multiply
constexpr auto two_matrices() {
    constexpr std::array<std::size_t, 3> dims{10, 20, 30};
    auto space = interval_split_space<3>{.n = 2};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

// 10 × 20 × 30 = 6000
static_assert(two_matrices().predicted_cost == 6000.0);

TEST(IntervalDP, TwoMatrices) {
    constexpr auto result = two_matrices();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 6000.0);
}

// Three matrices: [10, 30, 5, 60]
// Option 1: (A1 A2) A3 = 10*30*5 + 10*5*60 = 1500 + 3000 = 4500
// Option 2: A1 (A2 A3) = 30*5*60 + 10*30*60 = 9000 + 18000 = 27000
// Optimal = 4500
constexpr auto three_matrices() {
    constexpr std::array<std::size_t, 4> dims{10, 30, 5, 60};
    auto space = interval_split_space<4>{.n = 3};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

static_assert(three_matrices().predicted_cost == 4500.0);

TEST(IntervalDP, ThreeMatrices) {
    constexpr auto result = three_matrices();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 4500.0);
}

// =====================================================================
// Four matrices: [40, 20, 30, 10, 30]
// Known optimal: 26000
// =====================================================================

constexpr auto four_matrices() {
    constexpr std::array<std::size_t, 5> dims{40, 20, 30, 10, 30};
    auto space = interval_split_space<5>{.n = 4};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

static_assert(four_matrices().predicted_cost == 26000.0);

TEST(IntervalDP, FourMatrices) {
    constexpr auto result = four_matrices();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 26000.0);
}

// =====================================================================
// Empty space
// =====================================================================

constexpr auto empty_space() {
    constexpr std::array<std::size_t, 1> dims{10};
    auto space = interval_split_space<1>{.n = 0};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

static_assert(empty_space().predicted_cost == 0.0);

TEST(IntervalDP, EmptySpace) {
    constexpr auto result = empty_space();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 0.0);
}

// =====================================================================
// Stats consistency: evaluated + pruned == tested NOT required for interval_dp.
// For interval_dp: subproblems_evaluated = subproblems, candidates_evaluated = split evals.
// =====================================================================

TEST(IntervalDP, StatsConsistency) {
    constexpr auto result = cormen_result();
    // For 6 matrices: subproblems = sum_{len=2}^{6} (7-len) = 5+4+3+2+1 = 15
    EXPECT_EQ(result.stats.subproblems_evaluated, 15u);
    // Split evaluations: sum_{len=2}^{6} sum_{i} (len-1)
    // len=2: 5*1=5, len=3: 4*2=8, len=4: 3*3=9, len=5: 2*4=8, len=6: 1*5=5 = 35
    EXPECT_EQ(result.stats.candidates_evaluated, 35u);
}
