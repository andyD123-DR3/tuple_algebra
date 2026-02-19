// tests/test_exhaustive_search.cpp
// Tests for solver/algorithms/exhaustive_search.h

#include "ctdp/solver/algorithms/exhaustive_search.h"
#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/spaces/permutation_space.h"
#include "ctdp/solver/cost_models/additive.h"
#include <gtest/gtest.h>

using namespace ctdp;

enum class Strat { A, B };

// --- Exhaustive over small per_element_space ---
constexpr double cost_table[2][2] = {
    {10.0, 20.0},  // pos 0
    { 5.0, 15.0},  // pos 1
};

constexpr auto exhaustive_pes() {
    auto space = per_element_space<Strat, 2, 2>{
        .strategies = {Strat::A, Strat::B}
    };

    auto cost = additive_cost{
        [](std::size_t pos, Strat s) constexpr -> double {
            return cost_table[pos][static_cast<std::size_t>(s)];
        }
    };

    return exhaustive_search(space, cost);
}

// Optimal: pos0=A(10), pos1=A(5) → 15
static_assert(exhaustive_pes().predicted_cost == 15.0);
static_assert(exhaustive_pes().params[0] == Strat::A);
static_assert(exhaustive_pes().params[1] == Strat::A);

TEST(ExhaustiveSearch, PerElementSpace) {
    constexpr auto result = exhaustive_pes();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 15.0);
}

TEST(ExhaustiveSearch, StatsAllVisited) {
    constexpr auto result = exhaustive_pes();
    EXPECT_EQ(result.stats.candidates_total, 4u);  // 2^2
    EXPECT_EQ(result.stats.candidates_evaluated, 4u);     // no constraints
    EXPECT_EQ(result.stats.candidates_pruned, 0u);
}

// --- With constraint: exclude (A, A) ---
constexpr auto exhaustive_constrained() {
    auto space = per_element_space<Strat, 2, 2>{
        .strategies = {Strat::A, Strat::B}
    };

    auto cost = additive_cost{
        [](std::size_t pos, Strat s) constexpr -> double {
            return cost_table[pos][static_cast<std::size_t>(s)];
        }
    };

    auto constraint = [](std::array<Strat, 2> const& c) constexpr -> bool {
        return !(c[0] == Strat::A && c[1] == Strat::A);
    };

    return exhaustive_search(space, cost, constraint);
}

// (A,A)=15 excluded. Next best: (A,B)=25
static_assert(exhaustive_constrained().predicted_cost == 25.0);

TEST(ExhaustiveSearch, WithConstraint) {
    constexpr auto result = exhaustive_constrained();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 25.0);
    EXPECT_EQ(result.stats.candidates_pruned, 1u);
}

// --- Permutation space: find min-cost ordering ---
constexpr auto exhaustive_perm() {
    permutation_space<3> space{};

    // Cost = sum of |perm[i] - i|   (identity is optimal)
    auto cost = [](std::array<std::size_t, 3> const& p) constexpr -> double {
        double total = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            auto diff = p[i] > i ? p[i] - i : i - p[i];
            total += static_cast<double>(diff);
        }
        return total;
    };

    return exhaustive_search(space, cost);
}

// Identity permutation [0,1,2] has cost 0
static_assert(exhaustive_perm().predicted_cost == 0.0);

TEST(ExhaustiveSearch, PermutationSpace) {
    constexpr auto result = exhaustive_perm();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 0.0);
    EXPECT_EQ(result.stats.candidates_total, 6u);  // 3!
}

// --- Agrees with per_element_argmin on the same problem ---
constexpr auto agreement_test() {
    auto space = per_element_space<Strat, 2, 2>{
        .strategies = {Strat::A, Strat::B}
    };

    auto cost = additive_cost{
        [](std::size_t pos, Strat s) constexpr -> double {
            return cost_table[pos][static_cast<std::size_t>(s)];
        }
    };

    auto exhaustive = exhaustive_search(space, cost);
    auto argmin     = per_element_argmin(space, cost);

    return exhaustive.predicted_cost == argmin.predicted_cost;
}

static_assert(agreement_test());

TEST(ExhaustiveSearch, AgreesWithArgmin) {
    EXPECT_TRUE(agreement_test());
}
