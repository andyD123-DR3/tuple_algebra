// tests/test_select_and_run.cpp
// Tests for solver/algorithms/select_and_run.h
// Verifies concept-constrained overload dispatch.

#include "ctdp/solver/algorithms/select_and_run.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/cost_models/additive.h"
#include "ctdp/solver/cost_models/chain.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =====================================================================
// Overload 1: factored_space → per_element_argmin
// =====================================================================

enum class Strat { A, B };

constexpr auto sar_factored() {
    auto space = per_element_space<Strat, 2, 2>{
        .strategies = {Strat::A, Strat::B}
    };

    auto cost = additive_cost{
        [](std::size_t pos, Strat s) constexpr -> double {
            constexpr double t[2][2] = {{1.0, 5.0}, {3.0, 2.0}};
            return t[pos][static_cast<std::size_t>(s)];
        }
    };

    return select_and_run(space, cost);
}

// Optimal: A(1) + B(2) = 3
static_assert(sar_factored().predicted_cost == 3.0);

TEST(SelectAndRun, FactoredSpace) {
    constexpr auto result = sar_factored();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 3.0);
}

// =====================================================================
// Overload 2: interval space + interval cost → interval_dp
// =====================================================================

constexpr auto sar_interval() {
    constexpr std::array<std::size_t, 4> dims{10, 30, 5, 60};
    auto space = interval_split_space<4>{.n = 3};
    auto cost  = make_chain_cost(dims);
    return select_and_run(space, cost);
}

static_assert(sar_interval().predicted_cost == 4500.0);

TEST(SelectAndRun, IntervalSpace) {
    constexpr auto result = sar_interval();
    EXPECT_DOUBLE_EQ(result.predicted_cost, 4500.0);
}

// =====================================================================
// Both agree with direct calls
// =====================================================================

constexpr auto sar_agrees_with_direct() {
    auto space = per_element_space<Strat, 2, 2>{
        .strategies = {Strat::A, Strat::B}
    };

    auto cost = additive_cost{
        [](std::size_t pos, Strat s) constexpr -> double {
            constexpr double t[2][2] = {{1.0, 5.0}, {3.0, 2.0}};
            return t[pos][static_cast<std::size_t>(s)];
        }
    };

    auto via_sar    = select_and_run(space, cost);
    auto via_direct = per_element_argmin(space, cost);

    return via_sar.predicted_cost == via_direct.predicted_cost;
}

static_assert(sar_agrees_with_direct());

TEST(SelectAndRun, AgreesWithDirect) {
    EXPECT_TRUE(sar_agrees_with_direct());
}
