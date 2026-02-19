// tests/test_solver_concepts.cpp
// Tests for solver/concepts.h — concept satisfaction checks.

#include "ctdp/solver/concepts.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/spaces/interval_split_space.h"
#include "ctdp/solver/spaces/permutation_space.h"
#include <array>
#include <gtest/gtest.h>

using namespace ctdp;

// --- Test types ---
enum class Strat { A, B, C };
using PES = per_element_space<Strat, 4, 3>;
using ISS = interval_split_space<8>;
using PS  = permutation_space<4>;

// --- factored_space ---
static_assert(factored_space<PES>);
static_assert(!factored_space<ISS>);  // no dimension/branching
static_assert(!factored_space<PS>);

TEST(SolverConcepts, FactoredSpace) {
    EXPECT_TRUE(factored_space<PES>);
    EXPECT_FALSE(factored_space<ISS>);
}

// --- search_space ---
static_assert(search_space<PES>);
static_assert(search_space<ISS>);
static_assert(search_space<PS>);

TEST(SolverConcepts, SearchSpace) {
    EXPECT_TRUE(search_space<PES>);
    EXPECT_TRUE(search_space<ISS>);
    EXPECT_TRUE(search_space<PS>);
}

// --- cost_function_for ---
using Candidate4 = std::array<Strat, 4>;
constexpr auto good_cost = [](Candidate4 const&) -> double { return 1.0; };
constexpr auto int_cost  = [](Candidate4 const&) -> int { return 1; };  // not floating-point

static_assert(cost_function_for<decltype(good_cost), Candidate4>);
static_assert(!cost_function_for<decltype(int_cost), Candidate4>);

TEST(SolverConcepts, CostFunctionFor) {
    EXPECT_TRUE((cost_function_for<decltype(good_cost), Candidate4>));
    EXPECT_FALSE((cost_function_for<decltype(int_cost), Candidate4>));
}

// --- interval_cost ---
struct GoodIntervalCost {
    constexpr auto combine(std::size_t, std::size_t, std::size_t) const -> double { return 0.0; }
    constexpr auto leaf(std::size_t) const -> double { return 0.0; }
};

struct BadIntervalCost {
    constexpr auto combine(std::size_t, std::size_t, std::size_t) const -> int { return 0; }
    constexpr auto leaf(std::size_t) const -> double { return 0.0; }
};

static_assert(interval_cost<GoodIntervalCost>);
static_assert(!interval_cost<BadIntervalCost>);  // combine returns int

TEST(SolverConcepts, IntervalCost) {
    EXPECT_TRUE(interval_cost<GoodIntervalCost>);
    EXPECT_FALSE(interval_cost<BadIntervalCost>);
}

// --- dynamic_constraint_for ---
constexpr auto good_constraint = [](Candidate4 const&) -> bool { return true; };
constexpr auto bad_constraint  = [](Candidate4 const&) -> int { return 1; };

static_assert(dynamic_constraint_for<decltype(good_constraint), Candidate4>);
// int is convertible to bool, so this also satisfies:
static_assert(dynamic_constraint_for<decltype(bad_constraint), Candidate4>);

TEST(SolverConcepts, DynamicConstraintFor) {
    EXPECT_TRUE((dynamic_constraint_for<decltype(good_constraint), Candidate4>));
}
