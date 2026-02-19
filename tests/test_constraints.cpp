// tests/test_constraints.cpp
// Tests for solver/constraints/constraint.h — combinators and factories.

#include "ctdp/solver/constraints/constraint.h"
#include "ctdp/solver/concepts.h"
#include "ctdp/solver/algorithms/beam_search.h"
#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/spaces/heterogeneous_per_element_space.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =============================================================================
// Domain fixtures
// =============================================================================

enum class Strat { Fast, Medium, Safe };
enum class Field : std::size_t { Tag = 0, Length = 1, Value = 2 };

constexpr auto make_space() {
    return per_element_space<Strat, 3, 3, Field>{
        .descriptors = { Field::Tag, Field::Length, Field::Value },
        .strategies  = { Strat::Fast, Strat::Medium, Strat::Safe },
    };
}

constexpr auto whole_cost = [](auto const& c) constexpr -> double {
    constexpr double table[] = { 1.0, 3.0, 5.0 };
    double total = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        total += table[static_cast<std::size_t>(c[i])];
    return total;
};

constexpr auto element_cost = [](Field /*f*/, std::size_t choice) -> double {
    constexpr double table[] = { 1.0, 3.0, 5.0 };
    return table[choice];
};

// =============================================================================
// Combinator tests
// =============================================================================

TEST(Constraints, NotC) {
    auto always_true  = [](auto const&) constexpr { return true; };
    auto negated = not_c(always_true);

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    EXPECT_FALSE(negated(c));
}

TEST(Constraints, AllOf) {
    auto gt_zero = [](auto const& c) constexpr { return c[0] != Strat::Fast; };
    auto gt_one  = [](auto const& c) constexpr { return c[1] != Strat::Fast; };
    auto both = all_of(gt_zero, gt_one);

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    c[0] = Strat::Medium;
    c[1] = Strat::Medium;
    c[2] = Strat::Fast;
    EXPECT_TRUE(both(c));

    c[0] = Strat::Fast;
    EXPECT_FALSE(both(c));
}

TEST(Constraints, AnyOf) {
    auto is_fast_0 = [](auto const& c) constexpr { return c[0] == Strat::Fast; };
    auto is_fast_1 = [](auto const& c) constexpr { return c[1] == Strat::Fast; };
    auto either = any_of(is_fast_0, is_fast_1);

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    c[0] = Strat::Medium;
    c[1] = Strat::Medium;
    EXPECT_FALSE(either(c));

    c[1] = Strat::Fast;
    EXPECT_TRUE(either(c));
}

TEST(Constraints, CombinatorComposition) {
    // not(any_of(a, b)) == none_of(a, b) == all_of(not(a), not(b))
    auto is_fast_0 = [](auto const& c) constexpr { return c[0] == Strat::Fast; };
    auto is_fast_1 = [](auto const& c) constexpr { return c[1] == Strat::Fast; };
    auto none_fast_01 = not_c(any_of(is_fast_0, is_fast_1));

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    c[0] = Strat::Medium;
    c[1] = Strat::Safe;
    c[2] = Strat::Fast;  // position 2 is irrelevant
    EXPECT_TRUE(none_fast_01(c));

    c[0] = Strat::Fast;
    EXPECT_FALSE(none_fast_01(c));
}

// =============================================================================
// Dynamic constraint factory tests
// =============================================================================

TEST(Constraints, BudgetCap) {
    auto cap = budget_cap(whole_cost, 5.0);

    using Cand = typename decltype(make_space())::candidate_type;

    // All-Fast: cost 3.0 ≤ 5.0
    Cand c1{};
    c1[0] = Strat::Fast; c1[1] = Strat::Fast; c1[2] = Strat::Fast;
    EXPECT_TRUE(cap(c1));

    // All-Medium: cost 9.0 > 5.0
    Cand c2{};
    c2[0] = Strat::Medium; c2[1] = Strat::Medium; c2[2] = Strat::Medium;
    EXPECT_FALSE(cap(c2));
}

TEST(Constraints, MutualExclusion) {
    auto excl = mutual_exclusion(0, 1);

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    c[0] = Strat::Fast; c[1] = Strat::Fast;
    EXPECT_FALSE(excl(c));

    c[1] = Strat::Medium;
    EXPECT_TRUE(excl(c));
}

TEST(Constraints, ForbidAt) {
    auto no_safe_at_0 = forbid_at(std::size_t{0}, Strat::Safe);

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    c[0] = Strat::Safe;
    EXPECT_FALSE(no_safe_at_0(c));

    c[0] = Strat::Fast;
    EXPECT_TRUE(no_safe_at_0(c));
}

TEST(Constraints, RequireAt) {
    auto must_fast_2 = require_at(std::size_t{2}, Strat::Fast);

    using Cand = typename decltype(make_space())::candidate_type;
    Cand c{};
    c[2] = Strat::Medium;
    EXPECT_FALSE(must_fast_2(c));

    c[2] = Strat::Fast;
    EXPECT_TRUE(must_fast_2(c));
}

// =============================================================================
// Element predicate factory tests
// =============================================================================

TEST(Constraints, ForbidChoice) {
    auto no_choice_2 = forbid_choice(2);

    EXPECT_TRUE(no_choice_2(Field::Tag, std::size_t{0}));
    EXPECT_TRUE(no_choice_2(Field::Tag, std::size_t{1}));
    EXPECT_FALSE(no_choice_2(Field::Tag, std::size_t{2}));
}

TEST(Constraints, AllowOnly) {
    auto only_01 = allow_only({0, 1});

    EXPECT_TRUE(only_01(Field::Tag, std::size_t{0}));
    EXPECT_TRUE(only_01(Field::Tag, std::size_t{1}));
    EXPECT_FALSE(only_01(Field::Tag, std::size_t{2}));
}

// =============================================================================
// Concept satisfaction — factories produce valid constraints
// =============================================================================

TEST(Constraints, ConceptSatisfaction) {
    using Cand = typename decltype(make_space())::candidate_type;

    // Dynamic constraints satisfy dynamic_constraint_for.
    static_assert(dynamic_constraint_for<
        decltype(budget_cap(whole_cost, 5.0)), Cand>);
    static_assert(dynamic_constraint_for<
        decltype(mutual_exclusion(0, 1)), Cand>);
    static_assert(dynamic_constraint_for<
        decltype(forbid_at(std::size_t{0}, Strat::Fast)), Cand>);
    static_assert(dynamic_constraint_for<
        decltype(require_at(std::size_t{0}, Strat::Fast)), Cand>);

    // Element predicates satisfy element_predicate_for.
    static_assert(element_predicate_for<
        decltype(forbid_choice(2)), Field>);
    static_assert(element_predicate_for<
        decltype(allow_only({0, 1})), Field>);

    // Combinators preserve concept satisfaction.
    static_assert(dynamic_constraint_for<
        decltype(not_c(budget_cap(whole_cost, 5.0))), Cand>);
    static_assert(dynamic_constraint_for<
        decltype(all_of(mutual_exclusion(0, 1),
                        forbid_at(std::size_t{0}, Strat::Safe))), Cand>);
    static_assert(dynamic_constraint_for<
        decltype(any_of(require_at(std::size_t{0}, Strat::Fast),
                        require_at(std::size_t{0}, Strat::Medium))), Cand>);
}

// =============================================================================
// Integration: factories with beam_search
// =============================================================================

TEST(Constraints, BeamSearchWithFactories) {
    constexpr auto space = make_space();

    auto plan = beam_search(space, whole_cost,
        forbid_at(std::size_t{0}, Strat::Fast),
        mutual_exclusion(0, 1));

    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], Strat::Fast);
    EXPECT_NE(plan.params[0], plan.params[1]);

    // Best: Medium(3) at 0, Fast(1) at 1, Fast(1) at 2 = 5.0
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 5.0);
}

TEST(Constraints, BeamSearchWithAllOf) {
    constexpr auto space = make_space();

    auto combined = all_of(
        forbid_at(std::size_t{0}, Strat::Fast),
        forbid_at(std::size_t{1}, Strat::Fast));

    auto plan = beam_search(space, whole_cost, combined);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], Strat::Fast);
    EXPECT_NE(plan.params[1], Strat::Fast);

    // Medium(3) + Medium(3) + Fast(1) = 7.0
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 7.0);
}

TEST(Constraints, BeamSearchWithBudgetCap) {
    constexpr auto space = make_space();

    // Budget ≤ 3.0: only all-Fast (3.0) works.
    auto plan = beam_search(space, whole_cost,
        budget_cap(whole_cost, 3.0));

    EXPECT_TRUE(plan.is_feasible());
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 3.0);

    // Budget ≤ 2.0: impossible (min cost is 3.0).
    auto plan2 = beam_search(space, whole_cost,
        budget_cap(whole_cost, 2.0));
    EXPECT_TRUE(plan2.is_infeasible());
}

// =============================================================================
// Integration: element predicate factories with per_element_argmin
// =============================================================================

TEST(Constraints, ArgminWithForbidChoice) {
    constexpr auto space = make_space();

    // Forbid choice 0 (Fast) everywhere.
    constexpr auto plan = per_element_argmin(space, element_cost,
        forbid_choice(0));

    // Best remaining: Medium(3) × 3 = 9.0
    static_assert(plan.predicted_cost == 9.0);
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 9.0);
}

TEST(Constraints, ArgminWithAllowOnly) {
    constexpr auto space = make_space();

    // Allow only choices 0 and 1 (Fast and Medium).
    constexpr auto plan = per_element_argmin(space, element_cost,
        allow_only({0, 1}));

    // Best: Fast(1) × 3 = 3.0
    static_assert(plan.predicted_cost == 3.0);
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 3.0);
}

// =============================================================================
// Constexpr validation
// =============================================================================

constexpr auto constexpr_test() {
    auto space = per_element_space<Strat, 3, 3, Field>{
        .descriptors = { Field::Tag, Field::Length, Field::Value },
        .strategies  = { Strat::Fast, Strat::Medium, Strat::Safe },
    };
    return beam_search(space, whole_cost,
        forbid_at(std::size_t{0}, Strat::Fast));
}

static_assert(constexpr_test().is_feasible());
static_assert(constexpr_test().predicted_cost == 5.0);
