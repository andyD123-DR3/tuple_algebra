// tests/test_constrained_search.cpp
// Tests for beam_search and local_search — constrained solver algorithms.
// Covers: beam_search on factored spaces, local_search on neighbourhood
// spaces, select_and_run dispatch, neighbours() on per-element spaces.

#include "ctdp/solver/algorithms/beam_search.h"
#include "ctdp/solver/algorithms/local_search.h"
#include "ctdp/solver/algorithms/select_and_run.h"
#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/algorithms/exhaustive_search.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/spaces/heterogeneous_per_element_space.h"
#include "ctdp/solver/spaces/permutation_space.h"
#include "ctdp/solver/spaces/cartesian_space.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =============================================================================
// Domain: Strategy assignment with budget constraint
// =============================================================================

enum class Strat { Fast, Medium, Safe };

// Cost per (position, strategy): Fast=1.0, Medium=3.0, Safe=5.0
constexpr auto make_3x3_space() {
    return make_anonymous_space<Strat, 3>(
        std::array{ Strat::Fast, Strat::Medium, Strat::Safe });
}

constexpr auto whole_cost = [](auto const& candidate) constexpr -> double {
    double total = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        switch (candidate[i]) {
            case Strat::Fast:   total += 1.0; break;
            case Strat::Medium: total += 3.0; break;
            case Strat::Safe:   total += 5.0; break;
        }
    }
    return total;
};

// Budget constraint: total cost must be ≤ 10.0
constexpr auto budget_constraint = [](auto const& candidate) constexpr -> bool {
    double total = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        switch (candidate[i]) {
            case Strat::Fast:   total += 1.0; break;
            case Strat::Medium: total += 3.0; break;
            case Strat::Safe:   total += 5.0; break;
        }
    }
    return total <= 10.0;
};

// Tight budget: cost must be ≤ 5.0
constexpr auto tight_budget = [](auto const& candidate) constexpr -> bool {
    double total = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        switch (candidate[i]) {
            case Strat::Fast:   total += 1.0; break;
            case Strat::Medium: total += 3.0; break;
            case Strat::Safe:   total += 5.0; break;
        }
    }
    return total <= 5.0;
};

// =============================================================================
// Test: concept satisfaction — has_neighbours on per-element spaces
// =============================================================================

TEST(ConceptSatisfaction, PerElementSpaceHasNeighbours) {
    using PES = per_element_space<Strat, 3, 3>;
    static_assert(has_neighbours<PES>);
    static_assert(search_space<PES>);
    static_assert(factored_space<PES>);
}

TEST(ConceptSatisfaction, HeterogeneousSpaceHasNeighbours) {
    using HES = heterogeneous_per_element_space<Strat, 2, 3>;
    static_assert(has_neighbours<HES>);
    static_assert(factored_space<HES>);
}

// =============================================================================
// Test: neighbours() produces correct count
// =============================================================================

TEST(Neighbours, PerElementSpaceCount) {
    constexpr auto space = make_3x3_space();
    using candidate_type = decltype(space)::candidate_type;

    // Start from all-Fast candidate
    candidate_type start{};
    start[0] = Strat::Fast;
    start[1] = Strat::Fast;
    start[2] = Strat::Fast;

    std::size_t count = 0;
    space.neighbours(start, [&count](auto const&) { ++count; });

    // 3 positions × 2 alternative strategies = 6 neighbours
    EXPECT_EQ(count, 6u);
}

TEST(Neighbours, HeterogeneousSpaceCount) {
    enum class HS { A, B, C };
    constexpr auto space = make_heterogeneous_space(
        position_choices{ 0ul, std::array{ HS::A, HS::B, HS::C } },  // 3 choices
        position_choices{ 1ul, std::array{ HS::A, HS::B } }          // 2 choices
    );

    using candidate_type = decltype(space)::candidate_type;
    candidate_type start{};
    start[0] = HS::A;
    start[1] = HS::A;

    std::size_t count = 0;
    space.neighbours(start, [&count](auto const&) { ++count; });

    // pos 0: 2 alternatives, pos 1: 1 alternative → 3 neighbours
    EXPECT_EQ(count, 3u);
}

// =============================================================================
// Test: beam_search — unconstrained matches per_element_argmin
// =============================================================================

TEST(BeamSearch, UnconstrainedMatchesArgmin) {
    constexpr auto space = make_3x3_space();

    // beam_search without constraints should find the same optimum
    // as per_element_argmin.
    constexpr auto beam_plan = beam_search(space, whole_cost);
    constexpr auto argmin_plan = per_element_argmin(space, whole_cost);

    static_assert(beam_plan.predicted_cost == argmin_plan.predicted_cost);
    EXPECT_DOUBLE_EQ(beam_plan.predicted_cost, argmin_plan.predicted_cost);

    // All-Fast = 3.0
    EXPECT_DOUBLE_EQ(beam_plan.predicted_cost, 3.0);
    EXPECT_EQ(beam_plan.params[0], Strat::Fast);
    EXPECT_EQ(beam_plan.params[1], Strat::Fast);
    EXPECT_EQ(beam_plan.params[2], Strat::Fast);
}

// =============================================================================
// Test: beam_search — with budget constraint
// =============================================================================

TEST(BeamSearch, WithBudgetConstraint) {
    constexpr auto space = make_3x3_space();

    // Budget ≤ 10: unconstrained optimum (3.0) is feasible, so same result.
    constexpr auto plan_loose = beam_search(space, whole_cost, budget_constraint);
    EXPECT_DOUBLE_EQ(plan_loose.predicted_cost, 3.0);
    EXPECT_TRUE(plan_loose.is_feasible());
}

TEST(BeamSearch, WithTightBudgetConstraint) {
    constexpr auto space = make_3x3_space();

    // Budget ≤ 5: all-Fast (3.0) is still feasible.
    constexpr auto plan_tight = beam_search(space, whole_cost, tight_budget);
    EXPECT_DOUBLE_EQ(plan_tight.predicted_cost, 3.0);
    EXPECT_TRUE(plan_tight.is_feasible());
}

// =============================================================================
// Test: beam_search — constraint that forces non-trivial solution
// =============================================================================

TEST(BeamSearch, ConstraintForcesNonTrivial) {
    constexpr auto space = make_3x3_space();

    // Constraint: position 0 must NOT be Fast.
    constexpr auto no_fast_at_0 = [](auto const& c) constexpr -> bool {
        return c[0] != Strat::Fast;
    };

    constexpr auto plan = beam_search(space, whole_cost, no_fast_at_0);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], Strat::Fast);

    // Best: Medium at 0 (3), Fast at 1,2 (1+1) = 5.0
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 5.0);
    EXPECT_EQ(plan.params[0], Strat::Medium);
    EXPECT_EQ(plan.params[1], Strat::Fast);
    EXPECT_EQ(plan.params[2], Strat::Fast);
}

// =============================================================================
// Test: beam_search — multiple constraints
// =============================================================================

TEST(BeamSearch, MultipleConstraints) {
    constexpr auto space = make_3x3_space();

    // Constraint 1: no Fast at position 0
    constexpr auto no_fast_0 = [](auto const& c) constexpr -> bool {
        return c[0] != Strat::Fast;
    };
    // Constraint 2: no Fast at position 1
    constexpr auto no_fast_1 = [](auto const& c) constexpr -> bool {
        return c[1] != Strat::Fast;
    };

    constexpr auto plan = beam_search(space, whole_cost, no_fast_0, no_fast_1);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], Strat::Fast);
    EXPECT_NE(plan.params[1], Strat::Fast);

    // Best: Medium at 0,1 (3+3), Fast at 2 (1) = 7.0
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 7.0);
}

// =============================================================================
// Test: beam_search — all infeasible
// =============================================================================

TEST(BeamSearch, AllInfeasible) {
    constexpr auto space = make_3x3_space();

    // Impossible constraint: total must be ≤ 0
    constexpr auto impossible = [](auto const&) constexpr -> bool {
        return false;
    };

    constexpr auto plan = beam_search(space, whole_cost, impossible);
    EXPECT_TRUE(plan.is_infeasible());
}

// =============================================================================
// Test: beam_search — narrow beam width
// =============================================================================

TEST(BeamSearch, NarrowBeam) {
    constexpr auto space = make_3x3_space();

    // BeamWidth=1 is greedy: at each level, keep only the single best.
    constexpr auto plan = beam_search<1>(space, whole_cost);
    EXPECT_TRUE(plan.is_feasible());
    // Greedy should still find all-Fast for this simple problem.
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 3.0);
}

// =============================================================================
// Test: beam_search — stats populated correctly
// =============================================================================

TEST(BeamSearch, StatsPopulated) {
    constexpr auto space = make_3x3_space();

    auto plan = beam_search(space, whole_cost);
    EXPECT_GT(plan.stats.candidates_total, 0u);
    EXPECT_GT(plan.stats.candidates_evaluated, 0u);
    EXPECT_EQ(plan.stats.subproblems_total, 3u);  // 3 dimensions
    EXPECT_EQ(plan.stats.subproblems_evaluated, 3u);
    EXPECT_EQ(plan.stats.beam_width_used, 32u);  // default
}

// =============================================================================
// Test: beam_search on heterogeneous space — FIX-like demo
// =============================================================================

TEST(BeamSearch, HeterogeneousSpace) {
    enum class Field : std::size_t { Tag = 0, Value = 1 };
    enum class Strategy { Baseline, Inline, SIMD };

    constexpr auto space = make_heterogeneous_space(
        position_choices{ Field::Tag,   std::array{ Strategy::Baseline, Strategy::Inline } },
        position_choices{ Field::Value, std::array{ Strategy::Baseline, Strategy::SIMD } }
    );

    // Costs: Baseline=10, Inline=3, SIMD=2
    auto cost_fn = [](auto const& c) constexpr -> double {
        double total = 0.0;
        auto score = [](Strategy s) constexpr -> double {
            switch (s) {
                case Strategy::Baseline: return 10.0;
                case Strategy::Inline:   return 3.0;
                case Strategy::SIMD:     return 2.0;
            }
            return 10.0;
        };
        total += score(c[0]);
        total += score(c[1]);
        return total;
    };

    // Constraint: no SIMD at position 0 (Tag field doesn't support SIMD).
    // This is automatically satisfied since Tag only offers Baseline/Inline.
    // Add a global constraint: total cost must be ≤ 12.
    auto budget = [&cost_fn](auto const& c) constexpr -> bool {
        return cost_fn(c) <= 12.0;
    };

    auto plan = beam_search(space, cost_fn, budget);
    EXPECT_TRUE(plan.is_feasible());

    // Best: Inline(3) + SIMD(2) = 5.0, which is within budget.
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 5.0);
}

// =============================================================================
// Test: beam_search — constexpr validation
// =============================================================================

constexpr auto beam_constexpr_test() {
    auto space = make_anonymous_space<Strat, 3>(
        std::array{ Strat::Fast, Strat::Medium, Strat::Safe });
    return beam_search(space, whole_cost);
}
static_assert(beam_constexpr_test().predicted_cost == 3.0);
static_assert(beam_constexpr_test().is_feasible());

// =============================================================================
// Test: local_search — steepest descent on per_element_space
// =============================================================================

TEST(LocalSearch, PerElementSteepestDescent) {
    constexpr auto space = make_3x3_space();
    using candidate_type = decltype(space)::candidate_type;

    // Start from all-Safe (worst: cost=15)
    candidate_type start{};
    start[0] = Strat::Safe;
    start[1] = Strat::Safe;
    start[2] = Strat::Safe;

    auto plan = local_search(space, whole_cost, start);
    EXPECT_TRUE(plan.is_feasible());

    // Should descend to all-Fast (cost=3)
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 3.0);
    EXPECT_EQ(plan.params[0], Strat::Fast);
    EXPECT_EQ(plan.params[1], Strat::Fast);
    EXPECT_EQ(plan.params[2], Strat::Fast);
}

// =============================================================================
// Test: local_search — with constraint
// =============================================================================

TEST(LocalSearch, WithConstraint) {
    constexpr auto space = make_3x3_space();
    using candidate_type = decltype(space)::candidate_type;

    // Start from all-Safe
    candidate_type start{};
    start[0] = Strat::Safe;
    start[1] = Strat::Safe;
    start[2] = Strat::Safe;

    // Constraint: position 0 must not be Fast.
    auto no_fast_0 = [](auto const& c) constexpr -> bool {
        return c[0] != Strat::Fast;
    };

    auto plan = local_search(space, whole_cost, start, no_fast_0);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], Strat::Fast);

    // Should descend to Medium at 0, Fast at 1,2 → cost 5.0
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 5.0);
}

// =============================================================================
// Test: local_search — default initial (all-default-constructed)
// =============================================================================

TEST(LocalSearch, DefaultInitial) {
    constexpr auto space = make_3x3_space();

    // Default candidate has default-initialized enum values.
    // For this space, the default enum value (0) maps to Strat::Fast.
    auto plan = local_search(space, whole_cost);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 3.0);
}

// =============================================================================
// Test: local_search — infeasible initial
// =============================================================================

TEST(LocalSearch, InfeasibleInitial) {
    constexpr auto space = make_3x3_space();
    using candidate_type = decltype(space)::candidate_type;

    candidate_type start{};
    start[0] = Strat::Safe;

    // Impossible constraint.
    auto impossible = [](auto const&) constexpr -> bool { return false; };

    auto plan = local_search(space, whole_cost, start, impossible);
    EXPECT_TRUE(plan.is_infeasible());
}

// =============================================================================
// Test: local_search — on permutation space
// =============================================================================

TEST(LocalSearch, PermutationSpace) {
    permutation_space<4> space;
    using candidate_type = decltype(space)::candidate_type;

    // Cost: sum of |perm[i] - target[i]| where target = {3,2,1,0}
    auto cost_fn = [](candidate_type const& c) constexpr -> double {
        constexpr std::array<std::size_t, 4> target = {3, 2, 1, 0};
        double total = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            auto diff = (c[i] > target[i]) ? c[i] - target[i] : target[i] - c[i];
            total += static_cast<double>(diff);
        }
        return total;
    };

    // Start from identity {0,1,2,3}; optimal is {3,2,1,0} (cost=0).
    auto plan = local_search(space, cost_fn, space.identity());
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 0.0);
}

// =============================================================================
// Test: local_search — stats populated
// =============================================================================

TEST(LocalSearch, StatsPopulated) {
    constexpr auto space = make_3x3_space();
    using candidate_type = decltype(space)::candidate_type;

    candidate_type start{};
    start[0] = Strat::Safe;
    start[1] = Strat::Safe;
    start[2] = Strat::Safe;

    auto plan = local_search(space, whole_cost, start);
    EXPECT_GT(plan.stats.local_search_moves, 0u);
    EXPECT_GT(plan.stats.candidates_evaluated, 0u);
    EXPECT_GT(plan.stats.candidates_total, 0u);
}

// =============================================================================
// Test: local_search — constexpr validation
// =============================================================================

constexpr auto local_constexpr_test() {
    auto space = make_anonymous_space<Strat, 3>(
        std::array{ Strat::Fast, Strat::Medium, Strat::Safe });
    using candidate_type = decltype(space)::candidate_type;
    candidate_type start{};
    start[0] = Strat::Safe;
    start[1] = Strat::Safe;
    start[2] = Strat::Safe;
    return local_search(space, whole_cost, start);
}
static_assert(local_constexpr_test().predicted_cost == 3.0);
static_assert(local_constexpr_test().is_feasible());

// =============================================================================
// Test: select_and_run — factored + constraint dispatches to beam_search
// =============================================================================

TEST(SelectAndRun, FactoredWithConstraintUsesBeamSearch) {
    constexpr auto space = make_3x3_space();

    auto no_fast_0 = [](auto const& c) constexpr -> bool {
        return c[0] != Strat::Fast;
    };

    // select_and_run should dispatch to beam_search (overload 1c).
    auto plan = select_and_run(space, whole_cost, no_fast_0);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], Strat::Fast);

    // Medium at 0 (3), Fast at 1,2 (1+1) = 5.0
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 5.0);

    // Verify beam_search stats are populated.
    EXPECT_EQ(plan.stats.beam_width_used, 32u);
}

// =============================================================================
// Test: beam_search agrees with exhaustive on small space
// =============================================================================

TEST(BeamSearch, AgreesWithExhaustive) {
    constexpr auto space = make_3x3_space();

    auto no_fast_0 = [](auto const& c) constexpr -> bool {
        return c[0] != Strat::Fast;
    };

    auto beam_plan = beam_search(space, whole_cost, no_fast_0);
    auto exhaustive_plan = exhaustive_search(space, whole_cost, no_fast_0);

    EXPECT_DOUBLE_EQ(beam_plan.predicted_cost, exhaustive_plan.predicted_cost);
}

// =============================================================================
// Test: beam_search — mutual exclusion constraint (cross-position)
// =============================================================================

TEST(BeamSearch, MutualExclusionConstraint) {
    constexpr auto space = make_3x3_space();

    // Cannot use the same strategy at positions 0 and 1.
    constexpr auto mutual_excl = [](auto const& c) constexpr -> bool {
        return c[0] != c[1];
    };

    constexpr auto plan = beam_search(space, whole_cost, mutual_excl);
    EXPECT_TRUE(plan.is_feasible());
    EXPECT_NE(plan.params[0], plan.params[1]);

    // Best feasible: Fast at 0, Medium at 1, Fast at 2 → 1+3+1=5
    // (or Medium at 0, Fast at 1, Fast at 2 → 3+1+1=5)
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 5.0);
}

// =============================================================================
// Test: local_search — heterogeneous space
// =============================================================================

TEST(LocalSearch, HeterogeneousSpace) {
    enum class HS { A, B, C };
    constexpr auto space = make_heterogeneous_space(
        position_choices{ 0ul, std::array{ HS::A, HS::B, HS::C } },
        position_choices{ 1ul, std::array{ HS::A, HS::B } }
    );
    using candidate_type = decltype(space)::candidate_type;

    // Cost: C < B < A (i.e., higher enum value = lower cost)
    auto cost_fn = [](auto const& c) constexpr -> double {
        auto score = [](HS h) constexpr -> double {
            switch (h) {
                case HS::A: return 10.0;
                case HS::B: return 5.0;
                case HS::C: return 1.0;
            }
            return 10.0;
        };
        return score(c[0]) + score(c[1]);
    };

    // Start from all-A (worst)
    candidate_type start{};
    start[0] = HS::A;
    start[1] = HS::A;

    auto plan = local_search(space, cost_fn, start);
    EXPECT_TRUE(plan.is_feasible());

    // Best: C at pos 0 (1), B at pos 1 (5) = 6.0
    // (pos 1 doesn't have C, only A and B)
    EXPECT_DOUBLE_EQ(plan.predicted_cost, 6.0);
    EXPECT_EQ(plan.params[0], HS::C);
    EXPECT_EQ(plan.params[1], HS::B);
}
