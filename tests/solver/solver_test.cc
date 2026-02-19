// tests/solver/solver_test.cpp
//
// Google Tests for ctdp solver library.
//
// Tests plan, plan_compose, plan_set, plan_traversal, solve_stats,
// and candidate_traits.
//
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#include "plan.h"
#include "plan_compose.h"
#include "plan_set.h"
#include "plan_traversal.h"
#include "solve_stats.h"
#include "candidate_traits.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

using namespace ctdp;

// ============================================================================
// Test candidate types
// ============================================================================

// Simple per-element candidate for testing
template<typename Strategy, size_t N>
struct test_candidate {
    std::array<Strategy, N> assignment{};

    constexpr bool operator==(test_candidate const&) const = default;
};

// candidate_traits specialisation
template<typename Strategy, size_t N>
struct candidate_traits<test_candidate<Strategy, N>> {
    template<typename F>
    static constexpr void for_each_assignment(
        test_candidate<Strategy, N> const& candidate,
        auto const& descriptors,
        F fn
    ) {
        for (size_t i = 0; i < N; ++i) {
            fn(descriptors[i], candidate.assignment[i]);
        }
    }
};

enum class Strategy { A, B, C };

using TestCand3 = test_candidate<Strategy, 3>;
using TestCand5 = test_candidate<Strategy, 5>;

// Custom dominance for tests that want to keep all plans
struct never_dominates {
    template<typename C>
    constexpr bool operator()(plan<C> const&, plan<C> const&) const {
        return false;
    }
};

// ============================================================================
// solve_stats
// ============================================================================

TEST(SolveStats, DefaultConstruction) {
    constexpr solve_stats s;
    static_assert(s.subproblems_total == 0);
    static_assert(s.subproblems_evaluated == 0);
    static_assert(s.candidates_total == 0);
    static_assert(s.memo_hits == 0);
    static_assert(s.memo_misses == 0);
    static_assert(s.max_recursion_depth == 0);

    EXPECT_EQ(s.subproblems_total, 0u);
}

TEST(SolveStats, Addition) {
    constexpr auto combined = []() constexpr {
        solve_stats a;
        a.subproblems_total = 10;
        a.subproblems_evaluated = 8;
        a.memo_hits = 5;
        a.memo_misses = 3;
        a.memo_table_size = 100;
        a.max_recursion_depth = 10;

        solve_stats b;
        b.subproblems_total = 20;
        b.subproblems_evaluated = 15;
        b.memo_hits = 10;
        b.memo_misses = 5;
        b.memo_table_size = 200;
        b.max_recursion_depth = 5;

        return a + b;
    }();

    // Counters are summed
    static_assert(combined.subproblems_total == 30);
    static_assert(combined.subproblems_evaluated == 23);
    static_assert(combined.memo_hits == 15);
    static_assert(combined.memo_misses == 8);

    // Max fields take the maximum
    static_assert(combined.memo_table_size == 200);
    static_assert(combined.max_recursion_depth == 10);

    EXPECT_EQ(combined.subproblems_total, 30u);
}

TEST(SolveStats, PlusEquals) {
    solve_stats a;
    a.subproblems_total = 10;
    a.candidates_evaluated = 50;

    solve_stats b;
    b.subproblems_total = 20;
    b.candidates_evaluated = 100;

    a += b;

    EXPECT_EQ(a.subproblems_total, 30u);
    EXPECT_EQ(a.candidates_evaluated, 150u);
}

TEST(SolveStats, CacheHitRate) {
    constexpr auto rate = []() constexpr {
        solve_stats s;
        s.memo_hits = 80;
        s.memo_misses = 20;
        return s.cache_hit_rate();
    }();

    static_assert(rate == 0.8);
    EXPECT_DOUBLE_EQ(rate, 0.8);
}

TEST(SolveStats, CacheHitRateZero) {
    constexpr solve_stats s;
    static_assert(s.cache_hit_rate() == 0.0);
}

TEST(SolveStats, PruningRate) {
    constexpr auto rate = []() constexpr {
        solve_stats s;
        s.candidates_total = 100;
        s.candidates_pruned = 25;
        return s.pruning_rate();
    }();

    static_assert(rate == 0.25);
}

TEST(SolveStats, SubproblemCoverage) {
    constexpr auto cov = []() constexpr {
        solve_stats s;
        s.subproblems_total = 200;
        s.subproblems_evaluated = 100;
        return s.subproblem_coverage();
    }();

    static_assert(cov == 0.5);
}

TEST(SolveStats, AvgCandidatesPerSubproblem) {
    constexpr auto avg = []() constexpr {
        solve_stats s;
        s.subproblems_evaluated = 10;
        s.candidates_evaluated = 50;
        return s.avg_candidates_per_subproblem();
    }();

    static_assert(avg == 5.0);
}

TEST(SolveStats, Equality) {
    constexpr solve_stats a;
    constexpr solve_stats b;
    static_assert(a == b);

    EXPECT_EQ(a, b);
}

// ============================================================================
// plan — Construction and Basic Queries
// ============================================================================

TEST(Plan, DefaultIsInfeasible) {
    constexpr plan<int> p;
    static_assert(p.is_infeasible());
    static_assert(!p.is_feasible());
    static_assert(p.predicted_cost == std::numeric_limits<double>::infinity());

    EXPECT_TRUE(p.is_infeasible());
}

TEST(Plan, ConstructWithCandidateAndCost) {
    constexpr plan<int> p{42, 3.14};
    static_assert(p.is_feasible());
    static_assert(p.params == 42);
    static_assert(p.predicted_cost == 3.14);

    EXPECT_TRUE(p.is_feasible());
    EXPECT_EQ(p.params, 42);
    EXPECT_DOUBLE_EQ(p.predicted_cost, 3.14);
}

TEST(Plan, ConstructWithStats) {
    constexpr auto p = []() constexpr {
        solve_stats s;
        s.subproblems_total = 100;
        return plan<int>{42, 1.5, s};
    }();

    static_assert(p.is_feasible());
    static_assert(p.stats.subproblems_total == 100);
}

TEST(Plan, ZeroCostIsFeasible) {
    constexpr plan<int> p{0, 0.0};
    static_assert(p.is_feasible());
}

// ============================================================================
// plan — Comparison
// ============================================================================

TEST(Plan, OrderByCost) {
    constexpr plan<int> a{1, 2.0};
    constexpr plan<int> b{2, 5.0};
    constexpr plan<int> c{3, 2.0};

    static_assert(a < b);
    static_assert(b > a);
    static_assert(a <= c);
    static_assert(a >= c);

    EXPECT_LT(a, b);
    EXPECT_GT(b, a);
}

TEST(Plan, Equality) {
    constexpr plan<int> a{42, 3.0};
    constexpr plan<int> b{42, 3.0};
    constexpr plan<int> c{99, 3.0};

    static_assert(a == b);
    static_assert(a != c);
    static_assert(a.cost_equal(c));  // same cost, different candidate

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_TRUE(a.cost_equal(c));
}

TEST(Plan, InfeasibleComparisonOrder) {
    constexpr plan<int> feasible{1, 100.0};
    constexpr plan<int> infeasible;

    // Infeasible (infinity) is always worse
    static_assert(feasible < infeasible);
    static_assert(infeasible > feasible);
}

// ============================================================================
// plan — Helper Functions
// ============================================================================

TEST(Plan, MakeInfeasible) {
    constexpr auto p = make_infeasible_plan<int>();
    static_assert(p.is_infeasible());
}

TEST(Plan, IsBetter) {
    constexpr plan<int> a{1, 2.0};
    constexpr plan<int> b{2, 5.0};

    static_assert(is_better(a, b));
    static_assert(!is_better(b, a));
    static_assert(!is_better(a, a));
}

TEST(Plan, IsAtLeastAsGood) {
    constexpr plan<int> a{1, 2.0};
    constexpr plan<int> b{2, 5.0};
    constexpr plan<int> c{3, 2.0};

    static_assert(is_at_least_as_good(a, b));
    static_assert(is_at_least_as_good(a, c));
    static_assert(!is_at_least_as_good(b, a));
}

TEST(Plan, MinMaxPlan) {
    constexpr plan<int> a{1, 2.0};
    constexpr plan<int> b{2, 5.0};

    static_assert(min_plan(a, b).predicted_cost == 2.0);
    static_assert(max_plan(a, b).predicted_cost == 5.0);
}

// ============================================================================
// plan_compose
// ============================================================================

TEST(PlanCompose, Additive) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<double> b{2.0, 4.0};

    constexpr auto combined = compose_additive(a, b);

    static_assert(combined.is_feasible());
    static_assert(combined.combined_cost == 7.0);

    EXPECT_TRUE(combined.is_feasible());
    EXPECT_DOUBLE_EQ(combined.combined_cost, 7.0);
}

TEST(PlanCompose, AdditiveThreePlans) {
    constexpr plan<int> a{1, 2.0};
    constexpr plan<int> b{2, 3.0};
    constexpr plan<int> c{3, 5.0};

    constexpr auto combined = compose_additive(a, b, c);

    static_assert(combined.combined_cost == 10.0);
}

TEST(PlanCompose, AdditiveInfeasiblePropagates) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<double> b;  // infeasible

    constexpr auto combined = compose_additive(a, b);

    static_assert(!combined.is_feasible());
    EXPECT_FALSE(combined.is_feasible());
}

TEST(PlanCompose, Max) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<double> b{2.0, 7.0};

    constexpr auto combined = compose_max(a, b);

    static_assert(combined.combined_cost == 7.0);
    EXPECT_DOUBLE_EQ(combined.combined_cost, 7.0);
}

TEST(PlanCompose, MaxInfeasible) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<int> b;  // infeasible

    constexpr auto combined = compose_max(a, b);
    static_assert(!combined.is_feasible());
}

TEST(PlanCompose, Custom) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<int> b{2, 7.0};

    constexpr auto combined = compose_custom(
        [](double x, double y) { return 0.6 * x + 0.4 * y; },
        a, b
    );

    // 0.6 * 3.0 + 0.4 * 7.0 = 1.8 + 2.8 = 4.6
    static_assert(combined.combined_cost == 4.6);
    EXPECT_DOUBLE_EQ(combined.combined_cost, 4.6);
}

TEST(PlanCompose, Convenience) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<double> b{2.0, 4.0};

    constexpr auto combined = compose(a, b);
    static_assert(combined.combined_cost == 7.0);
}

// ============================================================================
// composite_plan
// ============================================================================

TEST(CompositePlan, SubPlanAccess) {
    constexpr plan<int> a{1, 3.0};
    constexpr plan<double> b{2.0, 4.0};

    constexpr auto combined = compose_additive(a, b);

    static_assert(combined.get<0>().predicted_cost == 3.0);
    static_assert(combined.get<1>().predicted_cost == 4.0);
    static_assert(combined.get<0>().params == 1);
    static_assert(combined.get<1>().params == 2.0);
}

TEST(CompositePlan, DefaultIsInfeasible) {
    constexpr composite_plan<int, double> cp;
    static_assert(!cp.is_feasible());
}

// ============================================================================
// plan_set
// ============================================================================

TEST(PlanSet, DefaultEmpty) {
    constexpr plan_set<int, 10> ps;
    static_assert(ps.empty());
    static_assert(ps.size() == 0);
    static_assert(ps.capacity() == 10);
    static_assert(!ps.has_feasible());

    EXPECT_TRUE(ps.empty());
}

TEST(PlanSet, InsertAndBest) {
    // With cost_dominance, lower cost dominates higher, so only best survives
    constexpr auto ps = []() constexpr {
        plan_set<int, 10> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});  // dominates 5.0
        ps.insert(plan<int>{3, 8.0});  // dominated by 2.0
        return ps;
    }();

    static_assert(ps.size() == 1);
    static_assert(ps.best().predicted_cost == 2.0);
    static_assert(ps.best().params == 2);

    EXPECT_EQ(ps.size(), 1u);
    EXPECT_DOUBLE_EQ(ps.best().predicted_cost, 2.0);
}

TEST(PlanSet, InsertAndBestNoDominance) {
    // With never_dominates, all feasible plans are kept sorted by cost
    constexpr auto ps = []() constexpr {
        plan_set<int, 10, never_dominates> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});
        ps.insert(plan<int>{3, 8.0});
        return ps;
    }();

    static_assert(ps.size() == 3);
    static_assert(ps.best().predicted_cost == 2.0);
    static_assert(ps.worst().predicted_cost == 8.0);

    EXPECT_EQ(ps.size(), 3u);
}

TEST(PlanSet, SortedOrder) {
    constexpr auto ps = []() constexpr {
        plan_set<int, 10, never_dominates> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});
        ps.insert(plan<int>{3, 8.0});
        ps.insert(plan<int>{4, 1.0});
        return ps;
    }();

    static_assert(ps[0].predicted_cost == 1.0);
    static_assert(ps[1].predicted_cost == 2.0);
    static_assert(ps[2].predicted_cost == 5.0);
    static_assert(ps[3].predicted_cost == 8.0);
}

TEST(PlanSet, RejectsInfeasible) {
    constexpr auto ps = []() constexpr {
        plan_set<int, 10> ps;
        auto ok = ps.insert(plan<int>{});  // infeasible
        return std::pair{ps.size(), ok};
    }();

    static_assert(ps.first == 0);  // not inserted
    static_assert(ps.second == false);
}

TEST(PlanSet, CapacityEviction) {
    constexpr auto ps = []() constexpr {
        plan_set<int, 3, never_dominates> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});
        ps.insert(plan<int>{3, 8.0});
        ps.insert(plan<int>{4, 1.0});  // should evict worst (8.0)
        return ps;
    }();

    static_assert(ps.size() == 3);
    static_assert(ps.best().predicted_cost == 1.0);
    static_assert(ps.worst().predicted_cost == 5.0);
}

TEST(PlanSet, DominanceRejection) {
    constexpr auto result = []() constexpr {
        plan_set<int, 10> ps;
        ps.insert(plan<int>{1, 2.0});
        auto ok = ps.insert(plan<int>{2, 5.0});  // dominated by 2.0
        return std::pair{ps.size(), ok};
    }();

    static_assert(result.first == 1);
    static_assert(result.second == false);
}

TEST(PlanSet, DominanceEviction) {
    constexpr auto ps = []() constexpr {
        plan_set<int, 10> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 8.0});
        ps.insert(plan<int>{3, 1.0});  // dominates both
        return ps;
    }();

    static_assert(ps.size() == 1);
    static_assert(ps.best().predicted_cost == 1.0);
}

TEST(PlanSet, Clear) {
    constexpr auto ps = []() constexpr {
        plan_set<int, 10> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});
        ps.clear();
        return ps;
    }();

    static_assert(ps.empty());
}

TEST(PlanSet, Merge) {
    constexpr auto result = []() constexpr {
        plan_set<int, 10> a;
        a.insert(plan<int>{1, 5.0});
        a.insert(plan<int>{2, 3.0});

        plan_set<int, 10> b;
        b.insert(plan<int>{3, 1.0});
        b.insert(plan<int>{4, 4.0});

        auto inserted = a.merge(b);
        return std::pair{a.size(), inserted};
    }();

    // 1.0 dominates 3.0, 4.0, 5.0 → only 1.0 survives
    static_assert(result.first == 1);
    static_assert(result.second == 1);
}

TEST(PlanSet, Iteration) {
    constexpr auto total_cost = []() constexpr {
        plan_set<int, 10> ps;
        ps.insert(plan<int>{1, 3.0});
        ps.insert(plan<int>{2, 5.0});
        ps.insert(plan<int>{3, 7.0});

        // These will be dominated, so only 3.0 survives
        // Wait - cost_dominance is strict <, so 5.0 is dominated by 3.0
        // Actually that IS cost_dominance - all are dominated by 3.0
        double total = 0.0;
        for (auto it = ps.begin(); it != ps.end(); ++it) {
            total += it->predicted_cost;
        }
        return total;
    }();

    // With default cost_dominance: 3.0 dominates 5.0 and 7.0
    static_assert(total_cost == 3.0);
}

// ============================================================================
// candidate_traits
// ============================================================================

TEST(CandidateTraits, HasCandidateTraitsConcept) {
    static_assert(has_candidate_traits<TestCand3>);
    static_assert(has_candidate_traits<TestCand5>);
}

TEST(CandidateTraits, ForEachAssignment) {
    constexpr auto count = []() constexpr {
        TestCand3 cand;
        cand.assignment = {Strategy::A, Strategy::B, Strategy::C};
        std::array<int, 3> descs = {10, 20, 30};

        int count = 0;
        candidate_traits<TestCand3>::for_each_assignment(
            cand, descs.data(),
            [&count](auto const&, auto const&) { ++count; }
        );
        return count;
    }();

    static_assert(count == 3);
}

TEST(CandidateTraits, CountAssignments) {
    constexpr auto n = []() constexpr {
        TestCand3 cand;
        cand.assignment = {Strategy::A, Strategy::B, Strategy::C};
        std::array<int, 3> descs = {10, 20, 30};
        return count_assignments(cand, descs.data());
    }();

    static_assert(n == 3);
}

TEST(CandidateTraits, IsEmptyCandidate) {
    // Not truly "empty" since our test candidate always has N assignments
    constexpr auto empty = []() constexpr {
        TestCand3 cand;
        std::array<int, 3> descs = {10, 20, 30};
        return is_empty_candidate(cand, descs.data());
    }();

    static_assert(!empty);
}

// ============================================================================
// plan_traversal
// ============================================================================

TEST(PlanTraversal, ForEachAssignment) {
    constexpr auto sum_descs = []() constexpr {
        TestCand3 cand;
        cand.assignment = {Strategy::A, Strategy::B, Strategy::C};
        plan<TestCand3> p{cand, 1.0};

        std::array<int, 3> descs = {10, 20, 30};
        int sum = 0;
        for_each_assignment(p, descs.data(),
            [&sum](int const& d, Strategy const&) { sum += d; }
        );
        return sum;
    }();

    static_assert(sum_descs == 60);
    EXPECT_EQ(sum_descs, 60);
}

TEST(PlanTraversal, AssignmentCount) {
    constexpr auto count = []() constexpr {
        TestCand5 cand;
        plan<TestCand5> p{cand, 1.0};
        std::array<int, 5> descs = {1, 2, 3, 4, 5};
        return assignment_count(p, descs.data());
    }();

    static_assert(count == 5);
}

TEST(PlanTraversal, ForEachSubPlan) {
    constexpr auto count = []() constexpr {
        plan<int> a{1, 3.0};
        plan<double> b{2.0, 4.0};
        auto combined = compose_additive(a, b);

        int count = 0;
        for_each_sub_plan(combined,
            [&count](auto const&) { ++count; }
        );
        return count;
    }();

    static_assert(count == 2);
}

TEST(PlanTraversal, SubPlanCount) {
    constexpr auto cp = []() constexpr {
        plan<int> a{1, 1.0};
        plan<double> b{2.0, 2.0};
        plan<char> c{'x', 3.0};
        return compose_additive(a, b, c);
    }();

    static_assert(sub_plan_count(cp) == 3);
}

// ============================================================================
// plan_set with custom dominance (never_dominates defined above)
// ============================================================================

TEST(PlanSet, CustomDominanceKeepsAll) {
    constexpr auto ps = []() constexpr {
        plan_set<int, 10, never_dominates> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});
        ps.insert(plan<int>{3, 8.0});
        return ps;
    }();

    static_assert(ps.size() == 3);
    // Still sorted by cost
    static_assert(ps[0].predicted_cost == 2.0);
    static_assert(ps[1].predicted_cost == 5.0);
    static_assert(ps[2].predicted_cost == 8.0);
}

TEST(PlanSet, EqualityWithCandidates) {
    constexpr auto ps1 = []() constexpr {
        plan_set<int, 10, never_dominates> ps;
        ps.insert(plan<int>{1, 5.0});
        ps.insert(plan<int>{2, 2.0});
        return ps;
    }();

    constexpr auto ps2 = []() constexpr {
        plan_set<int, 10, never_dominates> ps;
        ps.insert(plan<int>{2, 2.0});
        ps.insert(plan<int>{1, 5.0});
        return ps;
    }();

    static_assert(ps1 == ps2);  // Same content, sorted
}
