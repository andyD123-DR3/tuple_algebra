// tests/test_pareto.cpp
// Tests for ctdp/core/cost_vector.h and ctdp/core/pareto.h
//
// 15 tests:
//   cost_vector:
//     1. dominance — strict improvement in at least one objective
//     2. non-dominance — worse in at least one objective
//     3. equal vectors — not dominated (requires strict)
//     4. incomparable — each better in different objectives
//     5. single-objective — degenerates to scalar comparison
//     6. element access — operator[] read/write
//   pareto_frontier:
//     7. trivial — single point is its own frontier
//     8. dominated removal — 3 points, 1 dominated
//     9. all incomparable — everything on frontier
//    10. chain dominance — a > b > c, only a survives
//    11. empty input — empty frontier
//   lex_select:
//    12. single priority — picks best on that objective
//    13. tiebreak — first priority tied, second breaks it
//    14. single point — returns it
//   pareto_search:
//    15. end-to-end with descriptor_space

#include "ctdp/core/cost_vector.h"
#include "ctdp/core/pareto.h"
#include "ctdp/space/descriptor.h"
#include "ctdp/space/space.h"

#include <gtest/gtest.h>

namespace {

using namespace ctdp;
using namespace ctdp::space;

// =============================================================================
// cost_vector tests
// =============================================================================

TEST(CostVector, DominanceStrictImprovement) {
    constexpr cost_vector<3> a{{1.0, 2.0, 3.0}};
    constexpr cost_vector<3> b{{1.0, 3.0, 3.0}};
    static_assert(a.dominates(b));
    EXPECT_TRUE(a.dominates(b));
}

TEST(CostVector, NonDominanceWorseInOne) {
    constexpr cost_vector<3> a{{2.0, 1.0, 1.0}};
    constexpr cost_vector<3> b{{1.0, 3.0, 3.0}};
    static_assert(!a.dominates(b));
    static_assert(!b.dominates(a));
    EXPECT_FALSE(a.dominates(b));
    EXPECT_FALSE(b.dominates(a));
}

TEST(CostVector, EqualNotDominated) {
    constexpr cost_vector<3> a{{1.0, 2.0, 3.0}};
    constexpr cost_vector<3> b{{1.0, 2.0, 3.0}};
    static_assert(!a.dominates(b));
    static_assert(!b.dominates(a));
    static_assert(a == b);
    EXPECT_FALSE(a.dominates(b));
}

TEST(CostVector, Incomparable) {
    constexpr cost_vector<2> a{{1.0, 3.0}};
    constexpr cost_vector<2> b{{2.0, 1.0}};
    static_assert(a.incomparable(b));
    static_assert(b.incomparable(a));
    EXPECT_TRUE(a.incomparable(b));
}

TEST(CostVector, SingleObjective) {
    constexpr cost_vector<1> a{{1.0}};
    constexpr cost_vector<1> b{{2.0}};
    static_assert(a.dominates(b));
    static_assert(!b.dominates(a));
    EXPECT_TRUE(a.dominates(b));
}

TEST(CostVector, ElementAccess) {
    cost_vector<3> v{{10.0, 20.0, 30.0}};
    EXPECT_DOUBLE_EQ(v[0], 10.0);
    EXPECT_DOUBLE_EQ(v[1], 20.0);
    EXPECT_DOUBLE_EQ(v[2], 30.0);
    v[1] = 99.0;
    EXPECT_DOUBLE_EQ(v[1], 99.0);
    static_assert(cost_vector<3>::dimensions() == 3);
}

// =============================================================================
// pareto_frontier tests
// =============================================================================

TEST(ParetoFrontier, SinglePoint) {
    constexpr_vector<evaluated_point<int, 2>, 10> pts;
    const_cast<constexpr_vector<evaluated_point<int, 2>, 10>&>(pts)
        .push_back({42, {{1.0, 2.0}}});

    auto frontier = pareto_frontier(pts);
    EXPECT_EQ(frontier.size(), 1u);
    EXPECT_EQ(frontier[0].candidate, 42);
}

TEST(ParetoFrontier, DominatedRemoval) {
    constexpr_vector<evaluated_point<int, 2>, 10> pts;
    auto& mpts = const_cast<constexpr_vector<evaluated_point<int, 2>, 10>&>(pts);
    mpts.push_back({1, {{1.0, 3.0}}});  // on frontier (best in obj 0)
    mpts.push_back({2, {{3.0, 1.0}}});  // on frontier (best in obj 1)
    mpts.push_back({3, {{2.0, 4.0}}});  // dominated by point 1

    auto frontier = pareto_frontier(pts);
    EXPECT_EQ(frontier.size(), 2u);
    // Points 1 and 2 should survive
    bool found1 = false, found2 = false;
    for (std::size_t i = 0; i < frontier.size(); ++i) {
        if (frontier[i].candidate == 1) found1 = true;
        if (frontier[i].candidate == 2) found2 = true;
    }
    EXPECT_TRUE(found1);
    EXPECT_TRUE(found2);
}

TEST(ParetoFrontier, AllIncomparable) {
    constexpr_vector<evaluated_point<int, 2>, 10> pts;
    auto& mpts = const_cast<constexpr_vector<evaluated_point<int, 2>, 10>&>(pts);
    mpts.push_back({1, {{1.0, 4.0}}});
    mpts.push_back({2, {{2.0, 3.0}}});
    mpts.push_back({3, {{3.0, 2.0}}});
    mpts.push_back({4, {{4.0, 1.0}}});

    auto frontier = pareto_frontier(pts);
    EXPECT_EQ(frontier.size(), 4u);
}

TEST(ParetoFrontier, ChainDominance) {
    constexpr_vector<evaluated_point<int, 2>, 10> pts;
    auto& mpts = const_cast<constexpr_vector<evaluated_point<int, 2>, 10>&>(pts);
    mpts.push_back({1, {{1.0, 1.0}}});  // dominates all
    mpts.push_back({2, {{2.0, 2.0}}});  // dominated by 1
    mpts.push_back({3, {{3.0, 3.0}}});  // dominated by 1 and 2

    auto frontier = pareto_frontier(pts);
    EXPECT_EQ(frontier.size(), 1u);
    EXPECT_EQ(frontier[0].candidate, 1);
}

TEST(ParetoFrontier, EmptyInput) {
    constexpr_vector<evaluated_point<int, 2>, 10> pts;
    auto frontier = pareto_frontier(pts);
    EXPECT_EQ(frontier.size(), 0u);
}

// =============================================================================
// lex_select tests
// =============================================================================

TEST(LexSelect, SinglePriority) {
    constexpr_vector<evaluated_point<int, 3>, 10> frontier;
    auto& mf = const_cast<constexpr_vector<evaluated_point<int, 3>, 10>&>(frontier);
    mf.push_back({1, {{5.0, 1.0, 3.0}}});  // best on obj 1
    mf.push_back({2, {{1.0, 5.0, 3.0}}});  // best on obj 0
    mf.push_back({3, {{3.0, 3.0, 1.0}}});  // best on obj 2

    auto winner = lex_select<1>(frontier);  // priority: obj 1
    EXPECT_EQ(winner.candidate, 1);         // point 1 has obj[1]=1.0

    auto winner2 = lex_select<0>(frontier); // priority: obj 0
    EXPECT_EQ(winner2.candidate, 2);        // point 2 has obj[0]=1.0
}

TEST(LexSelect, Tiebreak) {
    constexpr_vector<evaluated_point<int, 3>, 10> frontier;
    auto& mf = const_cast<constexpr_vector<evaluated_point<int, 3>, 10>&>(frontier);
    mf.push_back({1, {{1.0, 5.0, 3.0}}});  // tied on obj 0, worse on obj 1
    mf.push_back({2, {{1.0, 2.0, 3.0}}});  // tied on obj 0&1 with 3, worse on obj 2
    mf.push_back({3, {{1.0, 2.0, 1.0}}});  // tied on obj 0&1 with 2, best on obj 2

    // lex_select<0, 1, 2>:
    //   start best=0 (cand 1)
    //   i=1 (cand 2): obj[0] tie, obj[1] 2.0 < 5.0 → best=1
    //   i=2 (cand 3): obj[0] tie, obj[1] tie, obj[2] 1.0 < 3.0 → best=2
    // Winner: candidate 3 (best on third priority after two ties)
    auto winner = lex_select<0, 1, 2>(frontier);
    EXPECT_EQ(winner.candidate, 3);

    // With different priority: obj 1 first, then obj 2
    // Both cand 2 and 3 have obj[1]=2.0 (best), tie on obj[2]: 3.0 vs 1.0
    // Candidate 3 wins again
    auto winner2 = lex_select<1, 2>(frontier);
    EXPECT_EQ(winner2.candidate, 3);
}

TEST(LexSelect, SinglePoint) {
    constexpr_vector<evaluated_point<int, 2>, 10> frontier;
    auto& mf = const_cast<constexpr_vector<evaluated_point<int, 2>, 10>&>(frontier);
    mf.push_back({42, {{1.0, 2.0}}});

    auto winner = lex_select<0>(frontier);
    EXPECT_EQ(winner.candidate, 42);
}

// =============================================================================
// pareto_search — end-to-end with descriptor_space
// =============================================================================

TEST(ParetoSearch, EndToEndDescriptorSpace) {
    // 2D space: power_2 tile size × enum strategy
    enum class Strat : int { fast, balanced, safe };

    auto space = descriptor_space("demo",
        power_2("tile", 8, 64),      // {8, 16, 32, 64} = 4 values
        make_enum_vals("strat", {Strat::fast, Strat::balanced, Strat::safe}));
    // 4 × 3 = 12 points

    // Multi-objective cost: (latency, error)
    auto cost = [](auto const& pt) -> cost_vector<2> {
        auto tile = std::get<0>(pt);
        auto strat = std::get<1>(pt);
        double latency = 100.0 / tile;  // larger tile = lower latency
        double error = (strat == Strat::fast) ? 10.0
                     : (strat == Strat::balanced) ? 5.0
                     : 1.0;  // safe = lowest error
        return {{latency, error}};
    };

    auto frontier = pareto_search<2, 20>(space, cost);

    // The frontier should contain the Pareto-optimal points.
    // fast+large_tile has low latency but high error.
    // safe+small_tile has high latency but low error.
    // These are incomparable → both on frontier.
    EXPECT_GE(frontier.size(), 2u);
    EXPECT_LE(frontier.size(), 12u);

    // Lexicographic select: latency first
    auto low_latency = lex_select<0>(frontier);
    EXPECT_EQ(std::get<0>(low_latency.candidate), 64);  // largest tile

    // Lexicographic select: error first
    auto low_error = lex_select<1>(frontier);
    EXPECT_EQ(std::get<1>(low_error.candidate), Strat::safe);
}

// =============================================================================
// weighted_select test
// =============================================================================

TEST(WeightedSelect, BasicWeighting) {
    constexpr_vector<evaluated_point<int, 2>, 10> frontier;
    auto& mf = const_cast<constexpr_vector<evaluated_point<int, 2>, 10>&>(frontier);
    mf.push_back({1, {{1.0, 10.0}}});  // scalarised: 0.5*1 + 0.5*10 = 5.5
    mf.push_back({2, {{5.0, 2.0}}});   // scalarised: 0.5*5 + 0.5*2 = 3.5
    mf.push_back({3, {{3.0, 5.0}}});   // scalarised: 0.5*3 + 0.5*5 = 4.0

    auto winner = weighted_select(frontier, cost_vector<2>{{0.5, 0.5}});
    EXPECT_EQ(winner.candidate, 2);  // lowest scalarised cost

    // Change weights to favour objective 0 heavily
    auto winner2 = weighted_select(frontier, cost_vector<2>{{0.9, 0.1}});
    EXPECT_EQ(winner2.candidate, 1);  // obj[0]=1.0 is best
}

} // namespace
