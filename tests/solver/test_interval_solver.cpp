#include "ctdp/solver/solver.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <optional>

namespace {

using ctdp::interval_dp;
using ctdp::interval_split_space;
using ctdp::make_chain_cost;
using ctdp::solver::interval_context;
using ctdp::solver::reconstruct_interval_rooted_plan;
using ctdp::solver::memo::triangular_memo;
using ctdp::solver::plans::interval_partition_plan;
using ctdp::solver::algorithms::interval_memo;
using ctdp::solver::algorithms::interval_recurrence;
using ctdp::solver::algorithms::interval_solver;

struct matrix_chain_recurrence {
    using value_type = double;

    std::array<std::size_t, 7> dims{};

    [[nodiscard]] std::optional<double> base_case(interval_context ctx) const {
        if (ctx.size() == 1) {
            return 0.0;
        }
        return std::nullopt;
    }

    [[nodiscard]] double combine(interval_partition_plan const& plan,
                                 double left,
                                 double right) const {
        return left + right + static_cast<double>(
            dims[plan.whole.start()] * dims[plan.split] * dims[plan.whole.end()]);
    }
};

struct left_size_score_recurrence {
    using value_type = int;

    [[nodiscard]] std::optional<int> base_case(interval_context ctx) const {
        if (ctx.size() == 1) {
            return 0;
        }
        return std::nullopt;
    }

    [[nodiscard]] int combine(interval_partition_plan const& plan,
                              int left,
                              int right) const {
        return left + right + static_cast<int>(plan.left_size());
    }
};

struct leftmost_split_only {
    template<class Emit>
    void for_each(interval_context ctx, Emit&& emit) const {
        emit(ctx.start() + 1);
    }
};

struct counting_recurrence {
    using value_type = int;

    std::size_t* base_calls{};
    std::size_t* combine_calls{};

    [[nodiscard]] std::optional<int> base_case(interval_context ctx) const {
        ++(*base_calls);
        if (ctx.size() == 1) {
            return 1;
        }
        return std::nullopt;
    }

    [[nodiscard]] int combine(interval_partition_plan const&,
                              int left,
                              int right) const {
        ++(*combine_calls);
        return left + right;
    }
};

static_assert(interval_recurrence<matrix_chain_recurrence>);
static_assert(interval_recurrence<left_size_score_recurrence>);
static_assert(interval_memo<triangular_memo<double>, double>);
static_assert(interval_memo<triangular_memo<int>, int>);

TEST(IntervalSolver, MatchesIntervalDPOnCormenMatrixChain) {
    constexpr std::array<std::size_t, 7> dims{30, 35, 15, 5, 10, 20, 25};

    matrix_chain_recurrence recurrence{dims};
    interval_solver<matrix_chain_recurrence> solver{recurrence};
    triangular_memo<double> memo{6};

    auto value = solver.solve(interval_context{0, 6}, memo);

    auto space = interval_split_space<7>{.n = 6};
    auto dp_result = interval_dp(space, make_chain_cost(dims));

    EXPECT_DOUBLE_EQ(value, dp_result.predicted_cost);
    EXPECT_DOUBLE_EQ(value, 15125.0);
}

TEST(IntervalSolver, SolveRootedMatchesIntervalDpRootedReconstruction) {
    constexpr std::array<std::size_t, 7> dims{30, 35, 15, 5, 10, 20, 25};

    matrix_chain_recurrence recurrence{dims};
    interval_solver<matrix_chain_recurrence> solver{recurrence};
    triangular_memo<double> memo{6};

    auto rooted = solver.solve_rooted<7>(interval_context{0, 6}, memo);

    auto dp_result = interval_dp(interval_split_space<7>{.n = 6}, make_chain_cost(dims));
    auto expected = reconstruct_interval_rooted_plan(dp_result);

    EXPECT_EQ(rooted.params, expected.params);
    EXPECT_TRUE(rooted.params.is_legal());
    EXPECT_TRUE(rooted.params.is_canonical());
    EXPECT_DOUBLE_EQ(rooted.predicted_cost, expected.predicted_cost);
    EXPECT_EQ(rooted.stats.subproblems_total, 21u);
    EXPECT_EQ(rooted.stats.subproblems_evaluated, 21u);
    EXPECT_EQ(rooted.stats.candidates_total, 35u);
    EXPECT_EQ(rooted.stats.candidates_evaluated, 35u);
    EXPECT_EQ(rooted.params.split(0, 6), 3u);
}

TEST(IntervalSolver, SolveRootedRebasesNonZeroStartIntervalsToLocalRootedCoordinates) {
    constexpr std::array<std::size_t, 5> sub_dims{35, 15, 5, 10, 20};
    constexpr std::array<std::size_t, 7> full_dims{30, 35, 15, 5, 10, 20, 25};

    matrix_chain_recurrence recurrence{full_dims};
    interval_solver<matrix_chain_recurrence> solver{recurrence};
    triangular_memo<double> memo{6};

    auto rooted = solver.solve_rooted<5>(interval_context{1, 5}, memo);
    auto expected = reconstruct_interval_rooted_plan(
        interval_dp(interval_split_space<5>{.n = 4}, make_chain_cost(sub_dims)));

    EXPECT_EQ(rooted.params, expected.params);
    EXPECT_TRUE(rooted.params.is_legal());
    EXPECT_TRUE(rooted.params.is_canonical());
    EXPECT_EQ(rooted.params.root_interval().start(), 0u);
    EXPECT_EQ(rooted.params.root_interval().end(), 4u);
    EXPECT_DOUBLE_EQ(rooted.predicted_cost, 7125.0);
    EXPECT_DOUBLE_EQ(rooted.predicted_cost, expected.predicted_cost);
    EXPECT_EQ(rooted.stats.subproblems_total, 10u);
    EXPECT_EQ(rooted.stats.subproblems_evaluated, 10u);
    EXPECT_EQ(rooted.stats.candidates_total, 10u);
    EXPECT_EQ(rooted.stats.candidates_evaluated, 10u);
    EXPECT_EQ(rooted.params.split(0, 4), 2u);
    EXPECT_EQ(rooted.params.split(0, 2), 1u);
    EXPECT_EQ(rooted.params.split(2, 4), 3u);
}

TEST(IntervalSolver, SolveWithStatsTracksCanonicalCounts) {
    constexpr std::array<std::size_t, 7> dims{30, 35, 15, 5, 10, 20, 25};

    matrix_chain_recurrence recurrence{dims};
    interval_solver<matrix_chain_recurrence> solver{recurrence};
    triangular_memo<double> memo{6};

    auto result = solver.solve_with_stats(interval_context{0, 6}, memo);

    EXPECT_DOUBLE_EQ(result.value, 15125.0);
    EXPECT_EQ(result.stats.subproblems_total, 21u);
    EXPECT_EQ(result.stats.subproblems_evaluated, 21u);
    EXPECT_EQ(result.stats.subproblems_cached, 21u);
    EXPECT_EQ(result.stats.candidates_total, 35u);
    EXPECT_EQ(result.stats.candidates_evaluated, 35u);
    EXPECT_EQ(result.stats.candidates_pruned, 0u);
    EXPECT_EQ(result.stats.memo_misses, 21u);
    EXPECT_GT(result.stats.memo_hits, 0u);
    EXPECT_EQ(result.stats.memo_table_size, 21u);
    EXPECT_EQ(result.stats.max_recursion_depth, 6u);
}

TEST(IntervalSolver, CompareControlsOptimisationDirection) {
    left_size_score_recurrence recurrence;

    interval_solver<left_size_score_recurrence> min_solver{recurrence};
    triangular_memo<int> min_memo{4};

    interval_solver<left_size_score_recurrence,
                    ctdp::solver::policies::all_binary_splits,
                    std::greater<>> max_solver{recurrence, {}, {}};
    triangular_memo<int> max_memo{4};

    auto min_value = min_solver.solve(interval_context{0, 4}, min_memo);
    auto max_value = max_solver.solve(interval_context{0, 4}, max_memo);

    EXPECT_EQ(min_value, 3);
    EXPECT_EQ(max_value, 6);
}

TEST(IntervalSolver, SolveRootedReflectsCompareDirectionAndSplitPolicy) {
    left_size_score_recurrence recurrence;

    interval_solver<left_size_score_recurrence, leftmost_split_only> leftmost_solver{recurrence};
    triangular_memo<int> leftmost_memo{4};
    auto leftmost = leftmost_solver.solve_rooted<4>(interval_context{0, 4}, leftmost_memo);

    interval_solver<left_size_score_recurrence,
                    ctdp::solver::policies::all_binary_splits,
                    std::greater<>> max_solver{recurrence, {}, {}};
    triangular_memo<int> max_memo{4};
    auto maximum = max_solver.solve_rooted<4>(interval_context{0, 4}, max_memo);

    EXPECT_DOUBLE_EQ(leftmost.predicted_cost, 3.0);
    EXPECT_TRUE(leftmost.params.is_canonical());
    EXPECT_EQ(leftmost.params.split(0, 4), 1u);
    EXPECT_EQ(leftmost.params.split(1, 4), 2u);
    EXPECT_EQ(leftmost.params.split(2, 4), 3u);

    EXPECT_DOUBLE_EQ(maximum.predicted_cost, 6.0);
    EXPECT_TRUE(maximum.params.is_canonical());
    EXPECT_EQ(maximum.params.split(0, 4), 3u);
    EXPECT_EQ(maximum.params.split(0, 3), 2u);
    EXPECT_EQ(maximum.params.split(0, 2), 1u);
}

TEST(IntervalSolver, SolveRootedRebasesRestrictedPoliciesForNonZeroStartIntervals) {
    left_size_score_recurrence recurrence;
    interval_solver<left_size_score_recurrence, leftmost_split_only> solver{recurrence};
    triangular_memo<int> memo{6};

    auto rooted = solver.solve_rooted<4>(interval_context{2, 6}, memo);

    EXPECT_DOUBLE_EQ(rooted.predicted_cost, 3.0);
    EXPECT_TRUE(rooted.params.is_legal());
    EXPECT_TRUE(rooted.params.is_canonical());
    EXPECT_EQ(rooted.params.root_interval().start(), 0u);
    EXPECT_EQ(rooted.params.root_interval().end(), 4u);
    EXPECT_EQ(rooted.params.split(0, 4), 1u);
    EXPECT_EQ(rooted.params.split(1, 4), 2u);
    EXPECT_EQ(rooted.params.split(2, 4), 3u);
    EXPECT_EQ(rooted.stats.subproblems_total, 7u);
    EXPECT_EQ(rooted.stats.candidates_total, 3u);
}

TEST(IntervalSolver, CustomSplitPolicyRestrictsSearch) {
    left_size_score_recurrence recurrence;
    interval_solver<left_size_score_recurrence, leftmost_split_only> solver{recurrence};
    triangular_memo<int> memo{4};

    auto value = solver.solve(interval_context{0, 4}, memo);

    EXPECT_EQ(value, 3);
}

TEST(IntervalSolver, CustomSplitPolicyStatsReflectRestrictedSearch) {
    left_size_score_recurrence recurrence;
    interval_solver<left_size_score_recurrence, leftmost_split_only> solver{recurrence};
    triangular_memo<int> memo{4};

    auto result = solver.solve_with_stats(interval_context{0, 4}, memo);

    EXPECT_EQ(result.value, 3);
    EXPECT_EQ(result.stats.subproblems_total, 7u);
    EXPECT_EQ(result.stats.subproblems_evaluated, 7u);
    EXPECT_EQ(result.stats.subproblems_cached, 7u);
    EXPECT_EQ(result.stats.candidates_total, 3u);
    EXPECT_EQ(result.stats.candidates_evaluated, 3u);
    EXPECT_EQ(result.stats.memo_misses, 7u);
    EXPECT_EQ(result.stats.memo_hits, 0u);
    EXPECT_EQ(result.stats.memo_table_size, 7u);
    EXPECT_EQ(result.stats.max_recursion_depth, 4u);
}

TEST(IntervalSolver, ReusesMemoizedRootAndSubproblems) {
    std::size_t base_calls = 0;
    std::size_t combine_calls = 0;

    counting_recurrence recurrence{&base_calls, &combine_calls};
    interval_solver<counting_recurrence> solver{recurrence};
    triangular_memo<int> memo{4};

    auto first = solver.solve(interval_context{0, 4}, memo);
    auto base_after_first = base_calls;
    auto combine_after_first = combine_calls;

    auto second = solver.solve(interval_context{0, 4}, memo);

    EXPECT_EQ(first, 4);
    EXPECT_EQ(second, 4);
    EXPECT_EQ(memo.size(), 10u);
    EXPECT_EQ(base_after_first, 10u);
    EXPECT_EQ(combine_after_first, 10u);
    EXPECT_EQ(base_calls, base_after_first);
    EXPECT_EQ(combine_calls, combine_after_first);
}

TEST(IntervalSolver, SolveWithStatsReportsRootMemoHitOnRepeatSolve) {
    counting_recurrence recurrence{nullptr, nullptr};
    interval_solver<counting_recurrence> solver{recurrence};
    triangular_memo<int> memo{4};

    std::size_t base_calls = 0;
    std::size_t combine_calls = 0;
    recurrence.base_calls = &base_calls;
    recurrence.combine_calls = &combine_calls;
    solver = interval_solver<counting_recurrence>{recurrence};

    auto first = solver.solve_with_stats(interval_context{0, 4}, memo);
    auto second = solver.solve_with_stats(interval_context{0, 4}, memo);

    EXPECT_EQ(first.value, 4);
    EXPECT_EQ(first.stats.subproblems_evaluated, 10u);
    EXPECT_EQ(first.stats.memo_misses, 10u);
    EXPECT_EQ(first.stats.candidates_evaluated, 10u);
    EXPECT_EQ(first.stats.memo_table_size, 10u);

    EXPECT_EQ(second.value, 4);
    EXPECT_EQ(second.stats.subproblems_total, 0u);
    EXPECT_EQ(second.stats.subproblems_evaluated, 0u);
    EXPECT_EQ(second.stats.subproblems_cached, 10u);
    EXPECT_EQ(second.stats.candidates_total, 0u);
    EXPECT_EQ(second.stats.candidates_evaluated, 0u);
    EXPECT_EQ(second.stats.memo_hits, 1u);
    EXPECT_EQ(second.stats.memo_misses, 0u);
    EXPECT_EQ(second.stats.memo_table_size, 10u);
    EXPECT_EQ(second.stats.max_recursion_depth, 1u);
}

} // namespace






