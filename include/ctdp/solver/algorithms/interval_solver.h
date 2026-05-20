// ctdp/solver/algorithms/interval_solver.h
// Narrow Stage 1 interval solver substrate.

#ifndef CTDP_SOLVER_ALGORITHMS_INTERVAL_SOLVER_H
#define CTDP_SOLVER_ALGORITHMS_INTERVAL_SOLVER_H

#include "../interval_rooted_candidate.h"
#include "../interval_context.h"
#include "../plans/interval_partition_plan.h"
#include "../policies/all_binary_splits.h"
#include "../../core/solve_stats.h"

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>

namespace ctdp::solver::algorithms {

namespace detail {

struct no_choice_tracking {
    constexpr void store([[maybe_unused]] std::size_t i,
                         [[maybe_unused]] std::size_t j,
                         [[maybe_unused]] std::size_t k) const noexcept {}
};

template<std::size_t MaxN>
struct interval_choice_table {
    std::array<std::size_t, MaxN * MaxN> best_split{};

    [[nodiscard]] static constexpr std::size_t index(std::size_t i, std::size_t j) noexcept {
        return i * MaxN + (j - 1);
    }

    [[nodiscard]] constexpr bool contains(std::size_t i, std::size_t j) const noexcept {
        assert(i < j && j <= MaxN && "Choice lookup requires interval inside table bounds");
        return best_split[index(i, j)] != 0;
    }

    constexpr void store(std::size_t i, std::size_t j, std::size_t k) noexcept {
        assert(i + 1 < j && j <= MaxN && "Choice storage requires an internal interval inside table bounds");
        assert(i < k && k < j && "Stored split must be interior to the interval");
        best_split[index(i, j)] = k + 1;
    }

    [[nodiscard]] constexpr std::size_t split(std::size_t i, std::size_t j) const noexcept {
        assert(contains(i, j) && "split requires a stored internal interval choice");
        return best_split[index(i, j)] - 1;
    }
};

template<std::size_t MaxN>
struct rebased_choice_tracker {
    std::size_t origin{};
    interval_choice_table<MaxN> local_choices{};

    constexpr void store(std::size_t i, std::size_t j, std::size_t k) noexcept {
        assert(i >= origin && j >= origin && k >= origin && "Rebased choice coordinates must not underflow");
        local_choices.store(i - origin, j - origin, k - origin);
    }
};

template<std::size_t MaxN>
[[nodiscard]] constexpr auto reconstruct_rooted_candidate(
    ctdp::solver::interval_context ctx,
    rebased_choice_tracker<MaxN> const& choices)
    -> ctdp::solver::interval_rooted_candidate<MaxN>
{
    assert(ctx.size() <= MaxN && "Reconstruction capacity MaxN must cover the rooted interval size");

    return ctdp::solver::reconstruct_interval_rooted_candidate<MaxN>(
        ctx.size(),
        [&choices](std::size_t i, std::size_t j) constexpr -> std::size_t {
            return choices.local_choices.split(i, j);
        });
}

} // namespace detail

template<class Value>
struct interval_solve_result {
    Value value;
    ctdp::solve_stats stats;
};

template<class R>
concept interval_recurrence =
    requires(R const& r,
             ctdp::solver::interval_context ctx,
             ctdp::solver::plans::interval_partition_plan const& plan,
             typename R::value_type const& left,
             typename R::value_type const& right) {
        typename R::value_type;

        { r.base_case(ctx) }
            -> std::same_as<std::optional<typename R::value_type>>;

        { r.combine(plan, left, right) }
            -> std::same_as<typename R::value_type>;
    };

template<class M, class Value>
concept interval_memo =
    requires(M& m, M const& cm, ctdp::solver::interval_context ctx, Value v) {
        { cm.lookup(ctx) } -> std::same_as<std::optional<Value>>;
        { m.store(ctx, std::move(v)) } -> std::same_as<void>;
    };

template<class Recurrence,
         class SplitPolicy = ctdp::solver::policies::all_binary_splits,
         class Compare = std::less<>>
requires interval_recurrence<Recurrence>
class interval_solver {
public:
    using value_type = typename Recurrence::value_type;
    using result_type = interval_solve_result<value_type>;

    explicit interval_solver(Recurrence recurrence,
                             SplitPolicy split_policy = {},
                             Compare compare = {})
        : recurrence_(std::move(recurrence)),
          split_policy_(std::move(split_policy)),
          compare_(std::move(compare)) {}

    template<class Memo>
    requires interval_memo<Memo, value_type>
    [[nodiscard]] value_type solve(ctdp::solver::interval_context ctx,
                                   Memo& memo) const {
        return solve_with_stats(ctx, memo).value;
    }

    template<class Memo>
    requires interval_memo<Memo, value_type>
    [[nodiscard]] result_type solve_with_stats(ctdp::solver::interval_context ctx,
                                               Memo& memo) const {
        assert(ctx.i < ctx.j && "Empty interval not allowed");

        ctdp::solve_stats stats{};
        detail::no_choice_tracking choices{};
        auto value = solve_impl(ctx, memo, choices, stats, 1);

        stats.subproblems_total = stats.subproblems_evaluated;
        if constexpr (requires(Memo const& m) {
            { m.size() } -> std::convertible_to<std::size_t>;
        }) {
            stats.memo_table_size = memo.size();
            stats.subproblems_cached = memo.size();
        }

        return {std::move(value), stats};
    }

    template<std::size_t MaxN, class Memo>
    requires interval_memo<Memo, value_type>
          && std::convertible_to<value_type, double>
    [[nodiscard]] auto solve_rooted(ctdp::solver::interval_context ctx,
                                    Memo& memo) const
        -> ctdp::solver::interval_rooted_plan<MaxN>
    {
        assert(ctx.size() <= MaxN && "solve_rooted requires MaxN to cover the rooted interval size");

        ctdp::solve_stats stats{};
        detail::rebased_choice_tracker<MaxN> choices{ctx.i};
        auto value = solve_impl(ctx, memo, choices, stats, 1);

        stats.subproblems_total = stats.subproblems_evaluated;
        if constexpr (requires(Memo const& m) {
            { m.size() } -> std::convertible_to<std::size_t>;
        }) {
            stats.memo_table_size = memo.size();
            stats.subproblems_cached = memo.size();
        }

        auto rooted = detail::reconstruct_rooted_candidate<MaxN>(ctx, choices);
        return ctdp::solver::interval_rooted_plan<MaxN>{
            std::move(rooted),
            static_cast<double>(value),
            stats};
    }

    [[nodiscard]] Recurrence const& recurrence() const { return recurrence_; }
    [[nodiscard]] SplitPolicy const& split_policy() const { return split_policy_; }
    [[nodiscard]] Compare const& compare() const { return compare_; }

private:
    template<class Memo, class ChoiceTracker>
    requires interval_memo<Memo, value_type>
    [[nodiscard]] value_type solve_impl(ctdp::solver::interval_context ctx,
                                        Memo& memo,
                                        ChoiceTracker& choices,
                                        ctdp::solve_stats& stats,
                                        std::size_t depth) const {
        if (depth > stats.max_recursion_depth) {
            stats.max_recursion_depth = depth;
        }

        if (auto cached = memo.lookup(ctx)) {
            ++stats.memo_hits;
            return *cached;
        }

        ++stats.memo_misses;
        ++stats.subproblems_evaluated;

        if (auto base = recurrence_.base_case(ctx)) {
            memo.store(ctx, *base);
            ++stats.subproblems_cached;
            return *base;
        }

        std::optional<value_type> best;
        std::optional<std::size_t> best_split;

        split_policy_.for_each(ctx, [&](std::size_t k) {
            auto plan = ctdp::solver::plans::interval_partition_plan::from_split(ctx, k);

            value_type left = solve_impl(plan.left_ctx, memo, choices, stats, depth + 1);
            value_type right = solve_impl(plan.right_ctx, memo, choices, stats, depth + 1);
            ++stats.candidates_total;
            ++stats.candidates_evaluated;
            value_type candidate = recurrence_.combine(plan, left, right);

            if (!best || compare_(candidate, *best)) {
                best = std::move(candidate);
                best_split = k;
            }
        });

        assert(best.has_value() && "No valid splits found - check split policy or base_case");
        assert(best_split.has_value() && "Internal intervals must record a winning split");
        choices.store(ctx.i, ctx.j, *best_split);
        memo.store(ctx, *best);
        ++stats.subproblems_cached;
        return *best;
    }

    Recurrence recurrence_;
    SplitPolicy split_policy_;
    Compare compare_;
};

} // namespace ctdp::solver::algorithms

#endif // CTDP_SOLVER_ALGORITHMS_INTERVAL_SOLVER_H



