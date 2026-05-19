// ctdp/solver/algorithms/interval_solver.h
// Narrow Stage 1 interval solver substrate.

#ifndef CTDP_SOLVER_ALGORITHMS_INTERVAL_SOLVER_H
#define CTDP_SOLVER_ALGORITHMS_INTERVAL_SOLVER_H

#include "../interval_context.h"
#include "../plans/interval_partition_plan.h"
#include "../policies/all_binary_splits.h"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>

namespace ctdp::solver::algorithms {

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
        assert(ctx.i < ctx.j && "Empty interval not allowed");

        if (auto cached = memo.lookup(ctx)) {
            return *cached;
        }

        if (auto base = recurrence_.base_case(ctx)) {
            memo.store(ctx, *base);
            return *base;
        }

        std::optional<value_type> best;

        split_policy_.for_each(ctx, [&](std::size_t k) {
            auto plan = ctdp::solver::plans::interval_partition_plan::from_split(ctx, k);

            value_type left = solve(plan.left_ctx, memo);
            value_type right = solve(plan.right_ctx, memo);
            value_type candidate = recurrence_.combine(plan, left, right);

            if (!best || compare_(candidate, *best)) {
                best = std::move(candidate);
            }
        });

        assert(best.has_value() && "No valid splits found - check split policy or base_case");
        memo.store(ctx, *best);
        return *best;
    }

    [[nodiscard]] Recurrence const& recurrence() const { return recurrence_; }
    [[nodiscard]] SplitPolicy const& split_policy() const { return split_policy_; }
    [[nodiscard]] Compare const& compare() const { return compare_; }

private:
    Recurrence recurrence_;
    SplitPolicy split_policy_;
    Compare compare_;
};

} // namespace ctdp::solver::algorithms

#endif // CTDP_SOLVER_ALGORITHMS_INTERVAL_SOLVER_H
