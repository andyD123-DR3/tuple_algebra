// interval_partition_plan.hpp - Plan term for binary interval partition
// Sprint 8: Integration layer for descriptor + context
// Author: Andrew Drakeford

#ifndef CT_DP_PLAN_INTERVAL_PARTITION_PLAN_HPP
#define CT_DP_PLAN_INTERVAL_PARTITION_PLAN_HPP

#include "ct_dp/solver/interval_context.hpp"
#include "ct_dp/space/binary_cut_desc.hpp"
#include <cassert>
#include <cstddef>

namespace ct_dp {
    namespace plan {

        struct interval_partition_plan {
            solver::interval_context whole;
            size_t split;
            solver::interval_context left_ctx;
            solver::interval_context right_ctx;

            template<size_t Len>
            [[nodiscard]] static constexpr interval_partition_plan make(
                solver::interval_context ctx,
                space::binary_cut_desc<Len> desc,
                size_t ordinal
            ) noexcept {
                assert(ctx.size() == Len &&
                    "Context size must match descriptor length (PRECONDITION)");
                size_t rel = desc.ordinal_to_relative_cut(ordinal);
                size_t k = ctx.start() + rel;
                return { ctx, k, ctx.left(k), ctx.right(k) };
            }

            // Sprint 9: Runtime plan construction from split point
            [[nodiscard]] static constexpr interval_partition_plan from_split(
                solver::interval_context ctx,
                size_t k
            ) noexcept {
                assert(k > ctx.i && k < ctx.j && "Invalid split point");
                return { ctx, k, solver::interval_context{ctx.i, k}, solver::interval_context{k, ctx.j} };
            }

            constexpr bool is_legal() const noexcept {
                return whole.i < split && split < whole.j &&
                    left_ctx.i == whole.i && left_ctx.j == split &&
                    right_ctx.i == split && right_ctx.j == whole.j;
            }

            constexpr size_t left_size() const noexcept { return left_ctx.size(); }
            constexpr size_t right_size() const noexcept { return right_ctx.size(); }

            constexpr bool preserves_size() const noexcept {
                return left_size() + right_size() == whole.size();
            }
        };

    } // namespace plan
} // namespace ct_dp

#endif // CT_DP_PLAN_INTERVAL_PARTITION_PLAN_HPP