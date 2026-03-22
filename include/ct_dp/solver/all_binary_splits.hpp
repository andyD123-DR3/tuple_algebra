#ifndef CT_DP_SOLVER_ALL_BINARY_SPLITS_HPP
#define CT_DP_SOLVER_ALL_BINARY_SPLITS_HPP

#include "ct_dp/solver/interval_context.hpp"
#include <cstddef>

namespace ct_dp {
namespace solver {

/// Default split policy: emit all k in (i, j)
///
/// SplitPolicy Concept Requirements:
/// 1. Method: for_each(interval_context ctx, Emit&& emit) const
/// 2. Emits: Zero or more split points k
/// 3. Range: Each k satisfies ctx.i < k < ctx.j
/// 4. Legality: Each k is valid for the problem
/// 5. Feasibility: combine(plan_from(k), ...) will succeed
/// 6. Uniqueness: No duplicates (unless problem-specific)
///
/// This default policy emits all binary splits in order.
struct all_binary_splits {
    template<class Emit>
    void for_each(interval_context ctx, Emit&& emit) const {
        for (size_t k = ctx.i + 1; k < ctx.j; ++k) {
            emit(k);
        }
    }
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_ALL_BINARY_SPLITS_HPP
