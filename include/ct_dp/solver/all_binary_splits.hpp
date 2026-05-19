#ifndef CT_DP_SOLVER_ALL_BINARY_SPLITS_HPP
#define CT_DP_SOLVER_ALL_BINARY_SPLITS_HPP

#include "ct_dp/solver/interval_context.hpp"

#include <cstddef>

namespace ct_dp {
namespace solver {

/// Default split policy: emit all interior cuts k in (i, j).
///
/// For interval [i, j), emits the ordered sequence:
///   i + 1, i + 2, ..., j - 1
///
/// Stage 1 scope: this is the canonical "all legal binary splits" policy for
/// ordered interval decomposition. More selective policies are deferred.
struct all_binary_splits {
    template <class Emit>
    void for_each(interval_context ctx, Emit&& emit) const {
        for (size_t k = ctx.i + 1; k < ctx.j; ++k) {
            emit(k);
        }
    }
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_ALL_BINARY_SPLITS_HPP

