#ifndef CT_DP_SOLVER_TRIANGULAR_MEMO_HPP
#define CT_DP_SOLVER_TRIANGULAR_MEMO_HPP

#include "ct_dp/solver/interval_context.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace ct_dp {
namespace solver {

/// Dense triangular memo for interval subproblems.
///
/// Construct with endpoint_limit = N to support all intervals [i, j) such that
/// 0 <= i < j <= N.
///
/// Storage size is N * (N + 1) / 2 entries, one for every valid interval.
template <class Value>
class triangular_memo {
public:
    explicit triangular_memo(size_t endpoint_limit)
        : n_(endpoint_limit),
          data_(n_ * (n_ + 1) / 2) {}

    std::optional<Value> lookup(interval_context ctx) const {
        assert(valid(ctx) && "Interval out of memo bounds");
        return data_[index(ctx.i, ctx.j)];
    }

    void store(interval_context ctx, Value v) {
        assert(valid(ctx) && "Interval out of memo bounds");
        data_[index(ctx.i, ctx.j)] = std::move(v);
    }

    void clear() {
        std::fill(data_.begin(), data_.end(), std::nullopt);
    }

    size_t size() const {
        return static_cast<size_t>(std::count_if(data_.begin(), data_.end(),
            [](const auto& v) { return v.has_value(); }));
    }

    size_t capacity() const {
        return n_;
    }

private:
    size_t n_;
    std::vector<std::optional<Value>> data_;

    bool valid(interval_context ctx) const {
        return ctx.i < ctx.j && ctx.j <= n_;
    }

    size_t index(size_t i, size_t j) const {
        assert(i < j && "Index requires i < j");
        return j * (j - 1) / 2 + i;
    }
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_TRIANGULAR_MEMO_HPP


