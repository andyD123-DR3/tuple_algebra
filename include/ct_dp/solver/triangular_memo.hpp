#ifndef CT_DP_SOLVER_TRIANGULAR_MEMO_HPP
#define CT_DP_SOLVER_TRIANGULAR_MEMO_HPP

#include "ct_dp/solver/interval_context.hpp"
#include <vector>
#include <optional>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace ct_dp {
namespace solver {

/// Dense triangular memo for bounded problems
///
/// Storage: n(n+1)/2 entries for all valid i < j pairs where j <= n
/// Sizing: For problem [0,N), construct triangular_memo<Value>{N}
/// Access: O(1) lookup/store
/// Allocation: One vector allocation, none in hot path
///
/// Example:
///   triangular_memo<int> memo{10};  // Supports endpoints [0, 10)
///   memo.store(interval_context{0, 4}, 42);
///   auto val = memo.lookup(interval_context{0, 4});  // Returns optional(42)
template<class Value>
class triangular_memo {
public:
    /// Construct memo for endpoints in [0, endpoint_limit)
    /// 
    /// @param endpoint_limit Largest allowed endpoint (exclusive upper bound)
    /// 
    /// For a problem [0, N), pass N as endpoint_limit.
    /// Storage allocated: N(N+1)/2 entries for all pairs (i,j) with i < j <= N
    explicit triangular_memo(size_t endpoint_limit)
        : n_(endpoint_limit),
          data_(n_ * (n_ + 1) / 2) {}
    
    /// Lookup value for interval [i, j)
    /// 
    /// Precondition: 0 <= i < j <= n_
    /// 
    /// @return optional<Value> containing stored value, or nullopt if not set
    std::optional<Value> lookup(interval_context ctx) const {
        assert(valid(ctx) && "Interval out of memo bounds");
        return data_[index(ctx.i, ctx.j)];
    }
    
    /// Store value for interval [i, j)
    /// 
    /// Precondition: 0 <= i < j <= n_
    void store(interval_context ctx, Value v) {
        assert(valid(ctx) && "Interval out of memo bounds");
        data_[index(ctx.i, ctx.j)] = std::move(v);
    }
    
    /// Clear all stored values
    void clear() {
        std::fill(data_.begin(), data_.end(), std::nullopt);
    }
    
    /// Count number of stored (non-nullopt) values
    /// 
    /// WARNING: This is O(n) where n = capacity*(capacity+1)/2
    /// Use for debugging/testing only, not in hot paths.
    /// If frequent size queries are needed, maintain a counter.
    size_t size() const {
        return std::count_if(data_.begin(), data_.end(),
            [](const auto& v) { return v.has_value(); });
    }
    
    /// Get endpoint limit (max j + 1)
    size_t capacity() const {
        return n_;
    }
    
private:
    size_t n_;  // endpoint_limit: largest valid j
    std::vector<std::optional<Value>> data_;
    
    /// Check if interval is within bounds
    bool valid(interval_context ctx) const {
        return ctx.i < ctx.j && ctx.j <= n_;
    }
    
    /// Dense triangular indexing: maps (i,j) with i<j to [0, n(n-1)/2)
    /// 
    /// Formula: For interval [i, j), index = j*(j-1)/2 + i
    /// 
    /// This creates a dense mapping of all valid (i,j) pairs with i < j.
    /// Example for n=4:
    ///   [0,1) -> 0
    ///   [0,2) -> 1, [1,2) -> 2
    ///   [0,3) -> 3, [1,3) -> 4, [2,3) -> 5
    ///   [0,4) -> 6, [1,4) -> 7, [2,4) -> 8, [3,4) -> 9
    ///   Total: 10 entries = 4*3/2
    /// 
    /// Note: For very large j (> 2^32 on 64-bit), j*(j-1) may overflow size_t.
    /// In practice, interval DP problems have n << 10^6, so overflow is not a concern.
    size_t index(size_t i, size_t j) const {
        assert(i < j && "Index requires i < j");
        return j * (j - 1) / 2 + i;
    }
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_TRIANGULAR_MEMO_HPP
