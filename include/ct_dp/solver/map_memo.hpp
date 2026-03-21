#ifndef CT_DP_SOLVER_MAP_MEMO_HPP
#define CT_DP_SOLVER_MAP_MEMO_HPP

#include "ct_dp/solver/interval_context.hpp"
#include <map>
#include <optional>
#include <utility>
#include <cstddef>

namespace ct_dp {
namespace solver {

/// Sparse memo for unbounded or very large problems
///
/// Use when:
/// - Problem size unknown at compile time or construction time
/// - Sparse subproblem graph (many states unreachable)
/// - Prototyping before optimizing with triangular_memo
/// - Problem requires unbounded endpoint values
///
/// Trade-offs vs triangular_memo:
/// - Allocation: O(k) for k stored states (vs O(n²) pre-allocated)
/// - Lookup: O(log k) (vs O(1))
/// - Storage: Only stores reached states (vs all valid pairs)
/// - No size limit (vs bounded by endpoint_limit)
///
/// Example:
///   map_memo<int> memo;
///   memo.store(interval_context{0, 1000000}, 42);  // OK, unbounded
template<class Value>
class map_memo {
public:
    /// Lookup value for interval [i, j)
    /// 
    /// @return optional<Value> containing stored value, or nullopt if not set
    std::optional<Value> lookup(interval_context ctx) const {
        auto it = cache_.find({ctx.i, ctx.j});
        return it == cache_.end() 
            ? std::nullopt
            : std::optional<Value>{it->second};
    }
    
    /// Store value for interval [i, j)
    void store(interval_context ctx, Value v) {
        cache_[{ctx.i, ctx.j}] = std::move(v);
    }
    
    /// Clear all stored values
    void clear() {
        cache_.clear();
    }
    
    /// Count number of stored values
    size_t size() const {
        return cache_.size();
    }
    
    /// Check if interval has been stored
    bool contains(interval_context ctx) const {
        return cache_.find({ctx.i, ctx.j}) != cache_.end();
    }
    
private:
    std::map<std::pair<size_t, size_t>, Value> cache_;
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_MAP_MEMO_HPP
