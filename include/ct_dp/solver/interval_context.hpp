// interval_context.hpp - Runtime interval state
// Sprint 8: Minimal passive interval record
// Author: Andrew Drakeford

#ifndef CT_DP_SOLVER_INTERVAL_CONTEXT_HPP
#define CT_DP_SOLVER_INTERVAL_CONTEXT_HPP

#include <cassert>
#include <cstddef>

namespace ct_dp {
namespace solver {

/**
 * @brief Runtime representation of an interval [i, j) being solved
 * 
 * Design: Passive interval record with minimal helper methods.
 * Does NOT own descriptor application logic (that lives in interval_partition_plan).
 * 
 * Semantic Law: interval_context{i,j} names interval [i,j) of length Len = j-i.
 */
struct interval_context {
    size_t i;  ///< Interval start (inclusive)
    size_t j;  ///< Interval end (exclusive)
    
    /**
     * @brief Construct interval [start, end)
     * 
     * Precondition: start < end.
     * Sprint 8 models only non-empty intervals; empty intervals are intentionally
     * excluded from this type for now. Revisit if Sprint 9 requires zero-length
     * base cases.
     * 
     * @param start Interval start (inclusive)
     * @param end Interval end (exclusive)
     */
    constexpr interval_context(size_t start, size_t end) noexcept
        : i(start), j(end) {
        assert(i < j && "Empty intervals are invalid");
    }
    
    /**
     * @brief Get interval length
     * @return j - i
     */
    constexpr size_t size() const noexcept { 
        return j - i; 
    }
    
    /**
     * @brief Get interval start
     * @return i
     */
    constexpr size_t start() const noexcept { 
        return i; 
    }
    
    /**
     * @brief Get interval end
     * @return j
     */
    constexpr size_t end() const noexcept { 
        return j; 
    }
    
    /**
     * @brief Create left subinterval [i, k)
     * 
     * Geometric operation - no descriptor coupling.
     * 
     * Precondition: i < k < j (k must be strictly interior)
     * 
     * @param k Cut point
     * @return Left subinterval [i, k)
     */
    [[nodiscard]] constexpr interval_context left(size_t k) const noexcept {
        assert(k > i && k < j && "Cut point must be strictly interior");
        return {i, k};
    }
    
    /**
     * @brief Create right subinterval [k, j)
     * 
     * Geometric operation - no descriptor coupling.
     * 
     * Precondition: i < k < j (k must be strictly interior)
     * 
     * @param k Cut point
     * @return Right subinterval [k, j)
     */
    [[nodiscard]] constexpr interval_context right(size_t k) const noexcept {
        assert(k > i && k < j && "Cut point must be strictly interior");
        return {k, j};
    }
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_INTERVAL_CONTEXT_HPP
