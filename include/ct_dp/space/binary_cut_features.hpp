// binary_cut_features.hpp - Feature encodings for binary_cut_desc
// Sprint 8: Standalone feature encoding helper
// Author: Andrew Drakeford
//
// NOTE: This is a standalone helper, not a specialization of feature_bridge.
// Integration into the canonical feature_bridge.hpp can be done in a future commit.

#ifndef CT_DP_SPACE_BINARY_CUT_FEATURES_HPP
#define CT_DP_SPACE_BINARY_CUT_FEATURES_HPP

#include "ct_dp/space/binary_cut_desc.hpp"
#include <array>
#include <cassert>
#include <cstddef>

namespace ct_dp {
namespace space {

/**
 * @brief Feature encoding helper for binary_cut_desc
 * 
 * Encoding Strategy:
 * - Option A (Fixed-Length): Use encode_onehot or encode_integer
 *   For per-length models where Len is constant
 * - Option B (Variable-Length): Use encode_normalized or encode_full
 *   For cross-length models where Len varies
 * 
 * Representation Guarantee: All encodings return fixed-size containers
 * (std::array or scalar). No dynamic allocation in hot path.
 */
template<size_t Len>
struct binary_cut_features {
    
    /**
     * @brief One-hot encoding (OPTION A: Fixed-length training)
     * 
     * Returns: std::array<T, Len-1> (compile-time sized, zero allocation)
     * Use case: Per-length models (separate SVR for each Len)
     * 
     * Encoding: [0,0,...,1,...,0] with single 1 at position (rel-1)
     * 
     * @tparam T Feature type (default: double)
     * @param rel Relative cut position [1, Len-1]
     * @return One-hot feature vector
     */
    template<typename T = double>
    static constexpr auto encode_onehot(size_t rel) noexcept {
        assert(rel > 0 && rel < Len && "Relative cut must be in [1, Len-1]");
        std::array<T, Len-1> result{};
        result[rel - 1] = T{1};
        return result;
    }
    
    /**
     * @brief Integer encoding (OPTION A: Fixed-length training)
     * 
     * Returns: size_t (single feature)
     * Use case: Ordinal regression models
     * 
     * @param rel Relative cut position [1, Len-1]
     * @return Relative cut as integer feature
     */
    static constexpr auto encode_integer(size_t rel) noexcept {
        assert(rel > 0 && rel < Len && "Relative cut must be in [1, Len-1]");
        return rel;
    }
    
    /**
     * @brief Normalized encoding (OPTION B: Variable-length training)
     * 
     * Returns: std::array<T, 3> = {length, rel, ratio}
     * Use case: Cross-length models (single SVR for multiple Len)
     * 
     * Features:
     * - [0]: Interval length (Len)
     * - [1]: Relative cut (rel)
     * - [2]: Normalized ratio (rel / Len) ∈ (0, 1)
     * 
     * @tparam T Feature type (default: double)
     * @param rel Relative cut position [1, Len-1]
     * @return Normalized feature vector
     */
    template<typename T = double>
    static constexpr auto encode_normalized(size_t rel) noexcept {
        assert(rel > 0 && rel < Len && "Relative cut must be in [1, Len-1]");
        return std::array<T, 3>{
            static_cast<T>(Len),              // Context: interval length
            static_cast<T>(rel),              // Choice: relative cut
            static_cast<T>(rel) / Len         // Normalized ratio
        };
    }
    
    /**
     * @brief Full encoding (OPTION B: Variable-length with asymmetry)
     * 
     * Returns: std::array<T, 5> = {length, rel, ratio, left_size, right_size}
     * Use case: Asymmetry-aware cross-length models
     * 
     * Features:
     * - [0]: Total length (Len)
     * - [1]: Cut position (rel)
     * - [2]: Normalized ratio (rel / Len)
     * - [3]: Left subinterval size (rel)
     * - [4]: Right subinterval size (Len - rel)
     * 
     * @tparam T Feature type (default: double)
     * @param rel Relative cut position [1, Len-1]
     * @return Full feature vector with asymmetry info
     */
    template<typename T = double>
    static constexpr auto encode_full(size_t rel) noexcept {
        assert(rel > 0 && rel < Len && "Relative cut must be in [1, Len-1]");
        return std::array<T, 5>{
            static_cast<T>(Len),              // Total length
            static_cast<T>(rel),              // Cut position
            static_cast<T>(rel) / Len,        // Normalized ratio
            static_cast<T>(rel),              // Left size
            static_cast<T>(Len - rel)         // Right size
        };
    }
};

} // namespace space
} // namespace ct_dp

// USAGE GUIDANCE:
// 
// Fixed-length models (one model per Len):
//   auto features = binary_cut_features<10>::encode_onehot(5);
//   auto features = binary_cut_features<10>::encode_integer(5);
//
// Variable-length models (one model across many Len):
//   auto features = binary_cut_features<10>::encode_normalized(5);
//   auto features = binary_cut_features<10>::encode_full(5);
//
// Scalability:
// - One-hot: Suitable for Len ≤ 100 (sparse beyond this)
// - Integer: Suitable for any Len
// - Normalized/Full: Suitable for any Len (fixed feature count)

#endif // CT_DP_SPACE_BINARY_CUT_FEATURES_HPP
