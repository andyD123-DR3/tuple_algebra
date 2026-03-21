// binary_cut_desc.hpp - Descriptor for relative binary cuts
// Sprint 8: Split constructor integration
// Author: Andrew Drakeford

#ifndef CT_DP_SPACE_BINARY_CUT_DESC_HPP
#define CT_DP_SPACE_BINARY_CUT_DESC_HPP

#include <array>
#include <cassert>
#include <cstddef>

namespace ct_dp {
namespace space {

/**
 * @brief Descriptor for relative binary cuts over a problem fragment of length Len
 * 
 * Semantic Law: binary_cut_desc<Len> is a descriptor for relative cut choices 
 * over a problem fragment of length Len; it does not encode absolute interval identity.
 * 
 * Valid relative cuts are {1, 2, ..., Len-1}, measured from the fragment start.
 * 
 * Object Model: Value-oriented constexpr descriptor
 * Usage: binary_cut_desc<10> desc; desc.size();
 *        OR: binary_cut_desc<10>{}.size();
 * 
 * @tparam Len Logical length of the problem fragment (Len >= 2)
 */
template<size_t Len>
struct binary_cut_desc {
    static_assert(Len >= 2, "Binary cut requires length >= 2");
    
    /**
     * @brief Number of valid cut choices (dimension of choice space)
     * @return Len - 1 (cuts at relative positions 1..Len-1)
     */
    constexpr size_t size() const noexcept { 
        return Len - 1; 
    }
    
    /**
     * @brief Rank of the descriptor (number of decision dimensions)
     * @return 1 (single-dimensional choice)
     */
    constexpr size_t rank() const noexcept { 
        return 1; 
    }
    
    /**
     * @brief Shape of the choice space
     * @return Array containing {Len-1}
     */
    constexpr auto shape() const noexcept { 
        return std::array<size_t, 1>{Len - 1}; 
    }
    
    /**
     * @brief Get the interval length this descriptor is parameterized for
     * @return Len
     */
    constexpr size_t interval_length() const noexcept { 
        return Len; 
    }
    
    /**
     * @brief Get the count of valid cuts (same as size())
     * @return Len - 1
     */
    constexpr size_t cut_count() const noexcept { 
        return Len - 1; 
    }
    
    /**
     * @brief Convert ordinal index to relative cut position
     * 
     * Precondition: ord < Len - 1
     * 
     * @param ord Ordinal index [0, Len-2]
     * @return Relative cut position [1, Len-1]
     */
    constexpr size_t ordinal_to_relative_cut(size_t ord) const noexcept {
        assert(ord < Len - 1 && "Ordinal must be in [0, Len-2]");
        return ord + 1;
    }
    
    /**
     * @brief Convert relative cut position to ordinal index
     * 
     * Precondition: rel > 0 && rel < Len
     * 
     * @param rel Relative cut position [1, Len-1]
     * @return Ordinal index [0, Len-2]
     */
    constexpr size_t relative_cut_to_ordinal(size_t rel) const noexcept {
        assert(rel > 0 && rel < Len && "Relative cut must be in [1, Len-1]");
        return rel - 1;
    }
};

} // namespace space
} // namespace ct_dp

#endif // CT_DP_SPACE_BINARY_CUT_DESC_HPP
