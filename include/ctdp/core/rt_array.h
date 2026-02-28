// core/rt_array.h — Fixed-capacity runtime array (non-constexpr counterpart to std::array)
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// constexpr graphs use std::array<T, MaxV> where MaxV is a template parameter.
// Runtime graphs don't know MaxV at compile time, so they need a dynamically-
// sized container that still owns its storage (no heap allocation during
// element access).
//
// rt_array<T> is a fixed-capacity vector: size set at construction, elements
// accessed in O(1).  It wraps std::vector but presents the same interface as
// std::array for algorithm code that indexes by node_id/edge_id.
//
// The key invariant: once constructed with a capacity, rt_array does not grow.
// Algorithms that call make_node_array<T>(g) get either a std::array (constexpr)
// or an rt_array (runtime) — the indexing interface is identical.

#ifndef CTDP_CORE_RT_ARRAY_H
#define CTDP_CORE_RT_ARRAY_H

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ctdp {

/// Fixed-capacity runtime array.
///
/// Constructed with a size (and optional fill value).  Supports O(1)
/// indexed access.  Does not support push_back or resize — capacity
/// is fixed at construction.
///
/// This type is the runtime counterpart to std::array<T, N>.
/// graph_traits<runtime_graph> maps node_array/edge_array to rt_array<T>.
template<typename T>
class rt_array {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = T const&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    rt_array() = default;

    /// Construct with `n` value-initialised elements.
    explicit rt_array(std::size_t n)
        : data_(n) {}

    /// Construct with `n` copies of `fill`.
    rt_array(std::size_t n, T const& fill)
        : data_(n, fill) {}

    // -----------------------------------------------------------------
    // Size queries
    // -----------------------------------------------------------------

    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    // -----------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------

    reference operator[](std::size_t i) { return data_[i]; }
    const_reference operator[](std::size_t i) const { return data_[i]; }

    reference at(std::size_t i) {
        if (i >= data_.size())
            throw std::out_of_range("rt_array::at: index out of bounds");
        return data_[i];
    }
    const_reference at(std::size_t i) const {
        if (i >= data_.size())
            throw std::out_of_range("rt_array::at: index out of bounds");
        return data_[i];
    }

    // -----------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------

    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }

    // -----------------------------------------------------------------
    // Data access
    // -----------------------------------------------------------------

    T* data() noexcept { return data_.data(); }
    T const* data() const noexcept { return data_.data(); }

    // -----------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------

    [[nodiscard]] bool operator==(rt_array const& other) const {
        return data_ == other.data_;
    }

private:
    std::vector<T> data_{};
};

} // namespace ctdp

#endif // CTDP_CORE_RT_ARRAY_H
