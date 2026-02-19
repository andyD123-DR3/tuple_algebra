// core/constexpr_vector.h - Fixed-capacity dynamic array for compile-time DP
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// std::vector cannot be used as a constexpr variable because it allocates.
// constexpr_vector provides vector-like semantics with fixed maximum capacity.
//
// IMPLEMENTATION STRATEGY:
// Thin wrapper around std::array<T, MaxN> + size tracking.
// This is simpler and safer than managing raw storage:
// - No placement new/delete
// - No manual destructor calls
// - No reinterpret_cast
// - std::array handles all memory management
//
// Trade-off: All MaxN elements are default-constructed even if unused.
// This is acceptable for compile-time DP where MaxN is typically small (<1000).
//
// Key properties:
// - Fixed capacity (MaxN) known at compile time
// - Dynamic size (0 to MaxN) tracked
// - No heap allocation (std::array storage)
// - Constexpr-compatible
// - STL-compatible interface

#ifndef CTDP_CORE_CONSTEXPR_VECTOR_H
#define CTDP_CORE_CONSTEXPR_VECTOR_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ctdp {

/// Fixed-capacity dynamic array for compile-time dynamic programming.
///
/// Implemented as std::array<T, MaxN> + size tracking.
/// Provides vector-like interface with fixed capacity.
template<typename T, size_t MaxN>
class constexpr_vector {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = T const&;
    using pointer = T*;
    using const_pointer = T const*;
    using iterator = typename std::array<T, MaxN>::iterator;
    using const_iterator = typename std::array<T, MaxN>::const_iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    
private:
    std::array<T, MaxN> data_{};  // All elements exist (default-constructed)
    size_type size_ = 0;           // Active size [0, MaxN]
    
public:
    // =============================================================================
    // Constructors
    // =============================================================================
    
    constexpr constexpr_vector() = default;
    
    constexpr constexpr_vector(size_type n, T const& value = T{}) : size_(n) {
        if (n > MaxN) {
            throw std::length_error("constexpr_vector: size exceeds capacity");
        }
        for (size_type i = 0; i < n; ++i) {
            data_[i] = value;
        }
    }
    
    constexpr constexpr_vector(std::initializer_list<T> init) : size_(init.size()) {
        if (init.size() > MaxN) {
            throw std::length_error("constexpr_vector: initializer_list exceeds capacity");
        }
        std::copy(init.begin(), init.end(), data_.begin());
    }
    
    // =============================================================================
    // Element Access
    // =============================================================================
    
    constexpr reference at(size_type pos) {
        if (pos >= size_) {
            throw std::out_of_range("constexpr_vector::at");
        }
        return data_[pos];
    }
    
    constexpr const_reference at(size_type pos) const {
        if (pos >= size_) {
            throw std::out_of_range("constexpr_vector::at");
        }
        return data_[pos];
    }
    
    constexpr reference operator[](size_type pos) { return data_[pos]; }
    constexpr const_reference operator[](size_type pos) const { return data_[pos]; }
    
    constexpr reference front() { return data_[0]; }
    constexpr const_reference front() const { return data_[0]; }
    
    constexpr reference back() { return data_[size_ - 1]; }
    constexpr const_reference back() const { return data_[size_ - 1]; }
    
    constexpr T* data() noexcept { return data_.data(); }
    constexpr T const* data() const noexcept { return data_.data(); }
    
    // =============================================================================
    // Iterators (only expose active range [0, size_))
    // =============================================================================
    
    constexpr iterator begin() noexcept { return data_.begin(); }
    constexpr const_iterator begin() const noexcept { return data_.begin(); }
    constexpr const_iterator cbegin() const noexcept { return data_.cbegin(); }
    
    constexpr iterator end() noexcept { return data_.begin() + size_; }
    constexpr const_iterator end() const noexcept { return data_.begin() + size_; }
    constexpr const_iterator cend() const noexcept { return data_.cbegin() + size_; }
    
    constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    
    constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }
    
    // =============================================================================
    // Capacity
    // =============================================================================
    
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr size_type max_size() const noexcept { return MaxN; }
    [[nodiscard]] constexpr size_type capacity() const noexcept { return MaxN; }
    
    // =============================================================================
    // Modifiers
    // =============================================================================
    
    constexpr void clear() noexcept { size_ = 0; }
    
    constexpr iterator insert(const_iterator pos, T const& value) {
        size_type index = pos - begin();
        if (size_ >= MaxN) {
            throw std::length_error("constexpr_vector::insert: capacity exceeded");
        }
        
        // Shift elements right
        for (size_type i = size_; i > index; --i) {
            data_[i] = std::move(data_[i - 1]);
        }
        
        data_[index] = value;
        ++size_;
        return begin() + index;
    }
    
    constexpr iterator insert(const_iterator pos, T&& value) {
        size_type index = pos - begin();
        if (size_ >= MaxN) {
            throw std::length_error("constexpr_vector::insert: capacity exceeded");
        }
        
        for (size_type i = size_; i > index; --i) {
            data_[i] = std::move(data_[i - 1]);
        }
        
        data_[index] = std::move(value);
        ++size_;
        return begin() + index;
    }
    
    constexpr iterator erase(const_iterator pos) {
        size_type index = pos - begin();
        if (index >= size_) {
            throw std::out_of_range("constexpr_vector::erase: position out of range");
        }
        
        // Shift elements left
        for (size_type i = index; i + 1 < size_; ++i) {
            data_[i] = std::move(data_[i + 1]);
        }
        
        --size_;
        return begin() + index;
    }
    
    constexpr iterator erase(const_iterator first, const_iterator last) {
        size_type start = first - begin();
        size_type count = last - first;
        
        if (count == 0) return begin() + start;
        if (start + count > size_) {
            throw std::out_of_range("constexpr_vector::erase: range exceeds size");
        }
        
        // Shift remaining elements left
        for (size_type i = start; i + count < size_; ++i) {
            data_[i] = std::move(data_[i + count]);
        }
        
        size_ -= count;
        return begin() + start;
    }
    
    constexpr void push_back(T const& value) {
        if (size_ >= MaxN) {
            throw std::length_error("constexpr_vector::push_back: capacity exceeded");
        }
        data_[size_++] = value;
    }
    
    constexpr void push_back(T&& value) {
        if (size_ >= MaxN) {
            throw std::length_error("constexpr_vector::push_back: capacity exceeded");
        }
        data_[size_++] = std::move(value);
    }
    
    template<typename... Args>
    constexpr reference emplace_back(Args&&... args) {
        if (size_ >= MaxN) {
            throw std::length_error("constexpr_vector::emplace_back: capacity exceeded");
        }
        data_[size_] = T(std::forward<Args>(args)...);
        return data_[size_++];
    }
    
    constexpr void pop_back() {
        if (size_ == 0) {
            throw std::out_of_range("constexpr_vector::pop_back: empty");
        }
        --size_;
    }
    
    constexpr void resize(size_type n, T const& value = T{}) {
        if (n > MaxN) {
            throw std::length_error("constexpr_vector::resize: size exceeds capacity");
        }
        
        if (n > size_) {
            // Fill new elements
            for (size_type i = size_; i < n; ++i) {
                data_[i] = value;
            }
        }
        size_ = n;
    }
    
    // =============================================================================
    // Comparison
    // =============================================================================
    
    constexpr bool operator==(constexpr_vector const& other) const {
        if (size_ != other.size_) return false;
        for (size_type i = 0; i < size_; ++i) {
            if (!(data_[i] == other.data_[i])) return false;
        }
        return true;
    }
    
    constexpr auto operator<=>(constexpr_vector const& other) const 
        requires std::three_way_comparable<T>
    {
        return std::lexicographical_compare_three_way(
            begin(), end(),
            other.begin(), other.end()
        );
    }
};

// No deduction guide: constexpr_vector{1,2,3} would deduce MaxN=3 with zero
// headroom for push_back. Always specify capacity explicitly.

} // namespace ctdp

#endif // CTDP_CORE_CONSTEXPR_VECTOR_H
