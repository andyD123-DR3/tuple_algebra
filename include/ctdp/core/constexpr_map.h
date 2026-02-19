// core/constexpr_map.h - Fixed-capacity sorted associative map for compile-time DP
// Part of the compile-time DP library (C++20)
//
// DESIGN DECISION: Sorted vector vs hash table
// 
// Sorted vector (CHOSEN):
// - O(log n) find via binary search
// - O(n) insert (acceptable for DP subproblem counts < 1000)
// - Deterministic iteration order (critical for reproducibility)
// - Simpler implementation, cache-friendly
// - No hash function required (avoids constexpr hash complexity)
//
// Hash table (REJECTED for now):
// - O(1) average find/insert
// - Non-deterministic iteration (iteration order depends on hash values)
// - More complex implementation (open addressing, probing, rehashing)
// - Requires constexpr hash function (non-trivial for custom key types)
//
// For DP memoization with bounded subproblem counts, sorted vector wins.

#ifndef CTDP_CORE_CONSTEXPR_MAP_H
#define CTDP_CORE_CONSTEXPR_MAP_H

#include <algorithm>
#include <array>
#include <concepts>
#include <stdexcept>
#include <utility>

namespace ctdp {

/// Fixed-capacity sorted associative map for compile-time memoization.
///
/// Implements a sorted vector of (key, value) pairs with binary search lookup.
/// Designed for DP memoization where:
/// - Subproblem count is bounded and known at compile time
/// - Reproducibility requires deterministic iteration order
/// - O(log n) lookup is acceptable for bounded n (< 1000)
///
/// Template parameters:
/// - Key: Must be totally ordered (std::totally_ordered)
/// - Value: Arbitrary value type (regular type recommended)
/// - MaxN: Maximum capacity (fixed at compile time)
///
/// Complexity:
/// - find: O(log n) binary search
/// - insert: O(n) linear search + shift (pessimistic for unsorted inserts)
/// - iteration: O(n), deterministic ascending key order
///
/// Example:
/// ```cpp
/// constexpr_map<std::pair<size_t, size_t>, double, 100> memo;
/// memo.insert({i, j}, cost);
/// if (auto it = memo.find({i, j}); it != memo.end()) {
///     return it->second; // Cache hit
/// }
/// ```
template<typename Key, typename Value, size_t MaxN>
    requires std::totally_ordered<Key>
class constexpr_map {
public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<Key, Value>;
    using size_type = size_t;
    
private:
    // Internal storage: value-initialized std::array (consistent with constexpr_vector)
    std::array<std::pair<Key, Value>, MaxN> storage_{};
    size_type size_ = 0;
    
public:
    // =============================================================================
    // Iterator Support
    // =============================================================================
    
    class iterator {
        std::pair<Key, Value>* ptr_;
        
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<Key, Value>;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::random_access_iterator_tag;
        
        constexpr iterator() : ptr_(nullptr) {}
        constexpr explicit iterator(std::pair<Key, Value>* p) : ptr_(p) {}
        
        constexpr auto operator*() const -> value_type& {
            return *ptr_;
        }
        
        constexpr auto operator->() const -> value_type* {
            return ptr_;
        }
        
        constexpr iterator& operator++() { ++ptr_; return *this; }
        constexpr iterator operator++(int) { auto tmp = *this; ++ptr_; return tmp; }
        constexpr iterator& operator--() { --ptr_; return *this; }
        constexpr iterator operator--(int) { auto tmp = *this; --ptr_; return tmp; }
        
        constexpr iterator& operator+=(difference_type n) { ptr_ += n; return *this; }
        constexpr iterator& operator-=(difference_type n) { ptr_ -= n; return *this; }
        
        constexpr iterator operator+(difference_type n) const { return iterator(ptr_ + n); }
        constexpr iterator operator-(difference_type n) const { return iterator(ptr_ - n); }
        
        constexpr difference_type operator-(iterator const& other) const {
            return ptr_ - other.ptr_;
        }
        
        constexpr auto operator[](difference_type n) const -> value_type& {
            return ptr_[n];
        }
        
        constexpr bool operator==(iterator const& other) const = default;
        constexpr auto operator<=>(iterator const& other) const = default;
    };
    
    class const_iterator {
        std::pair<Key, Value> const* ptr_;
        
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<Key, Value>;
        using pointer = value_type const*;
        using reference = value_type const&;
        using iterator_category = std::random_access_iterator_tag;
        
        constexpr const_iterator() : ptr_(nullptr) {}
        constexpr explicit const_iterator(std::pair<Key, Value> const* p) : ptr_(p) {}
        constexpr const_iterator(iterator it) : ptr_(it.operator->()) {}
        
        constexpr auto operator*() const -> value_type const& {
            return *ptr_;
        }
        
        constexpr auto operator->() const -> value_type const* {
            return ptr_;
        }
        
        constexpr const_iterator& operator++() { ++ptr_; return *this; }
        constexpr const_iterator operator++(int) { auto tmp = *this; ++ptr_; return tmp; }
        constexpr const_iterator& operator--() { --ptr_; return *this; }
        constexpr const_iterator operator--(int) { auto tmp = *this; --ptr_; return tmp; }
        
        constexpr const_iterator& operator+=(difference_type n) { ptr_ += n; return *this; }
        constexpr const_iterator& operator-=(difference_type n) { ptr_ -= n; return *this; }
        
        constexpr const_iterator operator+(difference_type n) const {
            return const_iterator(ptr_ + n);
        }
        constexpr const_iterator operator-(difference_type n) const {
            return const_iterator(ptr_ - n);
        }
        
        constexpr difference_type operator-(const_iterator const& other) const {
            return ptr_ - other.ptr_;
        }
        
        constexpr auto operator[](difference_type n) const -> value_type const& {
            return ptr_[n];
        }
        
        constexpr bool operator==(const_iterator const& other) const = default;
        constexpr auto operator<=>(const_iterator const& other) const = default;
    };
    
    // =============================================================================
    // Constructors
    // =============================================================================
    
    constexpr constexpr_map() = default;
    
    // =============================================================================
    // Capacity
    // =============================================================================
    
    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0;
    }
    
    [[nodiscard]] constexpr size_type size() const noexcept {
        return size_;
    }
    
    [[nodiscard]] constexpr size_type max_size() const noexcept {
        return MaxN;
    }
    
    [[nodiscard]] constexpr size_type capacity() const noexcept {
        return MaxN;
    }
    
    // =============================================================================
    // Iterators
    // =============================================================================
    
    constexpr iterator begin() noexcept {
        return iterator(storage_.data());
    }
    
    constexpr const_iterator begin() const noexcept {
        return const_iterator(storage_.data());
    }
    
    constexpr const_iterator cbegin() const noexcept {
        return const_iterator(storage_.data());
    }
    
    constexpr iterator end() noexcept {
        return iterator(storage_.data() + size_);
    }
    
    constexpr const_iterator end() const noexcept {
        return const_iterator(storage_.data() + size_);
    }
    
    constexpr const_iterator cend() const noexcept {
        return const_iterator(storage_.data() + size_);
    }
    
    // =============================================================================
    // Lookup
    // =============================================================================
    
    /// Binary search for key. Returns iterator to element or end() if not found.
    constexpr iterator find(Key const& k) {
        auto it = std::lower_bound(begin(), end(), k,
            [](auto const& entry, Key const& key) {
                return entry.first < key;
            });
        
        if (it != end() && it->first == k) {
            return it;
        }
        return end();
    }
    
    constexpr const_iterator find(Key const& k) const {
        auto it = std::lower_bound(begin(), end(), k,
            [](auto const& entry, Key const& key) {
                return entry.first < key;
            });
        
        if (it != end() && it->first == k) {
            return it;
        }
        return end();
    }
    
    /// Check if key exists in map.
    constexpr bool contains(Key const& k) const {
        return find(k) != end();
    }
    
    /// Number of elements with key k (0 or 1 for unique keys).
    constexpr size_type count(Key const& k) const {
        return contains(k) ? 1 : 0;
    }
    
    // =============================================================================
    // Modifiers
    // =============================================================================
    
    /// Insert key-value pair if key doesn't exist. Returns (iterator, bool).
    /// - iterator: Points to inserted element or existing element with same key
    /// - bool: true if insertion occurred, false if key already existed
    ///
    /// Complexity: O(log n) find + O(n) shift in worst case
    constexpr std::pair<iterator, bool> insert(Key const& k, Value const& v) {
        // Binary search for insertion point
        auto it = std::lower_bound(begin(), end(), k,
            [](auto const& entry, Key const& key) {
                return entry.first < key;
            });
        
        // Key already exists
        if (it != end() && it->first == k) {
            return {it, false};
        }
        
        // Capacity check
        if (size_ >= MaxN) {
            throw std::length_error("constexpr_map: capacity exceeded");
        }
        
        // Shift elements to make room
        auto pos = it - begin();
        for (size_type i = size_; i > static_cast<size_type>(pos); --i) {
            storage_[i] = storage_[i - 1];
        }
        
        // Insert new element
        storage_[pos] = std::pair<Key, Value>{k, v};
        ++size_;
        
        return {iterator(storage_.data() + pos), true};
    }
    
    /// Insert or assign value. Returns (iterator, bool).
    /// - If key exists: update value, return (iterator to element, false)
    /// - If key doesn't exist: insert, return (iterator to element, true)
    constexpr std::pair<iterator, bool> insert_or_assign(Key const& k, Value const& v) {
        auto [it, inserted] = insert(k, v);
        if (!inserted) {
            it->second = v;
        }
        return {it, inserted};
    }
    
    /// Emplace key-value pair (perfect forwarding support).
    template<typename... Args>
    constexpr std::pair<iterator, bool> emplace(Key const& k, Args&&... args) {
        return insert(k, Value(std::forward<Args>(args)...));
    }
    
    /// Erase element at iterator position.
    constexpr iterator erase(iterator pos) {
        auto idx = pos - begin();
        
        // Shift elements to close gap
        for (size_type i = idx; i < size_ - 1; ++i) {
            storage_[i] = storage_[i + 1];
        }
        
        --size_;
        return iterator(storage_.data() + idx);
    }
    
    /// Erase element by key. Returns number of elements removed (0 or 1).
    constexpr size_type erase(Key const& k) {
        auto it = find(k);
        if (it == end()) {
            return 0;
        }
        erase(it);
        return 1;
    }
    
    /// Clear all elements.
    constexpr void clear() noexcept {
        size_ = 0;
    }
    
    // =============================================================================
    // Element Access
    // =============================================================================
    
    /// Access or insert element with bounds checking.
    /// - If key exists: return reference to value
    /// - If key doesn't exist: insert default-constructed value, return reference
    constexpr Value& operator[](Key const& k) {
        auto [it, inserted] = insert(k, Value{});
        return it->second;
    }
    
    /// Access element with bounds checking. Throws if key doesn't exist.
    constexpr Value& at(Key const& k) {
        auto it = find(k);
        if (it == end()) {
            throw std::out_of_range("constexpr_map::at: key not found");
        }
        return it->second;
    }
    
    constexpr Value const& at(Key const& k) const {
        auto it = find(k);
        if (it == end()) {
            throw std::out_of_range("constexpr_map::at: key not found");
        }
        return it->second;
    }
};

} // namespace ctdp

#endif // CTDP_CORE_CONSTEXPR_MAP_H
