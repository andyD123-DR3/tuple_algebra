// ctdp/space/permutation.h — Permutation ordinate type
//
// permutation_value<N>: a permutation of N elements [0, N).
// permutation_desc<N>:  dimension_descriptor for ordering search.
//
// Feature encoding: pairwise precedence — N*(N-1)/2 binary features,
// one per pair (i,j) where i<j, indicating whether element i appears
// before element j. Kendall tau distance = Hamming distance of features.
//
// Usage:
//   auto perm = make_permutation<5>("lane_order");
//   // perm.cardinality() == 120 (5!)
//   // perm.feature_width() == 10 (5*4/2)
//   // perm.value_at(0) == identity {0,1,2,3,4}
//   // perm.value_at(119) == reverse {4,3,2,1,0}
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_PERMUTATION_H
#define CTDP_SPACE_PERMUTATION_H

#include "ctdp/space/concepts.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// Compile-time factorial
// ═══════════════════════════════════════════════════════════════════════

namespace detail {

template <std::size_t N>
constexpr std::size_t factorial() {
    std::size_t r = 1;
    for (std::size_t i = 2; i <= N; ++i) r *= i;
    return r;
}

// Factorial lookup: fact[k] = k!
template <std::size_t N>
constexpr auto make_factorial_table() {
    std::array<std::size_t, N + 1> t{};
    t[0] = 1;
    for (std::size_t i = 1; i <= N; ++i) t[i] = t[i - 1] * i;
    return t;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// permutation_value<N> — a permutation of {0, 1, ..., N-1}
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
struct permutation_value {
    std::array<std::size_t, N> order;

    // Default: identity permutation {0, 1, ..., N-1}
    constexpr permutation_value() : order{} {
        for (std::size_t i = 0; i < N; ++i) order[i] = i;
    }

    // Construct from array
    constexpr explicit permutation_value(std::array<std::size_t, N> arr)
        : order(arr) {}

    constexpr std::size_t operator[](std::size_t i) const { return order[i]; }

    constexpr bool is_valid() const {
        std::array<bool, N> seen{};
        for (std::size_t i = 0; i < N; ++i) {
            if (order[i] >= N) return false;
            if (seen[order[i]]) return false;
            seen[order[i]] = true;
        }
        return true;
    }

    constexpr bool is_identity() const {
        for (std::size_t i = 0; i < N; ++i)
            if (order[i] != i) return false;
        return true;
    }

    constexpr auto operator<=>(const permutation_value&) const = default;
};

// ═══════════════════════════════════════════════════════════════════════
// permutation_desc<N> — dimension descriptor for permutation search
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
struct permutation_desc {
    static_assert(N >= 1, "Permutation requires N >= 1");
    static_assert(N <= 20, "N! overflows std::size_t for N > 20");

    static constexpr dim_kind kind = dim_kind::permutation;
    using value_type = permutation_value<N>;

    std::string_view name;

    // ── Core protocol ───────────────────────────────────────────────

    constexpr std::size_t cardinality() const {
        return detail::factorial<N>();
    }

    constexpr encoding_hint default_encoding() const {
        return encoding_hint::precedence;
    }

    // ── Factoradic unrank: index → permutation (lexicographic) ──────

    constexpr value_type value_at(std::size_t index) const {
        assert(index < cardinality());
        constexpr auto fact = detail::make_factorial_table<N>();

        // Available elements
        std::array<std::size_t, N> avail{};
        for (std::size_t i = 0; i < N; ++i) avail[i] = i;

        std::array<std::size_t, N> result{};
        std::size_t remaining = N;
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t f = fact[remaining - 1];
            std::size_t digit = index / f;
            index %= f;
            result[i] = avail[digit];
            // Remove chosen element by shifting
            for (std::size_t j = digit; j + 1 < remaining; ++j)
                avail[j] = avail[j + 1];
            --remaining;
        }
        return value_type(result);
    }

    // ── Factoradic rank: permutation → index ────────────────────────

    constexpr std::size_t index_of(const value_type& v) const {
        assert(v.is_valid());
        constexpr auto fact = detail::make_factorial_table<N>();

        std::array<std::size_t, N> avail{};
        for (std::size_t i = 0; i < N; ++i) avail[i] = i;

        std::size_t index = 0;
        std::size_t remaining = N;
        for (std::size_t i = 0; i < N; ++i) {
            // Find position of v[i] in available elements
            std::size_t digit = 0;
            for (std::size_t j = 0; j < remaining; ++j) {
                if (avail[j] == v[i]) { digit = j; break; }
            }
            index += digit * fact[remaining - 1];
            // Remove chosen element
            for (std::size_t j = digit; j + 1 < remaining; ++j)
                avail[j] = avail[j + 1];
            --remaining;
        }
        return index;
    }

    // ── Feature encoding: pairwise precedence ───────────────────────

    constexpr std::size_t feature_width() const {
        return N * (N - 1) / 2;
    }

    void write_features(value_type val, double* out) const {
        // Precompute inverse: pos[element] = position
        std::array<std::size_t, N> pos{};
        for (std::size_t i = 0; i < N; ++i)
            pos[val[i]] = i;

        std::size_t k = 0;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j)
                out[k++] = (pos[i] < pos[j]) ? 1.0 : 0.0;
    }

    // ── Encoding override ───────────────────────────────────────────

    constexpr permutation_desc encoded_as(encoding_hint h) const {
        if (h != encoding_hint::precedence)
            throw std::invalid_argument(
                "permutation_desc only supports precedence encoding");
        return *this;
    }

    // ── Neighbour generation (optional) ─────────────────────────────

    template <typename F>
    void neighbours(const value_type& v, F&& fn) const {
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j) {
                auto nb = v;
                std::swap(nb.order[i], nb.order[j]);
                fn(nb);
            }
    }
};

// ── Factory ─────────────────────────────────────────────────────────

template <std::size_t N>
constexpr auto make_permutation(std::string_view name) {
    return permutation_desc<N>{name};
}

// ── Concept check ───────────────────────────────────────────────────

static_assert(dimension_descriptor<permutation_desc<1>>);
static_assert(dimension_descriptor<permutation_desc<4>>);
static_assert(dimension_descriptor<permutation_desc<8>>);

} // namespace ctdp::space

#endif // CTDP_SPACE_PERMUTATION_H
