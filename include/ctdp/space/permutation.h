#ifndef CTDP_SPACE_PERMUTATION_H
#define CTDP_SPACE_PERMUTATION_H

// ctdp/space/permutation.h — Permutation ordinate type
//
// permutation_value<N>: a permutation of N items [0, N).
// permutation_desc<N>:  dimension_descriptor for ordering search.
//
// Feature encoding: pairwise precedence — N*(N-1)/2 binary features,
// one per pair (i,j) where i<j, indicating whether i appears before j.
// Hamming distance in feature space matches Kendall-tau inversion count.

#include "ctdp/space/concepts.h"

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace ctdp::space {

namespace detail {

template <std::size_t N>
constexpr std::size_t factorial() {
    std::size_t r = 1;
    for (std::size_t i = 2; i <= N; ++i) r *= i;
    return r;
}

template <std::size_t N>
constexpr auto make_factorial_table() {
    std::array<std::size_t, N + 1> t{};
    t[0] = 1;
    for (std::size_t i = 1; i <= N; ++i) t[i] = t[i - 1] * i;
    return t;
}

} // namespace detail

template <std::size_t N>
struct permutation_value {
    std::array<std::size_t, N> order{};

    constexpr permutation_value() {
        for (std::size_t i = 0; i < N; ++i) order[i] = i;
    }

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
        for (std::size_t i = 0; i < N; ++i) {
            if (order[i] != i) return false;
        }
        return true;
    }

    constexpr bool contains(std::size_t item) const {
        for (auto x : order) {
            if (x == item) return true;
        }
        return false;
    }

    constexpr auto operator<=>(const permutation_value&) const = default;
};

template <std::size_t N>
struct permutation_desc {
    static_assert(N >= 1, "permutation_desc requires N >= 1");
    static_assert(N <= 20, "permutation_desc: N! overflows std::size_t for N > 20");

    static constexpr dim_kind kind = dim_kind::permutation;
    using value_type = permutation_value<N>;

    std::string_view name;
    encoding_hint encoding = encoding_hint::precedence;

    constexpr explicit permutation_desc(std::string_view n) : name(n) {}

    constexpr permutation_desc encoded_as(encoding_hint e) const {
        if (e != encoding_hint::precedence) {
            throw std::invalid_argument(
                "permutation_desc only supports precedence encoding");
        }
        auto copy = *this;
        copy.encoding = e;
        return copy;
    }

    constexpr encoding_hint default_encoding() const { return encoding; }

    constexpr std::size_t cardinality() const {
        return detail::factorial<N>();
    }

    constexpr value_type value_at(std::size_t index) const {
        if (index >= cardinality()) {
            throw std::out_of_range("permutation_desc::value_at(): index out of range");
        }

        constexpr auto fact = detail::make_factorial_table<N>();
        std::array<std::size_t, N> avail{};
        for (std::size_t i = 0; i < N; ++i) avail[i] = i;

        std::array<std::size_t, N> result{};
        std::size_t remaining = N;
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t f = fact[remaining - 1];
            std::size_t digit = index / f;
            index %= f;
            result[i] = avail[digit];
            for (std::size_t j = digit; j + 1 < remaining; ++j)
                avail[j] = avail[j + 1];
            --remaining;
        }
        return value_type(result);
    }

    constexpr bool contains(const value_type& v) const {
        return v.is_valid();
    }

    constexpr std::size_t index_of(const value_type& v) const {
        if (!v.is_valid()) return cardinality();

        constexpr auto fact = detail::make_factorial_table<N>();
        std::array<std::size_t, N> avail{};
        for (std::size_t i = 0; i < N; ++i) avail[i] = i;

        std::size_t index = 0;
        std::size_t remaining = N;
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t digit = 0;
            for (std::size_t j = 0; j < remaining; ++j) {
                if (avail[j] == v[i]) {
                    digit = j;
                    break;
                }
            }
            index += digit * fact[remaining - 1];
            for (std::size_t j = digit; j + 1 < remaining; ++j)
                avail[j] = avail[j + 1];
            --remaining;
        }
        return index;
    }

    constexpr std::size_t feature_width() const {
        return N * (N - 1) / 2;
    }

    void write_features(value_type val, double* out) const {
        std::array<std::size_t, N> pos{};
        for (std::size_t i = 0; i < N; ++i)
            pos[val[i]] = i;

        std::size_t k = 0;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                out[k++] = (pos[i] < pos[j]) ? 1.0 : 0.0;
            }
        }
    }

    template <typename F>
    void neighbours(const value_type& v, F&& fn) const {
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                auto nb = v;
                std::swap(nb.order[i], nb.order[j]);
                fn(nb);
            }
        }
    }
};

template <std::size_t N>
constexpr auto make_permutation(std::string_view name) {
    return permutation_desc<N>{name};
}

static_assert(dimension_descriptor<permutation_desc<1>>);
static_assert(dimension_descriptor<permutation_desc<4>>);
static_assert(dimension_descriptor<permutation_desc<8>>);

} // namespace ctdp::space

#endif // CTDP_SPACE_PERMUTATION_H
