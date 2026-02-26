#ifndef CTDP_SPACE_DESCRIPTOR_H
#define CTDP_SPACE_DESCRIPTOR_H

// ctdp v0.7.2 — Dimension descriptors + descriptor_space + feature_bridge
//
// Predefined ordinate types with defaults:
//   positive_int("N")           [1,64] step 1    raw
//   power_2("TM")               [2,64]           log2
//   make_int_set("X", {vals})   explicit          raw
//   bool_flag("V")              {false,true}      binary
//   make_enum_vals("S", {vals}) explicit enum      one_hot
//
// descriptor_space: canonical space generated from descriptors.
//   Point type = std::tuple<ValueTypes...>  (heterogeneous, tuple-like)
//   enumerate(fn) = Cartesian product       (callback, constexpr-safe)
//   Named access: space.get_dim_as_int(point, "TK")
//
// feature_bridge: Layer 2 encoding, descriptor → span<double>.
//   default_bridge(space) reads encoding hints from descriptors.
//   bridge.with(dim_idx, hint) returns new bridge with override.
//   bridge.write_features(point, span<double>).
//   Bridge::num_features = model vector length (≠ Space::rank when one_hot used).

#include "ctdp/space/space.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// dim_kind + encoding_hint
// ═══════════════════════════════════════════════════════════════════════

enum class dim_kind : std::uint8_t {
    int_set, power_of_two, int_range, bool_flag, enum_set,
};

enum class encoding_hint : std::uint8_t {
    raw, log2, one_hot, binary, normalised,
};

// ═══════════════════════════════════════════════════════════════════════
// Predefined ordinate types
// ═══════════════════════════════════════════════════════════════════════

struct positive_int {
    std::string_view name;
    int lo = 1, hi = 64, step = 1;
    encoding_hint encoding = encoding_hint::raw;
    static constexpr dim_kind kind = dim_kind::int_range;
    using value_type = int;

    constexpr explicit positive_int(std::string_view n) : name(n) {}
    constexpr positive_int(std::string_view n, int l, int h) : name(n), lo(l), hi(h) {}
    constexpr positive_int(std::string_view n, int l, int h, int s)
        : name(n), lo(l), hi(h), step(s) {}

    constexpr positive_int encoded_as(encoding_hint e) const {
        auto c = *this; c.encoding = e; return c;
    }
    constexpr encoding_hint default_encoding() const { return encoding; }
    constexpr std::size_t cardinality() const {
        if (step <= 0 || hi < lo) return 0;
        return static_cast<std::size_t>((hi - lo) / step) + 1;
    }
    constexpr int value_at(std::size_t i) const {
        return lo + static_cast<int>(i) * step;
    }
    constexpr bool contains(int v) const {
        return v >= lo && v <= hi && (v - lo) % step == 0;
    }
    constexpr std::size_t index_of(int v) const {
        if (!contains(v)) return cardinality();
        return static_cast<std::size_t>((v - lo) / step);
    }
};

struct power_2 {
    std::string_view name;
    int lo = 2, hi = 64;
    encoding_hint encoding = encoding_hint::log2;
    static constexpr dim_kind kind = dim_kind::power_of_two;
    using value_type = int;

    // Precondition: lo and hi must be powers of two, lo > 0, lo <= hi
    constexpr explicit power_2(std::string_view n) : name(n) {
        validate();
    }
    constexpr power_2(std::string_view n, int l, int h) : name(n), lo(l), hi(h) {
        validate();
    }

    constexpr power_2 encoded_as(encoding_hint e) const {
        auto c = *this; c.encoding = e; return c;
    }
    constexpr encoding_hint default_encoding() const { return encoding; }
    constexpr std::size_t cardinality() const {
        std::size_t n = 0;
        for (int v = lo; v <= hi; v *= 2) ++n;
        return n;
    }
    constexpr int value_at(std::size_t i) const {
        int v = lo; for (std::size_t j = 0; j < i; ++j) v *= 2; return v;
    }
    constexpr bool contains(int v) const {
        return v >= lo && v <= hi && v > 0 && (v & (v-1)) == 0;
    }
    constexpr std::size_t index_of(int v) const {
        if (!contains(v)) return cardinality();
        std::size_t idx = 0;
        for (int x = lo; x < v; x *= 2) ++idx;
        return idx;
    }

private:
    static constexpr bool is_pow2(int v) { return v > 0 && (v & (v-1)) == 0; }
    constexpr void validate() const {
        // In constexpr context, throw is not allowed → triggers compile error.
        // At runtime, throws std::logic_error with diagnostic.
        if (!is_pow2(lo) || !is_pow2(hi) || lo > hi) {
            throw std::logic_error("power_2: lo/hi must be powers of two and lo <= hi");
        }
    }
};

template <std::size_t N>
struct int_set {
    std::string_view name;
    std::array<int, N> values{};
    encoding_hint encoding = encoding_hint::raw;
    static constexpr dim_kind kind = dim_kind::int_set;
    using value_type = int;

    constexpr int_set encoded_as(encoding_hint e) const {
        auto c = *this; c.encoding = e; return c;
    }
    constexpr encoding_hint default_encoding() const { return encoding; }
    constexpr std::size_t cardinality() const { return N; }
    constexpr int value_at(std::size_t i) const { return values[i]; }
    constexpr bool contains(int v) const {
        for (std::size_t i = 0; i < N; ++i) if (values[i] == v) return true;
        return false;
    }
    constexpr std::size_t index_of(int v) const {
        for (std::size_t i = 0; i < N; ++i) if (values[i] == v) return i;
        return N;
    }
};

template <std::size_t N>
constexpr auto make_int_set(std::string_view name, const int (&vals)[N]) {
    int_set<N> d; d.name = name;
    for (std::size_t i = 0; i < N; ++i) d.values[i] = vals[i];
    return d;
}

struct bool_flag {
    std::string_view name;
    encoding_hint encoding = encoding_hint::binary;
    static constexpr dim_kind kind = dim_kind::bool_flag;
    using value_type = bool;

    constexpr explicit bool_flag(std::string_view n) : name(n) {}
    constexpr bool_flag encoded_as(encoding_hint e) const {
        auto c = *this; c.encoding = e; return c;
    }
    constexpr encoding_hint default_encoding() const { return encoding; }
    static constexpr std::size_t cardinality() { return 2; }
    static constexpr bool value_at(std::size_t i) { return i != 0; }
    static constexpr bool contains(bool) { return true; }
    static constexpr std::size_t index_of(bool v) { return v ? 1u : 0u; }
};

template <typename E, std::size_t N>
    requires std::is_enum_v<E>
struct enum_vals {
    std::string_view name;
    std::array<E, N> values{};
    encoding_hint encoding = encoding_hint::one_hot;
    static constexpr dim_kind kind = dim_kind::enum_set;
    using value_type = E;

    constexpr enum_vals encoded_as(encoding_hint e) const {
        auto c = *this; c.encoding = e; return c;
    }
    constexpr encoding_hint default_encoding() const { return encoding; }
    constexpr std::size_t cardinality() const { return N; }
    constexpr E value_at(std::size_t i) const { return values[i]; }
    constexpr bool contains(E v) const {
        for (std::size_t i = 0; i < N; ++i) if (values[i] == v) return true;
        return false;
    }
    constexpr std::size_t index_of(E v) const {
        for (std::size_t i = 0; i < N; ++i) if (values[i] == v) return i;
        return N;
    }
    constexpr int value_as_int(std::size_t i) const {
        return static_cast<int>(values[i]);
    }
};

template <typename E, std::size_t N> requires std::is_enum_v<E>
constexpr auto make_enum_vals(std::string_view name, const E (&vals)[N]) {
    enum_vals<E, N> d; d.name = name;
    for (std::size_t i = 0; i < N; ++i) d.values[i] = vals[i];
    return d;
}

// ═══════════════════════════════════════════════════════════════════════
// dimension_descriptor concept
// ═══════════════════════════════════════════════════════════════════════

template <typename D>
concept dimension_descriptor = requires(const D& d) {
    { d.name } -> std::convertible_to<std::string_view>;
    { D::kind } -> std::convertible_to<dim_kind>;
    { d.default_encoding() } -> std::convertible_to<encoding_hint>;
    { d.cardinality() } -> std::convertible_to<std::size_t>;
    typename D::value_type;
};

static_assert(dimension_descriptor<positive_int>);
static_assert(dimension_descriptor<power_2>);
static_assert(dimension_descriptor<int_set<3>>);
static_assert(dimension_descriptor<bool_flag>);

// ═══════════════════════════════════════════════════════════════════════
// descriptor_space — canonical space from descriptors
//
// Point type: std::tuple<D0::value_type, D1::value_type, ...>
// Enumerate: Cartesian product, callback, constexpr-safe.
// This is the primary space representation; hand-written spaces
// are adapters that delegate to descriptor_space or use filter_section.
// ═══════════════════════════════════════════════════════════════════════

template <typename... Descriptors>
    requires (dimension_descriptor<Descriptors> && ...)
struct descriptor_space {
    static constexpr std::size_t rank = sizeof...(Descriptors);
    using point_type = std::tuple<typename Descriptors::value_type...>;
    using descriptors_tuple = std::tuple<Descriptors...>;

    std::string_view name_;
    descriptors_tuple descs_;

    constexpr descriptor_space(std::string_view space_name, Descriptors... ds)
        : name_(space_name), descs_(ds...) {}

    constexpr std::string_view space_name() const { return name_; }

    static constexpr auto dimension_names_from(const descriptors_tuple& d) {
        std::array<std::string_view, rank> names{};
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((names[Is] = std::get<Is>(d).name), ...);
        }(std::make_index_sequence<rank>{});
        return names;
    }

    constexpr auto dimension_names() const {
        return dimension_names_from(descs_);
    }

    constexpr std::size_t cardinality() const {
        std::size_t result = 1;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((result = detail::saturating_mul(result,
                        std::get<Is>(descs_).cardinality())), ...);
        }(std::make_index_sequence<rank>{});
        return result;
    }

    // Callback enumerate — constexpr-safe Cartesian product
    template <typename F>
    constexpr void enumerate(F&& fn) const {
        point_type pt{};
        enumerate_impl<0>(fn, pt);
    }

    // Positional access
    template <std::size_t I>
    static constexpr auto get(const point_type& pt) {
        return std::get<I>(pt);
    }

    // Runtime named access
    int get_dim_as_int(const point_type& pt, std::string_view dim) const {
        auto names = dimension_names();
        int result = -1;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((names[Is] == dim
                ? (result = static_cast<int>(std::get<Is>(pt)), true)
                : false), ...);
        }(std::make_index_sequence<rank>{});
        return result;
    }

    // Query helpers
    constexpr const descriptors_tuple& descriptors() const { return descs_; }

    constexpr encoding_hint encoding_for(std::string_view dim) const {
        encoding_hint r = encoding_hint::raw;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(descs_).name == dim
                ? (r = std::get<Is>(descs_).default_encoding(), true) : false), ...);
        }(std::make_index_sequence<rank>{});
        return r;
    }

    constexpr dim_kind kind_of(std::string_view dim) const {
        dim_kind r = dim_kind::int_set;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(descs_).name == dim
                ? (r = std::remove_cvref_t<decltype(std::get<Is>(descs_))>::kind, true)
                : false), ...);
        }(std::make_index_sequence<rank>{});
        return r;
    }

    constexpr std::size_t cardinality_of(std::string_view dim) const {
        std::size_t r = 0;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(descs_).name == dim
                ? (r = std::get<Is>(descs_).cardinality(), true) : false), ...);
        }(std::make_index_sequence<rank>{});
        return r;
    }

    constexpr std::size_t feature_width() const {
        std::size_t w = 0;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((w += (std::get<Is>(descs_).default_encoding() == encoding_hint::one_hot
                    ? std::get<Is>(descs_).cardinality() : 1)), ...);
        }(std::make_index_sequence<rank>{});
        return w;
    }

    // ── Indexable tier: point_at(flat_index) ──────────────────────────
    //
    // Closed-form: decompose flat index into per-dimension indices via
    // div/mod (row-major, last dimension varies fastest).
    constexpr point_type point_at(std::size_t flat) const {
        point_type pt{};
        point_at_impl(flat, pt, std::make_index_sequence<rank>{});
        return pt;
    }

    // ── Factored tier: per-dimension enumeration ─────────────────────
    //
    // Beam search, branch-and-bound, etc. can walk dimensions one at a
    // time, extending partial assignments.

    static constexpr std::size_t num_dims() { return rank; }

    constexpr std::size_t dim_cardinality(std::size_t d) const {
        std::size_t r = 0;
        dim_dispatch(d, [&](const auto& desc) { r = desc.cardinality(); });
        return r;
    }

    // dim_value<I>(j): typed access to j-th value of dimension I.
    template <std::size_t I>
    constexpr auto dim_value(std::size_t j) const {
        return std::get<I>(descs_).value_at(j);
    }

private:
    // point_at helper: decompose flat index, last dimension fastest
    template <std::size_t... Is>
    constexpr void point_at_impl(std::size_t flat, point_type& pt,
                                  std::index_sequence<Is...>) const {
        // Compute strides (product of cardinalities of later dimensions)
        std::array<std::size_t, rank> strides{};
        strides[rank - 1] = 1;
        for (std::size_t i = rank - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * dim_card_at(i);
        }
        // Decompose
        ((std::get<Is>(pt) = std::get<Is>(descs_).value_at(
            (flat / strides[Is]) % std::get<Is>(descs_).cardinality())), ...);
    }

    // Helper: cardinality of dimension d (runtime index)
    constexpr std::size_t dim_card_at(std::size_t d) const {
        std::size_t r = 0;
        dim_dispatch(d, [&](const auto& desc) { r = desc.cardinality(); });
        return r;
    }

    // Runtime dimension dispatch: call fn with the descriptor at index d
    template <typename Fn>
    constexpr void dim_dispatch(std::size_t d, Fn&& fn) const {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((void)(Is == d ? (fn(std::get<Is>(descs_)), true) : false), ...);
        }(std::make_index_sequence<rank>{});
    }

    template <std::size_t I, typename F>
    constexpr void enumerate_impl(F& fn, point_type& pt) const {
        if constexpr (I == rank) {
            fn(std::as_const(pt));
        } else {
            const auto& desc = std::get<I>(descs_);
            for (std::size_t j = 0; j < desc.cardinality(); ++j) {
                std::get<I>(pt) = desc.value_at(j);
                enumerate_impl<I + 1>(fn, pt);
            }
        }
    }
};

// Deduction guide
template <typename... Ds>
descriptor_space(std::string_view, Ds...) -> descriptor_space<Ds...>;

// ═══════════════════════════════════════════════════════════════════════
// Descriptor product — flattened tuple, concatenated descriptors
//
// Unlike product_space (pair-based), this produces a single flat
// descriptor_space whose point type is tuple_cat of both.
// This is the preferred product for descriptor-based spaces.
// ═══════════════════════════════════════════════════════════════════════

template <typename... Ds1, typename... Ds2>
auto descriptor_product(const descriptor_space<Ds1...>& a,
                         const descriptor_space<Ds2...>& b,
                         std::string_view name) {
    return std::apply([&](const auto&... d1) {
        return std::apply([&](const auto&... d2) {
            return descriptor_space(name, d1..., d2...);
        }, b.descriptors());
    }, a.descriptors());
}

// ═══════════════════════════════════════════════════════════════════════
// described_space_traits — attach descriptors to hand-written spaces
//
// Specialise this for hand-written struct-point spaces to enable
// descriptor queries (encoding, kind, cardinality) without changing
// the space itself.
// ═══════════════════════════════════════════════════════════════════════

template <typename Space> struct described_space_traits;

template <typename Space>
concept described_space = search_space<Space> && requires {
    { described_space_traits<Space>::descriptors() };
};

// Convenience queries for described_space
namespace detail {
template <typename... Ds>
constexpr encoding_hint find_enc(std::string_view n, const std::tuple<Ds...>& d) {
    encoding_hint r = encoding_hint::raw;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(d).name == n
            ? (r = std::get<Is>(d).default_encoding(), true) : false), ...);
    }(std::index_sequence_for<Ds...>{});
    return r;
}
template <typename... Ds>
constexpr dim_kind find_knd(std::string_view n, const std::tuple<Ds...>& d) {
    dim_kind r = dim_kind::int_set;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(d).name == n
            ? (r = std::remove_cvref_t<decltype(std::get<Is>(d))>::kind, true)
            : false), ...);
    }(std::index_sequence_for<Ds...>{});
    return r;
}
template <typename... Ds>
constexpr std::size_t find_card(std::string_view n, const std::tuple<Ds...>& d) {
    std::size_t r = 0;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((std::get<Is>(d).name == n
            ? (r = std::get<Is>(d).cardinality(), true) : false), ...);
    }(std::index_sequence_for<Ds...>{});
    return r;
}
template <typename... Ds>
constexpr std::size_t total_card(const std::tuple<Ds...>& d) {
    std::size_t r = 1;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((r = saturating_mul(r, std::get<Is>(d).cardinality())), ...);
    }(std::index_sequence_for<Ds...>{});
    return r;
}
template <typename... Ds>
constexpr std::size_t feat_width(const std::tuple<Ds...>& d) {
    std::size_t w = 0;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((w += (std::get<Is>(d).default_encoding() == encoding_hint::one_hot
                ? std::get<Is>(d).cardinality() : 1)), ...);
    }(std::index_sequence_for<Ds...>{});
    return w;
}
} // namespace detail

template <described_space S>
constexpr encoding_hint dim_encoding(std::string_view n) {
    return detail::find_enc(n, described_space_traits<S>::descriptors());
}
template <described_space S>
constexpr dim_kind dim_type(std::string_view n) {
    return detail::find_knd(n, described_space_traits<S>::descriptors());
}
template <described_space S>
constexpr std::size_t dim_cardinality(std::string_view n) {
    return detail::find_card(n, described_space_traits<S>::descriptors());
}
template <described_space S>
constexpr std::size_t total_feature_width() {
    return detail::feat_width(described_space_traits<S>::descriptors());
}

// Built-in traits for hand-written adapter spaces
template <> struct described_space_traits<gemm_tile_space> {
    static constexpr auto descriptors() {
        return std::tuple{power_2("TM"), power_2("TN"), power_2("TK")};
    }
};
template <> struct described_space_traits<loop_transform_space> {
    static constexpr auto descriptors() {
        constexpr transform_strategy s[] = {
            transform_strategy::NONE, transform_strategy::LOOP_UNROLLING,
            transform_strategy::SCALAR_EXPANSION, transform_strategy::REDUCTION_TREE,
            transform_strategy::SOFTWARE_PIPELINING};
        return std::tuple{make_enum_vals("strategy", s),
                          make_int_set("unroll", {1, 2, 4, 8, 16})};
    }
};
template <> struct described_space_traits<bool_option_space> {
    static constexpr auto descriptors() {
        return std::tuple{bool_flag("enabled")};
    }
};

static_assert(described_space<gemm_tile_space>);
static_assert(described_space<loop_transform_space>);
static_assert(described_space<bool_option_space>);

// ═══════════════════════════════════════════════════════════════════════
// feature_bridge — Layer 2 encoding
//
// Reads descriptor encoding hints, writes point → span<double>.
// Supports override via .with(dim_idx, hint).
//
// num_features = model vector length (≠ rank when one_hot used).
// This is the ONLY place that knows about numeric encoding.
// Space knows structure; Bridge knows features; Model knows neither.
// ═══════════════════════════════════════════════════════════════════════

namespace detail {
constexpr int ilog2(int v) {
    int r = 0; while (v > 1) { v >>= 1; ++r; } return r;
}
} // namespace detail

template <typename... Descriptors>
struct feature_bridge {
    static constexpr std::size_t ndims = sizeof...(Descriptors);
    using descriptors_tuple = std::tuple<Descriptors...>;
    using point_type = std::tuple<typename Descriptors::value_type...>;

    descriptors_tuple descs_;
    std::array<encoding_hint, ndims> encodings_{};
    std::array<std::size_t, ndims> cardinalities_{};
    std::array<std::size_t, ndims> offsets_{};
    std::size_t total_features_ = 0;

    // Construct from descriptors tuple (no default construction needed)
    explicit feature_bridge(descriptors_tuple descs)
        : descs_(std::move(descs))
    {
        init_from_descs();
    }

    std::size_t num_features() const { return total_features_; }

    // Override encoding for one dimension, returns new bridge
    feature_bridge with(std::size_t dim_idx, encoding_hint enc) const {
        if (dim_idx >= ndims) {
            throw std::out_of_range("feature_bridge::with(): dim_idx out of range");
        }
        auto copy = *this;
        copy.encodings_[dim_idx] = enc;
        copy.recompute_offsets();
        return copy;
    }

    void write_features(const point_type& pt, std::span<double> out) const {
        write_impl(pt, out.data(), std::index_sequence_for<Descriptors...>{});
    }

    // Convenience: allocate and return feature vector
    std::vector<double> encode(const point_type& pt) const {
        std::vector<double> out(total_features_, 0.0);
        write_features(pt, out);
        return out;
    }

private:
    void init_from_descs() {
        [this]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((encodings_[Is] = std::get<Is>(descs_).default_encoding()), ...);
            ((cardinalities_[Is] = std::get<Is>(descs_).cardinality()), ...);
        }(std::index_sequence_for<Descriptors...>{});
        recompute_offsets();
    }

    void recompute_offsets() {
        std::size_t off = 0;
        for (std::size_t i = 0; i < ndims; ++i) {
            offsets_[i] = off;
            off += (encodings_[i] == encoding_hint::one_hot)
                   ? cardinalities_[i] : 1;
        }
        total_features_ = off;
    }

    template <std::size_t... Is>
    void write_impl(const point_type& pt, double* out,
                    std::index_sequence<Is...>) const {
        (write_dim<Is>(pt, out + offsets_[Is]), ...);
    }

    template <std::size_t I>
    void write_dim(const point_type& pt, double* out) const {
        using D = std::remove_cvref_t<decltype(std::get<I>(descs_))>;
        const auto& desc = std::get<I>(descs_);
        auto enc = encodings_[I];
        auto val = std::get<I>(pt);

        if constexpr (D::kind == dim_kind::bool_flag) {
            *out = val ? 1.0 : 0.0;
        } else if constexpr (D::kind == dim_kind::enum_set) {
            auto idx = desc.index_of(val);
            if (enc == encoding_hint::one_hot) {
                for (std::size_t k = 0; k < desc.cardinality(); ++k)
                    out[k] = (k == idx) ? 1.0 : 0.0;
                // idx == cardinality → all zeros (invalid value)
            } else {
                // Guard: invalid value → -1.0 (not UB)
                *out = (idx < desc.cardinality())
                    ? static_cast<double>(desc.value_as_int(idx))
                    : -1.0;
            }
        } else {
            // int-valued: positive_int, power_2, int_set
            if (enc == encoding_hint::log2) {
                *out = static_cast<double>(detail::ilog2(val));
            } else if (enc == encoding_hint::normalised) {
                if constexpr (requires { desc.lo; desc.hi; }) {
                    auto range = desc.hi - desc.lo;
                    *out = (range != 0)
                        ? static_cast<double>(val - desc.lo)
                          / static_cast<double>(range)
                        : 0.0;  // Fix 3.1: hi==lo → 0.0
                } else {
                    *out = static_cast<double>(val);
                }
            } else if (enc == encoding_hint::one_hot) {
                auto idx = desc.index_of(val);
                for (std::size_t k = 0; k < desc.cardinality(); ++k)
                    out[k] = (k == idx) ? 1.0 : 0.0;
                // idx == cardinality → all zeros (invalid value)
            } else {
                *out = static_cast<double>(val);
            }
        }
    }
};

// Factory: build bridge from a descriptor_space
template <typename... Ds>
auto default_bridge(const descriptor_space<Ds...>& space) {
    return feature_bridge<Ds...>(space.descriptors());
}

} // namespace ctdp::space

#endif // CTDP_SPACE_DESCRIPTOR_H
