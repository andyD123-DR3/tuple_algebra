// ctdp/space/feature_bridge.h — Layer 2 encoding: descriptor → features
//
// Reads descriptor encoding hints, writes point → span<double>.
// Supports override via .with(dim_idx, hint).
//
// num_features = model vector length (≠ rank when one_hot used).
// This is the ONLY place that knows about numeric encoding.
// Space knows structure; Bridge knows features; Model knows neither.
//
// The shared encode_scalar() function is the single source of truth
// for scalar encoding. Used by both feature_bridge::write_dim and
// conditional_dim::write_features — never duplicate this logic.
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_FEATURE_BRIDGE_H
#define CTDP_SPACE_FEATURE_BRIDGE_H

#include "ctdp/space/concepts.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ctdp::space {

namespace detail {

// ── Encoding helpers ─────────────────────────────────────────────────

constexpr int ilog2(int v) {
    int r = 0; while (v > 1) { v >>= 1; ++r; } return r;
}

// Prefer encoding_cardinality() when available (e.g. conditional_dim).
// Falls back to cardinality() for all other descriptors.
// This ensures feature width stability for conditional dimensions.
template <typename D>
constexpr std::size_t encoding_card_of(const D& d) {
    if constexpr (requires { d.encoding_cardinality(); }) {
        return d.encoding_cardinality();
    } else {
        return d.cardinality();
    }
}

// Compute the number of feature slots a descriptor occupies.
// Descriptors with feature_width() (e.g. partition) provide their own.
// Otherwise: one_hot → encoding_card_of slots, everything else → 1.
template <typename D>
constexpr std::size_t feature_width_of(const D& d, encoding_hint enc) {
    if constexpr (requires { d.feature_width(); }) {
        return d.feature_width();
    } else {
        return (enc == encoding_hint::one_hot)
            ? encoding_card_of(d) : 1;
    }
}

// ── Shared scalar encoding ──────────────────────────────────────────
//
// Single source of truth for encoding a descriptor value to features.
// Used by feature_bridge::write_dim and conditional_dim::write_features.
// Does NOT handle dim_kind::partition (those have write_features).

template <typename D, typename V>
void encode_scalar(const D& desc, V val, encoding_hint enc, double* out) {
    if constexpr (D::kind == dim_kind::bool_flag) {
        *out = val ? 1.0 : 0.0;
    } else if constexpr (D::kind == dim_kind::enum_set) {
        auto idx = desc.index_of(val);
        if (enc == encoding_hint::one_hot) {
            auto card = encoding_card_of(desc);
            for (std::size_t k = 0; k < card; ++k)
                out[k] = (k == idx) ? 1.0 : 0.0;
        } else if (enc == encoding_hint::normalised) {
            auto card = encoding_card_of(desc);
            if (idx >= card) {
                *out = -1.0;
            } else if (card <= 1) {
                *out = 0.0;
            } else {
                *out = static_cast<double>(idx)
                     / static_cast<double>(card - 1);
            }
        } else {
            auto card = encoding_card_of(desc);
            *out = (idx < card)
                ? static_cast<double>(desc.value_as_int(idx))
                : -1.0;
        }
    } else {
        // int-valued: positive_int, power_2, int_set
        if (enc == encoding_hint::log2) {
            *out = static_cast<double>(ilog2(val));
        } else if (enc == encoding_hint::normalised) {
            if constexpr (requires { desc.lo; desc.hi; }) {
                auto range = desc.hi - desc.lo;
                *out = (range != 0)
                    ? static_cast<double>(val - desc.lo)
                      / static_cast<double>(range)
                    : 0.0;
            } else {
                *out = static_cast<double>(val);
            }
        } else if (enc == encoding_hint::one_hot) {
            auto idx = desc.index_of(val);
            auto card = encoding_card_of(desc);
            for (std::size_t k = 0; k < card; ++k)
                out[k] = (k == idx) ? 1.0 : 0.0;
        } else {
            *out = static_cast<double>(val);
        }
    }
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// feature_bridge — Layer 2 encoding
// ═══════════════════════════════════════════════════════════════════════

template <typename... Descriptors>
struct feature_bridge {
    static constexpr std::size_t ndims = sizeof...(Descriptors);
    using descriptors_tuple = std::tuple<Descriptors...>;
    using point_type = std::tuple<typename Descriptors::value_type...>;

    descriptors_tuple descs_;
    std::array<encoding_hint, ndims> encodings_{};
    std::array<std::size_t, ndims> cardinalities_{};
    std::array<std::size_t, ndims> widths_{};
    std::array<std::size_t, ndims> offsets_{};
    std::size_t total_features_ = 0;

    // Construct from descriptors tuple (no default construction needed)
    explicit feature_bridge(descriptors_tuple descs)
        : descs_(std::move(descs))
    {
        init_from_descs();
    }

    std::size_t num_features() const { return total_features_; }

    // Override encoding for one dimension, returns new bridge.
    // NOTE: Only valid for scalar-encoded dimensions (int, enum, bool).
    // For descriptors with custom feature_width (e.g. partition), the
    // width is determined by the descriptor, not the encoding hint,
    // so .with() has no effect on those dimensions.
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
            ((cardinalities_[Is] = detail::encoding_card_of(std::get<Is>(descs_))), ...);
            ((widths_[Is] = detail::feature_width_of(std::get<Is>(descs_), encodings_[Is])), ...);
        }(std::index_sequence_for<Descriptors...>{});
        recompute_offsets();
    }

    void recompute_offsets() {
        std::size_t off = 0;
        for (std::size_t i = 0; i < ndims; ++i) {
            offsets_[i] = off;
            off += widths_[i];
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

        if constexpr (D::kind == dim_kind::partition) {
            // Pairwise co-membership encoding: N*(N-1)/2 binary features.
            static_assert(requires { desc.write_features(val, out); },
                "partition dim_kind requires write_features(value_type, double*)");
            desc.write_features(val, out);
        } else {
            detail::encode_scalar(desc, val, enc, out);
        }
    }
};

} // namespace ctdp::space

#endif // CTDP_SPACE_FEATURE_BRIDGE_H
