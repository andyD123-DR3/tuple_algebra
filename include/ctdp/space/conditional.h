// ctdp/space/conditional.h — Conditionally-present search dimension
//
// conditional_dim(active, descriptor) wraps a descriptor so that:
//   - When active:  behaves exactly like the wrapped descriptor
//   - When inactive: cardinality = 1, single default value, no search freedom
//
// Feature encoding stability:
//   Feature width is ALWAYS the same regardless of active/inactive state.
//   When active, features encode per the wrapped descriptor's rules.
//   When inactive, all feature slots are 0.0.
//   This guarantees that ML models trained on one space configuration
//   can still consume feature vectors from another without layout changes.
//
// The predicate is evaluated at construction time, not at search time.
// This is for conditions known before search begins: algebraic properties
// (all_associative), hardware capabilities (has_fma), intent requirements
// (deterministic_required). For conditions that depend on searched values,
// use conditional choose (tree_space) instead.
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_CONDITIONAL_H
#define CTDP_SPACE_CONDITIONAL_H

#include "ctdp/space/descriptor.h"

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// conditional_dim — property-driven dimension presence
//
// Wraps any dimension_descriptor. When the condition is false, the
// dimension contributes no search freedom (cardinality 1) but still
// occupies its slot in the point tuple with a default value, and still
// contributes its full feature width (all zeros) to the feature vector.
//
// Usage:
//   // Compile-time condition (constexpr bool from template traits):
//   conditional_dim(reduction_props::all_associative,
//       make_enum_vals("tree_shape", {flat, binary, canonical}))
//
//   // Runtime condition (hardware detection):
//   conditional_dim(hw.has_fma,
//       bool_flag("use_fma"))
//
// ═══════════════════════════════════════════════════════════════════════

template <typename Descriptor>
    requires dimension_descriptor<Descriptor>
struct conditional_dim {
    using value_type = typename Descriptor::value_type;
    static constexpr dim_kind kind = Descriptor::kind;

    Descriptor wrapped;
    bool active;

    // ── Construction ─────────────────────────────────────────────

    constexpr conditional_dim(bool condition, Descriptor desc)
        : wrapped(std::move(desc)), active(condition) {}

    // ── dimension_descriptor contract ────────────────────────────

    // Name is always available (for diagnostics, provenance)
    constexpr std::string_view get_name() const { return wrapped.name; }

    // Expose name member directly for dimension_descriptor concept
    // (concept checks d.name, not d.get_name())
    std::string_view name = wrapped.name;

    constexpr encoding_hint default_encoding() const {
        return wrapped.default_encoding();
    }

    /// When active: full cardinality. When inactive: 1 (no choice).
    constexpr std::size_t cardinality() const {
        return active ? wrapped.cardinality() : 1;
    }

    /// For feature encoding: ALWAYS returns the wrapped descriptor's full
    /// cardinality. This is the width stability guarantee — one-hot encoding
    /// always uses the same number of slots regardless of active/inactive.
    constexpr std::size_t encoding_cardinality() const {
        return wrapped.cardinality();
    }

    /// When active: delegate to wrapped. When inactive: return default value.
    constexpr value_type value_at(std::size_t i) const {
        if (active) return wrapped.value_at(i);
        // Inactive: only index 0 is legal, returns first value of wrapped
        return wrapped.value_at(0);
    }

    /// When active: delegate. When inactive: only the default value is legal.
    constexpr bool contains(value_type v) const {
        if (active) return wrapped.contains(v);
        return v == wrapped.value_at(0);
    }

    /// When active: delegate. When inactive: return out-of-range index
    /// so that one-hot encoding produces all zeros (no choice was made).
    constexpr std::size_t index_of(value_type v) const {
        if (active) return wrapped.index_of(v);
        // Inactive: return encoding_cardinality (out-of-range).
        // This causes the one-hot loop in feature_bridge to write all zeros.
        return wrapped.cardinality();
    }

    // ── Feature encoding support ─────────────────────────────────

    /// Feature width is ALWAYS the wrapped descriptor's feature width.
    /// This is the stability guarantee: active or not, same width.
    constexpr std::size_t feature_width() const {
        auto enc = wrapped.default_encoding();
        if (enc == encoding_hint::one_hot) return wrapped.cardinality();
        return 1;
    }

    /// Write features for this dimension.
    /// When active: encode normally per wrapped descriptor.
    /// When inactive: all zeros.
    void write_features(value_type val, double* out) const {
        auto w = feature_width();
        if (!active) {
            for (std::size_t k = 0; k < w; ++k) out[k] = 0.0;
            return;
        }
        // Active: delegate to the wrapped descriptor's encoding logic
        auto enc = wrapped.default_encoding();
        if (enc == encoding_hint::one_hot) {
            auto idx = wrapped.index_of(val);
            for (std::size_t k = 0; k < wrapped.cardinality(); ++k)
                out[k] = (k == idx) ? 1.0 : 0.0;
        } else if (enc == encoding_hint::binary) {
            // bool_flag style
            if constexpr (std::is_same_v<value_type, bool>) {
                *out = val ? 1.0 : 0.0;
            } else {
                *out = static_cast<double>(val);
            }
        } else if (enc == encoding_hint::log2) {
            *out = static_cast<double>(detail::ilog2(static_cast<int>(val)));
        } else if (enc == encoding_hint::normalised) {
            if constexpr (requires { wrapped.lo; wrapped.hi; }) {
                auto range = wrapped.hi - wrapped.lo;
                *out = (range != 0)
                    ? static_cast<double>(static_cast<int>(val) - wrapped.lo)
                      / static_cast<double>(range)
                    : 0.0;
            } else {
                *out = static_cast<double>(val);
            }
        } else {
            // raw
            *out = static_cast<double>(val);
        }
    }

    // ── Query ────────────────────────────────────────────────────

    constexpr bool is_active() const { return active; }
    constexpr const Descriptor& descriptor() const { return wrapped; }

    /// Default value (used when inactive, also the fallback for inactive plans)
    constexpr value_type default_value() const { return wrapped.value_at(0); }

    // ── feature_bridge compatibility ─────────────────────────────
    // Forward methods that feature_bridge::write_dim may call
    // depending on the wrapped descriptor's dim_kind.

    /// Forward value_as_int for enum_set descriptors
    constexpr int value_as_int(std::size_t i) const
        requires requires(const Descriptor& d) { d.value_as_int(i); }
    {
        if (active) return wrapped.value_as_int(i);
        return wrapped.value_as_int(0);
    }
};

// Deduction guide
template <typename D>
conditional_dim(bool, D) -> conditional_dim<D>;

// Verify concept satisfaction
static_assert(dimension_descriptor<conditional_dim<bool_flag>>);

// ═══════════════════════════════════════════════════════════════════════
// Factory helpers
// ═══════════════════════════════════════════════════════════════════════

/// Create a conditional dimension from a compile-time or runtime predicate.
template <typename Descriptor>
    requires dimension_descriptor<Descriptor>
constexpr auto make_conditional(bool condition, Descriptor desc) {
    return conditional_dim<Descriptor>(condition, std::move(desc));
}

/// Create a conditional dimension that is always active (identity wrapper).
template <typename Descriptor>
    requires dimension_descriptor<Descriptor>
constexpr auto always_active(Descriptor desc) {
    return conditional_dim<Descriptor>(true, std::move(desc));
}

/// Create a conditional dimension that is always inactive (dead dimension).
template <typename Descriptor>
    requires dimension_descriptor<Descriptor>
constexpr auto always_inactive(Descriptor desc) {
    return conditional_dim<Descriptor>(false, std::move(desc));
}

} // namespace ctdp::space

#endif // CTDP_SPACE_CONDITIONAL_H
