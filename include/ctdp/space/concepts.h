// ctdp/space/concepts.h — Vocabulary types for the space layer
//
// This header defines the minimum vocabulary needed by any component
// that participates in the space layer: the kind enum, the encoding
// hint enum, and the dimension_descriptor concept.
//
// No dependencies on space.h, descriptor types, or the feature bridge.
// Include this header when you need to define a new ordinate type or
// check concept satisfaction without pulling in the full descriptor
// and bridge machinery.
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_CONCEPTS_H
#define CTDP_SPACE_CONCEPTS_H

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// dim_kind — classifies the structural type of a dimension
// ═══════════════════════════════════════════════════════════════════════

enum class dim_kind : std::uint8_t {
    int_set, power_of_two, int_range, bool_flag, enum_set, partition,
    permutation,
};

// ═══════════════════════════════════════════════════════════════════════
// encoding_hint — how a dimension's value maps to feature slots
// ═══════════════════════════════════════════════════════════════════════

enum class encoding_hint : std::uint8_t {
    raw, log2, one_hot, binary, normalised, pairwise, precedence,
};

// ═══════════════════════════════════════════════════════════════════════
// dimension_descriptor — the contract every ordinate type must satisfy
//
// Any type D with:
//   d.name                 → std::string_view
//   D::kind                → dim_kind
//   d.default_encoding()   → encoding_hint
//   d.cardinality()        → std::size_t
//   D::value_type          (the point coordinate type)
// ═══════════════════════════════════════════════════════════════════════

template <typename D>
concept dimension_descriptor = requires(const D& d) {
    { d.name } -> std::convertible_to<std::string_view>;
    { D::kind } -> std::convertible_to<dim_kind>;
    { d.default_encoding() } -> std::convertible_to<encoding_hint>;
    { d.cardinality() } -> std::convertible_to<std::size_t>;
    typename D::value_type;
};

} // namespace ctdp::space

#endif // CTDP_SPACE_CONCEPTS_H
