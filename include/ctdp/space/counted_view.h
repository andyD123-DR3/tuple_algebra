// ctdp/space/counted_view.h — Tier-2 constrained space wrapper
//
// counted_view wraps a space and a point-callable predicate.
// It provides filtered enumeration (tier 1) and cached cardinality
// (tier 2). It satisfies countable_space and constrained_space.
//
// Declarative constraint types:
//   product_le_constraint  — product of named dimensions <= limit
//   multiple_of_constraint — named dimension value % factor == 0
//   all_of_constraint      — conjunction of child constraints
//
// Constraint factories validate dimension names at construction time
// and capture a lightweight copy of the space for evaluation. They
// do not retain a reference to the caller's space.
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_COUNTED_VIEW_H
#define CTDP_SPACE_COUNTED_VIEW_H

#include "ctdp/space/space.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// counted_view — filtered space with cached cardinality
//
// Wraps a space + predicate. Satisfies countable_space and
// constrained_space. Forwards rank and dimension_names from base
// (coordinate structure is preserved by filtering).
//
// Predicate contract: bool(const point_type&), pure, deterministic.
//
// Caching: first cardinality() call enumerates-and-counts, caches
// the result. Subsequent calls return cached value.
// Concurrent first-time calls on the same object are a data race;
// external synchronisation required.
//
// Copies share the cached value (correct given purity requirement).
// ═══════════════════════════════════════════════════════════════════════

template <typename Space, typename Pred>
struct counted_view {
    using point_type = typename Space::point_type;
    static constexpr std::size_t rank = Space::rank;

    Space base_;
    Pred pred_;
    mutable std::optional<std::size_t> cached_card_;

    counted_view(Space base, Pred pred)
        : base_(std::move(base))
        , pred_(std::move(pred))
    {}

    // Forwarded from base (coordinate structure preserved)
    constexpr auto space_name() const -> std::string_view {
        return base_.space_name();
    }
    constexpr auto dimension_names() const {
        return base_.dimension_names();
    }

    // Tier 1: filtered enumeration
    template <typename F>
    void enumerate(F&& fn) const {
        base_.enumerate([&](const point_type& p) {
            if (pred_(p)) fn(p);
        });
    }

    // Tier 2: cardinality (enumerate-and-count, cached)
    std::size_t cardinality() const {
        if (!cached_card_) {
            std::size_t count = 0;
            base_.enumerate([&](const point_type& p) {
                if (pred_(p)) ++count;
            });
            cached_card_ = count;
        }
        return *cached_card_;
    }

    // constrained_space: point validity
    bool is_valid(const point_type& p) const {
        return pred_(p);
    }

    // Access underlying space (for bridge construction)
    const Space& base() const { return base_; }
};

/// Factory function (deduces template args).
template <typename Space, typename Pred>
auto make_counted_view(Space space, Pred pred) {
    return counted_view<Space, Pred>(std::move(space), std::move(pred));
}

// ═══════════════════════════════════════════════════════════════════════
// Constraint vocabulary
// ═══════════════════════════════════════════════════════════════════════

/// Constraint kind enum for inspectability.
enum class constraint_kind : std::uint8_t {
    product_le,
    multiple_of,
    all_of,
};

// ═══════════════════════════════════════════════════════════════════════
// product_le_constraint — product of named dimensions <= limit
//
// Captures a lightweight copy of the space for get_dim_as_int at
// evaluation time. Validates dimension names at factory construction.
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N, typename Space>
struct product_le_constraint {
    Space space_;
    std::array<std::string_view, N> names_;
    std::size_t limit_;

    bool operator()(const typename Space::point_type& pt) const {
        std::size_t product = 1;
        for (std::size_t i = 0; i < N; ++i)
            product *= static_cast<std::size_t>(
                space_.get_dim_as_int(pt, names_[i]));
        return product <= limit_;
    }

    static constexpr constraint_kind kind() {
        return constraint_kind::product_le;
    }
    std::size_t limit() const { return limit_; }
    std::span<const std::string_view> dim_names() const {
        return {names_.data(), N};
    }
};

/// Factory: validate names, capture space by value.
/// Throws std::invalid_argument on unknown or duplicate dimension names.
template <typename Space, typename... Names>
auto make_product_le(const Space& space, std::size_t limit, Names... names) {
    constexpr auto NN = sizeof...(Names);
    std::array<std::string_view, NN> name_arr{names...};

    // Validate: all names exist
    for (const auto& n : name_arr)
        if (!detail::has_dim(space, n))
            throw std::invalid_argument(
                "make_product_le: unknown dimension");

    // Validate: no duplicates
    for (std::size_t i = 0; i < NN; ++i)
        for (std::size_t j = i + 1; j < NN; ++j)
            if (name_arr[i] == name_arr[j])
                throw std::invalid_argument(
                    "make_product_le: duplicate dimension");

    return product_le_constraint<NN, Space>{space, name_arr, limit};
}

// ═══════════════════════════════════════════════════════════════════════
// multiple_of_constraint — named dimension value % factor == 0
// ═══════════════════════════════════════════════════════════════════════

template <typename Space>
struct multiple_of_constraint {
    Space space_;
    std::string_view name_;
    int factor_;

    bool operator()(const typename Space::point_type& pt) const {
        return space_.get_dim_as_int(pt, name_) % factor_ == 0;
    }

    static constexpr constraint_kind kind() {
        return constraint_kind::multiple_of;
    }
    int factor() const { return factor_; }
    std::string_view dim_name() const { return name_; }
};

/// Factory: validate name and factor.
/// Throws std::invalid_argument on unknown name or zero factor.
template <typename Space>
auto make_multiple_of(const Space& space, std::string_view name, int factor) {
    if (factor == 0)
        throw std::invalid_argument(
            "make_multiple_of: factor must not be zero");
    if (!detail::has_dim(space, name))
        throw std::invalid_argument(
            "make_multiple_of: unknown dimension");

    return multiple_of_constraint<Space>{space, name, factor};
}

// ═══════════════════════════════════════════════════════════════════════
// all_of_constraint — conjunction of child constraints
//
// Children must be already-bound, point-callable predicates.
// Short-circuit evaluation: stops at first false (left to right).
// ═══════════════════════════════════════════════════════════════════════

template <typename... Constraints>
struct all_of_constraint {
    std::tuple<Constraints...> children_;

    template <typename Point>
    bool operator()(const Point& pt) const {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return (std::get<Is>(children_)(pt) && ...);
        }(std::index_sequence_for<Constraints...>{});
    }

    static constexpr constraint_kind kind() {
        return constraint_kind::all_of;
    }
    static constexpr std::size_t num_children() {
        return sizeof...(Constraints);
    }

    template <std::size_t I>
    const auto& child() const { return std::get<I>(children_); }
};

/// Factory: compose already-bound constraints.
template <typename... Cs>
auto make_all_of(Cs... constraints) {
    return all_of_constraint<Cs...>{{std::move(constraints)...}};
}

} // namespace ctdp::space

#endif // CTDP_SPACE_COUNTED_VIEW_H
