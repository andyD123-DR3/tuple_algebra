#ifndef CTDP_SPACE_SPACE_H
#define CTDP_SPACE_SPACE_H

// ctdp v0.7.2 — Search space concept, algebra, constexpr enumeration
//
// Three-layer architecture (this is Layer 1):
//   Layer 1 (Space):  structure — rank, names, enumeration, algebra
//   Layer 2 (Bridge): encoding — per-dimension, heterogeneous → doubles
//   Layer 3 (Model):  untyped — span<double>, knows nothing about spaces
//
// Access tier concepts (independent of structure):
//   search_space       — point_type, rank, space_name, dimension_names
//   countable_space    — + cardinality()
//   indexable_space    — + point_at(i)
//   factored_space     — + num_dims(), dim_cardinality(d), dim_value(d, i)
//
// Wrapper model:
//   valid_view         — predicate filter, tier 1 only (enumerate, no cardinality)
//   filter_section     — value filter, tier 1 (enumerate, O(n) cardinality)
//   section_space      — rank-reducing fix, inherits base tier
//   product_space      — Cartesian product, inherits min tier of operands
//
// Design invariants:
//   - Space owns structure. No feature encoding here.
//   - enumerate(fn) is the required protocol (callback, constexpr-safe).
//   - Space::rank = number of coordinates.  Bridge::num_features = vector length.
//   - Canonical points are tuple-like for descriptor spaces.
//   - Constraints live in Space (via valid_view), not in cost functions.
//   - Wrappers only expose capabilities they can truthfully implement.
//   - from_features is NOT required (Phase 1: search over discrete points).

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// Fixed string — compile-time string for NTTPs
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
struct fixed_string {
    char data[N]{};
    constexpr fixed_string() = default;
    constexpr fixed_string(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) data[i] = str[i];
    }
    constexpr std::string_view sv() const { return {data, N > 0 ? N-1 : 0}; }
    constexpr std::size_t size() const { return N > 0 ? N-1 : 0; }
    constexpr char operator[](std::size_t i) const { return data[i]; }
    constexpr bool operator==(const fixed_string&) const = default;
    template <std::size_t M>
    constexpr bool equals(const fixed_string<M>& o) const { return sv() == o.sv(); }
};
template <std::size_t N>
fixed_string(const char (&)[N]) -> fixed_string<N>;

// ═══════════════════════════════════════════════════════════════════════
// Tuple utilities — algebra support for tuple-based points
// ═══════════════════════════════════════════════════════════════════════

namespace detail {

// Remove element at index Skip from a tuple
template <std::size_t Skip, typename... Ts, std::size_t... Before, std::size_t... After>
constexpr auto tuple_remove_impl(const std::tuple<Ts...>& t,
        std::index_sequence<Before...>, std::index_sequence<After...>) {
    return std::make_tuple(std::get<Before>(t)...,
                           std::get<Skip + 1 + After>(t)...);
}

template <std::size_t Skip, typename... Ts>
constexpr auto tuple_remove(const std::tuple<Ts...>& t) {
    static_assert(Skip < sizeof...(Ts), "tuple_remove: index out of range");
    return tuple_remove_impl<Skip>(t,
        std::make_index_sequence<Skip>{},
        std::make_index_sequence<sizeof...(Ts) - Skip - 1>{});
}

template <std::size_t Skip, typename Tuple>
using tuple_remove_t = decltype(tuple_remove<Skip>(std::declval<Tuple>()));

// Insert value at index Pos into a tuple
template <std::size_t Pos, typename Val, typename... Ts,
          std::size_t... Before, std::size_t... After>
constexpr auto tuple_insert_impl(const std::tuple<Ts...>& t, const Val& val,
        std::index_sequence<Before...>, std::index_sequence<After...>) {
    return std::make_tuple(std::get<Before>(t)..., val,
                           std::get<Pos + After>(t)...);
}

template <std::size_t Pos, typename Val, typename... Ts>
constexpr auto tuple_insert(const std::tuple<Ts...>& t, const Val& val) {
    static_assert(Pos <= sizeof...(Ts), "tuple_insert: index out of range");
    return tuple_insert_impl<Pos>(t, val,
        std::make_index_sequence<Pos>{},
        std::make_index_sequence<sizeof...(Ts) - Pos>{});
}

// Find dimension index by name (works with both static and member dimension_names)
template <typename Space>
constexpr std::size_t find_dim(const Space& space, std::string_view name) {
    auto names = space.dimension_names();
    for (std::size_t i = 0; i < Space::rank; ++i)
        if (names[i] == name) return i;
    return Space::rank;
}

template <typename Space>
constexpr bool has_dim(const Space& space, std::string_view name) {
    return find_dim(space, name) < Space::rank;
}

// Qualify dimension name with space namespace
inline std::string qualify(std::string_view space, std::string_view dim) {
    std::string r;
    r.reserve(space.size() + 2 + dim.size());
    r += space; r += "::"; r += dim;
    return r;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// Concepts
// ═══════════════════════════════════════════════════════════════════════

// Cardinality sentinel: "unknown / too-large / saturated".
// Algorithms requiring finiteness must check != saturated (or <= some bound).
static constexpr std::size_t saturated = std::numeric_limits<std::size_t>::max();

// search_space: structure only. No features, no encoding.
template <typename S>
concept search_space = requires(const S& s) {
    typename S::point_type;
    { S::rank } -> std::convertible_to<std::size_t>;
    { s.space_name() } -> std::convertible_to<std::string_view>;
    { s.dimension_names() }
        -> std::same_as<std::array<std::string_view, S::rank>>;
};

// enumerable_space: finite, can enumerate via callback.
// enumerate(fn) is checked structurally at call site, not in concept
// (C++20 can't check "callable with any function" in requires).
template <typename S>
concept enumerable_space = search_space<S> && requires(const S& s) {
    { s.cardinality() } -> std::convertible_to<std::size_t>;
};

// has_enumerate: checked at algorithm/adaptor boundaries.
// Validates that the space can actually enumerate with a concrete callback.
template <typename S>
concept has_enumerate = requires(const S& s) {
    s.enumerate([](const typename S::point_type&) {});
};

// constrained_space: has validity predicate (legacy — prefer valid_view wrapper)
template <typename S>
concept constrained_space = search_space<S> &&
    requires(const S& s, const typename S::point_type& p) {
    { s.is_valid(p) } -> std::convertible_to<bool>;
};

// ─── Access tier concepts ───────────────────────────────────────────
//
// Tier 1: search_space           — enumerate(fn)
// Tier 2: countable_space        — + cardinality()
// Tier 3: indexable_space         — + point_at(i)
// Orthogonal: factored_space     — per-dimension enumeration
//
// descriptor_space satisfies all four.  valid_view satisfies only tier 1.
// Solvers declare which tier they need.  Wrappers expose only what they
// can truthfully implement.  Missing capability = concept failure at
// call site pointing at exactly what's missing.

// countable_space: same as enumerable_space (kept for naming consistency
// with the tier model; enumerable_space retained for backward compat).
template <typename S>
concept countable_space = search_space<S> && requires(const S& s) {
    { s.cardinality() } -> std::convertible_to<std::size_t>;
};

// indexable_space: random access by flat index.
template <typename S>
concept indexable_space = countable_space<S> && requires(const S& s) {
    { s.point_at(std::size_t{0}) } -> std::convertible_to<typename S::point_type>;
};

// factored_space: per-dimension enumeration of choices.
// Useful for beam search, branch-and-bound, anything that builds
// solutions dimension by dimension.  Filter destroys this — a
// valid_view over a factored_space is NOT factored because the
// predicate couples dimensions.
template <typename S>
concept factored_space = search_space<S> && requires(const S& s) {
    { s.num_dims() } -> std::convertible_to<std::size_t>;
    { s.dim_cardinality(std::size_t{0}) } -> std::convertible_to<std::size_t>;
    // dim_value(d, i) checked at call site — return type varies per dimension.
};

// ═══════════════════════════════════════════════════════════════════════
// valid_view — predicate-filtered wrapper (tier 1 only)
//
// Wraps any space with a predicate.  enumerate() only yields points
// that satisfy the predicate.  No cardinality (would require full
// enumeration to count — dishonest to expose without doing the work).
// No rank (filter couples dimensions — factored structure is destroyed).
//
// This is the recommended way to express constraints: validity lives
// in the space, cost functions assume all points are valid.
// ═══════════════════════════════════════════════════════════════════════

template <typename Space, typename Pred>
struct valid_view {
    using point_type = typename Space::point_type;
    // rank retained: points still have the same coordinate count.
    // Factored independence is destroyed (valid_view does NOT satisfy
    // factored_space) but coordinate structure is preserved.
    static constexpr std::size_t rank = Space::rank;

    Space base_;
    Pred  pred_;

    constexpr valid_view(Space base, Pred pred)
        : base_(std::move(base)), pred_(std::move(pred)) {}

    constexpr auto space_name() const -> std::string_view { return base_.space_name(); }

    // dimension_names() forwarded for diagnostics / bridge construction.
    // Does NOT imply factored structure — dimensions are coupled by pred_.
    constexpr auto dimension_names() const { return base_.dimension_names(); }

    template <typename F>
    constexpr void enumerate(F&& fn) const {
        base_.enumerate([&](const point_type& p) constexpr {
            if (pred_(p)) fn(p);
        });
    }

    // Access to underlying space (e.g. for bridge construction).
    constexpr const Space& base() const { return base_; }
};

// Factory (CTAD-friendly)
template <typename Space, typename Pred>
constexpr auto filter_valid(Space space, Pred pred) {
    return valid_view<Space, Pred>(std::move(space), std::move(pred));
}

// valid_view is tier 1 only: enumerate, no cardinality, no rank.
// Deliberately does NOT satisfy enumerable_space or countable_space.
// If a solver needs cardinality, materialise first.

// ═══════════════════════════════════════════════════════════════════════
// Arithmetic helpers
// ═══════════════════════════════════════════════════════════════════════

namespace detail {
// Saturating multiply: returns saturated (SIZE_MAX) on overflow instead of wrapping.
// Saturation is sticky: saturating_mul(saturated, n) == saturated for all n > 0.
constexpr std::size_t saturating_mul(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > std::numeric_limits<std::size_t>::max() / b)
        return std::numeric_limits<std::size_t>::max();
    return a * b;
}
} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// to_vector — build vector from callback enumerate (runtime convenience)
// ═══════════════════════════════════════════════════════════════════════

template <typename Space>
    requires has_enumerate<Space>
auto to_vector(const Space& space) {
    std::vector<typename Space::point_type> result;
    space.enumerate([&](const typename Space::point_type& p) {
        result.push_back(p);
    });
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// exhaustive_search — generic over any enumerable space
//
// Returns:
//   best point found, or default-initialized point if all infeasible.
//   cost = +infinity means no valid point was found.
// ═══════════════════════════════════════════════════════════════════════

template <typename Space, typename CostFn>
    requires has_enumerate<Space>
constexpr auto exhaustive_search(const Space& space, CostFn&& cost_fn)
    -> typename Space::point_type
{
    typename Space::point_type best{};
    double best_cost = std::numeric_limits<double>::infinity();
    bool found = false;

    space.enumerate([&](const typename Space::point_type& p) {
        if constexpr (constrained_space<Space>) {
            if (!space.is_valid(p)) return;
        }
        double c = cost_fn(p);
        if (!found || c < best_cost) {
            best = p; best_cost = c; found = true;
        }
    });
    return best;
}

template <typename Space, typename CostFn>
    requires has_enumerate<Space>
constexpr auto exhaustive_search_with_cost(const Space& space, CostFn&& cost_fn)
    -> std::pair<typename Space::point_type, double>
{
    typename Space::point_type best{};
    double best_cost = std::numeric_limits<double>::infinity();
    bool found = false;

    space.enumerate([&](const typename Space::point_type& p) {
        if constexpr (constrained_space<Space>) {
            if (!space.is_valid(p)) return;
        }
        double c = cost_fn(p);
        if (!found || c < best_cost) {
            best = p; best_cost = c; found = true;
        }
    });
    return {best, best_cost};
}

// ═══════════════════════════════════════════════════════════════════════
// section_space — genuine rank reduction on tuple-based points
//
// Fixes dimension DimIdx to a constant, removes it from the point type.
// embed() reconstructs the full point.
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t DimIdx, typename BaseSpace>
struct section_space {
    using base_point = typename BaseSpace::point_type;
    using fixed_type = std::tuple_element_t<DimIdx, base_point>;
    using point_type = detail::tuple_remove_t<DimIdx, base_point>;

    static constexpr std::size_t rank = BaseSpace::rank - 1;

    BaseSpace base_;
    fixed_type fixed_;

    constexpr section_space(BaseSpace base, fixed_type val)
        : base_(std::move(base)), fixed_(val) {}

    constexpr std::string_view space_name() const {
        return base_.space_name();
    }

    constexpr auto dimension_names() const {
        auto bn = base_.dimension_names();
        std::array<std::string_view, rank> result{};
        for (std::size_t i = 0, j = 0; i < BaseSpace::rank; ++i)
            if (i != DimIdx) result[j++] = bn[i];
        return result;
    }

    // Note: O(base cardinality) — enumerates and counts.
    // Not for hot paths. For descriptor spaces, could be computed as
    // base.cardinality() / descriptor[Dim].cardinality() when uniform.
    std::size_t cardinality() const {
        std::size_t n = 0;
        enumerate([&](const auto&) { ++n; });
        return n;
    }

    template <typename F>
    void enumerate(F&& fn) const {
        base_.enumerate([&](const base_point& p) {
            if (std::get<DimIdx>(p) == fixed_)
                fn(detail::tuple_remove<DimIdx>(p));
        });
    }

    // Embed: reconstruct full point from reduced + fixed value
    constexpr base_point embed(const point_type& reduced) const {
        return detail::tuple_insert<DimIdx>(reduced, fixed_);
    }
};

// Factory: section<DimIdx>(space, value)
template <std::size_t DimIdx, typename Space>
auto section(const Space& space,
             std::tuple_element_t<DimIdx, typename Space::point_type> value) {
    return section_space<DimIdx, Space>(space, value);
}

// ═══════════════════════════════════════════════════════════════════════
// filter_section — filter-based section for any space (no rank reduction)
//
// Works on hand-written struct-point spaces where tuple_remove
// isn't applicable. Point type unchanged; enumerate just filters.
// This is the adapter pattern for non-tuple spaces.
// ═══════════════════════════════════════════════════════════════════════

template <typename BaseSpace, typename Accessor, typename ValueType>
struct filter_section {
    using point_type = typename BaseSpace::point_type;
    static constexpr std::size_t rank = BaseSpace::rank;

    BaseSpace base_;
    Accessor accessor_;
    ValueType fixed_;

    constexpr std::string_view space_name() const { return base_.space_name(); }
    constexpr auto dimension_names() const { return base_.dimension_names(); }

    std::size_t cardinality() const {
        std::size_t n = 0;
        base_.enumerate([&](const point_type& p) {
            if (accessor_(p) == fixed_) ++n;
        });
        return n;
    }

    template <typename F>
    void enumerate(F&& fn) const {
        base_.enumerate([&](const point_type& p) {
            if (accessor_(p) == fixed_) fn(p);
        });
    }
};

template <typename Space, typename Accessor, typename Val>
auto make_filter(const Space& space, Accessor acc, Val value) {
    return filter_section<Space, Accessor, Val>{space, acc, value};
}

// ═══════════════════════════════════════════════════════════════════════
// product_space — Cartesian product of two spaces (pair-based)
//
// For hand-written struct spaces.  Descriptor spaces should use
// descriptor_product() in descriptor.h which flattens via tuple_cat.
// ═══════════════════════════════════════════════════════════════════════

template <typename SpaceA, typename SpaceB>
struct product_space {
    using point_type = std::pair<typename SpaceA::point_type,
                                 typename SpaceB::point_type>;
    static constexpr std::size_t rank = SpaceA::rank + SpaceB::rank;

    SpaceA a_;
    SpaceB b_;
    std::string name_;  // stored, not static

    product_space(SpaceA a, SpaceB b)
        : a_(std::move(a)), b_(std::move(b))
    {
        name_ += a_.space_name();
        name_ += "x";
        name_ += b_.space_name();
    }

    std::string_view space_name() const { return name_; }

    auto dimension_names() const {
        std::array<std::string_view, rank> names{};
        auto an = a_.dimension_names();
        auto bn = b_.dimension_names();
        for (std::size_t i = 0; i < SpaceA::rank; ++i) names[i] = an[i];
        for (std::size_t i = 0; i < SpaceB::rank; ++i)
            names[SpaceA::rank + i] = bn[i];
        return names;
    }

    std::size_t cardinality() const {
        return detail::saturating_mul(a_.cardinality(), b_.cardinality());
    }

    template <typename F>
    void enumerate(F&& fn) const {
        a_.enumerate([&](const typename SpaceA::point_type& pa) {
            b_.enumerate([&](const typename SpaceB::point_type& pb) {
                fn(point_type{pa, pb});
            });
        });
    }

    static constexpr const auto& first(const point_type& p) { return p.first; }
    static constexpr const auto& second(const point_type& p) { return p.second; }
};

template <typename A, typename B>
auto product(const A& a, const B& b) {
    return product_space<A, B>(a, b);
}

// ═══════════════════════════════════════════════════════════════════════
// Concrete adapter spaces — hand-written struct points
//
// These are adapters, not the core model.  Descriptor spaces
// (in descriptor.h) are the canonical representation.
// ═══════════════════════════════════════════════════════════════════════

struct tile_shape {
    int tm = 0, tn = 0, tk = 0;
    constexpr bool operator==(const tile_shape&) const = default;
    constexpr auto dims() const { return std::array<int,3>{tm, tn, tk}; }
};

struct gemm_tile_space {
    using point_type = tile_shape;
    static constexpr std::size_t rank = 3;

    static constexpr std::string_view space_name() { return "gemm"; }

    static constexpr auto dimension_names() {
        return std::array<std::string_view, 3>{"TM", "TN", "TK"};
    }

    static constexpr std::size_t cardinality() { return 6*6*6; }

    template <typename F>
    static constexpr void enumerate(F&& fn) {
        constexpr int vals[] = {2, 4, 8, 16, 32, 64};
        for (int tm : vals)
            for (int tn : vals)
                for (int tk : vals)
                    fn(tile_shape{tm, tn, tk});
    }

    static constexpr int get_dim(const tile_shape& p, std::string_view name) {
        if (name == "TM") return p.tm;
        if (name == "TN") return p.tn;
        if (name == "TK") return p.tk;
        return -1;
    }

    static auto qualified_names() {
        return std::array<std::string, 3>{"gemm::TM", "gemm::TN", "gemm::TK"};
    }
};

enum class transform_strategy : int {
    NONE = 0, LOOP_UNROLLING = 1, SCALAR_EXPANSION = 2,
    REDUCTION_TREE = 3, SOFTWARE_PIPELINING = 4
};

struct loop_transform_point {
    transform_strategy strategy = transform_strategy::NONE;
    int unroll_factor = 1;
    constexpr bool operator==(const loop_transform_point&) const = default;
};

struct loop_transform_space {
    using point_type = loop_transform_point;
    static constexpr std::size_t rank = 2;

    static constexpr std::string_view space_name() { return "loop"; }
    static constexpr auto dimension_names() {
        return std::array<std::string_view, 2>{"strategy", "unroll"};
    }
    static constexpr std::size_t cardinality() { return 5*5; }

    template <typename F>
    static constexpr void enumerate(F&& fn) {
        for (auto s : {transform_strategy::NONE,
                       transform_strategy::LOOP_UNROLLING,
                       transform_strategy::SCALAR_EXPANSION,
                       transform_strategy::REDUCTION_TREE,
                       transform_strategy::SOFTWARE_PIPELINING})
            for (int uf : {1, 2, 4, 8, 16})
                fn(loop_transform_point{s, uf});
    }
};

struct bool_option {
    bool value = false;
    constexpr bool operator==(const bool_option&) const = default;
};

struct bool_option_space {
    using point_type = bool_option;
    static constexpr std::size_t rank = 1;

    static constexpr std::string_view space_name() { return "flag"; }
    static constexpr auto dimension_names() {
        return std::array<std::string_view, 1>{"enabled"};
    }
    static constexpr std::size_t cardinality() { return 2; }

    template <typename F>
    static constexpr void enumerate(F&& fn) {
        fn(bool_option{false});
        fn(bool_option{true});
    }
};

// ═══════════════════════════════════════════════════════════════════════
// dispatch — instantiate an NTTP-parameterised executor from a point
//
// Usage:
//   template<auto Cfg> struct executor { static void run(); };
//   constexpr auto best = solve(...);
//   using optimal = dispatch<executor, best>;
//   optimal::run();
//
// The point must be a structural type (tuple of enums/ints/bools, or
// a struct with all-public members of structural type).
// ═══════════════════════════════════════════════════════════════════════

template <template<auto> class Executor, auto Point>
using dispatch = Executor<Point>;

// ═══════════════════════════════════════════════════════════════════════
// Concept verification
// ═══════════════════════════════════════════════════════════════════════

static_assert(search_space<gemm_tile_space>);
static_assert(enumerable_space<gemm_tile_space>);
static_assert(has_enumerate<gemm_tile_space>);
static_assert(search_space<loop_transform_space>);
static_assert(enumerable_space<loop_transform_space>);
static_assert(has_enumerate<loop_transform_space>);
static_assert(search_space<bool_option_space>);
static_assert(enumerable_space<bool_option_space>);
static_assert(has_enumerate<bool_option_space>);

// Saturation semantics: sticky through composition
static_assert(detail::saturating_mul(saturated, 1) == saturated);
static_assert(detail::saturating_mul(saturated, 42) == saturated);
static_assert(detail::saturating_mul(saturated, saturated) == saturated);
static_assert(detail::saturating_mul(0, saturated) == 0);
static_assert(detail::saturating_mul(100, 200) == 20000);

} // namespace ctdp::space

#endif // CTDP_SPACE_SPACE_H
