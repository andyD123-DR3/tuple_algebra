#pragma once

// ============================================================================
// CT-DP Phase 5 — Core plan constructors
// ============================================================================
//
// This header is part of the plan library and must be included through the
// umbrella header plan.h, not directly. Direct inclusion produces an error
// to prevent the is_wrapper ODR hazard: if a translation unit sees only the
// core header without the wrapper specialisations, is_plan_v returns wrong
// answers for wrapper types.
//
// Decoration policy
// -----------------
// There are exactly two decoration sites in the plan language:
//
//   1. Leaf attributes  (leaf<Kernel, Attrs...>)
//      Decorate the primitive implementation identity of a single kernel.
//      Initial attribute families: format, compute_strategy, reduce_topology,
//      traversal_hint. Each family must appear at most once per leaf.
//
//   2. Wrappers         (ct_dp::plan::wrap::*, defined in wrappers.h)
//      Decorate a whole plan subtree. Initial wrapper families: vectorise,
//      unroll, parallelise, prefetch.
//
// A given optimisation family must have exactly one canonical home — either
// as a leaf attribute or as a wrapper, but never both. Enforcement of this
// rule is deferred to the prototype branch and will be informed by the
// first worked DSL (SpMV).
// ============================================================================

#ifndef CTDP_PLAN_UMBRELLA_INCLUDE
#  error "Include ct_dp/plan/plan_ast.hpp rather than constructors.h directly"
#endif

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace ct_dp::plan {

// ----------------------------------------------------------------------------
// Constructor tags (metadata only — shape identity uses partial specialisation)
// ----------------------------------------------------------------------------

struct leaf_tag    {};
struct seq_tag     {};
struct nest_tag    {};
struct split_tag   {};
struct choose_tag  {};
struct wrapper_tag {};

// ----------------------------------------------------------------------------
// Forward declaration for wrapper detection
// ----------------------------------------------------------------------------
// Wrappers are defined in wrappers.h and specialise this trait. Users must
// include plan.h, which pulls in both headers, so by the time is_wrapper_v
// is queried all specialisations are visible.

template<class T>
struct is_wrapper : std::false_type {};

template<class T>
inline constexpr bool is_wrapper_v = is_wrapper<T>::value;

// ============================================================================
// The five core constructors
// ============================================================================

// 1. leaf<Kernel, Attrs...>
//
// Binds a semantic kernel identity to zero or more opaque attribute types.
// Attributes decorate primitive-local implementation choice only; whole-
// subtree transformations belong in wrappers.

template<class Kernel, class... Attrs>
struct leaf {
    using kind         = leaf_tag;
    using kernel_t     = Kernel;
    using attributes_t = std::tuple<Attrs...>;
    static constexpr std::size_t attribute_count = sizeof...(Attrs);
};

// 2. seq<Plans...>
//
// Ordered composition. Empty seq<> is legal and denotes the identity plan:
//   - recognised as a plan by is_plan_v
//   - will be a no-op in the instantiation layer when that lands
//   - canonicalisation will remove seq<> from any containing seq<..., seq<>, ...>
// Downstream layers honour this commitment.

template<class... Plans>
struct seq {
    using kind    = seq_tag;
    using plans_t = std::tuple<Plans...>;
    static constexpr std::size_t arity = sizeof...(Plans);
};

// 3. nest<Tessellation, Inner>
//
// Tessellated application of Inner over a structured domain. Tessellation is
// a separate primitive type (see tessellation.h, future) carrying domain
// reference, tile shape, traversal order, and band-level attributes.

template<class Tessellation, class Inner>
struct nest {
    using kind           = nest_tag;
    using tessellation_t = Tessellation;
    using inner_t        = Inner;
};

// 4. split<Key, GroupPlans...>
//
// Compile-time partition by grouping key. C++20 NTTPs require Key's type to
// be structural (all bases and non-static members public, no mutable
// members, no user-provided copy assignment). See test file for the
// canonical empty-struct key pattern.

template<auto Key, class... GroupPlans>
struct split {
    using kind     = split_tag;
    using key_t    = decltype(Key);
    static constexpr auto key = Key;
    using groups_t = std::tuple<GroupPlans...>;
    static constexpr std::size_t group_count = sizeof...(GroupPlans);
};

// 5. choose<Predicate, TruePlan, FalsePlan>
//
// Structural conditionality. Predicate exposes static constexpr bool value.

template<class Predicate, class TruePlan, class FalsePlan>
struct choose {
    using kind         = choose_tag;
    using predicate_t  = Predicate;
    using true_plan_t  = TruePlan;
    using false_plan_t = FalsePlan;
};

// ============================================================================
// Core shape traits (partial specialisation — single source of truth)
// ============================================================================
// These live in constructors.h so that is_core_plan can be built from them
// without depending on traits.h. Navigation traits stay in traits.h.

template<class T>
struct is_leaf : std::false_type {};
template<class K, class... A>
struct is_leaf<leaf<K, A...>> : std::true_type {};
template<class T>
inline constexpr bool is_leaf_v = is_leaf<T>::value;

template<class T>
struct is_seq : std::false_type {};
template<class... P>
struct is_seq<seq<P...>> : std::true_type {};
template<class T>
inline constexpr bool is_seq_v = is_seq<T>::value;

template<class T>
struct is_nest : std::false_type {};
template<class Tess, class Inner>
struct is_nest<nest<Tess, Inner>> : std::true_type {};
template<class T>
inline constexpr bool is_nest_v = is_nest<T>::value;

template<class T>
struct is_split : std::false_type {};
template<auto Key, class... P>
struct is_split<split<Key, P...>> : std::true_type {};
template<class T>
inline constexpr bool is_split_v = is_split<T>::value;

template<class T>
struct is_choose : std::false_type {};
template<class Pred, class A, class B>
struct is_choose<choose<Pred, A, B>> : std::true_type {};
template<class T>
inline constexpr bool is_choose_v = is_choose<T>::value;

// ============================================================================
// is_core_plan — now built from shape traits, not tag dispatch
// ============================================================================

template<class T>
struct is_core_plan
    : std::bool_constant<
          is_leaf_v<T>  || is_seq_v<T>   || is_nest_v<T> ||
          is_split_v<T> || is_choose_v<T>
      > {};

template<class T>
inline constexpr bool is_core_plan_v = is_core_plan<T>::value;

// ============================================================================
// is_plan — core plan or recognised wrapper
// ============================================================================

template<class T>
struct is_plan
    : std::bool_constant<is_core_plan_v<T> || is_wrapper_v<T>> {};

template<class T>
inline constexpr bool is_plan_v = is_plan<T>::value;

template<class T>
concept plan = is_plan_v<T>;

// Utility fold
template<class... Ts>
inline constexpr bool all_plans_v = (is_plan_v<Ts> && ...);

// ============================================================================
// Concept-constrained public builders
// ============================================================================
// Errors fire at the user's call site with "constraint plan<X> not
// satisfied", not deep inside implementation templates.

template<plan... Plans>
using seq_t = seq<Plans...>;

template<class Tessellation, plan Inner>
using nest_t = nest<Tessellation, Inner>;

template<auto Key, plan... GroupPlans>
    requires (sizeof...(GroupPlans) > 0)
using split_t = split<Key, GroupPlans...>;

template<class Predicate, plan TruePlan, plan FalsePlan>
using choose_t = choose<Predicate, TruePlan, FalsePlan>;

} // namespace ct_dp::plan
