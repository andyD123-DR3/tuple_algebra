#pragma once

#ifndef CTDP_PLAN_UMBRELLA_INCLUDE
#  error "Include ct_dp/plan/plan_ast.hpp rather than traits.h directly"
#endif

// ============================================================================
// CT-DP Phase 5 — Plan navigation traits
// ============================================================================
// Shape traits (is_leaf, is_seq, is_nest, is_split, is_choose) live in
// constructors.h as the single source of truth. This header provides
// navigation over the plan AST: child access, leaf internals, nest
// tessellation, choose predicate, and recursive wrapper stripping.
//
// plan_children is declared in wrappers.h (with wrapper specialisations)
// and specialised here for the five core constructors and for leaf (empty
// tuple). Together the two files cover the full plan universe so generic
// traversal works without special cases.
// ============================================================================

#include "constructors.hpp"
#include "wrappers.hpp"

#include <tuple>
#include <type_traits>

namespace ct_dp::plan {

// ============================================================================
// plan_children — total over the plan universe
// ============================================================================
// leaf returns std::tuple<> (no children). Core composite constructors
// return a tuple of their structural children. Wrappers are covered in
// wrappers.h with std::tuple<Inner>.

template<class K, class... A>
struct plan_children<leaf<K, A...>> {
    using type = std::tuple<>;
};

template<class... Plans>
struct plan_children<seq<Plans...>> {
    using type = std::tuple<Plans...>;
};

template<class Tessellation, class Inner>
struct plan_children<nest<Tessellation, Inner>> {
    using type = std::tuple<Inner>;
};

template<auto Key, class... GroupPlans>
struct plan_children<split<Key, GroupPlans...>> {
    using type = std::tuple<GroupPlans...>;
};

template<class Predicate, class TruePlan, class FalsePlan>
struct plan_children<choose<Predicate, TruePlan, FalsePlan>> {
    using type = std::tuple<TruePlan, FalsePlan>;
};

template<class Plan>
using plan_children_t = typename plan_children<Plan>::type;

// ============================================================================
// Leaf access
// ============================================================================

template<class Plan>
struct leaf_kernel;

template<class Kernel, class... Attrs>
struct leaf_kernel<leaf<Kernel, Attrs...>> { using type = Kernel; };

template<class Plan>
using leaf_kernel_t = typename leaf_kernel<Plan>::type;

template<class Plan>
struct leaf_attributes;

template<class Kernel, class... Attrs>
struct leaf_attributes<leaf<Kernel, Attrs...>> {
    using type = std::tuple<Attrs...>;
};

template<class Plan>
using leaf_attributes_t = typename leaf_attributes<Plan>::type;

// ============================================================================
// Nest access
// ============================================================================

template<class Plan>
struct nest_tessellation;

template<class Tessellation, class Inner>
struct nest_tessellation<nest<Tessellation, Inner>> {
    using type = Tessellation;
};

template<class Plan>
using nest_tessellation_t = typename nest_tessellation<Plan>::type;

// ============================================================================
// Choose access
// ============================================================================

template<class Plan>
struct choose_predicate;

template<class Predicate, class TruePlan, class FalsePlan>
struct choose_predicate<choose<Predicate, TruePlan, FalsePlan>> {
    using type = Predicate;
};

template<class Plan>
using choose_predicate_t = typename choose_predicate<Plan>::type;

// ============================================================================
// Recursive wrapper stripping
// ============================================================================
// wrapper_inner_t peels one wrapper. strip_wrappers_t recursively strips
// until a core plan is reached. Identity on non-wrappers.

template<class T>
struct strip_wrappers { using type = T; };

template<class T>
    requires is_wrapper_v<T>
struct strip_wrappers<T> {
    using type = typename strip_wrappers<wrapper_inner_t<T>>::type;
};

template<class T>
using strip_wrappers_t = typename strip_wrappers<T>::type;

// ============================================================================
// Convenience: nth child
// ============================================================================

template<std::size_t N, class Plan>
using nth_child_t = std::tuple_element_t<N, plan_children_t<Plan>>;

} // namespace ct_dp::plan
