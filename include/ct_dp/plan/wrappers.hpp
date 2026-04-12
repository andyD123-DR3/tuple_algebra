#pragma once

#ifndef CTDP_PLAN_UMBRELLA_INCLUDE
#  error "Include ct_dp/plan/plan_ast.hpp rather than wrappers.h directly"
#endif

// ============================================================================
// CT-DP Phase 5 — Plan wrappers
// ============================================================================
// Every wrapper:
//   - exposes kind = wrapper_tag and inner_t
//   - specialises is_wrapper, wrapper_inner, rewrap, and plan_children
// All specialisations live in this header so adding a wrapper family touches
// one file. The primary templates for wrapper_inner and rewrap include
// static_assert diagnostics for users who forget to register a new family.
// ============================================================================

#include "constructors.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace ct_dp::plan::wrap {

template<std::size_t Width, class Inner>
struct vectorise {
    static_assert(ct_dp::plan::is_plan_v<Inner>,
                  "vectorise inner must be a plan");
    using kind    = ct_dp::plan::wrapper_tag;
    using inner_t = Inner;
    static constexpr std::size_t width = Width;
};

template<std::size_t Factor, class Inner>
struct unroll {
    static_assert(ct_dp::plan::is_plan_v<Inner>,
                  "unroll inner must be a plan");
    using kind    = ct_dp::plan::wrapper_tag;
    using inner_t = Inner;
    static constexpr std::size_t factor = Factor;
};

template<class Policy, class Inner>
struct parallelise {
    static_assert(ct_dp::plan::is_plan_v<Inner>,
                  "parallelise inner must be a plan");
    using kind     = ct_dp::plan::wrapper_tag;
    using inner_t  = Inner;
    using policy_t = Policy;
};

template<std::size_t Distance, std::size_t Locality, class Inner>
struct prefetch {
    static_assert(ct_dp::plan::is_plan_v<Inner>,
                  "prefetch inner must be a plan");
    using kind    = ct_dp::plan::wrapper_tag;
    using inner_t = Inner;
    static constexpr std::size_t distance = Distance;
    static constexpr std::size_t locality = Locality;
};

} // namespace ct_dp::plan::wrap

namespace ct_dp::plan {

// ============================================================================
// Wrapper recognition
// ============================================================================

template<std::size_t W, class I>
struct is_wrapper<wrap::vectorise<W, I>> : std::true_type {};

template<std::size_t F, class I>
struct is_wrapper<wrap::unroll<F, I>> : std::true_type {};

template<class P, class I>
struct is_wrapper<wrap::parallelise<P, I>> : std::true_type {};

template<std::size_t D, std::size_t L, class I>
struct is_wrapper<wrap::prefetch<D, L, I>> : std::true_type {};

// ============================================================================
// wrapper_inner — single-step peel
// ============================================================================
// Primary template is defined (not just declared) so misuse produces a
// readable diagnostic rather than an incomplete-type error. For recursive
// peeling through arbitrary wrapper stacks, use strip_wrappers_t (traits.h).

template<class T>
struct wrapper_inner {
    static_assert(is_wrapper_v<T>,
        "wrapper_inner requires a recognised wrapper type; "
        "did you forget to specialise is_wrapper for a new wrapper family?");
};

template<std::size_t W, class I>
struct wrapper_inner<wrap::vectorise<W, I>> { using type = I; };

template<std::size_t F, class I>
struct wrapper_inner<wrap::unroll<F, I>> { using type = I; };

template<class P, class I>
struct wrapper_inner<wrap::parallelise<P, I>> { using type = I; };

template<std::size_t D, std::size_t L, class I>
struct wrapper_inner<wrap::prefetch<D, L, I>> { using type = I; };

template<class T>
using wrapper_inner_t = typename wrapper_inner<T>::type;

// ============================================================================
// rewrap — rebuild a wrapper around a replaced inner
// ============================================================================
// The Inner parameter in each specialisation is a pattern-match placeholder
// for the original inner type and is intentionally unused on the right-hand
// side. It is required so the primary template can match the wrapper shape.

template<class Wrapper, class NewInner>
struct rewrap {
    static_assert(is_wrapper_v<Wrapper>,
        "rewrap requires a recognised wrapper type as its first argument; "
        "did you forget to specialise is_wrapper for a new wrapper family?");
};

template<std::size_t W, class Inner, class NewInner>
struct rewrap<wrap::vectorise<W, Inner>, NewInner> {
    using type = wrap::vectorise<W, NewInner>;
};

template<std::size_t F, class Inner, class NewInner>
struct rewrap<wrap::unroll<F, Inner>, NewInner> {
    using type = wrap::unroll<F, NewInner>;
};

template<class P, class Inner, class NewInner>
struct rewrap<wrap::parallelise<P, Inner>, NewInner> {
    using type = wrap::parallelise<P, NewInner>;
};

template<std::size_t D, std::size_t L, class Inner, class NewInner>
struct rewrap<wrap::prefetch<D, L, Inner>, NewInner> {
    using type = wrap::prefetch<D, L, NewInner>;
};

template<class Wrapper, class NewInner>
using rewrap_t = typename rewrap<Wrapper, NewInner>::type;

// ============================================================================
// plan_children specialisations for wrappers
// ============================================================================
// Declared here (forward) and specialised below. The primary template and
// the core-constructor specialisations live in traits.h. Placing the
// wrapper specialisations here keeps all wrapper-specific machinery in one
// file so adding a wrapper family touches wrappers.h only.

template<class Plan>
struct plan_children;

template<std::size_t W, class I>
struct plan_children<wrap::vectorise<W, I>> { using type = std::tuple<I>; };

template<std::size_t F, class I>
struct plan_children<wrap::unroll<F, I>> { using type = std::tuple<I>; };

template<class P, class I>
struct plan_children<wrap::parallelise<P, I>> { using type = std::tuple<I>; };

template<std::size_t D, std::size_t L, class I>
struct plan_children<wrap::prefetch<D, L, I>> { using type = std::tuple<I>; };

} // namespace ct_dp::plan
