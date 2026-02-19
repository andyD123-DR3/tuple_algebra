// ct_dp/algebra/operations.h
//
// Typed operations for tuple algebra.
//
// These are stateless function objects used as template parameters throughout
// the algebra layer. Each operation is:
//   - constexpr: usable in compile-time and runtime contexts
//   - default-constructible: can be stored in tuples without state
//   - strongly typed: carries semantic information (is_identity, is_power, etc.)
//
// Algebraic property annotations (associative, commutative, idempotent) are
// descriptive metadata. They do NOT override the semantic requirements of
// parallel algorithms, and are NOT sufficient to justify parallelization for
// floating-point types.
//
// Part of P3666R2 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_OPERATIONS_H
#define CT_DP_ALGEBRA_OPERATIONS_H

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>

namespace ct_dp::algebra {

// ---------------------------------------------------------------------------
// Transform operations (unary: T -> T)
// ---------------------------------------------------------------------------

/// Identity transform: returns its argument unchanged.
/// Trait: is_identity = true. Cost: 0 (elided by optimiser).
struct identity_t {
    template<typename T>
    constexpr T operator()(T x) const noexcept { return x; }

    // Self-describing trait
    static constexpr bool is_identity = true;
    static constexpr bool is_power    = true;
    static constexpr int  exponent    = 1;
};

/// Constant transform: ignores input, returns a fixed value.
/// Used for count lanes: constant_t<1>{}(x) == 1 for all x.
template<auto V>
struct constant_t {
    template<typename T>
    constexpr auto operator()(T) const noexcept { return V; }

    static constexpr bool is_identity = false;
    static constexpr bool is_power    = false;
    static constexpr auto value       = V;
};

/// Power transform: computes x^P using binary exponentiation.
/// Trait: is_power = true, exponent = P.
/// power_t<0>: always 1.  power_t<1>: identity.  power_t<2>: x*x.
///
/// Exponentiation by squaring ensures O(log P) multiplications,
/// which matters both for runtime efficiency and consteval step budget.
template<int P>
struct power_t {
    static_assert(P >= 0, "Negative exponents not supported");

    template<typename T>
    constexpr T operator()(T x) const noexcept {
        if constexpr (P == 0) {
            return T{1};
        } else if constexpr (P == 1) {
            return x;
        } else if constexpr (P == 2) {
            return x * x;
        } else if constexpr (P % 2 == 0) {
            auto half = power_t<P / 2>{}(x);
            return half * half;
        } else {
            return x * power_t<P - 1>{}(x);
        }
    }

    static constexpr bool is_identity = (P == 1);
    static constexpr bool is_power    = true;
    static constexpr int  exponent    = P;
};

/// Negate transform: returns -x.
struct negate_t {
    template<typename T>
    constexpr T operator()(T x) const noexcept { return -x; }

    static constexpr bool is_identity = false;
    static constexpr bool is_power    = false;
};

/// Absolute value transform.
struct abs_t {
    template<typename T>
    constexpr T operator()(T x) const noexcept { return x < T{} ? -x : x; }

    static constexpr bool is_identity = false;
    static constexpr bool is_power    = false;
};

// ---------------------------------------------------------------------------
// Reduce operations (binary: (T, T) -> T)
//
// Each carries:
//   - declared_associative / declared_commutative / declared_idempotent:
//     mathematical intent annotations (true for the abstract algebra)
//   - identity<T>(): the identity element for type T
//
// The external trait variables (declares_associative_v etc.) gate these
// annotations on type safety — they are only true for types where the
// property is guaranteed (integral/enum types).
// ---------------------------------------------------------------------------

/// Minimum: returns the smaller of two values.
/// Mathematical properties: associative, commutative, idempotent.
/// Identity element: +infinity (largest representable value).
struct min_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a < b ? a : b; }

    // Declared mathematical properties (abstract algebra)
    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = true;

    /// Identity element for type T: the largest representable value.
    template<typename T>
    static constexpr T identity() noexcept {
        return std::numeric_limits<T>::max();
    }
};

/// Maximum: returns the larger of two values.
/// Mathematical properties: associative, commutative, idempotent.
/// Identity element: -infinity (lowest representable value).
struct max_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a > b ? a : b; }

    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = true;

    template<typename T>
    static constexpr T identity() noexcept {
        return std::numeric_limits<T>::lowest();
    }
};

/// Plus: addition. Wraps std::plus<> with algebraic property annotations.
/// Identity element: 0.
struct plus_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a + b; }

    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = false;

    template<typename T>
    static constexpr T identity() noexcept { return T{}; }
};

/// Multiplies: multiplication. Identity element: 1.
struct multiplies_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a * b; }

    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = false;

    template<typename T>
    static constexpr T identity() noexcept { return T{1}; }
};

/// Bitwise AND. Identity element: all-ones.
struct bit_and_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a & b; }

    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = true;

    template<typename T>
    static constexpr T identity() noexcept { return ~T{}; }
};

/// Bitwise OR. Identity element: 0.
struct bit_or_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a | b; }

    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = true;

    template<typename T>
    static constexpr T identity() noexcept { return T{}; }
};

// ---------------------------------------------------------------------------
// Transform trait queries (type-independent)
// ---------------------------------------------------------------------------

/// Is Op an identity transform? (exposition-only)
template<typename Op>
inline constexpr bool is_identity_transform_v =
    requires { { Op::is_identity } -> std::convertible_to<bool>; } && Op::is_identity;

/// Is Op a power transform? (exposition-only)
template<typename Op>
inline constexpr bool is_power_transform_v =
    requires { { Op::is_power } -> std::convertible_to<bool>; } && Op::is_power;

// ---------------------------------------------------------------------------
// Type-dependent algebraic property queries
//
// These are descriptive metadata. They do NOT change the semantic requirements
// of algorithms such as transform_reduce, and are NOT sufficient to establish
// that an operation is safe to use with parallel execution policies for a
// given T (for example, floating-point addition is not associative under
// IEEE 754, and min/max have NaN-handling ambiguities).
//
// For safety, the default specialisations only return true for integral
// and enum types. Users may provide further specialisations for types
// where the properties are known to hold.
// ---------------------------------------------------------------------------

namespace detail {
    template<typename T>
    inline constexpr bool is_exact_v = std::is_integral_v<T> || std::is_enum_v<T>;

    /// Does Op self-declare the given property?
    template<typename Op, typename = void>
    inline constexpr bool has_declared_assoc = false;
    template<typename Op>
    inline constexpr bool has_declared_assoc<Op, std::void_t<decltype(Op::declared_associative)>> =
        Op::declared_associative;

    template<typename Op, typename = void>
    inline constexpr bool has_declared_commut = false;
    template<typename Op>
    inline constexpr bool has_declared_commut<Op, std::void_t<decltype(Op::declared_commutative)>> =
        Op::declared_commutative;

    template<typename Op, typename = void>
    inline constexpr bool has_declared_idemp = false;
    template<typename Op>
    inline constexpr bool has_declared_idemp<Op, std::void_t<decltype(Op::declared_idempotent)>> =
        Op::declared_idempotent;
}  // namespace detail

/// True if Op declares associativity AND T is an exact type (integral/enum).
/// Users may specialise for additional type/op combinations.
template<typename Op, typename T>
inline constexpr bool declares_associative_v =
    detail::has_declared_assoc<Op> && detail::is_exact_v<T>;

/// True if Op declares commutativity AND T is an exact type.
template<typename Op, typename T>
inline constexpr bool declares_commutative_v =
    detail::has_declared_commut<Op> && detail::is_exact_v<T>;

/// True if Op declares idempotency AND T is an exact type.
template<typename Op, typename T>
inline constexpr bool declares_idempotent_v =
    detail::has_declared_idemp<Op> && detail::is_exact_v<T>;

// ---------------------------------------------------------------------------
// Convenience concepts
// ---------------------------------------------------------------------------

/// Does this operation act as identity (no-op transform)?
template<typename Op>
concept is_identity_op = is_identity_transform_v<Op>;

/// Is this operation a power transform?
template<typename Op>
concept is_power_op = is_power_transform_v<Op>;

/// Does this operation declare associativity for the abstract algebra?
/// (Type-independent; for type-safe checks use declares_associative_v<Op, T>.)
template<typename Op>
concept declares_associative = detail::has_declared_assoc<Op>;

/// Does this operation declare commutativity for the abstract algebra?
template<typename Op>
concept declares_commutative = detail::has_declared_commut<Op>;

/// Can this operation produce an identity element for type T?
template<typename Op, typename T>
concept has_identity = requires { { Op::template identity<T>() } -> std::same_as<T>; };

}  // namespace ct_dp::algebra

#endif  // CT_DP_ALGEBRA_OPERATIONS_H
