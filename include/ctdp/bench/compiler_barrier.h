#ifndef CTDP_BENCH_COMPILER_BARRIER_H
#define CTDP_BENCH_COMPILER_BARRIER_H

// ctdp::bench::compiler_barrier — Anti-elision primitives for benchmarking
//
// DoNotOptimize(uint64_t)  — force a value to be materialized in a register
// ClobberMemory()          — force all memory-resident state to be flushed
// result_token             — opaque type returned from benchmarked callables
// mix_token()              — combine multiple result_tokens into one
//
// Design: Based on Google Benchmark patterns, adapted for C++20.
// The result_token type prevents dead code elimination without
// requiring volatile or external linkage hacks.

#include <cstdint>
#include <type_traits>

namespace ctdp::bench {

// ─── Anti-elision barriers ──────────────────────────────────────────

/// Force the compiler to materialise `val` in a register.
/// Prevents dead-code elimination of benchmark computation results.
inline void DoNotOptimize(std::uint64_t val) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(val) : );
#elif defined(_MSC_VER)
    // MSVC: volatile read to prevent optimisation
    volatile auto sink = val;
    (void)sink;
#else
    volatile auto sink = val;
    (void)sink;
#endif
}

/// Overload for signed integers
inline void DoNotOptimize(std::int64_t val) noexcept {
    DoNotOptimize(static_cast<std::uint64_t>(val));
}

/// Overload for 32-bit unsigned
inline void DoNotOptimize(std::uint32_t val) noexcept {
    DoNotOptimize(static_cast<std::uint64_t>(val));
}

/// Overload for 32-bit signed
inline void DoNotOptimize(std::int32_t val) noexcept {
    DoNotOptimize(static_cast<std::uint64_t>(static_cast<std::uint32_t>(val)));
}

/// Overload for double
inline void DoNotOptimize(double val) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "x"(val) : );
#else
    volatile auto sink = val;
    (void)sink;
#endif
}

/// Overload for float
inline void DoNotOptimize(float val) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "x"(val) : );
#else
    volatile auto sink = val;
    (void)sink;
#endif
}

/// Overload for pointers
template <typename T>
inline void DoNotOptimize(T* val) noexcept {
    DoNotOptimize(reinterpret_cast<std::uint64_t>(val));
}

/// Force all memory-resident state to be considered clobbered.
/// Use after writing data structures to prevent reordering.
inline void ClobberMemory() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : : "memory");
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// ─── result_token ───────────────────────────────────────────────────

/// Opaque token returned by benchmarked callables.
/// Ensures the callable's result is consumed (not dead-code-eliminated)
/// while keeping the anti-elision concern out of the callable itself.
///
/// Usage:
///   auto fn = [&]() -> result_token {
///       auto r = do_work();
///       return result_token{static_cast<uint64_t>(r)};
///   };
struct result_token {
    std::uint64_t value{0};

    constexpr result_token() noexcept = default;
    constexpr explicit result_token(std::uint64_t v) noexcept : value{v} {}

    /// Implicit conversion from common types
    constexpr result_token(std::int64_t v) noexcept
        : value{static_cast<std::uint64_t>(v)} {}
    constexpr result_token(std::int32_t v) noexcept
        : value{static_cast<std::uint64_t>(static_cast<std::uint32_t>(v))} {}
    constexpr result_token(std::uint32_t v) noexcept
        : value{static_cast<std::uint64_t>(v)} {}
    constexpr result_token(bool v) noexcept
        : value{v ? 1u : 0u} {}

    /// Consume the token — forces materialisation
    void consume() const noexcept {
        DoNotOptimize(value);
    }
};

/// Combine multiple result_tokens into one (XOR mix).
/// Prevents the compiler from proving any individual token is unused.
[[nodiscard]] constexpr result_token mix_token(result_token a,
                                                result_token b) noexcept {
    return result_token{a.value ^ b.value};
}

/// Variadic mix for convenience
template <typename... Tokens>
    requires (std::is_same_v<Tokens, result_token> && ...)
[[nodiscard]] constexpr result_token mix_token(result_token first,
                                                Tokens... rest) noexcept {
    if constexpr (sizeof...(rest) == 0) {
        return first;
    } else {
        return mix_token(first, mix_token(rest...));
    }
}

} // namespace ctdp::bench

#endif // CTDP_BENCH_COMPILER_BARRIER_H
