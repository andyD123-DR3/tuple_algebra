#ifndef CTDP_CALIBRATOR_FIX_ET_PARSER_H
#define CTDP_CALIBRATOR_FIX_ET_PARSER_H

// ctdp::calibrator::fix — Expression-template FIX parser infrastructure
//
// Template-specialised parser for FIX protocol integer fields.
// Each field in the message is parsed by a compile-time-chosen
// strategy (Unrolled, SWAR, Loop, Generic).  The compiler sees
// the entire 12-field parse chain as a single function, enabling
// cross-field ILP and cache-line amortisation that runtime dispatch
// cannot achieve.
//
// Key lesson from Phase 10: runtime dispatch overestimates latency
// by 1.2–1.9× compared to template-specialised code.  Calibration
// must measure actual ET instantiations, not isolated strategies.
//
// Usage:
//
//   constexpr fix_config cfg = {S::U, S::U, S::S, S::L, ...};
//   auto tok = fix_et_parser<cfg>::parse(msg);
//
//   // Or with the measurement wrapper:
//   auto pctl = measure_config<cfg>(messages, 100'000);
//
// Integration:
//   Uses bench::result_token for anti-elision
//   Uses bench::percentile for p99 computation
//   Compatible with calibrator::Scenario concept

#include <ctdp/bench/compiler_barrier.h>
#include <ctdp/bench/percentile.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace ctdp::calibrator::fix {

// ═══════════════════════════════════════════════════════════════════
// Strategy enum and configuration
// ═══════════════════════════════════════════════════════════════════

/// Parsing strategy for a single integer field.
enum class Strategy : std::uint8_t {
    Unrolled = 0,   // U: fully unrolled, no branches
    SWAR     = 1,   // S: SWAR 4-at-a-time
    Loop     = 2,   // L: simple counted loop
    Generic  = 3    // G: loop with bounds checking
};

/// Number of strategies
inline constexpr int num_strategies = 4;

/// Short label for each strategy
[[nodiscard]] constexpr char strategy_char(Strategy s) noexcept {
    constexpr char labels[] = {'U', 'S', 'L', 'G'};
    return labels[static_cast<int>(s)];
}

/// Strategy from character label
[[nodiscard]] constexpr Strategy strategy_from_char(char c) noexcept {
    switch (c) {
        case 'U': return Strategy::Unrolled;
        case 'S': return Strategy::SWAR;
        case 'L': return Strategy::Loop;
        case 'G': return Strategy::Generic;
        default:  return Strategy::Generic;
    }
}

/// Number of integer fields in a FIX MarketData message.
inline constexpr int num_fields = 12;

/// A complete configuration: one strategy per field.
using fix_config = std::array<Strategy, num_fields>;

/// Digit counts for each field in a typical FIX MarketData message.
///
/// Represents: BodyLength(3), MsgSeqNum(6), AccountRef(4), Price(8),
///             OrderQty(6), SymbolSfx(4), Side(2), TransactTime(8),
///             TradeID(6), CheckSum(3), PartyRef(4), SecuritySfx(4)
inline constexpr std::array<int, num_fields> field_digits = {
    3, 6, 4, 8, 6, 4, 2, 8, 6, 3, 4, 4
};

/// Total digits across all fields (for message generation).
inline constexpr int total_digits = [] {
    int sum = 0;
    for (int d : field_digits) sum += d;
    return sum;
}();

/// Config → string representation (e.g. "UUSLSUUSSUUU")
[[nodiscard]] inline std::string config_to_string(fix_config const& cfg) {
    std::string s;
    s.reserve(num_fields);
    for (auto st : cfg) s += strategy_char(st);
    return s;
}

/// String → config
[[nodiscard]] inline fix_config config_from_string(std::string_view s) {
    fix_config cfg{};
    for (int i = 0; i < num_fields && i < static_cast<int>(s.size()); ++i) {
        cfg[static_cast<std::size_t>(i)] = strategy_from_char(s[static_cast<std::size_t>(i)]);
    }
    return cfg;
}


// ═══════════════════════════════════════════════════════════════════
// Per-strategy parsing implementations (template-specialised)
// ═══════════════════════════════════════════════════════════════════

// ─── Unrolled: compile-time digit count, no branches ────────────

template<int N>
[[gnu::always_inline]]
inline std::uint64_t parse_unrolled(const char* s) noexcept {
    std::uint64_t r = 0;
    // Fully unrolled via if constexpr chain
    if constexpr (N >= 1)  r = r * 10 + static_cast<std::uint64_t>(s[0]  - '0');
    if constexpr (N >= 2)  r = r * 10 + static_cast<std::uint64_t>(s[1]  - '0');
    if constexpr (N >= 3)  r = r * 10 + static_cast<std::uint64_t>(s[2]  - '0');
    if constexpr (N >= 4)  r = r * 10 + static_cast<std::uint64_t>(s[3]  - '0');
    if constexpr (N >= 5)  r = r * 10 + static_cast<std::uint64_t>(s[4]  - '0');
    if constexpr (N >= 6)  r = r * 10 + static_cast<std::uint64_t>(s[5]  - '0');
    if constexpr (N >= 7)  r = r * 10 + static_cast<std::uint64_t>(s[6]  - '0');
    if constexpr (N >= 8)  r = r * 10 + static_cast<std::uint64_t>(s[7]  - '0');
    if constexpr (N >= 9)  r = r * 10 + static_cast<std::uint64_t>(s[8]  - '0');
    if constexpr (N >= 10) r = r * 10 + static_cast<std::uint64_t>(s[9]  - '0');
    return r;
}

// ─── SWAR: 4-digit chunks via word-level parallelism ────────────

template<int N>
[[gnu::always_inline]]
inline std::uint64_t parse_swar(const char* s) noexcept {
    std::uint64_t result = 0;
    constexpr int full_chunks = N / 4;
    constexpr int tail = N - full_chunks * 4;

    // Process 4 digits at a time
    const char* p = s;
    for (int c = 0; c < full_chunks; ++c) {
        std::uint32_t chunk;
        std::memcpy(&chunk, p, 4);
        chunk -= 0x30303030u;
        // Little-endian byte order: byte0 is most significant digit
        std::uint64_t val =
            static_cast<std::uint64_t>(chunk & 0xFFu) * 1000 +
            static_cast<std::uint64_t>((chunk >> 8) & 0xFFu) * 100 +
            static_cast<std::uint64_t>((chunk >> 16) & 0xFFu) * 10 +
            static_cast<std::uint64_t>((chunk >> 24) & 0xFFu);
        result = result * 10000 + val;
        p += 4;
    }

    // Compile-time-known tail length
    if constexpr (tail >= 1) result = result * 10 + static_cast<std::uint64_t>(p[0] - '0');
    if constexpr (tail >= 2) result = result * 10 + static_cast<std::uint64_t>(p[1] - '0');
    if constexpr (tail >= 3) result = result * 10 + static_cast<std::uint64_t>(p[2] - '0');

    return result;
}

// ─── Loop: simple counted loop, compiler decides unrolling ──────

template<int N>
[[gnu::always_inline]]
inline std::uint64_t parse_loop(const char* s) noexcept {
    std::uint64_t result = 0;
    for (int i = 0; i < N; ++i) {
        result = result * 10 + static_cast<std::uint64_t>(s[i] - '0');
    }
    return result;
}

// ─── Generic: loop with bounds checking (most defensive) ────────

template<int N>
[[gnu::always_inline]]
inline std::uint64_t parse_generic(const char* s) noexcept {
    std::uint64_t result = 0;
    for (int i = 0; i < N; ++i) {
        char c = s[i];
        if (c < '0' || c > '9') break;
        result = result * 10 + static_cast<std::uint64_t>(c - '0');
    }
    return result;
}

// ─── Strategy dispatch (compile-time) ───────────────────────────

template<Strategy S, int Digits>
[[gnu::always_inline]]
inline std::uint64_t parse_field(const char* p) noexcept {
    if constexpr (S == Strategy::Unrolled) {
        return parse_unrolled<Digits>(p);
    } else if constexpr (S == Strategy::SWAR) {
        return parse_swar<Digits>(p);
    } else if constexpr (S == Strategy::Loop) {
        return parse_loop<Digits>(p);
    } else {
        return parse_generic<Digits>(p);
    }
}


// ═══════════════════════════════════════════════════════════════════
// Expression-template parser (the full 12-field ET chain)
// ═══════════════════════════════════════════════════════════════════

namespace detail {

/// Recursive field parser: processes field I, then I+1, etc.
/// Accumulates results via XOR to prevent dead code elimination.
template<fix_config Config, int I = 0>
[[gnu::always_inline]]
inline std::uint64_t parse_fields(const char* msg,
                                   const int* offsets,
                                   std::uint64_t acc) noexcept
{
    if constexpr (I < num_fields) {
        constexpr Strategy s = Config[static_cast<std::size_t>(I)];
        constexpr int digits = field_digits[static_cast<std::size_t>(I)];
        acc ^= parse_field<s, digits>(msg + offsets[I]);
        return parse_fields<Config, I + 1>(msg, offsets, acc);
    } else {
        return acc;
    }
}

} // namespace detail


/// Template-specialised FIX parser for a specific configuration.
///
/// The compiler sees the entire 12-field parse chain as one function.
/// This enables cross-field ILP, register allocation across the chain,
/// and cache-line amortisation — effects that runtime dispatch misses.
///
/// @tparam Config  Array of 12 strategies, one per field.
template<fix_config Config>
struct fix_et_parser {

    /// Parse a FIX message, returning an anti-elision token.
    ///
    /// @param msg      Pointer to message data (digit characters)
    /// @param offsets  Byte offsets to each field within the message
    [[nodiscard]] [[gnu::always_inline]]
    static bench::result_token parse(const char* msg,
                                      const int* offsets) noexcept
    {
        return bench::result_token{
            detail::parse_fields<Config>(msg, offsets, 0)
        };
    }
};


// ═══════════════════════════════════════════════════════════════════
// Message generation (for benchmarking)
// ═══════════════════════════════════════════════════════════════════

/// Pre-computed field offsets for the standard FIX message layout.
/// Fields are packed with SOH (0x01) delimiters between them.
[[nodiscard]] inline std::array<int, num_fields> compute_offsets() noexcept {
    std::array<int, num_fields> offsets{};
    int pos = 0;
    for (int i = 0; i < num_fields; ++i) {
        offsets[static_cast<std::size_t>(i)] = pos;
        pos += field_digits[static_cast<std::size_t>(i)] + 1; // +1 for SOH delimiter
    }
    return offsets;
}

inline const auto standard_offsets = compute_offsets();

/// Generate a pool of random FIX messages for benchmarking.
///
/// Each message contains 12 integer fields with the standard digit
/// counts, separated by SOH delimiters.  Messages are pre-generated
/// to avoid allocation during measurement.
///
/// @param count  Number of messages to generate
/// @param seed   PRNG seed for reproducibility
[[nodiscard]] inline std::vector<std::string>
generate_message_pool(std::size_t count, std::uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> digit_dist(0, 9);

    // Total message length: sum of digits + (num_fields - 1) delimiters + padding
    int msg_len = total_digits + num_fields; // includes trailing delimiter

    std::vector<std::string> pool;
    pool.reserve(count);

    for (std::size_t m = 0; m < count; ++m) {
        std::string msg(static_cast<std::size_t>(msg_len), '\x01');
        for (int f = 0; f < num_fields; ++f) {
            int off = standard_offsets[static_cast<std::size_t>(f)];
            int digits = field_digits[static_cast<std::size_t>(f)];
            for (int d = 0; d < digits; ++d) {
                msg[static_cast<std::size_t>(off + d)] =
                    static_cast<char>('0' + digit_dist(rng));
            }
        }
        pool.push_back(std::move(msg));
    }

    return pool;
}


// ═══════════════════════════════════════════════════════════════════
// Measurement infrastructure
// ═══════════════════════════════════════════════════════════════════

/// Measure the p99 latency of a specific ET parser configuration.
///
/// Performs `samples` parse invocations, records wall-clock time for
/// each, and computes the full percentile distribution.
///
/// @tparam Config   The parser configuration to measure
/// @param messages  Pre-generated message pool
/// @param samples   Number of parse invocations (100K minimum for p99)
/// @return          Complete percentile distribution
template<fix_config Config>
[[nodiscard]] bench::percentile_result measure_config(
    std::vector<std::string> const& messages,
    std::size_t samples = 100'000)
{
    std::vector<double> latencies;
    latencies.reserve(samples);

    auto const* offsets = standard_offsets.data();
    std::size_t pool_size = messages.size();

    // Warmup: 1000 invocations to prime branch predictor & icache
    for (std::size_t i = 0; i < 1000; ++i) {
        auto tok = fix_et_parser<Config>::parse(
            messages[i % pool_size].data(), offsets);
        bench::DoNotOptimize(tok.value);
        bench::ClobberMemory();
    }

    // Measurement: open-loop, fixed iteration count
    for (std::size_t i = 0; i < samples; ++i) {
        auto const& msg = messages[i % pool_size];

        auto t0 = std::chrono::steady_clock::now();
        auto tok = fix_et_parser<Config>::parse(msg.data(), offsets);
        auto t1 = std::chrono::steady_clock::now();

        bench::DoNotOptimize(tok.value);
        bench::ClobberMemory();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        latencies.push_back(ns);
    }

    return bench::compute_percentiles(
        std::span<const double>{latencies});
}


// ═══════════════════════════════════════════════════════════════════
// Constexpr PRNG for compile-time config generation
// ═══════════════════════════════════════════════════════════════════

/// SplitMix64 PRNG (constexpr-compatible).
constexpr std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// Generate a uniformly random configuration.
constexpr fix_config random_config(std::uint64_t& state) noexcept {
    fix_config cfg{};
    for (auto& s : cfg) {
        s = static_cast<Strategy>(splitmix64(state) % 4);
    }
    return cfg;
}

/// Generate N uniformly random configurations at compile time.
template<std::size_t N>
constexpr std::array<fix_config, N> generate_random_configs(
    std::uint64_t seed) noexcept
{
    std::array<fix_config, N> configs{};
    std::uint64_t state = seed;
    for (auto& cfg : configs) {
        cfg = random_config(state);
    }
    return configs;
}

// ─── Named baseline configurations ─────────────────────────────

/// All-Unrolled: maximum ILP, largest code size
inline constexpr fix_config all_unrolled = {
    Strategy::Unrolled, Strategy::Unrolled, Strategy::Unrolled,
    Strategy::Unrolled, Strategy::Unrolled, Strategy::Unrolled,
    Strategy::Unrolled, Strategy::Unrolled, Strategy::Unrolled,
    Strategy::Unrolled, Strategy::Unrolled, Strategy::Unrolled
};

/// All-SWAR: good throughput, moderate code size
inline constexpr fix_config all_swar = {
    Strategy::SWAR, Strategy::SWAR, Strategy::SWAR,
    Strategy::SWAR, Strategy::SWAR, Strategy::SWAR,
    Strategy::SWAR, Strategy::SWAR, Strategy::SWAR,
    Strategy::SWAR, Strategy::SWAR, Strategy::SWAR
};

/// All-Loop: minimal code, compiler decides unrolling
inline constexpr fix_config all_loop = {
    Strategy::Loop, Strategy::Loop, Strategy::Loop,
    Strategy::Loop, Strategy::Loop, Strategy::Loop,
    Strategy::Loop, Strategy::Loop, Strategy::Loop,
    Strategy::Loop, Strategy::Loop, Strategy::Loop
};

/// All-Generic: most defensive, smallest code
inline constexpr fix_config all_generic = {
    Strategy::Generic, Strategy::Generic, Strategy::Generic,
    Strategy::Generic, Strategy::Generic, Strategy::Generic,
    Strategy::Generic, Strategy::Generic, Strategy::Generic,
    Strategy::Generic, Strategy::Generic, Strategy::Generic
};


} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_ET_PARSER_H
