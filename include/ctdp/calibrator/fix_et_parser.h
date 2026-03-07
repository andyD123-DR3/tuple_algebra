#ifndef CTDP_CALIBRATOR_FIX_ET_PARSER_H
#define CTDP_CALIBRATOR_FIX_ET_PARSER_H

// ctdp::calibrator::fix -- Expression-template FIX parser infrastructure
//
// Template-specialised parser for FIX protocol integer fields.
// Each field in the message is parsed by a compile-time-chosen
// strategy (Unrolled, SWAR, Loop, Generic).  The compiler sees
// the entire 12-field parse chain as a single function, enabling
// cross-field ILP and cache-line amortisation that runtime dispatch
// cannot achieve.
//
// Key lesson from Phase 10: runtime dispatch overestimates latency
// by 1.2-1.9x compared to template-specialised code.  Calibration
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
#include <ctdp/bench/perf_counter.h>
#include <ctdp/bench/percentile.h>
#include <ctdp/bench/distribution_fit.h>

// Portable always-inline: GCC/Clang use the attribute, MSVC uses __forceinline.
#if defined(_MSC_VER)
#  define CTDP_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define CTDP_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#  define CTDP_ALWAYS_INLINE inline
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace ctdp::calibrator::fix {

// ===================================================================
// Strategy enum and calibrator data types are canonical in data_point.h.
#include <ctdp/calibrator/fix/data_point.h>


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

/// Config -> string representation (e.g. "UUSLSUUSSUUU")
[[nodiscard]] inline std::string config_to_string(fix_config const& cfg) {
    std::string s;
    s.reserve(num_fields);
    for (auto st : cfg) s += strategy_char(st);
    return s;
}

/// String -> config
[[nodiscard]] inline fix_config config_from_string(std::string_view s) {
    fix_config cfg{};
    for (int i = 0; i < num_fields && i < static_cast<int>(s.size()); ++i) {
        cfg[static_cast<std::size_t>(i)] = strategy_from_char(s[static_cast<std::size_t>(i)]).value_or(Strategy::Generic);
    }
    return cfg;
}


// ===================================================================
// Per-strategy parsing implementations (template-specialised)
// ===================================================================

// --- Unrolled: compile-time digit count, no branches ------------

template<int N>
CTDP_ALWAYS_INLINE std::uint64_t parse_unrolled(const char* s) noexcept {
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

// --- SWAR: 4-digit chunks via word-level parallelism ------------

template<int N>
CTDP_ALWAYS_INLINE std::uint64_t parse_swar(const char* s) noexcept {
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

// --- Loop: simple counted loop, compiler decides unrolling ------

template<int N>
CTDP_ALWAYS_INLINE std::uint64_t parse_loop(const char* s) noexcept {
    std::uint64_t result = 0;
    for (int i = 0; i < N; ++i) {
        result = result * 10 + static_cast<std::uint64_t>(s[i] - '0');
    }
    return result;
}

// --- Generic: loop with bounds checking (most defensive) --------

template<int N>
CTDP_ALWAYS_INLINE std::uint64_t parse_generic(const char* s) noexcept {
    std::uint64_t result = 0;
    for (int i = 0; i < N; ++i) {
        char c = s[i];
        if (c < '0' || c > '9') break;
        result = result * 10 + static_cast<std::uint64_t>(c - '0');
    }
    return result;
}

// --- Strategy dispatch (compile-time) ---------------------------

template<Strategy S, int Digits>
CTDP_ALWAYS_INLINE std::uint64_t parse_field(const char* p) noexcept {
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


// ===================================================================
// Expression-template parser (the full 12-field ET chain)
// ===================================================================

namespace detail {

/// Recursive field parser: processes field I, then I+1, etc.
/// Accumulates results via XOR to prevent dead code elimination.
template<fix_config Config, int I = 0>
CTDP_ALWAYS_INLINE std::uint64_t parse_fields(const char* msg,
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
/// and cache-line amortisation -- effects that runtime dispatch misses.
///
/// @tparam Config  Array of 12 strategies, one per field.
template<fix_config Config>
struct fix_et_parser {

    /// Parse a FIX message, returning an anti-elision token.
    ///
    /// @param msg      Pointer to message data (digit characters)
    /// @param offsets  Byte offsets to each field within the message
    [[nodiscard]] static inline
    bench::result_token parse(const char* msg,
                                      const int* offsets) noexcept
    {
        return bench::result_token{
            detail::parse_fields<Config>(msg, offsets, 0)
        };
    }
};


// ===================================================================
// Message generation (for benchmarking)
// ===================================================================

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


// ===================================================================
// Measurement infrastructure
// ===================================================================

// --- TSC calibration --------------------------------------------

/// Calibrate the TSC frequency by timing a known-duration busy-wait
/// against steady_clock.  Returns cycles per nanosecond.
///
/// Result is cached after first call -- TSC frequency is constant
/// within a process lifetime on modern x86 (invariant TSC).
[[nodiscard]] inline double calibrate_tsc() noexcept {
    static double cached = 0.0;
    if (cached > 0.0) return cached;

    using clock = std::chrono::steady_clock;
    constexpr auto target_ns = 10'000'000;  // 10 ms

    // Warmup: let the TSC stabilise
    auto warm = bench::rdtsc_start();
    bench::DoNotOptimize(warm);

    auto wall_start = clock::now();
    auto tsc_start  = bench::rdtsc_start();

    // Busy-wait for target duration
    for (;;) {
        auto now = clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - wall_start).count();
        if (elapsed >= target_ns) break;
    }

    auto tsc_end  = bench::rdtsc_end();
    auto wall_end = clock::now();

    double wall_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            wall_end - wall_start).count());
    double tsc_delta = static_cast<double>(tsc_end - tsc_start);

    cached = tsc_delta / wall_ns;  // cycles per nanosecond
    return cached;
}

/// Configuration for measurement runs.
struct measurement_config {
    std::size_t samples       = 100'000; ///< Number of timing samples
    std::size_t batch_size    = 64;      ///< Parses per timing sample
    std::size_t warmup_parses = 4'000;   ///< Warmup invocations
    double      cycles_per_ns = 0.0;     ///< TSC freq (0 = auto-calibrate)
};

// --- Per-invocation rdtsc measurement ---------------------------

/// Measure per-parse latency using rdtsc with no batching.
///
/// Each sample is a single parse timed by rdtsc.  Overhead is
/// ~8 ns per sample (the rdtsc pair), giving ~0.3 ns resolution
/// on a 3 GHz core.  Suitable for distribution analysis where
/// per-invocation variance matters.
///
/// @tparam Config   The parser configuration to measure
/// @param messages  Pre-generated message pool
/// @param cfg       Measurement parameters
/// @return          Complete percentile distribution (in nanoseconds)
template<fix_config Config>
[[nodiscard]] bench::percentile_result measure_config_rdtsc(
    std::vector<std::string> const& messages,
    measurement_config const& cfg)
{
    double cpns = cfg.cycles_per_ns;
    if (cpns <= 0.0) cpns = calibrate_tsc();

    std::vector<double> latencies;
    latencies.reserve(cfg.samples);

    auto const* offsets = standard_offsets.data();
    std::size_t pool_size = messages.size();

    // Warmup
    for (std::size_t i = 0; i < cfg.warmup_parses; ++i) {
        auto tok = fix_et_parser<Config>::parse(
            messages[i % pool_size].data(), offsets);
        bench::DoNotOptimize(tok.value);
        bench::ClobberMemory();
    }

    // Measurement: one parse per sample, rdtsc timing
    for (std::size_t i = 0; i < cfg.samples; ++i) {
        auto const& msg = messages[i % pool_size];

        auto tsc0 = bench::rdtsc_start();
        auto tok = fix_et_parser<Config>::parse(msg.data(), offsets);
        auto tsc1 = bench::rdtsc_end();

        bench::DoNotOptimize(tok.value);

        double cycles = static_cast<double>(tsc1 - tsc0);
        latencies.push_back(cycles / cpns);
    }

    return bench::compute_percentiles(
        std::span<const double>{latencies});
}

// --- Batched measurement ----------------------------------------

/// Measure per-parse latency using batched timing.
///
/// Each sample times `batch_size` consecutive parses in a tight loop,
/// then divides by batch_size.  This amortises clock overhead to
/// give sub-nanosecond resolution on mean latency.
///
/// The distribution reflects batch-averaged behaviour: it smooths
/// out per-invocation spikes but captures sustained performance
/// variations (icache eviction, frequency transitions, etc.).
///
/// @tparam Config   The parser configuration to measure
/// @param messages  Pre-generated message pool
/// @param cfg       Measurement parameters
/// @return          Complete percentile distribution (in nanoseconds)
template<fix_config Config>
[[nodiscard]] bench::percentile_result measure_config_batched(
    std::vector<std::string> const& messages,
    measurement_config const& cfg)
{
    double cpns = cfg.cycles_per_ns;
    if (cpns <= 0.0) cpns = calibrate_tsc();

    std::size_t batch = cfg.batch_size;
    if (batch == 0) batch = 1;

    std::vector<double> latencies;
    latencies.reserve(cfg.samples);

    auto const* offsets = standard_offsets.data();
    std::size_t pool_size = messages.size();

    // Warmup
    for (std::size_t i = 0; i < cfg.warmup_parses; ++i) {
        auto tok = fix_et_parser<Config>::parse(
            messages[i % pool_size].data(), offsets);
        bench::DoNotOptimize(tok.value);
        bench::ClobberMemory();
    }

    // Measurement: batch_size parses per sample
    std::size_t msg_idx = 0;
    for (std::size_t s = 0; s < cfg.samples; ++s) {
        auto tsc0 = bench::rdtsc_start();

        for (std::size_t b = 0; b < batch; ++b) {
            auto tok = fix_et_parser<Config>::parse(
                messages[msg_idx % pool_size].data(), offsets);
            bench::DoNotOptimize(tok.value);
            ++msg_idx;
        }

        auto tsc1 = bench::rdtsc_end();

        double total_cycles = static_cast<double>(tsc1 - tsc0);
        double per_parse_ns = total_cycles / (cpns * static_cast<double>(batch));
        latencies.push_back(per_parse_ns);
    }

    return bench::compute_percentiles(
        std::span<const double>{latencies});
}

// --- Default measure_config (batched, backward compatible) ------

/// Measure the p99 latency of a specific ET parser configuration.
///
/// Uses batched rdtsc timing: 64 parses per sample, 100K samples.
/// Each reported value is the amortised per-parse latency in
/// nanoseconds with sub-nanosecond resolution.
///
/// @tparam Config   The parser configuration to measure
/// @param messages  Pre-generated message pool
/// @param samples   Number of timing samples (100K minimum for p99)
/// @return          Complete percentile distribution
template<fix_config Config>
[[nodiscard]] bench::percentile_result measure_config(
    std::vector<std::string> const& messages,
    std::size_t samples = 100'000)
{
    measurement_config cfg;
    cfg.samples    = samples;
    cfg.batch_size = 64;
    return measure_config_batched<Config>(messages, cfg);
}

// --- Measurement with hardware counters (two-pass) ---------------
//
// Pass 1: Batched rdtsc timing -> percentile distribution (as above)
// Pass 2: Large batch with perf_event counter groups -> aggregate
//          IPC, L1D miss rate, icache miss rate, DTLB miss rate
//
// The counter pass runs the same workload but brackets it with
// Tier 1 (instructions, branches, cache) and Tier 2 (L1D, L1I, DTLB)
// counter groups.  Results are per-parse averages over the batch.

/// Aggregated per-configuration metrics: timing + hardware counters.
struct config_metrics {
    bench::percentile_result timing;  ///< Empirical percentiles from rdtsc
    bench::distribution_result dist;  ///< Fitted distribution (lognormal/gamma)

    // Distribution-derived percentiles (more stable than empirical)
    double fitted_p50  = 0.0;        ///< p50 from fitted distribution
    double fitted_p99  = 0.0;        ///< p99 from fitted distribution
    double fitted_p999 = 0.0;        ///< p99.9 from fitted distribution
    double fitted_cv   = 0.0;        ///< Coefficient of variation
    double fitted_tail = 0.0;        ///< Fitted tail ratio (p99/p50)

    // Per-parse averages from counter pass
    double ipc              = 0.0;    ///< instructions per cycle
    double instructions     = 0.0;    ///< instructions per parse
    double cycles           = 0.0;    ///< cycles per parse
    double l1d_miss_rate    = 0.0;    ///< L1D read miss / L1D read access
    double l1d_misses       = 0.0;    ///< L1D read misses per parse
    double l1i_miss_rate    = 0.0;    ///< icache miss rate
    double l1i_misses       = 0.0;    ///< icache misses per parse
    double dtlb_miss_rate   = 0.0;    ///< DTLB miss rate
    double dtlb_misses      = 0.0;    ///< DTLB misses per parse
    double branch_miss_rate = 0.0;    ///< branch miss rate
    double cache_miss_rate  = 0.0;    ///< LL cache miss rate

    bool tier1_available    = false;  ///< Tier 1 counters worked
    bool tier2_available    = false;  ///< Tier 2 counters worked
};

/// Measure a configuration with timing, distribution fit, and hardware counters.
///
/// Three passes:
///   Pass 1: batched rdtsc -> raw samples -> empirical percentiles + distribution fit
///   Pass 2: Tier 1 counter group (instructions, branches, cache)
///   Pass 3: Tier 2 counter group (L1D, L1I, DTLB)
///
/// The distribution fit uses ALL timing samples to estimate lognormal/gamma
/// parameters, then derives p99 analytically. This is more stable than the
/// empirical p99 because it leverages the full distribution shape.
///
/// @tparam Config   The parser configuration to measure
/// @param messages  Pre-generated message pool
/// @param cfg       Measurement parameters
/// @return          Combined timing + distribution + counter metrics
template<fix_config Config>
[[nodiscard]] config_metrics measure_config_with_counters(
    std::vector<std::string> const& messages,
    measurement_config const& cfg = {})
{
    config_metrics result;

    double cpns = cfg.cycles_per_ns;
    if (cpns <= 0.0) cpns = calibrate_tsc();

    std::size_t batch = cfg.batch_size;
    if (batch == 0) batch = 1;

    auto const* offsets = standard_offsets.data();
    std::size_t pool_size = messages.size();

    // ---- Pass 1: Timing (batched rdtsc) + distribution fit ----
    {
        std::vector<double> latencies;
        latencies.reserve(cfg.samples);

        // Warmup
        for (std::size_t i = 0; i < cfg.warmup_parses; ++i) {
            auto tok = fix_et_parser<Config>::parse(
                messages[i % pool_size].data(), offsets);
            bench::DoNotOptimize(tok.value);
            bench::ClobberMemory();
        }

        // Collect raw per-batch latencies
        std::size_t msg_idx = 0;
        for (std::size_t s = 0; s < cfg.samples; ++s) {
            auto tsc0 = bench::rdtsc_start();

            for (std::size_t b = 0; b < batch; ++b) {
                auto tok = fix_et_parser<Config>::parse(
                    messages[msg_idx % pool_size].data(), offsets);
                bench::DoNotOptimize(tok.value);
                ++msg_idx;
            }

            auto tsc1 = bench::rdtsc_end();

            double total_cycles = static_cast<double>(tsc1 - tsc0);
            double per_parse_ns = total_cycles /
                (cpns * static_cast<double>(batch));
            latencies.push_back(per_parse_ns);
        }

        // Empirical percentiles
        result.timing = bench::compute_percentiles(
            std::span<const double>{latencies});

        // Fit distribution to raw samples
        result.dist = bench::fit_distribution(
            std::span<const double>{latencies});

        // Extract fitted values
        result.fitted_p50  = result.dist.fitted_p50();
        result.fitted_p99  = result.dist.fitted_p99();
        result.fitted_p999 = result.dist.lognormal.p999();
        result.fitted_cv   = result.dist.fitted_cv();
        result.fitted_tail = result.dist.fitted_tail_ratio();
    }

    // Pass 2: counter groups over a single large batch
    std::size_t counter_iters = cfg.samples;  // same workload as timing

    // Warmup
    for (std::size_t i = 0; i < cfg.warmup_parses; ++i) {
        auto tok = fix_et_parser<Config>::parse(
            messages[i % pool_size].data(), offsets);
        bench::DoNotOptimize(tok.value);
        bench::ClobberMemory();
    }

    // Tier 1: instructions, branches, cache (aggregate)
    {
        bench::perf_counter_group tier1;
        result.tier1_available = tier1.tier1_available();

        if (result.tier1_available) {
            tier1.start();
            for (std::size_t i = 0; i < counter_iters; ++i) {
                auto tok = fix_et_parser<Config>::parse(
                    messages[i % pool_size].data(), offsets);
                bench::DoNotOptimize(tok.value);
            }
            tier1.stop();

            auto snap = tier1.snapshot();
            double n = static_cast<double>(counter_iters);

            result.ipc = (snap.tsc_cycles > 0)
                ? static_cast<double>(snap.instructions) /
                  static_cast<double>(snap.tsc_cycles)
                : 0.0;
            result.instructions = static_cast<double>(snap.instructions) / n;
            result.cycles = static_cast<double>(snap.tsc_cycles) / n;
            result.cache_miss_rate = (snap.cache_references > 0)
                ? static_cast<double>(snap.cache_misses) /
                  static_cast<double>(snap.cache_references)
                : 0.0;
            result.branch_miss_rate = (snap.branches > 0)
                ? static_cast<double>(snap.branch_misses) /
                  static_cast<double>(snap.branches)
                : 0.0;
        }
    }

    // Tier 2: L1D, L1I (icache), DTLB
    {
        bench::cache_hierarchy_group tier2;
        result.tier2_available = tier2.available();

        if (result.tier2_available) {
            // Second warmup to re-prime caches after Tier 1
            for (std::size_t i = 0; i < cfg.warmup_parses; ++i) {
                auto tok = fix_et_parser<Config>::parse(
                    messages[i % pool_size].data(), offsets);
                bench::DoNotOptimize(tok.value);
                bench::ClobberMemory();
            }

            tier2.start();
            for (std::size_t i = 0; i < counter_iters; ++i) {
                auto tok = fix_et_parser<Config>::parse(
                    messages[i % pool_size].data(), offsets);
                bench::DoNotOptimize(tok.value);
            }
            tier2.stop();

            bench::counter_snapshot snap;
            tier2.fill_snapshot(snap);
            double n = static_cast<double>(counter_iters);

            result.l1d_miss_rate = (snap.l1d_read_access > 0)
                ? static_cast<double>(snap.l1d_read_miss) /
                  static_cast<double>(snap.l1d_read_access)
                : 0.0;
            result.l1d_misses = static_cast<double>(snap.l1d_read_miss) / n;
            result.l1i_miss_rate = (snap.l1i_read_access > 0)
                ? static_cast<double>(snap.l1i_read_miss) /
                  static_cast<double>(snap.l1i_read_access)
                : 0.0;
            result.l1i_misses = static_cast<double>(snap.l1i_read_miss) / n;
            result.dtlb_miss_rate = (snap.dtlb_read_access > 0)
                ? static_cast<double>(snap.dtlb_read_miss) /
                  static_cast<double>(snap.dtlb_read_access)
                : 0.0;
            result.dtlb_misses = static_cast<double>(snap.dtlb_read_miss) / n;
        }
    }

    return result;
}


// ===================================================================
// Constexpr PRNG for compile-time config generation
// ===================================================================

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

// --- Named baseline configurations -----------------------------

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

#undef CTDP_ALWAYS_INLINE

#endif // CTDP_CALIBRATOR_FIX_ET_PARSER_H
