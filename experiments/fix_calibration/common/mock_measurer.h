#ifndef CTDP_FIX_EXPERIMENT_MOCK_MEASURER_H
#define CTDP_FIX_EXPERIMENT_MOCK_MEASURER_H

// experiments/fix_calibration/common/mock_measurer.h
//
// Deterministic mock measurement for testing the ML experiment pipeline.
//
// Design contract (dev plan v3):
//   - stable_noise(config, seed, sigma) → same config + same seed = identical result
//   - Hash-based, not RNG-state-based: no ordering dependency
//   - Two separate interfaces: mock_measurer for tests, compiled_measurer
//     (PR5) for real RDTSC.  NOT polymorphic.
//   - The experiment_runner<Measurer> template instantiates either.
//
// Synthetic cost model:
//   Per-field cost depends on Strategy × digit_count, modelling the real
//   observation that Unrolled wins for short fixed-width fields while
//   SWAR amortises better on wider fields.  The landscape has a genuine
//   mixed optimum (not all-one-strategy), matching production measurements.
//
//   true_p50 = Σ field_cost(strategy_i, digits_i)
//   true_p99 = true_p50 × tail_factor(config)
//   measured = true + stable_noise(config, seed, sigma)
//
// Dependencies: baselines.h (measurement_result), fix_et_parser.h
// C++20

#include "baselines.h"

#include <ctdp/calibrator/fix_et_parser.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace ctdp::experiment::fix_calibration {

namespace fix = ctdp::calibrator::fix;

// ─────────────────────────────────────────────────────────────────────
//  Hash utilities
//
//  splitmix64-based hash for deterministic noise.  Two independent
//  hash streams (for p50 and p99) from a single config+seed pair.
// ─────────────────────────────────────────────────────────────────────

namespace mock_detail {

/// splitmix64 finaliser — bijective 64-bit mix.
[[nodiscard]] inline constexpr uint64_t splitmix64(uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return z;
}

/// Hash a fix_config into a 64-bit digest, seeded.
[[nodiscard]] inline constexpr uint64_t hash_config(
    fix::fix_config const& cfg, uint64_t seed) noexcept
{
    uint64_t h = seed;
    for (std::size_t i = 0; i < static_cast<std::size_t>(fix::num_fields); ++i) {
        // Combine: field index, strategy value, running hash
        h ^= (static_cast<uint64_t>(cfg[i]) + 1) * (i + 1);
        h = splitmix64(h);
    }
    return h;
}

/// Convert a 64-bit hash to uniform double in (0, 1).
[[nodiscard]] inline constexpr double hash_to_uniform(uint64_t h) noexcept {
    // Use upper 52 bits → [1, 2) via IEEE754, then subtract 1.
    // Simpler: just use integer division.
    return static_cast<double>(h >> 11) / 9007199254740992.0; // 2^53
}

/// Box-Muller transform: two uniform values → one standard normal sample.
/// Uses the first hash for the pair.
[[nodiscard]] inline double hash_to_normal(uint64_t h) noexcept {
    uint64_t h2 = splitmix64(h);
    double u1 = hash_to_uniform(h);
    double u2 = hash_to_uniform(h2);
    // Clamp away from zero to avoid log(0)
    if (u1 < 1e-15) u1 = 1e-15;
    return std::sqrt(-2.0 * std::log(u1))
         * std::cos(2.0 * std::numbers::pi * u2);
}

} // namespace mock_detail

// ─────────────────────────────────────────────────────────────────────
//  stable_noise — deterministic Gaussian noise for a config
//
//  Guarantee: stable_noise(cfg, seed, sigma) always returns the same
//  value for the same (cfg, seed, sigma) triple.  No hidden state.
//
//  The `channel` parameter selects independent noise streams (0 for
//  p50, 1 for p99) so the two percentile noises are uncorrelated.
// ─────────────────────────────────────────────────────────────────────

[[nodiscard]] inline double stable_noise(
    fix::fix_config const& cfg,
    uint64_t seed,
    double sigma,
    uint64_t channel = 0) noexcept
{
    uint64_t h = mock_detail::hash_config(cfg, seed ^ (channel * 0xdeadbeefcafeULL));
    return mock_detail::hash_to_normal(h) * sigma;
}

// ─────────────────────────────────────────────────────────────────────
//  Synthetic cost model
//
//  Two components:
//    1. Per-field cost = f(Strategy, digits)
//    2. ILP adjacency bonus: consecutive Unrolled fields share decode
//       bandwidth and enable cross-field instruction-level parallelism
//       that SWAR/Loop/Generic cannot exploit.
//
//  Calibrated so that:
//    - all_generic ≈ 48 ns p50 (slowest)
//    - all_loop    ≈ 31 ns p50
//    - all_swar    ≈ 28 ns p50
//    - all_unrolled ≈ 28 ns p50 (fast per field but icache bloat)
//    - phase10g_optimal ≈ 25 ns p50 (genuine mixed optimum)
//    - tail_factor ranges from 1.20 (compact code) to 1.45 (bloated)
//
//  The crossover Unrolled → SWAR occurs around 5 digits: Unrolled wins
//  for short fixed-width fields, SWAR amortises better on wide fields.
//  The ILP bonus for consecutive Unrolled runs creates a genuine mixed
//  optimum that is NOT all-one-strategy.
// ─────────────────────────────────────────────────────────────────────

namespace mock_detail {

/// Per-field parse cost in nanoseconds.
[[nodiscard]] inline constexpr double field_cost(
    fix::Strategy strategy, int digits) noexcept
{
    double d = static_cast<double>(digits);
    switch (strategy) {
        case fix::Strategy::Unrolled:
            // Low linear cost but quadratic icache pressure per field.
            // Sweet spot: short fields (2–4 digits).
            // Crossover to SWAR at d ≈ 5.
            return 0.25 * d + 0.05 * d * d;

        case fix::Strategy::SWAR:
            // Fixed word-setup cost + small per-digit residual.
            // Best for d ≥ 6 where setup amortises.
            return 1.40 + 0.20 * d;

        case fix::Strategy::Loop:
            // Branch overhead + linear counted iteration.
            return 0.90 + 0.35 * d;

        case fix::Strategy::Generic:
            // Delimiter-scan fallback: high constant, mild slope.
            return 3.00 + 0.20 * d;

        default:
            return 5.0;
    }
}

/// ILP adjacency bonus: each consecutive pair of Unrolled fields
/// gains a fixed discount because the compiler can interleave their
/// decode logic.  This is the key non-linearity that makes a mixed
/// config better than all-one-strategy.
inline constexpr double ilp_bonus_per_pair = 0.35;

/// Count consecutive Unrolled pairs in a config.
[[nodiscard]] inline constexpr int count_unrolled_pairs(
    fix::fix_config const& cfg) noexcept
{
    int pairs = 0;
    for (std::size_t i = 1;
         i < static_cast<std::size_t>(fix::num_fields); ++i)
    {
        if (cfg[i] == fix::Strategy::Unrolled &&
            cfg[i - 1] == fix::Strategy::Unrolled)
            ++pairs;
    }
    return pairs;
}

/// Total p50 for a complete config.
[[nodiscard]] inline constexpr double config_p50(
    fix::fix_config const& cfg) noexcept
{
    double total = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(fix::num_fields); ++i) {
        total += field_cost(cfg[i], fix::field_digits[i]);
    }
    total -= ilp_bonus_per_pair * count_unrolled_pairs(cfg);
    return total;
}

/// Tail factor: p99 / p50.  Depends on code diversity (more strategies
/// mixed → more branch mispredictions in the tail) and total code size
/// (more unrolled → more icache misses in the tail).
[[nodiscard]] inline constexpr double tail_factor(
    fix::fix_config const& cfg) noexcept
{
    // Count how many distinct strategies are used.
    std::array<int, fix::NUM_STRATEGIES> counts{};
    for (auto s : cfg) counts[static_cast<std::size_t>(s)]++;

    int distinct = 0;
    int unrolled_count = counts[0];
    for (int c : counts) {
        if (c > 0) ++distinct;
    }

    // Base tail: 1.20 (tight code, single strategy)
    // +0.04 per additional distinct strategy (branch diversity in tail)
    // +0.02 per unrolled field beyond 6 (icache eviction in tail)
    double tf = 1.20;
    tf += 0.04 * static_cast<double>(distinct - 1);
    if (unrolled_count > 6)
        tf += 0.02 * static_cast<double>(unrolled_count - 6);
    return tf;
}

} // namespace mock_detail

// ─────────────────────────────────────────────────────────────────────
//  mock_measurer
//
//  Stateless (all parameters stored as members).
//  measure(cfg) returns a deterministic measurement_result.
//
//  Usage:
//    mock_measurer m{.seed = 42, .noise_sigma = 0.5};
//    auto r = m.measure(some_config);
//    // r.p50_ns, r.p99_ns — deterministic for (some_config, seed=42)
// ─────────────────────────────────────────────────────────────────────

struct mock_measurer {
    uint64_t seed         = 42;     ///< Noise seed
    double   noise_sigma  = 0.5;    ///< Noise std-dev in nanoseconds

    /// Measure a config.  Deterministic: same (cfg, seed) → same result.
    [[nodiscard]] measurement_result measure(
        fix::fix_config const& cfg) const noexcept
    {
        double true_p50 = mock_detail::config_p50(cfg);
        double tf       = mock_detail::tail_factor(cfg);
        double true_p99 = true_p50 * tf;

        double noise_p50 = stable_noise(cfg, seed, noise_sigma, /*channel=*/0);
        double noise_p99 = stable_noise(cfg, seed, noise_sigma, /*channel=*/1);

        // Clamp to positive (latency can't be negative).
        double p50 = std::max(0.1, true_p50 + noise_p50);
        double p99 = std::max(p50,  true_p99 + noise_p99); // p99 ≥ p50

        return {p50, p99};
    }

    /// Ground truth p50 (no noise) — for test assertions.
    [[nodiscard]] static constexpr double true_p50(
        fix::fix_config const& cfg) noexcept
    {
        return mock_detail::config_p50(cfg);
    }

    /// Ground truth p99 (no noise) — for test assertions.
    [[nodiscard]] static constexpr double true_p99(
        fix::fix_config const& cfg) noexcept
    {
        return mock_detail::config_p50(cfg) * mock_detail::tail_factor(cfg);
    }

    /// Ground truth tail factor — for test assertions.
    [[nodiscard]] static constexpr double true_tail_factor(
        fix::fix_config const& cfg) noexcept
    {
        return mock_detail::tail_factor(cfg);
    }
};

// ─────────────────────────────────────────────────────────────────────
//  zero_noise_measurer
//
//  Convenience: returns exact ground truth with no noise.
//  Useful for unit tests that need exact values.
// ─────────────────────────────────────────────────────────────────────

struct zero_noise_measurer {
    [[nodiscard]] measurement_result measure(
        fix::fix_config const& cfg) const noexcept
    {
        return {mock_detail::config_p50(cfg),
                mock_detail::config_p50(cfg) * mock_detail::tail_factor(cfg)};
    }
};

} // namespace ctdp::experiment::fix_calibration

#endif // CTDP_FIX_EXPERIMENT_MOCK_MEASURER_H
