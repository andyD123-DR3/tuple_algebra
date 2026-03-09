#ifndef CTDP_FIX_EXPERIMENT_SWEEP_CONFIGS_H
#define CTDP_FIX_EXPERIMENT_SWEEP_CONFIGS_H

// experiments/fix_calibration/common/sweep_configs.h
//
// Constexpr generation of all configs in the consensus subspace.
//
// 7 field positions are fixed by consensus from Programs A–E.
// 5 field positions are uncertain (positions 1, 6, 7, 8, 9).
// 4 strategies per uncertain position → 4^5 = 1,024 configs.
//
// Program F measures all 1,024 exhaustively to establish the exact
// optimum within the reduced consensus subspace.  This is NOT the
// full 4^12 space — the consensus locking restricts the search.
// The fixed-position audit checks whether the locked positions
// were safe to fix.

#include "baselines.h"
#include "experiment_config.h"

#include <ctdp/calibrator/fix_et_parser.h>

#include <array>
#include <cstddef>

namespace ctdp::fix_experiment {

namespace fix = ctdp::calibrator::fix;

namespace sweep_detail {

/// Build a single sweep config from a 5-digit base-4 index.
/// Digit i selects the strategy for CONSENSUS_UNCERTAIN[i].
/// Fixed positions come from CONSENSUS_FIXED.
[[nodiscard]] inline constexpr fix::fix_config
make_sweep_config(int index) noexcept {
    fix::fix_config cfg{};

    // Fill fixed positions from consensus
    for (int i = 0; i < N_FIXED; ++i)
        cfg[static_cast<std::size_t>(CONSENSUS_FIXED[i].position)] =
            CONSENSUS_FIXED[i].strategy;

    // Fill uncertain positions from the base-4 index
    int remaining = index;
    for (int i = 0; i < N_UNCERTAIN; ++i) {
        int s = remaining % fix::NUM_STRATEGIES;
        remaining /= fix::NUM_STRATEGIES;
        cfg[static_cast<std::size_t>(CONSENSUS_UNCERTAIN[i])] =
            static_cast<fix::Strategy>(s);
    }

    return cfg;
}

/// Generate all 1,024 sweep configs at compile time.
template<std::size_t... Is>
[[nodiscard]] constexpr auto
generate_sweep_impl(std::index_sequence<Is...>) noexcept {
    return std::array<fix::fix_config, sizeof...(Is)>{
        make_sweep_config(static_cast<int>(Is))...
    };
}

} // namespace sweep_detail

/// All 1,024 configs in the consensus subspace.
/// Fixed positions: {0:U, 2:U, 3:S, 4:S, 5:S, 10:S, 11:S}
/// Swept positions: {1, 6, 7, 8, 9} × {U, S, L, G}
inline constexpr auto sweep_configs =
    sweep_detail::generate_sweep_impl(
        std::make_index_sequence<static_cast<std::size_t>(N_SWEEP)>{});

static_assert(sweep_configs.size() == 1024);

// ── Fixed-position audit configs ─────────────────────────────────────
//
// For each of the 7 consensus-fixed positions, generate 3 one-flip
// variants (the 3 strategies that were NOT selected by consensus).
// Total: 7 × 3 = 21 audit configs.
//
// These are measured alongside the sweep to verify that the consensus
// locking was safe.  If any audit config beats the sweep winner, the
// consensus at that position is suspect.

inline constexpr std::size_t N_AUDIT = static_cast<std::size_t>(N_FIXED) * 3;

namespace sweep_detail {

[[nodiscard]] inline constexpr auto generate_audit_configs() noexcept {
    // Base config: sweep_configs[0] is as good a starting point as any
    // (all uncertain positions at strategy 0 = Unrolled).
    // The audit flips only fixed positions, so the uncertain positions
    // don't matter — we use the sweep winner from the runtime results.
    // However, for compile-time generation we need a fixed base.
    // We use the first sweep config; the program re-measures with the
    // actual sweep winner as base at runtime.
    std::array<fix::fix_config, N_FIXED * 3> audit{};
    auto base = sweep_configs[0];

    std::size_t idx = 0;
    for (int f = 0; f < N_FIXED; ++f) {
        int pos = CONSENSUS_FIXED[f].position;
        auto fixed_strat = CONSENSUS_FIXED[f].strategy;
        for (int s = 0; s < fix::NUM_STRATEGIES; ++s) {
            auto alt = static_cast<fix::Strategy>(s);
            if (alt == fixed_strat) continue;
            auto cfg = base;
            cfg[static_cast<std::size_t>(pos)] = alt;
            audit[idx++] = cfg;
        }
    }
    return audit;
}

} // namespace sweep_detail

/// 21 audit configs: one-flip variants at each consensus-fixed position.
inline constexpr auto audit_configs = sweep_detail::generate_audit_configs();

static_assert(audit_configs.size() == 21);

// ── Combined ancillary pool for dispatch table ───────────────────────
// Baselines (5) + audit (21) = 26 configs.
// Program F uses compiled_measurer_dual<sweep_configs, sweep_ancillary>.

inline constexpr auto sweep_ancillary = [] {
    std::array<fix::fix_config, num_baselines + N_AUDIT> arr{};
    std::size_t i = 0;
    for (const auto& cfg : baseline_configs) arr[i++] = cfg;
    for (const auto& cfg : audit_configs) arr[i++] = cfg;
    return arr;
}();

static_assert(sweep_ancillary.size() == 26);

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_SWEEP_CONFIGS_H
