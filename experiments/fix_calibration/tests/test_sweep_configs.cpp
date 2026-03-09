// experiments/fix_calibration/tests/test_sweep_configs.cpp
//
// Unit tests for sweep_configs.h:
//   - sweep_configs array properties (size, uniqueness, fixed positions)
//   - audit_configs array properties (size, correct flips)
//   - sweep_ancillary composition

#include "sweep_configs.h"
#include "baselines.h"
#include "experiment_config.h"

#include <gtest/gtest.h>

#include <set>
#include <cstdint>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

// ── Helper: pack config into uint32 for uniqueness check ─────────────

static std::uint32_t pack_config(const fix::fix_config& c) {
    std::uint32_t k = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(fix::num_fields); ++i)
        k |= (static_cast<std::uint32_t>(c[i]) & 0x3u) << (2u * i);
    return k;
}

// =================================================================
// sweep_configs basics
// =================================================================

TEST(SweepConfigs, CorrectSize) {
    EXPECT_EQ(fxe::sweep_configs.size(), 1024u);
}

TEST(SweepConfigs, AllUnique) {
    std::set<std::uint32_t> keys;
    for (const auto& cfg : fxe::sweep_configs)
        EXPECT_TRUE(keys.insert(pack_config(cfg)).second);
    EXPECT_EQ(keys.size(), 1024u);
}

TEST(SweepConfigs, FixedPositionsConstant) {
    // Every config in the sweep must have the consensus-fixed strategies
    for (std::size_t ci = 0; ci < fxe::sweep_configs.size(); ++ci) {
        const auto& cfg = fxe::sweep_configs[ci];
        for (int f = 0; f < fxe::N_FIXED; ++f) {
            int pos = fxe::CONSENSUS_FIXED[f].position;
            EXPECT_EQ(cfg[static_cast<std::size_t>(pos)],
                      fxe::CONSENSUS_FIXED[f].strategy)
                << "config[" << ci << "] position " << pos;
        }
    }
}

TEST(SweepConfigs, FirstConfigAllUncertainUnrolled) {
    // Index 0 should map all uncertain positions to strategy 0 (Unrolled)
    const auto& cfg = fxe::sweep_configs[0];
    for (int i = 0; i < fxe::N_UNCERTAIN; ++i) {
        int pos = fxe::CONSENSUS_UNCERTAIN[i];
        EXPECT_EQ(cfg[static_cast<std::size_t>(pos)], fxe::Strategy::Unrolled)
            << "uncertain position " << pos;
    }
}

TEST(SweepConfigs, LastConfigAllUncertainGeneric) {
    // Index 1023 = 4^5-1, all base-4 digits = 3 (Generic)
    const auto& cfg = fxe::sweep_configs[1023];
    for (int i = 0; i < fxe::N_UNCERTAIN; ++i) {
        int pos = fxe::CONSENSUS_UNCERTAIN[i];
        EXPECT_EQ(cfg[static_cast<std::size_t>(pos)], fxe::Strategy::Generic)
            << "uncertain position " << pos;
    }
}

TEST(SweepConfigs, UncertainPositionsCoverAllStrategies) {
    // Each uncertain position should see all 4 strategies across the sweep
    for (int i = 0; i < fxe::N_UNCERTAIN; ++i) {
        int pos = fxe::CONSENSUS_UNCERTAIN[i];
        std::set<fxe::Strategy> seen;
        for (const auto& cfg : fxe::sweep_configs)
            seen.insert(cfg[static_cast<std::size_t>(pos)]);
        EXPECT_EQ(seen.size(), 4u) << "uncertain position " << pos;
    }
}

// =================================================================
// audit_configs
// =================================================================

TEST(AuditConfigs, CorrectSize) {
    EXPECT_EQ(fxe::audit_configs.size(), 21u);  // 7 fixed × 3 alternatives
}

TEST(AuditConfigs, EachFlipsExactlyOneFixedPosition) {
    // Base is sweep_configs[0].  Each audit config should differ from
    // the base at exactly one fixed position, and that position should
    // have a non-consensus strategy.
    const auto& base = fxe::sweep_configs[0];
    std::size_t idx = 0;
    for (int f = 0; f < fxe::N_FIXED; ++f) {
        int pos = fxe::CONSENSUS_FIXED[f].position;
        auto fixed_strat = fxe::CONSENSUS_FIXED[f].strategy;
        for (int s = 0; s < fix::NUM_STRATEGIES; ++s) {
            auto alt = static_cast<fxe::Strategy>(s);
            if (alt == fixed_strat) continue;

            const auto& cfg = fxe::audit_configs[idx];
            // The flipped position should have the alternative strategy
            EXPECT_EQ(cfg[static_cast<std::size_t>(pos)], alt)
                << "audit[" << idx << "] pos " << pos;
            // All other positions should match the base
            for (int p = 0; p < fix::num_fields; ++p) {
                if (p == pos) continue;
                EXPECT_EQ(cfg[static_cast<std::size_t>(p)],
                          base[static_cast<std::size_t>(p)])
                    << "audit[" << idx << "] non-flipped pos " << p;
            }
            ++idx;
        }
    }
}

TEST(AuditConfigs, AllUnique) {
    std::set<std::uint32_t> keys;
    for (const auto& cfg : fxe::audit_configs)
        EXPECT_TRUE(keys.insert(pack_config(cfg)).second);
    EXPECT_EQ(keys.size(), 21u);
}

// =================================================================
// sweep_ancillary
// =================================================================

TEST(SweepAncillary, CorrectSize) {
    EXPECT_EQ(fxe::sweep_ancillary.size(), 26u);  // 5 baselines + 21 audit
}

TEST(SweepAncillary, FirstFiveAreBaselines) {
    for (std::size_t i = 0; i < fxe::num_baselines; ++i) {
        EXPECT_EQ(fxe::sweep_ancillary[i], fxe::baseline_configs[i])
            << "ancillary[" << i << "] should match baseline_configs[" << i << "]";
    }
}

TEST(SweepAncillary, Last21AreAudit) {
    for (std::size_t i = 0; i < fxe::N_AUDIT; ++i) {
        EXPECT_EQ(fxe::sweep_ancillary[fxe::num_baselines + i],
                  fxe::audit_configs[i])
            << "ancillary[" << (fxe::num_baselines + i)
            << "] should match audit_configs[" << i << "]";
    }
}
