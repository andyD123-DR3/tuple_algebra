// experiments/fix_calibration/tests/test_beam_search.cpp
//
// Unit tests for beam_search.h.
// Uses trivial mock models with known scoring functions so that
// search outcomes are deterministic and verifiable.

#include "beam_search.h"
#include "experiment_config.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <string>

namespace cfix = ctdp::calibrator::fix;
using namespace experiment;

// =================================================================
// Mock models
// =================================================================

// StrategyIndexSum: score = sum of strategy indices across all fields.
// Global minimum is all-Unrolled (score = 0).
// Global maximum is all-Generic (score = 36).
// Allows precise verification of beam search convergence.

struct StrategyIndexSum {
    double predict(const fix_point& p) const {
        double sum = 0.0;
        for (int i = 0; i < NUM_FIELDS; ++i)
            sum += static_cast<double>(p.config[static_cast<std::size_t>(i)]);
        return sum;
    }
};

// HammingToTarget: score = number of positions that differ from target.
// Global minimum = 0 (exact match).  This tests whether beam search
// can navigate to a specific configuration.

struct HammingToTarget {
    fix_config target;

    double predict(const fix_point& p) const {
        double dist = 0.0;
        for (int i = 0; i < NUM_FIELDS; ++i) {
            if (p.config[static_cast<std::size_t>(i)] !=
                target[static_cast<std::size_t>(i)])
                dist += 1.0;
        }
        return dist;
    }
};

// WeightedSum: score = sum of weighted strategy values.
// Weights vary per position, so the optimal strategy differs per field.

struct WeightedSum {
    // Weights[field][strategy] -- lower is better
    double weights[12][4] = {
        {1, 3, 5, 7},  {7, 1, 3, 5},  {5, 7, 1, 3},  {3, 5, 7, 1},
        {1, 3, 5, 7},  {7, 1, 3, 5},  {5, 7, 1, 3},  {3, 5, 7, 1},
        {1, 3, 5, 7},  {7, 1, 3, 5},  {5, 7, 1, 3},  {3, 5, 7, 1},
    };

    double predict(const fix_point& p) const {
        double sum = 0.0;
        for (int i = 0; i < NUM_FIELDS; ++i) {
            int s = static_cast<int>(p.config[static_cast<std::size_t>(i)]);
            sum += weights[i][s];
        }
        return sum;
    }

    // Expected optimal: the strategy with the lowest weight per field
    fix_config optimal() const {
        fix_config cfg{};
        for (int i = 0; i < NUM_FIELDS; ++i) {
            int best_s = 0;
            double best_w = weights[i][0];
            for (int s = 1; s < NUM_STRATEGIES; ++s) {
                if (weights[i][s] < best_w) {
                    best_w = weights[i][s];
                    best_s = s;
                }
            }
            cfg[static_cast<std::size_t>(i)] = static_cast<Strategy>(best_s);
        }
        return cfg;
    }
};

// ConstantScore: every config scores the same.
// All entries are ties -- tests that tie-breaking is deterministic
// and that the retained beam is in canonical config_key order.

struct ConstantScore {
    double predict(const fix_point& /*p*/) const {
        return 42.0;
    }
};

// =================================================================
// Helper: make seeds from the constexpr pool
// =================================================================

static std::vector<fix_config> make_seeds(std::size_t n) {
    constexpr auto pool = cfix::generate_random_configs<200>(TRAIN_SEED);
    std::vector<fix_config> seeds;
    seeds.reserve(n);
    for (std::size_t i = 0; i < n && i < pool.size(); ++i)
        seeds.push_back(pool[i]);
    return seeds;
}

static std::vector<fix_config> single_seed(const fix_config& cfg) {
    return {cfg};
}

// =================================================================
// Basic functionality
// =================================================================

TEST(BeamSearch, FindsAllUnrolledOptimum) {
    StrategyIndexSum model;
    auto seeds = make_seeds(20);
    auto result = beam_search(model, seeds, 20);

    ASSERT_FALSE(result.beam.empty());
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);
    for (int i = 0; i < NUM_FIELDS; ++i) {
        EXPECT_EQ(result.beam[0].config[static_cast<std::size_t>(i)],
                  Strategy::Unrolled) << "field " << i;
    }
}

TEST(BeamSearch, FindsSpecificTarget) {
    fix_config target = cfix::all_swar;
    HammingToTarget model{target};
    auto seeds = make_seeds(20);
    auto result = beam_search(model, seeds, 20);

    ASSERT_FALSE(result.beam.empty());
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);
    for (int i = 0; i < NUM_FIELDS; ++i) {
        EXPECT_EQ(result.beam[0].config[static_cast<std::size_t>(i)],
                  Strategy::SWAR) << "field " << i;
    }
}

TEST(BeamSearch, FindsMixedOptimum) {
    WeightedSum model;
    fix_config expected = model.optimal();
    auto seeds = make_seeds(20);
    auto result = beam_search(model, seeds, 20);

    ASSERT_FALSE(result.beam.empty());
    fix_point opt_pt{expected};
    double expected_score = model.predict(opt_pt);
    EXPECT_DOUBLE_EQ(result.beam[0].score, expected_score);
    for (int i = 0; i < NUM_FIELDS; ++i) {
        EXPECT_EQ(result.beam[0].config[static_cast<std::size_t>(i)],
                  expected[static_cast<std::size_t>(i)]) << "field " << i;
    }
}

// =================================================================
// Beam width behaviour
// =================================================================

TEST(BeamSearch, WiderBeamAtLeastAsGood) {
    StrategyIndexSum model;
    auto seeds = make_seeds(50);

    auto narrow = beam_search(model, seeds, 5);
    auto wide   = beam_search(model, seeds, 30);

    ASSERT_FALSE(narrow.beam.empty());
    ASSERT_FALSE(wide.beam.empty());
    EXPECT_LE(wide.beam[0].score, narrow.beam[0].score);
}

TEST(BeamSearch, BeamWidth1) {
    StrategyIndexSum model;
    auto seeds = make_seeds(10);
    auto result = beam_search(model, seeds, 1);

    ASSERT_EQ(result.beam.size(), 1u);
    // Greedy should still find all-Unrolled for this separable model
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);
}

TEST(BeamSearch, BeamSizeCapped) {
    StrategyIndexSum model;
    auto seeds = make_seeds(50);
    auto result = beam_search(model, seeds, 10);
    EXPECT_LE(result.beam.size(), 10u);
}

// =================================================================
// Output invariants
// =================================================================

TEST(BeamSearch, OutputSortedBestFirst) {
    StrategyIndexSum model;
    auto seeds = make_seeds(30);
    auto result = beam_search(model, seeds, 15);

    for (std::size_t i = 1; i < result.beam.size(); ++i) {
        EXPECT_TRUE(!beam_less(result.beam[i], result.beam[i - 1]))
            << "beam not sorted at position " << i;
    }
}

TEST(BeamSearch, NoDuplicatesInBeam) {
    StrategyIndexSum model;
    auto seeds = make_seeds(30);
    auto result = beam_search(model, seeds, 20);

    std::set<std::uint32_t> keys;
    for (const auto& entry : result.beam) {
        EXPECT_TRUE(keys.insert(entry.key).second)
            << "duplicate config in beam: "
            << cfix::config_to_string(entry.config);
    }
}

TEST(BeamSearch, ScoresMatchModel) {
    StrategyIndexSum model;
    auto seeds = make_seeds(20);
    auto result = beam_search(model, seeds, 10);

    for (const auto& entry : result.beam) {
        double expected = model.predict(fix_point{entry.config});
        EXPECT_DOUBLE_EQ(entry.score, expected)
            << "config " << cfix::config_to_string(entry.config);
    }
}

TEST(BeamSearch, KeysMatchConfigs) {
    // Every stored key must agree with config_key(config).
    StrategyIndexSum model;
    auto seeds = make_seeds(20);
    auto result = beam_search(model, seeds, 15);

    for (const auto& entry : result.beam) {
        EXPECT_EQ(entry.key, detail::config_key(entry.config))
            << "config " << cfix::config_to_string(entry.config);
    }
}

TEST(BeamSearch, TotalEvalsPositive) {
    StrategyIndexSum model;
    auto seeds = make_seeds(10);
    auto result = beam_search(model, seeds, 5);
    EXPECT_GT(result.total_evaluated, 0u);
    // At minimum: unique seeds + 12 field rounds * some candidates
    EXPECT_GE(result.total_evaluated, 1u);
}

// =================================================================
// Determinism
// =================================================================

TEST(BeamSearch, Deterministic) {
    StrategyIndexSum model;
    auto seeds = make_seeds(20);

    auto r1 = beam_search(model, seeds, 15);
    auto r2 = beam_search(model, seeds, 15);

    ASSERT_EQ(r1.beam.size(), r2.beam.size());
    EXPECT_EQ(r1.total_evaluated, r2.total_evaluated);
    for (std::size_t i = 0; i < r1.beam.size(); ++i) {
        EXPECT_EQ(r1.beam[i].key, r2.beam[i].key);
        EXPECT_DOUBLE_EQ(r1.beam[i].score, r2.beam[i].score);
    }
}

// =================================================================
// Edge cases
// =================================================================

TEST(BeamSearch, SingleSeed) {
    StrategyIndexSum model;
    auto result = beam_search(model, single_seed(cfix::all_generic), 10);
    ASSERT_FALSE(result.beam.empty());
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);
}

TEST(BeamSearch, SeedIsAlreadyOptimal) {
    StrategyIndexSum model;
    auto result = beam_search(model, single_seed(cfix::all_unrolled), 5);
    ASSERT_FALSE(result.beam.empty());
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);
}

TEST(BeamSearch, EmptySeeds) {
    // Empty seeds is valid: returns empty beam, zero evals.
    StrategyIndexSum model;
    std::vector<fix_config> no_seeds;
    auto result = beam_search(model, no_seeds, 5);
    EXPECT_TRUE(result.beam.empty());
    EXPECT_EQ(result.total_evaluated, 0u);
}

// =================================================================
// Duplicate-seed handling
// =================================================================

TEST(BeamSearch, DuplicateSeedsDeduped) {
    // 10 identical seeds should produce exactly 1 eval in seed phase.
    StrategyIndexSum model;
    std::vector<fix_config> seeds(10, cfix::all_loop);
    auto result = beam_search(model, seeds, 10);
    ASSERT_FALSE(result.beam.empty());
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);

    // Run with a single seed -- should get the same eval count
    auto result_single = beam_search(model, single_seed(cfix::all_loop), 10);
    EXPECT_EQ(result.total_evaluated, result_single.total_evaluated);
}

// =================================================================
// Exact evaluation count
// =================================================================

TEST(BeamSearch, ExactEvalCountSingleSeedWidth1) {
    // Single seed, width 1 (greedy):
    //   Seed phase: 1 unique seed = 1 eval
    //   Per field round: 1 beam entry * 4 strategies = up to 4 candidates,
    //     but the original strategy is already in the beam so 4 unique
    //     (the parent config with its field mutated to all 4 strategies).
    //   12 rounds * 4 evals = 48
    //   Total = 1 + 48 = 49
    StrategyIndexSum model;
    auto result = beam_search(model, single_seed(cfix::all_generic), 1);

    EXPECT_EQ(result.total_evaluated, 49u);
}

// =================================================================
// Tie-break determinism
// =================================================================

TEST(BeamSearch, TieBreakByConfigKey) {
    // ConstantScore: all candidates score identically.
    // The beam should be sorted by config_key among equals.
    ConstantScore model;
    auto seeds = make_seeds(30);
    auto result = beam_search(model, seeds, 15);

    ASSERT_GT(result.beam.size(), 1u);
    // All scores are 42.0
    for (const auto& entry : result.beam)
        EXPECT_DOUBLE_EQ(entry.score, 42.0);

    // Within equal scores, entries must be in ascending key order
    for (std::size_t i = 1; i < result.beam.size(); ++i) {
        EXPECT_LT(result.beam[i - 1].key, result.beam[i].key)
            << "tie-break violated at position " << i
            << ": key[" << (i-1) << "]=" << result.beam[i-1].key
            << " key[" << i << "]=" << result.beam[i].key;
    }
}

// =================================================================
// config_key structural tests
// =================================================================

TEST(ConfigKey, AllUnrolledIsZero) {
    EXPECT_EQ(detail::config_key(cfix::all_unrolled), 0u);
}

TEST(ConfigKey, AllGenericIsMaxPattern) {
    // All-Generic = strategy 3 in every field
    // 0b11 repeated 12 times = 0x00FFFFFF
    EXPECT_EQ(detail::config_key(cfix::all_generic), 0x00FFFFFFu);
}

TEST(ConfigKey, RoundTripsDistinctConfigs) {
    // Two configs that differ at one position must have different keys.
    fix_config a = cfix::all_unrolled;
    fix_config b = cfix::all_unrolled;
    b[5] = Strategy::SWAR;  // position 5 = SWAR (1), bits 10-11

    EXPECT_NE(detail::config_key(a), detail::config_key(b));
}

TEST(ConfigKey, SingleFieldEncoding) {
    // Strategy s at field i should set bits [2i, 2i+1] = s
    fix_config cfg = cfix::all_unrolled;  // key = 0
    cfg[3] = Strategy::Loop;  // Loop = 2, field 3 -> bits 6-7
    std::uint32_t expected = 2u << 6;
    EXPECT_EQ(detail::config_key(cfg), expected);
}

// =================================================================
// Hamming model with non-uniform target
// =================================================================

TEST(BeamSearch, ConvergesToMixedTarget) {
    fix_config target = {
        Strategy::Unrolled, Strategy::Unrolled,
        Strategy::SWAR,     Strategy::Loop,
        Strategy::SWAR,     Strategy::Unrolled,
        Strategy::Unrolled, Strategy::SWAR,
        Strategy::Unrolled, Strategy::Unrolled,
        Strategy::Unrolled, Strategy::Unrolled,
    };
    HammingToTarget model{target};
    auto seeds = make_seeds(20);
    auto result = beam_search(model, seeds, 20);

    ASSERT_FALSE(result.beam.empty());
    EXPECT_DOUBLE_EQ(result.beam[0].score, 0.0);
    for (int i = 0; i < NUM_FIELDS; ++i) {
        EXPECT_EQ(result.beam[0].config[static_cast<std::size_t>(i)],
                  target[static_cast<std::size_t>(i)]) << "field " << i;
    }
}
