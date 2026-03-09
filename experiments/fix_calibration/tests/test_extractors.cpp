// experiments/fix_calibration/tests/test_extractors.cpp
//
// Unit tests for the three feature extractors.
// Tests dimensions, invariants, known outputs, and edge cases.

#include "feature_extractors.h"
#include "experiment_config.h"
#include <gtest/gtest.h>
#include <numeric>
#include <cmath>

namespace cfix = ctdp::calibrator::fix;
using namespace experiment;

// Helper: build a config from a string like "UUSLSUUSUUUU".
// Fails loudly on invalid characters so test typos don't silently
// become Generic.
static fix_config make_config(const char* s) {
    fix_config cfg{};
    for (int i = 0; i < NUM_FIELDS; ++i) {
        auto opt = cfix::strategy_from_char(s[i]);
        EXPECT_TRUE(opt.has_value())
            << "invalid strategy char '" << s[i] << "' at position " << i
            << " in config string \"" << s << "\"";
        cfg[static_cast<std::size_t>(i)] = opt.value_or(Strategy::Generic);
    }
    return cfg;
}

// =================================================================
// onehot_extractor
// =================================================================

TEST(OnehotExtractor, CorrectDimension) {
    auto f = onehot_extractor::encode(make_config("UUUUUUUUUUUU"));
    EXPECT_EQ(f.size(), onehot_extractor::DIM);
    EXPECT_EQ(f.size(), 36u);
}

TEST(OnehotExtractor, AllUnrolled) {
    auto f = onehot_extractor::encode(make_config("UUUUUUUUUUUU"));
    for (int i = 0; i < NUM_FIELDS; ++i) {
        EXPECT_DOUBLE_EQ(f[static_cast<std::size_t>(i * 3 + 0)], 1.0)
            << "pos " << i << " U bit";
        EXPECT_DOUBLE_EQ(f[static_cast<std::size_t>(i * 3 + 1)], 0.0)
            << "pos " << i << " S bit";
        EXPECT_DOUBLE_EQ(f[static_cast<std::size_t>(i * 3 + 2)], 0.0)
            << "pos " << i << " L bit";
    }
}

TEST(OnehotExtractor, AllGeneric) {
    auto f = onehot_extractor::encode(make_config("GGGGGGGGGGGG"));
    for (std::size_t i = 0; i < f.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], 0.0) << "index " << i;
    }
}

TEST(OnehotExtractor, AllSWAR) {
    auto f = onehot_extractor::encode(make_config("SSSSSSSSSSSS"));
    for (int i = 0; i < NUM_FIELDS; ++i) {
        EXPECT_DOUBLE_EQ(f[static_cast<std::size_t>(i * 3 + 0)], 0.0);
        EXPECT_DOUBLE_EQ(f[static_cast<std::size_t>(i * 3 + 1)], 1.0);
        EXPECT_DOUBLE_EQ(f[static_cast<std::size_t>(i * 3 + 2)], 0.0);
    }
}

TEST(OnehotExtractor, MixedConfig) {
    auto f = onehot_extractor::encode(make_config("USGLGGGGGGGG"));
    // pos 0 = U: [1,0,0]
    EXPECT_DOUBLE_EQ(f[0], 1.0);
    EXPECT_DOUBLE_EQ(f[1], 0.0);
    EXPECT_DOUBLE_EQ(f[2], 0.0);
    // pos 1 = S: [0,1,0]
    EXPECT_DOUBLE_EQ(f[3], 0.0);
    EXPECT_DOUBLE_EQ(f[4], 1.0);
    EXPECT_DOUBLE_EQ(f[5], 0.0);
    // pos 2 = G: [0,0,0]
    EXPECT_DOUBLE_EQ(f[6], 0.0);
    EXPECT_DOUBLE_EQ(f[7], 0.0);
    EXPECT_DOUBLE_EQ(f[8], 0.0);
    // pos 3 = L: [0,0,1]
    EXPECT_DOUBLE_EQ(f[9], 0.0);
    EXPECT_DOUBLE_EQ(f[10], 0.0);
    EXPECT_DOUBLE_EQ(f[11], 1.0);
}

TEST(OnehotExtractor, PerPositionBlockSumsToZeroOrOne) {
    auto f = onehot_extractor::encode(make_config("USLGUSLGUSLS"));
    for (int i = 0; i < NUM_FIELDS; ++i) {
        double sum = f[static_cast<std::size_t>(i*3)]
                   + f[static_cast<std::size_t>(i*3+1)]
                   + f[static_cast<std::size_t>(i*3+2)];
        EXPECT_TRUE(sum == 0.0 || sum == 1.0)
            << "pos " << i << " block sum = " << sum;
    }
}

TEST(OnehotExtractor, ViaFixPoint) {
    fix_point pt{make_config("SSSSSSSSSSSS")};
    onehot_extractor ext;
    auto f = ext(pt);
    EXPECT_EQ(f.size(), 36u);
    EXPECT_DOUBLE_EQ(f[1], 1.0);  // pos 0, S bit
}

TEST(OnehotExtractor, FeatureName) {
    EXPECT_STREQ(onehot_extractor::feature_name(), "onehot_per_field");
    EXPECT_STREQ(onehot_extractor::name(), "onehot_extractor");
}

// =================================================================
// count_extractor
// =================================================================

TEST(CountExtractor, CorrectDimension) {
    auto f = count_extractor::encode(make_config("UUUUUUUUUUUU"));
    EXPECT_EQ(f.size(), count_extractor::DIM);
    EXPECT_EQ(f.size(), 40u);
}

TEST(CountExtractor, CountsSumToTwelve) {
    const char* configs[] = {
        "UUUUUUUUUUUU", "SSSSSSSSSSSS", "GGGGGGGGGGGG",
        "USGLSUUSLUSG", "ULUSSSGSSSSS", "GSGGULUGGGSU"
    };
    for (auto* cfg_str : configs) {
        auto f = count_extractor::encode(make_config(cfg_str));
        double sum = f[36] + f[37] + f[38] + f[39];
        EXPECT_DOUBLE_EQ(sum, 12.0) << "config: " << cfg_str;
    }
}

TEST(CountExtractor, AllUnrolledCounts) {
    auto f = count_extractor::encode(make_config("UUUUUUUUUUUU"));
    EXPECT_DOUBLE_EQ(f[36], 12.0);  // count_U
    EXPECT_DOUBLE_EQ(f[37], 0.0);   // count_S
    EXPECT_DOUBLE_EQ(f[38], 0.0);   // count_L
    EXPECT_DOUBLE_EQ(f[39], 0.0);   // count_G
}

TEST(CountExtractor, MixedCounts) {
    // UUSLSUUSLUSG: U=5, S=4, L=2, G=1
    auto f = count_extractor::encode(make_config("UUSLSUUSLUSG"));
    EXPECT_DOUBLE_EQ(f[36], 5.0);
    EXPECT_DOUBLE_EQ(f[37], 4.0);
    EXPECT_DOUBLE_EQ(f[38], 2.0);
    EXPECT_DOUBLE_EQ(f[39], 1.0);
}

TEST(CountExtractor, OnehotPrefixUnchanged) {
    auto cfg = make_config("ULUSSSGSSSSS");
    auto f_oh = onehot_extractor::encode(cfg);
    auto f_ct = count_extractor::encode(cfg);
    for (std::size_t i = 0; i < 36; ++i) {
        EXPECT_DOUBLE_EQ(f_oh[i], f_ct[i]) << "index " << i;
    }
}

TEST(CountExtractor, FeatureName) {
    EXPECT_STREQ(count_extractor::name(), "count_extractor");
}

// =================================================================
// transition_extractor
// =================================================================

TEST(TransitionExtractor, CorrectDimension) {
    auto f = transition_extractor::encode(make_config("UUUUUUUUUUUU"));
    EXPECT_EQ(f.size(), transition_extractor::DIM);
    EXPECT_EQ(f.size(), 56u);
}

TEST(TransitionExtractor, TransitionsSumToEleven) {
    const char* configs[] = {
        "UUUUUUUUUUUU", "SSSSSSSSSSSS", "USGLSUUSLUSG",
        "GSGGULUGGGSU", "ULULULULULUS", "SGLGLLSULGLS"
    };
    for (auto* cfg_str : configs) {
        auto f = transition_extractor::encode(make_config(cfg_str));
        double sum = 0.0;
        for (std::size_t i = 40; i < 56; ++i)
            sum += f[i];
        EXPECT_DOUBLE_EQ(sum, 11.0) << "config: " << cfg_str;
    }
}

TEST(TransitionExtractor, AllUnrolledTransitions) {
    auto f = transition_extractor::encode(make_config("UUUUUUUUUUUU"));
    EXPECT_DOUBLE_EQ(f[40], 11.0);  // U->U
    for (std::size_t i = 41; i < 56; ++i) {
        EXPECT_DOUBLE_EQ(f[i], 0.0) << "transition index " << (i - 40);
    }
}

TEST(TransitionExtractor, AlternatingConfig) {
    auto f = transition_extractor::encode(make_config("USUSUSUSUSUS"));
    // U->S: from=0,to=1, index=1, feature[41] = 6
    EXPECT_DOUBLE_EQ(f[41], 6.0);
    // S->U: from=1,to=0, index=4, feature[44] = 5
    EXPECT_DOUBLE_EQ(f[44], 5.0);
    double us = f[41], su = f[44];
    double total = 0;
    for (std::size_t i = 40; i < 56; ++i) total += f[i];
    EXPECT_DOUBLE_EQ(total, 11.0);
    EXPECT_DOUBLE_EQ(us + su, 11.0);
}

TEST(TransitionExtractor, KnownConfigTransitions) {
    // USGLGGGGGGGG
    // U->S(1), S->G(1), G->L(1), L->G(1), G->G(7)
    auto f = transition_extractor::encode(make_config("USGLGGGGGGGG"));
    EXPECT_DOUBLE_EQ(f[41], 1.0);   // U->S: 0*4+1=1
    EXPECT_DOUBLE_EQ(f[47], 1.0);   // S->G: 1*4+3=7
    EXPECT_DOUBLE_EQ(f[54], 1.0);   // G->L: 3*4+2=14
    EXPECT_DOUBLE_EQ(f[51], 1.0);   // L->G: 2*4+3=11
    EXPECT_DOUBLE_EQ(f[55], 7.0);   // G->G: 3*4+3=15
}

TEST(TransitionExtractor, CountPrefixUnchanged) {
    auto cfg = make_config("ULUSSSGSSSSS");
    auto f_ct = count_extractor::encode(cfg);
    auto f_tr = transition_extractor::encode(cfg);
    for (std::size_t i = 0; i < 40; ++i) {
        EXPECT_DOUBLE_EQ(f_ct[i], f_tr[i]) << "index " << i;
    }
}

TEST(TransitionExtractor, FeatureName) {
    EXPECT_STREQ(transition_extractor::name(), "transition_extractor");
}

// =================================================================
// Cross-extractor consistency
// =================================================================

TEST(Extractors, DimensionsAreConsistent) {
    EXPECT_EQ(onehot_extractor::DIM, 36u);
    EXPECT_EQ(count_extractor::DIM, onehot_extractor::DIM + 4);
    EXPECT_EQ(transition_extractor::DIM,
              count_extractor::DIM + NUM_STRATEGIES * NUM_STRATEGIES);
}

TEST(Extractors, AllExtractorsProduceCorrectDimsOnSameConfig) {
    auto cfg = make_config("GSGGULUGGGSU");
    auto f1 = onehot_extractor::encode(cfg);
    auto f2 = count_extractor::encode(cfg);
    auto f3 = transition_extractor::encode(cfg);
    EXPECT_EQ(f1.size(), 36u);
    EXPECT_EQ(f2.size(), 40u);
    EXPECT_EQ(f3.size(), 56u);
}
