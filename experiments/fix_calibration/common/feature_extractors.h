#ifndef EXPERIMENT_FIX_CALIBRATION_FEATURE_EXTRACTORS_H
#define EXPERIMENT_FIX_CALIBRATION_FEATURE_EXTRACTORS_H

// experiments/fix_calibration/common/feature_extractors.h
//
// Feature extractors for the FIX parser experiment programs.
// Three extractors, each building on the previous:
//
//   onehot_extractor      36 features   (12 positions x 3 dummies)
//   count_extractor       40 features   (+ 4 strategy counts)
//   transition_extractor  56 features   (+ 16 transition counts)
//
// Design notes:
//   - 3-dummy one-hot encoding (not 4-way symmetric) is retained to
//     match the existing baseline in fix_p99_beam_search.cpp.
//     The reference level is Generic (strategy index 3).
//     Symmetric 4-way expansion is a future variant.
//   - count_extractor makes global composition effects explicit for
//     the RBF kernel, which cannot easily extract them from 36 binary
//     dimensions.
//   - transition_extractor adds 16 strategy->strategy transition
//     counts.  These capture adjacency structure without imposing
//     fake ordinal distances that would poison the RBF kernel.
//
// Invariants (testable):
//   - onehot: dimension = 36.  Each position block sums to 0 or 1.
//   - count:  dimension = 40.  Counts sum to 12.
//   - transition: dimension = 56.  Transition counts sum to 11.

#include <ctdp/calibrator/fix_et_parser.h>
#include <array>
#include <cstddef>
#include <vector>

namespace experiment {

// -- Type aliases from the ctdp calibrator namespace ---------------
using fix_config  = ctdp::calibrator::fix::fix_config;
using Strategy    = ctdp::calibrator::fix::Strategy;
inline constexpr int NUM_FIELDS     = ctdp::calibrator::fix::num_fields;
inline constexpr int NUM_STRATEGIES = ctdp::calibrator::fix::NUM_STRATEGIES;

// -- fix_point: thin wrapper around fix_config --------------------
// Satisfies the point_like concept from ctdp::cost_models so these
// extractors can plug into the existing svr_trainer/linear_trainer.

struct fix_point {
    fix_config config;

    std::array<double, 12> dims() const {
        std::array<double, 12> d{};
        for (int i = 0; i < NUM_FIELDS; ++i)
            d[static_cast<std::size_t>(i)] =
                static_cast<double>(config[static_cast<std::size_t>(i)]);
        return d;
    }
};

// -- onehot_extractor ---------------------------------------------
// 36 features: 12 positions x 3 binary indicators.
// Strategies 0(U), 1(S), 2(L) get a 1-bit per position.
// Strategy 3(G) is the dropped reference level (all zeros).

struct onehot_extractor {
    static constexpr std::size_t DIM = 36;  // NUM_FIELDS * 3

    std::vector<double> operator()(const fix_point& p) const {
        return encode(p.config);
    }

    static std::vector<double> encode(const fix_config& cfg) {
        std::vector<double> f(DIM, 0.0);
        for (int i = 0; i < NUM_FIELDS; ++i) {
            int s = static_cast<int>(cfg[static_cast<std::size_t>(i)]);
            if (s < 3)
                f[static_cast<std::size_t>(i * 3 + s)] = 1.0;
        }
        return f;
    }

    static constexpr const char* feature_name() { return "onehot_per_field"; }
    static constexpr const char* name() { return "onehot_extractor"; }
};

// -- count_extractor -----------------------------------------------
// 40 features: 36 one-hot + [count_U, count_S, count_L, count_G].
// Invariant: counts sum to 12.

struct count_extractor {
    static constexpr std::size_t DIM = onehot_extractor::DIM + 4;  // 40

    std::vector<double> operator()(const fix_point& p) const {
        return encode(p.config);
    }

    static std::vector<double> encode(const fix_config& cfg) {
        auto f = onehot_extractor::encode(cfg);
        f.reserve(DIM);

        std::array<int, NUM_STRATEGIES> counts{};
        for (int i = 0; i < NUM_FIELDS; ++i)
            counts[static_cast<int>(cfg[static_cast<std::size_t>(i)])]++;

        for (int c = 0; c < NUM_STRATEGIES; ++c)
            f.push_back(static_cast<double>(counts[static_cast<std::size_t>(c)]));

        return f;
    }

    static constexpr const char* feature_name() { return "onehot_counts"; }
    static constexpr const char* name() { return "count_extractor"; }
};

// -- transition_extractor ------------------------------------------
// 56 features: 40 count + 16 transition counts.
// Transition counts: how many times each (from, to) strategy pair
// occurs across the 11 adjacent position pairs.
// Invariant: transition counts sum to 11.

struct transition_extractor {
    static constexpr std::size_t DIM =
        count_extractor::DIM + NUM_STRATEGIES * NUM_STRATEGIES;  // 56

    std::vector<double> operator()(const fix_point& p) const {
        return encode(p.config);
    }

    static std::vector<double> encode(const fix_config& cfg) {
        auto f = count_extractor::encode(cfg);
        f.reserve(DIM);

        std::array<int, NUM_STRATEGIES * NUM_STRATEGIES> trans{};
        for (int i = 0; i < NUM_FIELDS - 1; ++i) {
            int from = static_cast<int>(cfg[static_cast<std::size_t>(i)]);
            int to   = static_cast<int>(cfg[static_cast<std::size_t>(i + 1)]);
            trans[static_cast<std::size_t>(from * NUM_STRATEGIES + to)]++;
        }

        for (std::size_t t = 0; t < trans.size(); ++t)
            f.push_back(static_cast<double>(trans[t]));

        return f;
    }

    static constexpr const char* feature_name() { return "onehot_counts_transitions"; }
    static constexpr const char* name() { return "transition_extractor"; }
};

} // namespace experiment

#endif // EXPERIMENT_FIX_CALIBRATION_FEATURE_EXTRACTORS_H
