#ifndef CTDP_CALIBRATOR_FIX_COUNTER_PREPROCESSOR_H
#define CTDP_CALIBRATOR_FIX_COUNTER_PREPROCESSOR_H

// ============================================================
//  counter_preprocessor.h  –  CT-DP FIX Parser Optimiser
//
//  Prepares hardware counter features for Stage 2 SVR training.
//
//  Responsibilities:
//    1. Corpus splitting — partitions a DataPoint<N> corpus into
//       Ok / Suspicious / HardDeleted sets (calibrator_design_v2 §B4).
//    2. Per-counter z-score normalisation — fit on Ok training
//       data only; applies the same transform at inference time
//       (no leakage across CV folds).
//    3. 5-fold CV support — fold_mask() returns the Ok indices
//       for a given hold-out fold, ready for SVR fit/predict.
//    4. Feature matrix assembly — write_feature_matrix() fills a
//       flat row-major float buffer [n_samples × NUM_COUNTERS]
//       suitable for direct hand-off to epsilon_svr.h.
//    5. Target vector assembly — write_target_vector() fills a
//       double buffer with mean_ab_p99_ns() for the same rows.
//
//  Stage 2 design note (calibrator_design_v2 §stage2):
//    Stage 2 SVR is trained exclusively on measured counter vectors.
//    One-hot strategy features are NOT included in Stage 2 — they
//    would reintroduce the collinearity problems that motivated the
//    two-stage split.  Strategy features belong to Stage 1 only.
//
//  Normalisation:
//    z = (x - mean) / max(std, EPS_STD)
//    EPS_STD = 1.0 (one raw count) guards against dead counters
//    (e.g. branch-misses on a branchless inner loop ≈ 0 for all
//    configs; std ≈ 0 but the counter carries no information and
//    the normalised value should also be ≈ 0 rather than NaN).
//
//  Suspicious handling (calibrator_design_v2 §B4):
//    Suspicious points are excluded from fit() and from all
//    training folds.  They are available via suspicious_indices()
//    for CounterAdapter transfer experiments (§S5).
//
//  Dependencies: data_point.h (same directory), std only
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/data_point.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  CounterStats — per-counter mean and std, fitted on Ok training data
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr double EPS_STD = 1.0;   // minimum std (1 raw count)

struct CounterStats {
    std::array<double, NUM_COUNTERS> mean{};
    std::array<double, NUM_COUNTERS> std_dev{};
    bool fitted{false};

    // Normalise a raw counter vector in-place.
    void normalise(std::array<double, NUM_COUNTERS>& v) const noexcept {
        assert(fitted);
        for (std::size_t i = 0; i < NUM_COUNTERS; ++i)
            v[i] = (v[i] - mean[i]) / std::max(std_dev[i], EPS_STD);
    }

    // Normalise into a float output (for SVR feature matrix).
    void normalise_to_float(const std::array<double, NUM_COUNTERS>& raw,
                            std::span<float> out) const noexcept {
        assert(fitted);
        assert(out.size() >= NUM_COUNTERS);
        for (std::size_t i = 0; i < NUM_COUNTERS; ++i)
            out[i] = static_cast<float>(
                (raw[i] - mean[i]) / std::max(std_dev[i], EPS_STD));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  fit_counter_stats — compute mean/std from a set of Ok DataPoints
//
//  indices: which rows of corpus to include (training fold only —
//           do NOT pass in hold-out rows).
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] CounterStats
fit_counter_stats(std::span<const DataPoint<N>> corpus,
                  std::span<const int>          indices) {
    if (indices.empty())
        throw std::invalid_argument("fit_counter_stats: empty index set");

    CounterStats stats{};

    // Pass 1: mean
    for (int idx : indices) {
        const auto& cf = corpus[idx].counter_features();
        for (std::size_t c = 0; c < NUM_COUNTERS; ++c)
            stats.mean[c] += cf[c];
    }
    const double n = static_cast<double>(indices.size());
    for (std::size_t c = 0; c < NUM_COUNTERS; ++c)
        stats.mean[c] /= n;

    // Pass 2: variance → std
    for (int idx : indices) {
        const auto& cf = corpus[idx].counter_features();
        for (std::size_t c = 0; c < NUM_COUNTERS; ++c) {
            const double d = cf[c] - stats.mean[c];
            stats.std_dev[c] += d * d;
        }
    }
    for (std::size_t c = 0; c < NUM_COUNTERS; ++c)
        stats.std_dev[c] = std::sqrt(stats.std_dev[c] / n);

    stats.fitted = true;
    return stats;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CounterPreprocessor<N>
//
//  Owns the corpus split and the fitted CounterStats.  The typical
//  workflow for one CV fold:
//
//    CounterPreprocessor<4> pp;
//    pp.ingest(corpus);                     // split Ok/Suspicious/HardDeleted
//
//    for (int hold_out = 0; hold_out < 5; ++hold_out) {
//        auto train_idx = pp.train_indices(hold_out);
//        auto test_idx  = pp.test_indices(hold_out);
//
//        CounterStats stats = pp.fit(hold_out);  // fit on train fold only
//
//        auto [Xtr, ytr] = pp.feature_matrix(train_idx, stats);
//        auto [Xte, yte] = pp.feature_matrix(test_idx,  stats);
//
//        // hand Xtr/ytr to epsilon_svr.h ...
//    }
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
    requires (N >= 1 && N <= 16)
class CounterPreprocessor {
public:
    // ── Ingest ───────────────────────────────────────────────────────────────

    // Scan the corpus and build the Ok / Suspicious / HardDeleted index sets.
    // Idempotent: calling again replaces the previous split.
    void ingest(std::span<const DataPoint<N>> corpus) {
        corpus_     = corpus;
        ok_.clear();
        suspicious_.clear();
        hard_deleted_.clear();

        for (int i = 0; i < static_cast<int>(corpus.size()); ++i) {
            switch (corpus[i].status) {
                case DataPointStatus::Ok:
                    ok_.push_back(i);
                    break;
                case DataPointStatus::Suspicious:
                    suspicious_.push_back(i);
                    break;
                case DataPointStatus::HardDeleted:
                    hard_deleted_.push_back(i);
                    break;
            }
        }
    }

    // ── Corpus split accessors ────────────────────────────────────────────────

    [[nodiscard]] std::span<const int> ok_indices()           const noexcept { return ok_; }
    [[nodiscard]] std::span<const int> suspicious_indices()   const noexcept { return suspicious_; }
    [[nodiscard]] std::span<const int> hard_deleted_indices() const noexcept { return hard_deleted_; }

    [[nodiscard]] int n_ok()           const noexcept { return static_cast<int>(ok_.size()); }
    [[nodiscard]] int n_suspicious()   const noexcept { return static_cast<int>(suspicious_.size()); }
    [[nodiscard]] int n_hard_deleted() const noexcept { return static_cast<int>(hard_deleted_.size()); }
    [[nodiscard]] int n_total()        const noexcept { return static_cast<int>(corpus_.size()); }

    // ── 5-fold CV index sets ─────────────────────────────────────────────────
    //
    //  Only Ok points participate in CV.
    //  train_indices(k): all Ok points with fold_id != k
    //  test_indices(k):  all Ok points with fold_id == k

    [[nodiscard]] std::vector<int> train_indices(int hold_out_fold) const {
        check_fold(hold_out_fold);
        std::vector<int> idx;
        idx.reserve(ok_.size());
        for (int i : ok_)
            if (corpus_[i].fold_id != static_cast<uint8_t>(hold_out_fold))
                idx.push_back(i);
        return idx;
    }

    [[nodiscard]] std::vector<int> test_indices(int hold_out_fold) const {
        check_fold(hold_out_fold);
        std::vector<int> idx;
        for (int i : ok_)
            if (corpus_[i].fold_id == static_cast<uint8_t>(hold_out_fold))
                idx.push_back(i);
        return idx;
    }

    // ── Fit normaliser ───────────────────────────────────────────────────────
    //
    //  Fit on the training fold only (no leakage from hold-out rows).

    [[nodiscard]] CounterStats fit(int hold_out_fold) const {
        auto train = train_indices(hold_out_fold);
        return fit_counter_stats<N>(corpus_, train);
    }

    // Fit on ALL Ok data (use for final model after CV; not during evaluation).
    [[nodiscard]] CounterStats fit_all() const {
        return fit_counter_stats<N>(corpus_, ok_);
    }

    // ── Feature matrix assembly ──────────────────────────────────────────────
    //
    //  Returns (X, y) where:
    //    X is row-major [n × NUM_COUNTERS] normalised float features
    //    y is [n] double targets (mean_ab_p99_ns)
    //
    //  The stats argument must have been fitted on training data only.

    struct FeatureMatrix {
        std::vector<float>  X;  // row-major [n_rows × NUM_COUNTERS]
        std::vector<double> y;  // [n_rows]
        int n_rows{0};
        int n_cols{static_cast<int>(NUM_COUNTERS)};

        // Access row i, column j
        [[nodiscard]] float X_at(int i, int j) const noexcept {
            return X[i * n_cols + j];
        }
    };

    [[nodiscard]] FeatureMatrix
    feature_matrix(std::span<const int> indices,
                   const CounterStats&  stats) const {
        assert(stats.fitted);
        FeatureMatrix fm;
        fm.n_rows = static_cast<int>(indices.size());
        fm.X.resize(fm.n_rows * NUM_COUNTERS);
        fm.y.resize(fm.n_rows);

        for (int row = 0; row < fm.n_rows; ++row) {
            const auto& dp = corpus_[indices[row]];
            stats.normalise_to_float(
                dp.counter_features(),
                std::span<float>{ fm.X.data() + row * NUM_COUNTERS,
                                  NUM_COUNTERS });
            fm.y[row] = dp.latency_estimate();
        }
        return fm;
    }

    // Convenience: build train+test matrices for one fold in one call.
    struct FoldMatrices {
        FeatureMatrix train;
        FeatureMatrix test;
        CounterStats  stats;  // fitted on train only — keep for inference
    };

    [[nodiscard]] FoldMatrices make_fold(int hold_out_fold) const {
        FoldMatrices fm;
        auto train_idx = train_indices(hold_out_fold);
        auto test_idx  = test_indices(hold_out_fold);
        fm.stats = fit_counter_stats<N>(corpus_, train_idx);
        fm.train = feature_matrix(train_idx, fm.stats);
        fm.test  = feature_matrix(test_idx,  fm.stats);
        return fm;
    }

    // ── Summary ──────────────────────────────────────────────────────────────

    struct Summary {
        int n_total{};
        int n_ok{};
        int n_suspicious{};
        int n_hard_deleted{};
        std::array<int, 5> fold_sizes{};  // Ok points per fold
    };

    [[nodiscard]] Summary summary() const noexcept {
        Summary s;
        s.n_total        = n_total();
        s.n_ok           = n_ok();
        s.n_suspicious   = n_suspicious();
        s.n_hard_deleted = n_hard_deleted();
        for (int i : ok_)
            if (corpus_[i].fold_id < 5)
                ++s.fold_sizes[corpus_[i].fold_id];
        return s;
    }

private:
    std::span<const DataPoint<N>> corpus_;
    std::vector<int>              ok_;
    std::vector<int>              suspicious_;
    std::vector<int>              hard_deleted_;

    static void check_fold(int f) {
        if (f < 0 || f > 4)
            throw std::out_of_range("fold index must be 0..4");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  assign_folds — round-robin fold assignment for a freshly built corpus.
//
//  Shuffles the Ok-only indices (by plan_id for reproducibility, not by
//  randomness) then assigns fold_id = i % 5.  Suspicious and HardDeleted
//  points are assigned fold_id = 255 (sentinel — never used in CV).
//
//  Call this once after measuring all data points, before constructing
//  CounterPreprocessor.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
void assign_folds(std::span<DataPoint<N>> corpus) noexcept {
    // Collect Ok indices sorted by plan_id for deterministic assignment.
    std::vector<int> ok_idx;
    ok_idx.reserve(corpus.size());
    for (int i = 0; i < static_cast<int>(corpus.size()); ++i)
        if (corpus[i].is_ok()) ok_idx.push_back(i);

    std::sort(ok_idx.begin(), ok_idx.end(),
              [&](int a, int b){ return corpus[a].plan_id < corpus[b].plan_id; });

    for (int k = 0; k < static_cast<int>(ok_idx.size()); ++k)
        corpus[ok_idx[k]].fold_id = static_cast<uint8_t>(k % 5);

    // Sentinel for non-Ok points.
    for (auto& dp : corpus)
        if (!dp.is_ok()) dp.fold_id = 255;
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_COUNTER_PREPROCESSOR_H
