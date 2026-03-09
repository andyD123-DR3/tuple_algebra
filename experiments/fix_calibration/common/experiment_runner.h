#ifndef CTDP_FIX_EXPERIMENT_RUNNER_H
#define CTDP_FIX_EXPERIMENT_RUNNER_H

// experiments/fix_calibration/common/experiment_runner.h
//
// Two-phase experiment orchestration:
//
//   Phase 1 — run_discovery():
//     Train SVR on measured training configs, beam-search for candidates.
//     Returns discovery_result containing candidate configs + model metrics.
//     The caller writes candidates to generated/X_candidates.h via
//     candidate_writer.h, then recompiles for Phase 2.
//
//   Phase 2 — run_verification():
//     Measure beam-search winners + baselines through the real ET
//     dispatch table.  Produces the final ExperimentReport.
//
//   For testing (mock_measurer): both phases run in sequence in one
//   process — no recompilation needed.
//
// The runner is templated on Measurer (satisfies m.measure(fix_config)
// → measurement_result) and Extractor (feature encoding).
//
// Dependencies:
//   experiment_config.h  — seeds, N_TRAIN, BEAM_WIDTH
//   feature_extractors.h — fix_point, extractor types
//   beam_search.h        — beam_search()
//   baselines.h          — baselines[], measurement_result
//   report_types.h       — ExperimentReport, CandidateResult, etc.
//   svr_model.h          — svr_trainer, svr_model, observation
//   cross_validation.h   — k_fold_cv
// C++20

#include "baselines.h"
#include "beam_search.h"
#include "experiment_config.h"
#include "feature_extractors.h"
#include "report_types.h"

#include <ctdp/calibrator/fix_et_parser.h>
#include <ctdp/solver/cost_models/linear_model.h>
#include <ctdp/solver/cost_models/svr_model.h>
#include <ctdp/solver/cost_models/performance_model.h>
#include <ctdp/solver/cost_models/cross_validation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctdp::fix_experiment {

namespace fix = ctdp::calibrator::fix;
namespace cm  = ctdp::cost_models;

// ─────────────────────────────────────────────────────────────────────
//  experiment_params — what varies per program
// ─────────────────────────────────────────────────────────────────────

struct experiment_params {
    std::string program_id;          // "A", "B", ..., "F"
    std::string program_name;        // "A_p50_baseline"
    std::string target_description;  // "raw p50", "log(p99)"
    std::string target_units;        // "ns", "log(ns)"
    std::size_t beam_width  = BEAM_WIDTH;
    // beam_depth is always NUM_FIELDS (12) — the beam_search algorithm
    // sweeps all field positions left-to-right.  Not user-configurable.
};

// ─────────────────────────────────────────────────────────────────────
//  discovery_result — Phase 1 output
//
//  Carries everything Phase 2 needs without re-training:
//    - Candidate configs (beam-search winners)
//    - Model metrics (for the report)
//    - Predicted scores (for ranking in the report)
// ─────────────────────────────────────────────────────────────────────

struct discovery_result {
    std::vector<fix::fix_config> candidates;      // beam winners, best-first
    std::vector<double>          predicted_scores; // aligned with candidates
    ModelMetrics                 model;            // CV metrics from training
    std::string                  extractor_name;   // for the report
};

// ─────────────────────────────────────────────────────────────────────
//  run_discovery — Phase 1: train, search, return candidates
//
//  Template parameters:
//    Extractor — feature encoding (onehot_extractor, count_extractor, etc.)
//    Measurer  — anything with measure(fix_config) → measurement_result
//    TargetFn  — measurement_result → double (selects the training target)
//
//  Steps:
//    1. Generate N_TRAIN configs (constexpr, deterministic seed)
//    2. Measure each via measurer.measure(cfg)
//    3. Build observations: {fix_point{cfg}, target_fn(result)}
//    4. Train SVR with two-stage hyperparameter tuning
//    5. Beam search: seeds = top 2×beam_width by prediction + baselines
//    6. Return candidate configs + model metrics
// ─────────────────────────────────────────────────────────────────────

template<typename Extractor, typename Measurer, typename TargetFn>
discovery_result run_discovery(
    const experiment_params& params,
    Measurer& measurer,
    TargetFn target_fn)
{
    // ── Step 1: Training configs from shared pool ──
    // training_pool is the canonical constexpr array in experiment_config.h.
    // compiled_measurer dispatch tables must be instantiated over the same array.

    // ── Step 2: Measure training set ──
    struct train_entry {
        fix::fix_config    config;
        measurement_result measured;
    };
    std::vector<train_entry> training;
    training.reserve(N_TRAIN);
    for (const auto& cfg : training_pool) {
        auto mr = measurer.measure(cfg);
        training.push_back({cfg, mr});
    }

    // ── Step 3: Build observations ──
    using obs_t = cm::observation<fix_point>;
    std::vector<obs_t> observations;
    observations.reserve(training.size());
    for (const auto& te : training) {
        double target = target_fn(te.measured);
        observations.push_back(obs_t{fix_point{te.config}, target});
    }

    // ── Step 4: Train SVR ──
    cm::svr_trainer<fix_point, Extractor> trainer{Extractor{}};
    auto model = trainer.build(observations);
    auto q = model.quality();

    ModelMetrics metrics{};
    metrics.cv_rmse    = q.oos_rmse;  // may be NaN if CV failed
    metrics.cv_r2      = q.oos_r2;    // may be NaN if CV failed
    metrics.n_training = training.size();
    metrics.n_features = Extractor::DIM;
    metrics.n_folds    = 5;

    // ── Step 5: Beam search ──
    // Seeds: top 2×beam_width training configs by prediction + baselines
    struct scored_config {
        fix::fix_config config;
        double score;
    };
    std::vector<scored_config> scored;
    scored.reserve(training.size());
    for (const auto& te : training) {
        double s = model.predict(fix_point{te.config});
        scored.push_back({te.config, s});
    }
    // Deterministic sort: (score, config_key).  Stable across compilers.
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) {
            if (a.score < b.score) return true;
            if (b.score < a.score) return false;
            return detail::config_key(a.config) < detail::config_key(b.config);
        });

    std::vector<fix::fix_config> seeds;
    std::size_t n_seeds = std::min(scored.size(), 2 * params.beam_width);
    seeds.reserve(n_seeds + num_baselines);
    for (std::size_t i = 0; i < n_seeds; ++i)
        seeds.push_back(scored[i].config);
    // Always include baselines as seeds
    for (const auto& bl : baselines)
        seeds.push_back(bl.config);

    auto beam = beam_search(model, seeds, params.beam_width);

    // ── Step 6: Collect results ──
    std::size_t n_cand = std::min(beam.beam.size(), params.beam_width);
    std::vector<fix::fix_config> candidates;
    std::vector<double> predicted;
    candidates.reserve(n_cand);
    predicted.reserve(n_cand);
    for (std::size_t i = 0; i < n_cand; ++i) {
        candidates.push_back(beam.beam[i].config);
        predicted.push_back(beam.beam[i].score);
    }

    return discovery_result{
        std::move(candidates),
        std::move(predicted),
        metrics,
        Extractor::name()
    };
}

// ─────────────────────────────────────────────────────────────────────
//  run_verification — Phase 2: measure candidates + baselines → report
//
//  Measurer must be able to measure:
//    - All configs in discovery.candidates (in the dispatch table)
//    - All 5 baseline configs (in the dispatch table or mock)
//
//  Steps:
//    1. Measure each candidate → CandidateResult with p50 + p99
//    2. Measure each baseline → BaselineResult
//    3. Find best_measured_index
//    4. Populate and return ExperimentReport
// ─────────────────────────────────────────────────────────────────────

template<typename Measurer>
ExperimentReport run_verification(
    const experiment_params& params,
    Measurer& measurer,
    const discovery_result& discovery)
{
    ExperimentReport report;
    report.program_id         = params.program_id;
    report.program_name       = params.program_name;
    report.extractor_name     = discovery.extractor_name;
    report.target_description = params.target_description;
    report.target_units       = params.target_units;
    report.beam_width         = params.beam_width;
    report.beam_depth         = static_cast<std::size_t>(NUM_FIELDS);
    report.model              = discovery.model;

    // ── Measure candidates ──
    if (discovery.candidates.size() != discovery.predicted_scores.size()) {
        throw std::logic_error(
            "run_verification: candidates.size() ("
            + std::to_string(discovery.candidates.size())
            + ") != predicted_scores.size() ("
            + std::to_string(discovery.predicted_scores.size()) + ")");
    }
    report.candidates.reserve(discovery.candidates.size());
    for (std::size_t i = 0; i < discovery.candidates.size(); ++i) {
        const auto& cfg = discovery.candidates[i];
        auto mr = measurer.measure(cfg);

        CandidateResult cr;
        cr.config_index     = i;
        cr.config_label     = fix::config_to_string(cfg);
        cr.predicted_target = discovery.predicted_scores[i];
        cr.measured_p50_ns  = mr.p50_ns;
        cr.measured_p99_ns  = mr.p99_ns;
        cr.rank             = i + 1;  // 1-based, sorted by prediction
        report.candidates.push_back(std::move(cr));
    }

    // ── Find best measured (by p50) ──
    report.best_measured_index = 0;
    if (!report.candidates.empty()) {
        double best_p50 = report.candidates[0].measured_p50_ns;
        for (std::size_t i = 1; i < report.candidates.size(); ++i) {
            if (report.candidates[i].measured_p50_ns < best_p50) {
                best_p50 = report.candidates[i].measured_p50_ns;
                report.best_measured_index = i;
            }
        }
    }

    // ── Measure baselines ──
    report.baselines.reserve(num_baselines);
    for (const auto& bl : baselines) {
        auto mr = measurer.measure(bl.config);
        report.baselines.push_back(BaselineResult{
            std::string(bl.name),
            fix::config_to_string(bl.config),
            mr.p50_ns,
            mr.p99_ns
        });
    }

    return report;
}

// ─────────────────────────────────────────────────────────────────────
//  Convenience: run both phases in one call (for mock testing)
// ─────────────────────────────────────────────────────────────────────

template<typename Extractor, typename Measurer, typename TargetFn>
ExperimentReport run_experiment(
    const experiment_params& params,
    Measurer& measurer,
    TargetFn target_fn)
{
    auto discovery = run_discovery<Extractor>(params, measurer, target_fn);
    return run_verification(params, measurer, discovery);
}

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_RUNNER_H
