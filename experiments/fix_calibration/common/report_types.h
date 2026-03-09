#ifndef CTDP_FIX_EXPERIMENT_REPORT_TYPES_H
#define CTDP_FIX_EXPERIMENT_REPORT_TYPES_H

// report_types.h — Shared report model types for FIX calibration experiments
//
// Part of PR3 (step3-output).
//
// These types are the output contract between the experiment runner (PR5)
// and both output sinks (output_table.h, json_report.h).  They live in
// their own header so that neither output layer depends on the other.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctdp::fix_experiment {

// ── Per-candidate measurement result ─────────────────────────────────

/// Per-candidate measurement result.
struct CandidateResult {
    std::size_t config_index;      // index into candidate set
    std::string config_label;      // human-readable config summary
    double      predicted_target;  // model prediction (in target space — see target_units)
    double      measured_p50_ns;   // wall-clock p50 in nanoseconds
    double      measured_p99_ns;   // wall-clock p99 in nanoseconds
    std::size_t rank;              // 1-based rank by predicted_target
};

// ── Model quality metrics from cross-validation ──────────────────────

/// Model quality metrics from cross-validation.
struct ModelMetrics {
    double      cv_rmse;           // cross-validated RMSE
    double      cv_r2;             // cross-validated R²
    std::size_t n_training;        // number of training configs
    std::size_t n_features;        // feature dimensionality
    std::size_t n_folds;           // k in k-fold CV
};

// ── Complete experiment report ────────────────────────────────────────

/// Complete experiment report — the unit of output.
///
/// Invariants the producer must maintain:
///   - candidates is sorted by rank (ascending).
///   - best_measured_index < candidates.size() (unless candidates is empty).
///   - Every rank value is unique and in [1, candidates.size()].
struct ExperimentReport {
    std::string program_id;         // e.g. "A", "B", ..., "F"
    std::string program_name;       // e.g. "A_p50_baseline"
    std::string extractor_name;     // e.g. "onehot_extractor"
    std::string target_description; // e.g. "raw p50", "log(p99)"
    std::string target_units;       // e.g. "ns", "log(ns)" — disambiguates predicted_target
    std::size_t beam_width;
    std::size_t beam_depth;

    ModelMetrics model;
    std::vector<CandidateResult> candidates;  // sorted by rank

    // Index into candidates of the best result by *measured* p50.
    // May differ from index 0 (best predicted) — that is the mis-rank
    // the experiment is designed to detect.
    std::size_t best_measured_index;  // index into candidates
};

// ── Validation ────────────────────────────────────────────────────────

/// Throws std::out_of_range if best_measured_index is out of bounds
/// for a non-empty candidates vector.  Both output sinks call this
/// before indexing.
inline void validate_report(const ExperimentReport& r) {
    if (!r.candidates.empty() &&
        r.best_measured_index >= r.candidates.size()) {
        throw std::out_of_range(
            "ExperimentReport: best_measured_index ("
            + std::to_string(r.best_measured_index)
            + ") >= candidates.size() ("
            + std::to_string(r.candidates.size()) + ")");
    }
}

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_REPORT_TYPES_H
