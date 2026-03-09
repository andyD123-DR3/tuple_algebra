#ifndef CTDP_FIX_EXPERIMENT_OUTPUT_TABLE_H
#define CTDP_FIX_EXPERIMENT_OUTPUT_TABLE_H

// output_table.h — Console table display for FIX calibration experiments
//
// Part of PR3 (step3-output). Tested via mock pipeline in PR5 and
// directly by test_output.cpp.
//
// Public API:
//   print_report(report)         — one-arg, writes to std::cout
//   print_report_to(os, report)  — two-arg, writes to any ostream
//
// print_report() matches the display_fn signature void(const ExperimentReport&)
// expected by experiment_runner.  print_report_to() is the testable core;
// PR5's test_mock_pipeline.cpp captures output via ostringstream.

#include "report_types.h"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace ctdp::fix_experiment {

namespace detail {

// ── RAII stream-state guard ──────────────────────────────────────────
// Restores flags, precision and fill on scope exit so callers' streams
// are not left with mutated formatting state.  [Review issue 7]

class StreamStateGuard {
    std::ostream&           os_;
    std::ios_base::fmtflags flags_;
    std::streamsize         prec_;
    char                    fill_;
public:
    explicit StreamStateGuard(std::ostream& os)
        : os_(os), flags_(os.flags()), prec_(os.precision()), fill_(os.fill()) {}
    ~StreamStateGuard() {
        os_.flags(flags_);
        os_.precision(prec_);
        os_.fill(fill_);
    }
    StreamStateGuard(const StreamStateGuard&) = delete;
    StreamStateGuard& operator=(const StreamStateGuard&) = delete;
};

// ── Formatting helpers ───────────────────────────────────────────────

inline void print_separator(std::ostream& os, int width) {
    os << std::string(static_cast<std::size_t>(width), '-') << '\n';
}

inline void print_header_block(std::ostream& os, const ExperimentReport& r) {
    os << "\n";
    print_separator(os, 72);
    os << "  Program " << r.program_id << ": " << r.program_name << "\n";
    os << "  Extractor: " << r.extractor_name
       << "  |  Target: " << r.target_description;
    if (!r.target_units.empty()) {
        os << " (" << r.target_units << ")";
    }
    os << "\n";
    os << "  Beam: width=" << r.beam_width
       << " depth=" << r.beam_depth << "\n";
    print_separator(os, 72);
}

inline void print_model_block(std::ostream& os, const ModelMetrics& m) {
    os << "  Model: " << m.n_features << " features, "
       << m.n_training << " training configs, "
       << m.n_folds << "-fold CV\n";
    os << "  CV RMSE: " << std::fixed << std::setprecision(4) << m.cv_rmse
       << "  |  CV R\xC2\xB2: " << std::setprecision(4) << m.cv_r2 << "\n";
    os << "\n";
}

inline void print_candidate_table(std::ostream& os,
                                  const std::vector<CandidateResult>& cands,
                                  std::size_t best_idx) {
    // Header
    os << "  " << std::left
       << std::setw(6)  << "Rank"
       << std::setw(12) << "Predicted"
       << std::setw(12) << "Meas p50"
       << std::setw(12) << "Meas p99"
       << "Config\n";
    os << "  " << std::string(66, '-') << "\n";

    for (std::size_t i = 0; i < cands.size(); ++i) {
        const auto& c = cands[i];
        const char* marker = (i == best_idx) ? " *" : "";
        os << "  " << std::left << std::setw(6) << c.rank
           << std::right << std::fixed
           << std::setw(11) << std::setprecision(2) << c.predicted_target << " "
           << std::setw(11) << std::setprecision(2) << c.measured_p50_ns << " "
           << std::setw(11) << std::setprecision(2) << c.measured_p99_ns << " "
           << std::left << c.config_label << marker << "\n";
    }
    os << "\n";
}

inline void print_baseline_table(std::ostream& os,
                                 const std::vector<BaselineResult>& baselines) {
    if (baselines.empty()) return;
    os << "  Baselines:\n";
    os << "  " << std::left
       << std::setw(18) << "Name"
       << std::setw(12) << "Meas p50"
       << std::setw(12) << "Meas p99"
       << "Config\n";
    os << "  " << std::string(56, '-') << "\n";
    for (const auto& b : baselines) {
        os << "  " << std::left << std::setw(18) << b.name
           << std::right << std::fixed
           << std::setw(11) << std::setprecision(2) << b.measured_p50_ns << " "
           << std::setw(11) << std::setprecision(2) << b.measured_p99_ns << " "
           << std::left << b.config_label << "\n";
    }
    os << "\n";
}

inline void print_summary_block(std::ostream& os,
                                const ExperimentReport& r) {
    if (r.candidates.empty()) {
        os << "  (no candidates)\n";
        return;
    }
    const auto& best_pred = r.candidates.front();  // rank 1
    const auto& best_meas = r.candidates[r.best_measured_index];

    os << "  Best predicted: rank " << best_pred.rank
       << " -> p50=" << std::fixed << std::setprecision(2)
       << best_pred.measured_p50_ns
       << " ns, p99=" << best_pred.measured_p99_ns << " ns\n";

    if (r.best_measured_index != 0) {
        os << "  Best measured:  rank " << best_meas.rank
           << " -> p50=" << best_meas.measured_p50_ns
           << " ns, p99=" << best_meas.measured_p99_ns << " ns"
           << "  (model mis-rank)\n";
    }
    os << "\n";
}

} // namespace detail

// ── Public API ───────────────────────────────────────────────────────

/// Print a complete experiment report to a specific ostream.
/// Extra-param-first signature for binding / testability.
inline void print_report_to(std::ostream& os,
                            const ExperimentReport& report) {
    validate_report(report);
    detail::StreamStateGuard guard(os);

    detail::print_header_block(os, report);
    detail::print_model_block(os, report.model);
    detail::print_baseline_table(os, report.baselines);
    detail::print_candidate_table(os, report.candidates,
                                  report.best_measured_index);
    detail::print_summary_block(os, report);
}

/// Print a complete experiment report to std::cout.
/// Matches the display_fn signature: void(const ExperimentReport&).
inline void print_report(const ExperimentReport& report) {
    print_report_to(std::cout, report);
}

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_OUTPUT_TABLE_H
