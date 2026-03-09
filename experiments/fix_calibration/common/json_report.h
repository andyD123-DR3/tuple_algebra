#ifndef CTDP_FIX_EXPERIMENT_JSON_REPORT_H
#define CTDP_FIX_EXPERIMENT_JSON_REPORT_H

// json_report.h — Machine-readable JSON output for FIX calibration experiments
//
// Part of PR3 (step3-output). Tested via mock pipeline in PR5 and
// directly by test_output.cpp.
//
// Public API:
//   to_json(report)                   — serialise to std::string
//   write_json_report(report)         — one-arg, writes to results/<id>.json
//   write_json_report_to(dir, report) — two-arg, writes to dir/<id>.json
//
// write_json_report() matches void(const ExperimentReport&) for use as
// a callback in run_experiment().
//
// Filename uses program_id (single letter, always filesystem-safe)
// rather than program_name to avoid path-safety issues.  [Review issue 6]
//
// No external JSON library — minimal hand-rolled serialiser sufficient for
// flat numeric/string fields. Output is valid JSON consumable by Python's
// json.load() and compare_results.py (PR9).
//
// The results/ directory is gitignored; callers create it if absent.

#include "report_types.h"

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ctdp::fix_experiment {

namespace detail {

/// Escape a string for JSON (handles \, ", control chars).
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // control character — emit \u00XX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// Minimal JSON builder using std::ostringstream.
class JsonWriter {
    std::ostringstream os_;
    int indent_ = 0;
    bool need_comma_ = false;

    void newline() {
        os_ << '\n';
        for (int i = 0; i < indent_; ++i) os_ << "  ";
    }
    void comma_if_needed() {
        if (need_comma_) os_ << ',';
        need_comma_ = false;
    }

public:
    void begin_object() {
        comma_if_needed();
        os_ << '{';
        ++indent_;
        need_comma_ = false;
    }
    void end_object() {
        --indent_;
        newline();
        os_ << '}';
        need_comma_ = true;
    }

    void begin_array() {
        comma_if_needed();
        os_ << '[';
        ++indent_;
        need_comma_ = false;
    }
    void end_array() {
        --indent_;
        newline();
        os_ << ']';
        need_comma_ = true;
    }

    void key(const std::string& k) {
        comma_if_needed();
        newline();
        os_ << '"' << json_escape(k) << "\": ";
        need_comma_ = false;
    }

    void value_string(const std::string& v) {
        comma_if_needed();
        os_ << '"' << json_escape(v) << '"';
        need_comma_ = true;
    }
    void value_bool(bool v) {
        comma_if_needed();
        os_ << (v ? "true" : "false");
        need_comma_ = true;
    }
    void value_double(double v, int precision = 4) {
        comma_if_needed();
        os_ << std::fixed << std::setprecision(precision) << v;
        need_comma_ = true;
    }
    void value_size(std::size_t v) {
        comma_if_needed();
        os_ << v;
        need_comma_ = true;
    }

    std::string str() const { return os_.str(); }
};

inline void serialise_model(JsonWriter& w, const ModelMetrics& m) {
    w.begin_object();
    w.key("cv_rmse");      w.value_double(m.cv_rmse);
    w.key("cv_r2");        w.value_double(m.cv_r2);
    w.key("n_training");   w.value_size(m.n_training);
    w.key("n_features");   w.value_size(m.n_features);
    w.key("n_folds");      w.value_size(m.n_folds);
    w.end_object();
}

inline void serialise_candidate(JsonWriter& w, const CandidateResult& c) {
    w.begin_object();
    w.key("rank");             w.value_size(c.rank);
    w.key("config_index");     w.value_size(c.config_index);
    w.key("config_label");     w.value_string(c.config_label);
    w.key("predicted_target"); w.value_double(c.predicted_target, 4);
    w.key("measured_p50_ns");  w.value_double(c.measured_p50_ns, 2);
    w.key("measured_p99_ns");  w.value_double(c.measured_p99_ns, 2);
    w.end_object();
}

inline void serialise_baseline(JsonWriter& w, const BaselineResult& b) {
    w.begin_object();
    w.key("name");             w.value_string(b.name);
    w.key("config_label");     w.value_string(b.config_label);
    w.key("measured_p50_ns");  w.value_double(b.measured_p50_ns, 2);
    w.key("measured_p99_ns");  w.value_double(b.measured_p99_ns, 2);
    w.end_object();
}

} // namespace detail

// ── Serialisation ────────────────────────────────────────────────────

/// Serialise an ExperimentReport to a JSON string.
inline std::string to_json(const ExperimentReport& r) {
    validate_report(r);
    detail::JsonWriter w;

    w.begin_object();

    w.key("program_id");         w.value_string(r.program_id);
    w.key("program_name");       w.value_string(r.program_name);
    w.key("extractor_name");     w.value_string(r.extractor_name);
    w.key("target_description"); w.value_string(r.target_description);
    w.key("target_units");       w.value_string(r.target_units);
    w.key("beam_width");         w.value_size(r.beam_width);
    w.key("beam_depth");         w.value_size(r.beam_depth);
    w.key("exhaustive");         w.value_bool(r.exhaustive);

    if (!r.exhaustive) {
        w.key("model");
        detail::serialise_model(w, r.model);
    }

    w.key("baselines");
    w.begin_array();
    for (const auto& b : r.baselines) {
        detail::serialise_baseline(w, b);
    }
    w.end_array();

    w.key("candidates");
    w.begin_array();
    for (const auto& c : r.candidates) {
        detail::serialise_candidate(w, c);
    }
    w.end_array();

    w.key("best_measured_index"); w.value_size(r.best_measured_index);

    // Convenience: repeat best-measured values at top level for quick access
    if (!r.candidates.empty()) {
        const auto& best = r.candidates[r.best_measured_index];
        w.key("best_p50_ns"); w.value_double(best.measured_p50_ns, 2);
        w.key("best_p99_ns"); w.value_double(best.measured_p99_ns, 2);
    }

    w.end_object();

    return w.str();
}

// ── File output ──────────────────────────────────────────────────────

/// Write an ExperimentReport to dir/<program_id>.json.
/// Creates the directory if it does not exist.
/// Extra-param-first signature for binding / explicit directory choice.
inline void write_json_report_to(const std::filesystem::path& results_dir,
                                 const ExperimentReport& r) {
    std::filesystem::create_directories(results_dir);

    // Use program_id (single letter, always filesystem-safe) for filename.
    auto path = results_dir / (r.program_id + ".json");
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error(
            "write_json_report: cannot open " + path.string());
    }
    ofs << to_json(r) << '\n';
}

/// Write an ExperimentReport to results/<program_id>.json.
/// Matches the callback signature: void(const ExperimentReport&).
inline void write_json_report(const ExperimentReport& r) {
    write_json_report_to("results", r);
}

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_JSON_REPORT_H
