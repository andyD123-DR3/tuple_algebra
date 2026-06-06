#pragma once

#include "spmv_design_space/facts.hpp"
#include "spmv_design_space/problem.hpp"
#include "spmv_design_space/search.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ctdp::spmv_dsl {

inline std::string default_platform_label() {
#if defined(_WIN32)
    std::string os = "windows";
#elif defined(__APPLE__)
    std::string os = "macos";
#elif defined(__linux__)
    std::string os = "linux";
#else
    std::string os = "unknown-os";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    return os + "/arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return os + "/x64";
#elif defined(__i386__) || defined(_M_IX86)
    return os + "/x86";
#else
    return os + "/unknown-arch";
#endif
}

inline std::string plan_family_label(const plan_descriptor& p) {
    std::ostringstream os;
    os << to_string(p.storage)
       << " / " << to_string(p.decomposition)
       << " / " << to_string(p.threading)
       << " / " << to_string(p.executor);
    return os.str();
}

inline std::string plan_structural_family_label(const plan_descriptor& p) {
    std::ostringstream os;
    os << to_string(p.storage)
       << " / " << to_string(p.decomposition)
       << " / " << to_string(p.threading);
    return os.str();
}

inline std::string plan_execution_family_label(const plan_descriptor& p) {
    std::ostringstream os;
    os << to_string(p.storage)
       << " / " << to_string(p.decomposition)
       << " / " << to_string(p.simd);
    return os.str();
}

inline std::string csv_escape(std::string_view value) {
    const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string_view::npos;
    if (!needs_quotes) {
        return std::string(value);
    }

    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            out.push_back('"');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}


struct search_run_count_summary {
    std::string platform;
    std::string problem_kind;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t rows = 0;
    std::size_t nnz = 0;
    std::string observation_mode;
    std::string search_mode;
    std::size_t generated_candidates = 0;
    std::size_t legal_candidates = 0;
    std::size_t executed_candidates = 0;
    std::size_t skipped_candidates = 0;
    std::size_t strict_conforming_candidates = 0;
    std::size_t relaxed_executable_candidates = 0;
    std::size_t conformance_passed_candidates = 0;
    std::size_t structural_gate_passed = 0;
    std::size_t target_gate_passed = 0;
    std::size_t numerical_gate_passed = 0;
    std::size_t observation_gate_passed = 0;
};

inline search_run_count_summary make_search_run_count_summary(
    const stencil_problem& problem,
    const sparse_facts& facts,
    const search_options& options,
    const std::vector<candidate_result>& results,
    std::string_view platform = {}) {

    search_run_count_summary row;
    row.platform = platform.empty() ? default_platform_label() : std::string(platform);
    row.problem_kind = problem.has_irregular_remainder() ? "2d_stencil_plus_remainder" : "2d_stencil";
    row.width = problem.width;
    row.height = problem.height;
    row.rows = facts.rows;
    row.nnz = facts.nnz;
    row.observation_mode = std::string(to_string(options.timing_observation));
    row.search_mode = std::string(to_string(options.scope));
    row.generated_candidates = results.size();

    for (const auto& result : results) {
        if (result.legality.legal()) {
            ++row.legal_candidates;
        }
        if (result.executed) {
            ++row.executed_candidates;
        } else {
            ++row.skipped_candidates;
        }
        if (result.strict_conforming) {
            ++row.strict_conforming_candidates;
        }
        if (result.relaxed_executable) {
            ++row.relaxed_executable_candidates;
        }
        if (result.conformance_passed) {
            ++row.conformance_passed_candidates;
        }
        if (result.legality.structurally_legal) {
            ++row.structural_gate_passed;
        }
        if (result.legality.target_legal) {
            ++row.target_gate_passed;
        }
        if (result.legality.numerically_legal) {
            ++row.numerical_gate_passed;
        }
        if (result.legality.observation_legal) {
            ++row.observation_gate_passed;
        }
    }
    return row;
}

inline void write_search_run_count_summary_csv_header(std::ostream& os) {
    os << "platform,problem_kind,width,height,rows,nnz,observation_mode,search_mode,"
          "generated_candidates,legal_candidates,executed_candidates,skipped_candidates,"
          "strict_conforming_candidates,relaxed_executable_candidates,conformance_passed_candidates,"
          "structural_gate_passed,target_gate_passed,numerical_gate_passed,observation_gate_passed\n";
}

inline void write_search_run_count_summary_csv_row(std::ostream& os,
                                                   const search_run_count_summary& row) {
    os << csv_escape(row.platform) << ','
       << csv_escape(row.problem_kind) << ','
       << row.width << ','
       << row.height << ','
       << row.rows << ','
       << row.nnz << ','
       << csv_escape(row.observation_mode) << ','
       << csv_escape(row.search_mode) << ','
       << row.generated_candidates << ','
       << row.legal_candidates << ','
       << row.executed_candidates << ','
       << row.skipped_candidates << ','
       << row.strict_conforming_candidates << ','
       << row.relaxed_executable_candidates << ','
       << row.conformance_passed_candidates << ','
       << row.structural_gate_passed << ','
       << row.target_gate_passed << ','
       << row.numerical_gate_passed << ','
       << row.observation_gate_passed << '\n';
}

inline void write_search_run_count_summary_csv(std::ostream& os,
                                               const std::vector<search_run_count_summary>& rows) {
    write_search_run_count_summary_csv_header(os);
    for (const auto& row : rows) {
        write_search_run_count_summary_csv_row(os, row);
    }
}

inline void write_search_run_count_summary_markdown(
    std::ostream& os,
    const std::vector<search_run_count_summary>& rows) {

    os << "# SpMV candidate-count summary\n\n";
    os << "This table records how much of the generated plan slice was legal, executed, "
          "strict-conforming, or skipped for each sweep case.\n\n";
    os << "| Platform | Size | Generated | Legal | Executed | Skipped | Strict-conforming | Relaxed-executable | Structural | Target | Numerical | Observation |\n";
    os << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& row : rows) {
        os << "| " << row.platform
           << " | " << row.width << "x" << row.height
           << " | " << row.generated_candidates
           << " | " << row.legal_candidates
           << " | " << row.executed_candidates
           << " | " << row.skipped_candidates
           << " | " << row.strict_conforming_candidates
           << " | " << row.relaxed_executable_candidates
           << " | " << row.structural_gate_passed
           << " | " << row.target_gate_passed
           << " | " << row.numerical_gate_passed
           << " | " << row.observation_gate_passed << " |\n";
    }
    os << "\nInterpretation: these counts make the generated-slice boundary explicit.  "
          "The optimiser exhaustively measures the executable part of this slice; "
          "it is not claiming to search every possible sparse implementation.\n";
}

struct selected_plan_summary {
    std::string platform;
    std::string problem_kind;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t rows = 0;
    std::size_t nnz = 0;
    std::string observation_mode;
    std::string search_mode;
    std::string selected;
    std::string family;
    std::string layout;
    std::string decomposition;
    std::string threading;
    std::string simd;
    std::string fusion;
    std::string reduction;
    double median_ns = 0.0;
    double mean_ns = 0.0;
    std::string rho_bits;
    std::string sigma_bits;
    std::string alpha_bits;
    std::string x_next_hash;
    std::string observation_hash;
    bool uses_threads = false;
    bool found = false;
};

inline selected_plan_summary make_selected_plan_summary(
    const stencil_problem& problem,
    const sparse_facts& facts,
    const search_options& options,
    const candidate_result* selected,
    std::string_view platform = {}) {

    selected_plan_summary row;
    row.platform = platform.empty() ? default_platform_label() : std::string(platform);
    row.problem_kind = problem.has_irregular_remainder() ? "2d_stencil_plus_remainder" : "2d_stencil";
    row.width = problem.width;
    row.height = problem.height;
    row.rows = facts.rows;
    row.nnz = facts.nnz;
    row.observation_mode = std::string(to_string(options.timing_observation));
    row.search_mode = std::string(to_string(options.scope));

    if (selected == nullptr) {
        row.selected = "<none>";
        return row;
    }

    row.found = true;
    row.selected = selected->plan.name;
    row.family = plan_family_label(selected->plan);
    row.layout = std::string(to_string(selected->plan.storage));
    row.decomposition = std::string(to_string(selected->plan.decomposition));
    row.threading = std::string(to_string(selected->plan.threading));
    row.simd = std::string(to_string(selected->plan.simd));
    row.fusion = std::string(to_string(selected->plan.fusion));
    row.reduction = std::string(to_string(selected->plan.reduction));
    row.median_ns = selected->median_ns;
    row.mean_ns = selected->mean_ns;
    row.rho_bits = hex64(selected->rho_bits);
    row.sigma_bits = hex64(selected->sigma_bits);
    row.alpha_bits = hex64(selected->alpha_bits);
    row.x_next_hash = hex64(selected->x_next_hash);
    row.observation_hash = hex64(selected->observation_hash);
    row.uses_threads = uses_parallel_threading(selected->plan.threading);
    return row;
}

inline void write_selected_plan_summary_csv_header(std::ostream& os) {
    os << "platform,problem_kind,width,height,rows,nnz,observation_mode,search_mode,"
          "selected,family,layout,decomposition,threading,simd,fusion,reduction,"
          "median_ns,mean_ns,rho_bits,sigma_bits,alpha_bits,x_next_hash,observation_hash,uses_threads\n";
}

inline void write_selected_plan_summary_csv_row(std::ostream& os, const selected_plan_summary& row) {
    os << csv_escape(row.platform) << ','
       << csv_escape(row.problem_kind) << ','
       << row.width << ','
       << row.height << ','
       << row.rows << ','
       << row.nnz << ','
       << csv_escape(row.observation_mode) << ','
       << csv_escape(row.search_mode) << ','
       << csv_escape(row.selected) << ','
       << csv_escape(row.family) << ','
       << csv_escape(row.layout) << ','
       << csv_escape(row.decomposition) << ','
       << csv_escape(row.threading) << ','
       << csv_escape(row.simd) << ','
       << csv_escape(row.fusion) << ','
       << csv_escape(row.reduction) << ','
       << std::fixed << std::setprecision(1) << row.median_ns << ','
       << std::fixed << std::setprecision(1) << row.mean_ns << ','
       << row.rho_bits << ','
       << row.sigma_bits << ','
       << row.alpha_bits << ','
       << row.x_next_hash << ','
       << row.observation_hash << ','
       << (row.uses_threads ? "yes" : "no") << '\n';
}

inline void write_selected_plan_summary_csv(std::ostream& os,
                                            const std::vector<selected_plan_summary>& rows) {
    write_selected_plan_summary_csv_header(os);
    for (const auto& row : rows) {
        write_selected_plan_summary_csv_row(os, row);
    }
}

inline void write_selected_plan_summary_markdown(std::ostream& os,
                                                 const std::vector<selected_plan_summary>& rows) {
    os << "# SpMV selected strict plans by size\n\n";
    os << "This table is intended for talk/report use.  It records the fastest "
          "strict-conforming measured candidate in each sweep case.\n\n";
    os << "| Platform | Size | Layout | Decomposition | SIMD | Fusion | Median ns | Observation hash | Selected plan |\n";
    os << "|---|---:|---|---|---|---|---:|---|---|\n";
    for (const auto& row : rows) {
        os << "| " << row.platform
           << " | " << row.width << "x" << row.height
           << " | " << row.layout
           << " | " << row.decomposition
           << " | " << row.simd
           << " | " << row.fusion
           << " | " << std::fixed << std::setprecision(1) << row.median_ns
           << " | `" << row.observation_hash << "`"
           << " | `" << row.selected << "` |\n";
    }

    os << "\nInterpretation: the selected implementation strategy may vary by "
          "platform and size, but strict rows are comparable by their observed "
          "solver-state fingerprint.\n";
}

struct candidate_plan_summary {
    std::string platform;
    std::string problem_kind;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t rows = 0;
    std::size_t nnz = 0;
    std::string observation_mode;
    std::string search_mode;
    std::string selected;
    std::string structural_family;
    std::string execution_family;
    std::string family;
    std::string contract;
    std::string layout;
    std::string decomposition;
    std::string ordering;
    std::string colouring;
    std::string threading;
    std::string simd;
    std::string fusion;
    std::string preconditioner;
    std::string reduction;
    std::string executor;
    bool legal = false;
    bool strict_conforming = false;
    bool relaxed_executable = false;
    bool conformance_passed = false;
    double best_ns = 0.0;
    double median_ns = 0.0;
    double mean_ns = 0.0;
    double rho_abs_delta = 0.0;
    double sigma_abs_delta = 0.0;
    double alpha_abs_delta = 0.0;
    double x_next_max_abs_delta = 0.0;
    std::string rho_bits;
    std::string sigma_bits;
    std::string alpha_bits;
    std::string x_next_hash;
    std::string observation_hash;
    std::string execution_path;
    std::string plan_name;
};

inline candidate_plan_summary make_candidate_plan_summary(
    const stencil_problem& problem,
    const sparse_facts& facts,
    const search_options& options,
    const candidate_result& result,
    bool selected,
    std::string_view platform = {}) {

    candidate_plan_summary row;
    row.platform = platform.empty() ? default_platform_label() : std::string(platform);
    row.problem_kind = problem.has_irregular_remainder() ? "2d_stencil_plus_remainder" : "2d_stencil";
    row.width = problem.width;
    row.height = problem.height;
    row.rows = facts.rows;
    row.nnz = facts.nnz;
    row.observation_mode = std::string(to_string(options.timing_observation));
    row.search_mode = std::string(to_string(options.scope));
    row.selected = selected ? "yes" : "no";
    row.structural_family = plan_structural_family_label(result.plan);
    row.execution_family = plan_execution_family_label(result.plan);
    row.family = plan_family_label(result.plan);
    row.contract = std::string(to_string(result.plan.contract));
    row.layout = std::string(to_string(result.plan.storage));
    row.decomposition = std::string(to_string(result.plan.decomposition));
    row.ordering = std::string(to_string(result.plan.ordering));
    row.colouring = std::string(to_string(result.plan.colouring));
    row.threading = std::string(to_string(result.plan.threading));
    row.simd = std::string(to_string(result.plan.simd));
    row.fusion = std::string(to_string(result.plan.fusion));
    row.preconditioner = std::string(to_string(result.plan.preconditioner));
    row.reduction = std::string(to_string(result.plan.reduction));
    row.executor = std::string(to_string(result.plan.executor));
    row.legal = result.legality.legal();
    row.strict_conforming = result.strict_conforming;
    row.relaxed_executable = result.relaxed_executable;
    row.conformance_passed = result.conformance_passed;
    row.best_ns = result.best_ns;
    row.median_ns = result.median_ns;
    row.mean_ns = result.mean_ns;
    row.rho_abs_delta = result.rho_abs_delta;
    row.sigma_abs_delta = result.sigma_abs_delta;
    row.alpha_abs_delta = result.alpha_abs_delta;
    row.x_next_max_abs_delta = result.x_next_max_abs_delta;
    row.rho_bits = hex64(result.rho_bits);
    row.sigma_bits = hex64(result.sigma_bits);
    row.alpha_bits = hex64(result.alpha_bits);
    row.x_next_hash = hex64(result.x_next_hash);
    row.observation_hash = hex64(result.observation_hash);
    row.execution_path = result.execution_path;
    row.plan_name = result.plan.name;
    return row;
}

inline std::vector<candidate_plan_summary> make_candidate_plan_summaries(
    const stencil_problem& problem,
    const sparse_facts& facts,
    const search_options& options,
    const std::vector<candidate_result>& results,
    const candidate_result* selected,
    std::string_view platform = {}) {

    std::vector<candidate_plan_summary> rows;
    for (const auto& result : results) {
        if (!result.executed) {
            continue;
        }
        rows.push_back(make_candidate_plan_summary(problem, facts, options, result, selected == &result, platform));
    }
    return rows;
}

inline void write_candidate_plan_summary_csv_header(std::ostream& os) {
    os << "platform,problem_kind,width,height,rows,nnz,observation_mode,search_mode,"
          "selected,structural_family,execution_family,family,contract,layout,decomposition,"
          "ordering,colouring,threading,simd,fusion,preconditioner,reduction,executor,"
          "legal,strict_conforming,relaxed_executable,conformance_passed,best_ns,median_ns,mean_ns,"
          "rho_abs_delta,sigma_abs_delta,alpha_abs_delta,x_next_max_abs_delta,"
          "rho_bits,sigma_bits,alpha_bits,x_next_hash,observation_hash,execution_path,plan_name\n";
}

inline void write_candidate_plan_summary_csv_row(std::ostream& os, const candidate_plan_summary& row) {
    os << csv_escape(row.platform) << ','
       << csv_escape(row.problem_kind) << ','
       << row.width << ','
       << row.height << ','
       << row.rows << ','
       << row.nnz << ','
       << csv_escape(row.observation_mode) << ','
       << csv_escape(row.search_mode) << ','
       << row.selected << ','
       << csv_escape(row.structural_family) << ','
       << csv_escape(row.execution_family) << ','
       << csv_escape(row.family) << ','
       << csv_escape(row.contract) << ','
       << csv_escape(row.layout) << ','
       << csv_escape(row.decomposition) << ','
       << csv_escape(row.ordering) << ','
       << csv_escape(row.colouring) << ','
       << csv_escape(row.threading) << ','
       << csv_escape(row.simd) << ','
       << csv_escape(row.fusion) << ','
       << csv_escape(row.preconditioner) << ','
       << csv_escape(row.reduction) << ','
       << csv_escape(row.executor) << ','
       << (row.legal ? "yes" : "no") << ','
       << (row.strict_conforming ? "yes" : "no") << ','
       << (row.relaxed_executable ? "yes" : "no") << ','
       << (row.conformance_passed ? "yes" : "no") << ','
       << std::fixed << std::setprecision(1) << row.best_ns << ','
       << std::fixed << std::setprecision(1) << row.median_ns << ','
       << std::fixed << std::setprecision(1) << row.mean_ns << ','
       << std::setprecision(17) << row.rho_abs_delta << ','
       << std::setprecision(17) << row.sigma_abs_delta << ','
       << std::setprecision(17) << row.alpha_abs_delta << ','
       << std::setprecision(17) << row.x_next_max_abs_delta << ','
       << row.rho_bits << ','
       << row.sigma_bits << ','
       << row.alpha_bits << ','
       << row.x_next_hash << ','
       << row.observation_hash << ','
       << csv_escape(row.execution_path) << ','
       << csv_escape(row.plan_name) << '\n';
}

inline void write_candidate_plan_summary_csv(std::ostream& os,
                                             const std::vector<candidate_plan_summary>& rows) {
    write_candidate_plan_summary_csv_header(os);
    for (const auto& row : rows) {
        write_candidate_plan_summary_csv_row(os, row);
    }
}

struct candidate_family_summary {
    std::string structural_family;
    std::size_t measured_candidates = 0;
    std::size_t strict_conforming = 0;
    std::size_t selected_wins = 0;
    double best_median_ns = std::numeric_limits<double>::infinity();
    std::string best_size;
    std::string best_simd;
    std::string best_fusion;
    std::string best_reduction;
    std::string best_observation_hash;
    std::string best_plan;
};

inline std::vector<candidate_family_summary> summarize_candidate_families(
    const std::vector<candidate_plan_summary>& rows) {

    std::vector<candidate_family_summary> families;
    for (const auto& row : rows) {
        auto it = std::find_if(families.begin(), families.end(), [&](const candidate_family_summary& f) {
            return f.structural_family == row.structural_family;
        });
        if (it == families.end()) {
            candidate_family_summary f;
            f.structural_family = row.structural_family;
            families.push_back(f);
            it = std::prev(families.end());
        }
        ++it->measured_candidates;
        if (row.strict_conforming) {
            ++it->strict_conforming;
        }
        if (row.selected == "yes") {
            ++it->selected_wins;
        }
        if (row.median_ns < it->best_median_ns) {
            it->best_median_ns = row.median_ns;
            it->best_size = std::to_string(row.width) + "x" + std::to_string(row.height);
            it->best_simd = row.simd;
            it->best_fusion = row.fusion;
            it->best_reduction = row.reduction;
            it->best_observation_hash = row.observation_hash;
            it->best_plan = row.plan_name;
        }
    }

    std::sort(families.begin(), families.end(), [](const candidate_family_summary& a,
                                                   const candidate_family_summary& b) {
        if (a.selected_wins != b.selected_wins) {
            return a.selected_wins > b.selected_wins;
        }
        if (a.strict_conforming != b.strict_conforming) {
            return a.strict_conforming > b.strict_conforming;
        }
        return a.best_median_ns < b.best_median_ns;
    });
    return families;
}

inline void write_candidate_family_summary_markdown(
    std::ostream& os,
    const std::vector<candidate_plan_summary>& rows) {

    const auto families = summarize_candidate_families(rows);
    os << "# SpMV measured-candidate structural families\n\n";
    os << "Family = storage / decomposition / threading.  This groups every executed candidate, "
          "not only the selected winners, so it shows what the optimiser considered.\n\n";
    os << "| Structural family | Measured candidates | Strict candidates | Selected wins | Best size | Best SIMD | Best fusion | Best reduction | Best median ns | Best observation hash | Best plan |\n";
    os << "|---|---:|---:|---:|---:|---|---|---|---:|---|---|\n";
    for (const auto& f : families) {
        os << "| " << f.structural_family
           << " | " << f.measured_candidates
           << " | " << f.strict_conforming
           << " | " << f.selected_wins
           << " | " << f.best_size
           << " | " << f.best_simd
           << " | " << f.best_fusion
           << " | " << f.best_reduction
           << " | " << std::fixed << std::setprecision(1) << f.best_median_ns
           << " | `" << f.best_observation_hash << "`"
           << " | `" << f.best_plan << "` |\n";
    }

    os << "\nInterpretation: this is a design-space map, not merely a winner list.  "
          "It helps explain where CSR/DIA/matrix-free, flat/recursive, SIMD, "
          "fusion and reduction choices are competitive under the selected contract.\n";
}

struct family_summary {
    std::string family;
    std::size_t wins = 0;
    double best_median_ns = 0.0;
    std::string best_size;
    std::string best_simd;
    std::string best_observation_hash;
    std::string best_plan;
};

inline std::vector<family_summary> summarize_selected_families(
    const std::vector<selected_plan_summary>& rows) {

    std::vector<family_summary> families;
    for (const auto& row : rows) {
        if (!row.found) {
            continue;
        }
        auto it = std::find_if(families.begin(), families.end(), [&](const family_summary& f) {
            return f.family == row.family;
        });
        if (it == families.end()) {
            family_summary f;
            f.family = row.family;
            f.best_median_ns = row.median_ns;
            families.push_back(f);
            it = std::prev(families.end());
        }
        ++it->wins;
        if (it->wins == 1 || row.median_ns < it->best_median_ns) {
            it->best_median_ns = row.median_ns;
            it->best_size = std::to_string(row.width) + "x" + std::to_string(row.height);
            it->best_simd = row.simd;
            it->best_observation_hash = row.observation_hash;
            it->best_plan = row.selected;
        }
    }

    std::sort(families.begin(), families.end(), [](const family_summary& a, const family_summary& b) {
        if (a.wins != b.wins) {
            return a.wins > b.wins;
        }
        return a.best_median_ns < b.best_median_ns;
    });
    return families;
}

inline void write_family_summary_markdown(std::ostream& os,
                                          const std::vector<selected_plan_summary>& rows) {
    const auto families = summarize_selected_families(rows);
    os << "# SpMV selected-plan structural families\n\n";
    os << "Family = storage / decomposition / threading / executor.  SIMD and fusion "
          "remain displayed as selected leaves inside the winning member.\n\n";
    os << "| Family | Wins | Best size | Best SIMD | Best median ns | Best observation hash | Best plan |\n";
    os << "|---|---:|---:|---|---:|---|---|\n";
    for (const auto& f : families) {
        os << "| " << f.family
           << " | " << f.wins
           << " | " << f.best_size
           << " | " << f.best_simd
           << " | " << std::fixed << std::setprecision(1) << f.best_median_ns
           << " | `" << f.best_observation_hash << "`"
           << " | `" << f.best_plan << "` |\n";
    }
}

} // namespace ctdp::spmv_dsl
