#pragma once

#include "spmv_design_space/facts.hpp"
#include "spmv_design_space/problem.hpp"
#include "spmv_design_space/search.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <iterator>
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
