#pragma once

#include "spmv_design_space/executor.hpp"
#include "spmv_design_space/measurement.hpp"
#include "spmv_design_space/plan_tree.hpp"
#include "spmv_design_space/recursive_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace ctdp::spmv_dsl {

struct search_options {
    std::size_t iterations = 31;
    std::size_t warmup = 3;
    std::size_t threads = 0; // 0 means choose a sensible demonstrator default.
    std::size_t task_grain = 2048;
    recursive_search_options recursive{};
    candidate_scope scope = candidate_scope::strict_conforming_only;
    hardware_profile hardware{};
};

inline bool relaxed_search_enabled(const search_options& options) noexcept {
    return options.scope == candidate_scope::strict_and_relaxed_executable;
}

inline double max_abs_delta(const std::vector<double>& a, const std::vector<double>& b) {
    const auto n = std::min(a.size(), b.size());
    double out = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out = std::max(out, std::abs(a[i] - b[i]));
    }
    return out;
}

inline std::size_t default_worker_count() {
    const auto hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 4;
    }
    return std::max<std::size_t>(2, std::min<std::size_t>(4, hw));
}

inline std::size_t effective_worker_count(const search_options& options) {
    return options.threads == 0 ? default_worker_count() : std::max<std::size_t>(1, options.threads);
}

inline std::vector<candidate_result> run_design_space_search(
    const stencil_problem& problem,
    const expression_contract& base_contract,
    double alpha,
    const search_options& options = {}) {

    const auto facts = analyse_problem(problem);
    const auto reference = execute_strict_reference(problem, alpha);
    auto hardware = options.hardware;
    if (hardware.max_worker_threads == 0) {
        hardware.max_worker_threads = effective_worker_count(options);
    }
    persistent_row_pool pool{effective_worker_count(options)};
    execution_context threaded_ctx{.pool = &pool, .task_grain = std::max<std::size_t>(1, options.task_grain)};
    auto plans = generate_candidate_plans(facts);
    std::vector<candidate_result> results;
    results.reserve(plans.size());

    for (const auto& plan : plans) {
        const auto contract = contract_for_plan(base_contract, plan.contract);
        candidate_result result;
        result.plan = plan;
        result.plan_tree = plan_tree::to_string(plan_tree::as_plan_tree(plan));
        result.legality = analyse_legality(contract, facts, plan, hardware);

        if (plan.contract == contract_level::strict_expression &&
            plan.storage == storage_kind::matrix_free_stencil &&
            plan.decomposition == decomposition_kind::recursive_grid_bisection) {
            const auto recursive = search_recursive_spmv_plan(facts, options.recursive);
            result.recursive_search_tree = recursive_tree_string(recursive);
            result.recursive_search_trace = recursive_trace_string(recursive);
        }

        result.relaxed_executable = result.legality.structurally_legal &&
            result.legality.numerically_legal && result.legality.target_legal;
        const bool should_execute = result.legality.legal() ||
            (relaxed_search_enabled(options) && result.relaxed_executable);

        if (should_execute) {
            auto last = execute_plan(problem, plan, alpha, &threaded_ctx);
            result.executed = true;
            result.rho = last.rho;
            result.rho_abs_delta = std::abs(last.rho - reference.rho);
            result.rho_rel_delta = reference.rho == 0.0 ? result.rho_abs_delta : result.rho_abs_delta / std::abs(reference.rho);
            result.x_next_max_abs_delta = max_abs_delta(last.x_next, reference.x_next);
            result.execution_path = last.execution_path;
            result.conformance_passed = conforms_to_reference(last, reference, plan);
            result.strict_conforming = plan.contract == contract_level::strict_expression &&
                result.legality.legal() && result.conformance_passed;
            const auto timings = measure([&] {
                auto r = execute_plan(problem, plan, alpha, &threaded_ctx);
                // Prevent the call from being trivially discarded.
                if (r.x_next.empty()) {
                    throw std::runtime_error("empty execution result");
                }
            }, options.iterations, options.warmup);
            result.best_ns = timings.best_ns;
            result.median_ns = timings.median_ns;
            result.mean_ns = timings.mean_ns;
        }

        results.push_back(result);
    }

    return results;
}

inline std::vector<candidate_result> run_design_space_search(
    const stencil_problem& problem,
    const expression_contract& base_contract,
    double alpha,
    std::size_t iterations) {
    search_options options;
    options.iterations = iterations;
    return run_design_space_search(problem, base_contract, alpha, options);
}

inline const candidate_result* select_best_legal(const std::vector<candidate_result>& results) {
    const candidate_result* best_strict = nullptr;
    for (const auto& r : results) {
        if (!r.strict_conforming) {
            continue;
        }
        if (best_strict == nullptr || r.median_ns < best_strict->median_ns) {
            best_strict = &r;
        }
    }
    return best_strict;
}

inline const candidate_result* select_fastest_executed(const std::vector<candidate_result>& results) {
    const candidate_result* best = nullptr;
    for (const auto& r : results) {
        if (!r.executed) {
            continue;
        }
        if (best == nullptr || r.median_ns < best->median_ns) {
            best = &r;
        }
    }
    return best;
}

inline const candidate_result* select_fastest_non_strict_executed(const std::vector<candidate_result>& results) {
    const candidate_result* best = nullptr;
    for (const auto& r : results) {
        if (!r.executed || r.strict_conforming) {
            continue;
        }
        if (best == nullptr || r.median_ns < best->median_ns) {
            best = &r;
        }
    }
    return best;
}

inline bool selected_plan_uses_threads(const std::vector<candidate_result>& results) {
    const auto* best = select_best_legal(results);
    return best != nullptr && uses_parallel_threading(best->plan.threading);
}

inline void print_indented_block(std::ostream& os, std::string_view label, const std::string& text) {
    if (text.empty()) {
        return;
    }
    os << "      " << label << ":\n";
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        os << "        " << line << "\n";
    }
}

inline std::string wisdom_symbol(fusion_kind fusion) {
    switch (fusion) {
    case fusion_kind::r_z_p_u: return "r_z_p_u";
    case fusion_kind::rz_p_u: return "rz_p_u";
    case fusion_kind::r_zp_u: return "r_zp_u";
    case fusion_kind::r_z_pu: return "r_z_pu";
    case fusion_kind::rzp_u: return "rzp_u";
    case fusion_kind::rz_pu: return "rz_pu";
    case fusion_kind::r_zpu: return "r_zpu";
    case fusion_kind::rzpu: return "rzpu";
    }
    return "unknown";
}

inline std::string emit_plan_wisdom(const plan_descriptor& p) {
    std::ostringstream os;
    os << "using selected_spmv_plan = spmv_plan<\n"
       << "    storage::" << to_string(p.storage) << ",\n"
       << "    decomposition::" << to_string(p.decomposition) << ",\n"
       << "    ordering::" << to_string(p.ordering) << ",\n"
       << "    colouring::" << to_string(p.colouring) << ",\n"
       << "    preconditioner::" << to_string(p.preconditioner) << ",\n"
       << "    threading::" << to_string(p.threading) << ",\n"
       << "    simd::" << to_string(p.simd) << ",\n"
       << "    fusion::" << wisdom_symbol(p.fusion) << ",\n"
       << "    reduction::" << to_string(p.reduction) << ",\n"
       << "    executor::" << to_string(p.executor) << ">;";
    return os.str();
}

inline void print_report(
    std::ostream& os,
    const stencil_problem& problem,
    const sparse_facts& facts,
    const std::vector<candidate_result>& results,
    const search_options& options = {}) {

    os << "Sparse Expression Decomposition DSL Demonstrator\n";
    os << "================================================\n\n";
    os << "Expression:\n";
    os << "  r    = b - A*x\n";
    os << "  z    = M^{-1}r\n";
    os << "  rho  = canonical_dot_row_major(r,z)\n";
    os << "  x'   = x + alpha*z\n\n";

    os << "Facts:\n";
    os << "  grid: " << problem.width << " x " << problem.height << "\n";
    os << "  rows: " << facts.rows << "\n";
    os << "  nnz: " << facts.nnz << "\n";
    os << "  stencil_like: " << (facts.stencil_like ? "yes" : "no") << "\n";
    os << "  connected_components: " << facts.connected_components << "\n";
    os << "  estimated_colour_count: " << facts.estimated_colour_count << "\n";
    os << "  irregular_remainder_nnz: " << facts.irregular_remainder_nnz << "\n\n";

    auto reported_hardware = options.hardware;
    if (reported_hardware.max_worker_threads == 0) {
        reported_hardware.max_worker_threads = effective_worker_count(options);
    }
    os << "Hardware profile:\n";
    os << "  " << describe_hardware_profile(reported_hardware) << "\n\n";

    os << "Objective and gates:\n";
    os << "  objective: minimise measured median_ns\n";
    os << "  structural gate: matrix facts must make the storage/evaluator legal\n";
    os << "  target gate: SIMD/threading must fit the declared hardware profile\n";
    os << "  numerical gate: strict contracts preserve preconditioner binding and observed reduction\n";
    os << "  conformance gate: executed strict candidates must match the reference result bitwise\n\n";

    const bool relaxed_candidates_executed = std::any_of(results.begin(), results.end(), [](const candidate_result& r) {
        return r.executed && !r.legality.legal() && r.relaxed_executable;
    });

    os << "Selection rule:\n";
    os << "  strict_best = lowest measured median_ns among candidates passing all gates\n";
    if (relaxed_candidates_executed) {
        os << "  relaxed comparison = executable non-strict candidates are measured but cannot win strict_best\n";
    }
    os << "  optimality_scope = generated candidates x declared hardware profile x measured objective\n\n";

    os << "Candidates:\n";
    for (const auto& r : results) {
        os << "  - " << describe_plan(r.plan) << "\n";
        os << "      legal: " << (r.legality.legal() ? "yes" : "no") << "\n";
        os << "      legality_gates: structural=" << (r.legality.structurally_legal ? "pass" : "fail")
           << " target=" << (r.legality.target_legal ? "pass" : "fail")
           << " numerical=" << (r.legality.numerically_legal ? "pass" : "fail")
           << " observation=" << (r.legality.observation_legal ? "pass" : "fail") << "\n";
        os << "      reason: " << r.legality.reason << "\n";
        print_indented_block(os, "plan_tree", r.plan_tree);
        print_indented_block(os, "recursive_search_tree", r.recursive_search_tree);
        print_indented_block(os, "recursive_search_trace", r.recursive_search_trace);
        os << "      relaxed_executable: " << (r.relaxed_executable ? "yes" : "no") << "\n";
        os << "      executed: " << (r.executed ? "yes" : "no") << "\n";
        if (r.executed) {
            os << "      strict_conforming: " << (r.strict_conforming ? "yes" : "no") << "\n";
            os << "      conformance: " << (r.conformance_passed ? "PASS" : "FAIL") << "\n";
            os << "      best_ns: " << std::fixed << std::setprecision(1) << r.best_ns << "\n";
            os << "      median_ns: " << std::fixed << std::setprecision(1) << r.median_ns << "\n";
            os << "      mean_ns: " << std::fixed << std::setprecision(1) << r.mean_ns << "\n";
            os << "      rho: " << std::setprecision(17) << r.rho << "\n";
            os << "      rho_abs_delta: " << std::setprecision(17) << r.rho_abs_delta << "\n";
            os << "      rho_rel_delta: " << std::setprecision(17) << r.rho_rel_delta << "\n";
            os << "      x_next_max_abs_delta: " << std::setprecision(17) << r.x_next_max_abs_delta << "\n";
            os << "      executed_by: " << r.execution_path << "\n";
        }
    }

    if (const auto* best = select_best_legal(results)) {
        os << "\nStrict selected:\n";
        os << "  " << describe_plan(best->plan) << "\n";
        os << "  selected_by: lowest measured median_ns among strict-conforming candidates\n";
        os << "  median_ns: " << std::fixed << std::setprecision(1) << best->median_ns << "\n";
        os << "  uses_threads: " << (uses_parallel_threading(best->plan.threading) ? "yes" : "no") << "\n";
        os << "  generated_wisdom:\n";
        std::istringstream wisdom(emit_plan_wisdom(best->plan));
        for (std::string line; std::getline(wisdom, line);) {
            os << "    " << line << "\n";
        }
    } else {
        os << "\nStrict selected:\n  no strict-conforming candidate\n";
    }

    if (const auto* fastest = select_fastest_executed(results)) {
        os << "\nFastest executed candidate:\n";
        os << "  " << describe_plan(fastest->plan) << "\n";
        os << "  median_ns: " << std::fixed << std::setprecision(1) << fastest->median_ns << "\n";
        os << "  strict_conforming: " << (fastest->strict_conforming ? "yes" : "no") << "\n";
    }

    if (relaxed_candidates_executed) {
        if (const auto* non_strict = select_fastest_non_strict_executed(results)) {
            os << "\nFastest non-strict executed candidate:\n";
            os << "  " << describe_plan(non_strict->plan) << "\n";
            os << "  median_ns: " << std::fixed << std::setprecision(1) << non_strict->median_ns << "\n";
            os << "  rho_abs_delta: " << std::setprecision(17) << non_strict->rho_abs_delta << "\n";
            os << "  x_next_max_abs_delta: " << std::setprecision(17) << non_strict->x_next_max_abs_delta << "\n";
        }
    }
}

} // namespace ctdp::spmv_dsl
