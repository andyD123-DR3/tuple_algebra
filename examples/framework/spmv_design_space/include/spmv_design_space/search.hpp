#pragma once

#include "spmv_design_space/executor.hpp"
#include "spmv_design_space/measurement.hpp"
#include "spmv_design_space/plan_tree.hpp"
#include "spmv_design_space/recursive_search.hpp"

#include <algorithm>
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
};

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
    persistent_row_pool pool{effective_worker_count(options)};
    execution_context threaded_ctx{.pool = &pool, .task_grain = std::max<std::size_t>(1, options.task_grain)};
    auto plans = generate_candidate_plans(facts);
    std::vector<candidate_result> results;
    results.reserve(plans.size());

    for (const auto& plan : plans) {
        const auto contract = contract_for_plan(base_contract, plan.contract);
        candidate_result result;
        result.plan = plan;
        result.legality = analyse_legality(contract, facts, plan);

        if (plan.contract == contract_level::strict_expression &&
            plan.storage == storage_kind::matrix_free_stencil &&
            plan.decomposition == decomposition_kind::recursive_grid_bisection) {
            const auto recursive = search_recursive_spmv_plan(facts, options.recursive);
            result.plan_tree = plan_tree::to_string<plan_tree::matrix_free_recursive_candidate>();
            result.recursive_search_tree = recursive_tree_string(recursive);
            result.recursive_search_trace = recursive_trace_string(recursive);
        }

        if (result.legality.legal()) {
            auto last = execute_plan(problem, plan, alpha, &threaded_ctx);
            result.rho = last.rho;
            result.execution_path = last.execution_path;
            result.conformance_passed = conforms_to_reference(last, reference, plan);
            const auto timings = measure([&] {
                auto r = execute_plan(problem, plan, alpha, &threaded_ctx);
                // Prevent the call from being trivially discarded.
                if (r.x_next.empty()) {
                    throw std::runtime_error("empty execution result");
                }
            }, options.iterations, options.warmup);
            result.median_ns = timings.median_ns;
            result.p99_ns = timings.p99_ns;
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
    const candidate_result* best_any = nullptr;
    for (const auto& r : results) {
        if (!r.legality.legal() || !r.conformance_passed) {
            continue;
        }
        if (best_any == nullptr || r.p99_ns < best_any->p99_ns) {
            best_any = &r;
        }
        if (r.plan.contract == contract_level::strict_expression &&
            (best_strict == nullptr || r.p99_ns < best_strict->p99_ns)) {
            best_strict = &r;
        }
    }
    return best_strict != nullptr ? best_strict : best_any;
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

inline void print_report(
    std::ostream& os,
    const stencil_problem& problem,
    const sparse_facts& facts,
    const std::vector<candidate_result>& results) {

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
    os << "  estimated_colour_count: " << facts.estimated_colour_count << "\n\n";

    os << "Selection rule:\n";
    os << "  best = lowest measured p99 among legal, conforming, strict-expression candidates\n";
    os << "  This proves optimality only over the generated candidate set and measured objective.\n\n";

    os << "Candidates:\n";
    for (const auto& r : results) {
        os << "  - " << describe_plan(r.plan) << "\n";
        os << "      legal: " << (r.legality.legal() ? "yes" : "no") << "\n";
        os << "      reason: " << r.legality.reason << "\n";
        print_indented_block(os, "plan_tree", r.plan_tree);
        print_indented_block(os, "recursive_search_tree", r.recursive_search_tree);
        print_indented_block(os, "recursive_search_trace", r.recursive_search_trace);
        if (r.legality.legal()) {
            os << "      conformance: " << (r.conformance_passed ? "PASS" : "FAIL") << "\n";
            os << "      median_ns: " << std::fixed << std::setprecision(1) << r.median_ns << "\n";
            os << "      p99_ns: " << std::fixed << std::setprecision(1) << r.p99_ns << "\n";
            os << "      rho: " << std::setprecision(17) << r.rho << "\n";
            os << "      executed_by: " << r.execution_path << "\n";
        }
    }

    if (const auto* best = select_best_legal(results)) {
        os << "\nSelected:\n";
        os << "  " << describe_plan(best->plan) << "\n";
        os << "  selected_by: lowest measured p99 among legal conforming strict-expression candidates\n";
        os << "  uses_threads: " << (uses_parallel_threading(best->plan.threading) ? "yes" : "no") << "\n";
    } else {
        os << "\nSelected:\n  no legal conforming candidate\n";
    }
}

} // namespace ctdp::spmv_dsl
