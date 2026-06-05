#include "spmv_design_space/search.hpp"
#include "spmv_design_space/plan_tree.hpp"

#include "test_support.hpp"
#include <iostream>
#include <sstream>
#include <string>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(10, 10);
    const auto reference = execute_strict_reference(problem, 0.125);

    const auto hybrid_problem = make_stencil_with_remainder_problem(10, 10, 7);
    const auto hybrid_reference = execute_strict_reference(hybrid_problem, 0.125);
    plan_descriptor hybrid{
        .name = "hybrid strict",
        .storage = storage_kind::dia_csr_remainder,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes4,
        .executor = executor_kind::hybrid_dia_csr_executor
    };
    const auto hybrid_result = execute_plan(hybrid_problem, hybrid, 0.125);
    SPMV_REQUIRE(conforms_to_reference(hybrid_result, hybrid_reference, hybrid));
    SPMV_REQUIRE(hybrid_result.execution_path.find("hybrid_dia_csr_executor") != std::string::npos);

    plan_descriptor csr{
        .name = "csr strict",
        .storage = storage_kind::csr,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .executor = executor_kind::csr_executor
    };
    const auto csr_result = execute_plan(problem, csr, 0.125);
    SPMV_REQUIRE(conforms_to_reference(csr_result, reference, csr));


    plan_descriptor dia{
        .name = "dia strict",
        .storage = storage_kind::dia,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes4,
        .executor = executor_kind::dia_executor
    };
    const auto dia_result = execute_plan(problem, dia, 0.125);
    SPMV_REQUIRE(conforms_to_reference(dia_result, reference, dia));
    SPMV_REQUIRE(dia_result.execution_path.find("dia_executor") != std::string::npos);

    plan_descriptor fused{
        .name = "matrix free fused strict",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::blocked_rows,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes8,
        .fusion = fusion_kind::rzpu,
        .executor = executor_kind::matrix_free_executor
    };
    const auto fused_result = execute_plan(problem, fused, 0.125);
    SPMV_REQUIRE(conforms_to_reference(fused_result, reference, fused));
    SPMV_REQUIRE(fused_result.execution_path.find("[RZPU]") != std::string::npos);

    const fusion_kind partitions[] = {
        fusion_kind::r_z_p_u,
        fusion_kind::rz_p_u,
        fusion_kind::r_zp_u,
        fusion_kind::r_z_pu,
        fusion_kind::rzp_u,
        fusion_kind::rz_pu,
        fusion_kind::r_zpu,
        fusion_kind::rzpu
    };
    for (const auto partition : partitions) {
        plan_descriptor partition_plan{
            .name = "matrix free fusion partition strict",
            .storage = storage_kind::matrix_free_stencil,
            .decomposition = decomposition_kind::blocked_rows,
            .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
            .simd = simd_kind::lanes4,
            .fusion = partition,
            .executor = executor_kind::matrix_free_executor
        };
        const auto partition_result = execute_plan(problem, partition_plan, 0.125);
        SPMV_REQUIRE(conforms_to_reference(partition_result, reference, partition_plan));
    }

    plan_descriptor mf{
        .name = "matrix free strict",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::recursive_grid_bisection,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .executor = executor_kind::matrix_free_executor
    };
    const auto mf_result = execute_plan(problem, mf, 0.125);
    SPMV_REQUIRE(conforms_to_reference(mf_result, reference, mf));


    persistent_row_pool pool{2};
    execution_context ctx{.pool = &pool, .task_grain = 16};

    plan_descriptor red_black{
        .name = "matrix free red-black strict",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::blocked_rows,
        .colouring = colouring_kind::red_black_stencil,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .threading = threading_kind::colour_phases,
        .simd = simd_kind::lanes8,
        .executor = executor_kind::matrix_free_executor
    };
    const auto rb_result = execute_plan(problem, red_black, 0.125, &ctx);
    SPMV_REQUIRE(conforms_to_reference(rb_result, reference, red_black));
    SPMV_REQUIRE(!rb_result.visited_rows.empty());
    SPMV_REQUIRE(rb_result.visited_rows[0] == 0);
    SPMV_REQUIRE(rb_result.visited_rows[1] != 1); // red-black phase order is not plain row-major order.
    SPMV_REQUIRE(rb_result.execution_path.find("red_black_stencil") != std::string::npos);
    SPMV_REQUIRE(rb_result.execution_path.find("colour_phases(2T)") != std::string::npos);
    SPMV_REQUIRE(rb_result.execution_path.find("lanes8(8 lanes)") != std::string::npos);
    SPMV_REQUIRE(rb_result.worker_count == 2);
    SPMV_REQUIRE(rb_result.simd_lanes == 8);

    const expression_contract contract{};
    search_options options;
    options.iterations = 5;
    options.warmup = 1;
    options.threads = 2;
    options.task_grain = 16;
    const auto results = run_design_space_search(problem, contract, 0.125, options);
    SPMV_REQUIRE(!results.empty());
    const auto* selected = select_best_legal(results);
    SPMV_REQUIRE(selected != nullptr);
    const auto selected_tree = plan_tree::as_plan_tree(selected->plan);
    SPMV_REQUIRE(plan_tree::get_leaf<plan_tree::role::storage>(selected_tree).value == selected->plan.storage);
    SPMV_REQUIRE(plan_tree::get_leaf<plan_tree::role::fusion>(selected_tree).value == selected->plan.fusion);
    SPMV_REQUIRE(plan_tree::get_leaf<plan_tree::role::reduction>(selected_tree).value == selected->plan.reduction);
    SPMV_REQUIRE(!selected->plan_tree.empty());
    SPMV_REQUIRE(selected->plan_tree.find("leaf<storage::") != std::string::npos);

    bool saw_illegal = false;
    bool saw_legal_conforming = false;
    bool saw_dia_candidate = false;
    bool saw_fused_candidate = false;
    for (const auto& r : results) {
        saw_illegal = saw_illegal || !r.legality.legal();
        saw_dia_candidate = saw_dia_candidate || (r.plan.storage == storage_kind::dia && r.conformance_passed);
        saw_fused_candidate = saw_fused_candidate || (r.plan.fusion == fusion_kind::rzpu && r.conformance_passed);
        saw_legal_conforming = saw_legal_conforming || r.strict_conforming;
        if (r.strict_conforming) {
            SPMV_REQUIRE(r.executed);
            SPMV_REQUIRE(r.best_ns >= 0.0);
            SPMV_REQUIRE(r.median_ns >= 0.0);
            SPMV_REQUIRE(r.mean_ns >= 0.0);
        }
    }
    SPMV_REQUIRE(saw_illegal);
    SPMV_REQUIRE(saw_legal_conforming);
    SPMV_REQUIRE(saw_dia_candidate);
    SPMV_REQUIRE(saw_fused_candidate);

    search_options relaxed_options = options;
    relaxed_options.scope = candidate_scope::strict_and_relaxed_executable;
    const auto relaxed_results = run_design_space_search(problem, contract, 0.125, relaxed_options);
    SPMV_REQUIRE(select_best_legal(relaxed_results) != nullptr);
    SPMV_REQUIRE(select_fastest_executed(relaxed_results) != nullptr);
    SPMV_REQUIRE(select_fastest_non_strict_executed(relaxed_results) != nullptr);

    bool saw_executed_unordered_witness = false;
    bool saw_unexecuted_structural_rejection = false;
    for (const auto& r : relaxed_results) {
        if (r.plan.reduction == reduction_kind::thread_local_unordered_witness) {
            saw_executed_unordered_witness = saw_executed_unordered_witness ||
                (r.executed && r.relaxed_executable && !r.strict_conforming);
            SPMV_REQUIRE(!r.legality.legal());
            SPMV_REQUIRE(!r.legality.observation_legal);
        }
        if (r.plan.preconditioner == preconditioner_kind::coloured_smoother_solver_family) {
            saw_unexecuted_structural_rejection = saw_unexecuted_structural_rejection ||
                (!r.executed && !r.relaxed_executable);
        }
        if (r.executed && !r.strict_conforming) {
            SPMV_REQUIRE(r.best_ns >= 0.0);
            SPMV_REQUIRE(r.median_ns >= 0.0);
            SPMV_REQUIRE(r.x_next_max_abs_delta >= 0.0);
        }
    }
    SPMV_REQUIRE(saw_executed_unordered_witness);
    SPMV_REQUIRE(saw_unexecuted_structural_rejection);


    const auto larger_problem = make_stencil_problem(64, 64);
    search_options large_options;
    large_options.iterations = 3;
    large_options.warmup = 1;
    large_options.threads = 2;
    large_options.task_grain = 512;
    const auto larger_results = run_design_space_search(larger_problem, contract, 0.125, large_options);
    SPMV_REQUIRE(select_best_legal(larger_results) != nullptr);
    bool saw_threaded_candidate = false;
    for (const auto& r : larger_results) {
        saw_threaded_candidate = saw_threaded_candidate ||
            (r.legality.legal() && uses_parallel_threading(r.plan.threading));
    }
    SPMV_REQUIRE(saw_threaded_candidate);

    search_options single_thread_options;
    single_thread_options.iterations = 2;
    single_thread_options.warmup = 1;
    single_thread_options.threads = 1;
    single_thread_options.hardware.max_worker_threads = 1;
    single_thread_options.hardware.allow_parallel_candidates = false;
    single_thread_options.hardware.max_simd_lanes = 4;
    const auto hardware_results = run_design_space_search(problem, contract, 0.125, single_thread_options);
    bool saw_target_rejection = false;
    for (const auto& r : hardware_results) {
        saw_target_rejection = saw_target_rejection || !r.legality.target_legal;
        if (r.plan.simd == simd_kind::lanes8) {
            SPMV_REQUIRE(!r.legality.target_legal);
        }
        if (uses_parallel_threading(r.plan.threading)) {
            SPMV_REQUIRE(!r.legality.target_legal);
        }
    }
    SPMV_REQUIRE(saw_target_rejection);

    const auto hybrid_search_problem = make_stencil_with_remainder_problem(16, 16, 9);
    search_options hybrid_options;
    hybrid_options.iterations = 2;
    hybrid_options.warmup = 1;
    hybrid_options.threads = 1;
    const auto hybrid_search_results = run_design_space_search(hybrid_search_problem, contract, 0.125, hybrid_options);
    bool saw_legal_hybrid_candidate = false;
    bool saw_regular_stencil_rejected = false;
    for (const auto& r : hybrid_search_results) {
        saw_legal_hybrid_candidate = saw_legal_hybrid_candidate ||
            (r.plan.storage == storage_kind::dia_csr_remainder && r.strict_conforming);
        saw_regular_stencil_rejected = saw_regular_stencil_rejected ||
            ((r.plan.storage == storage_kind::dia || r.plan.storage == storage_kind::matrix_free_stencil) &&
             !r.legality.structurally_legal);
    }
    SPMV_REQUIRE(saw_legal_hybrid_candidate);
    SPMV_REQUIRE(saw_regular_stencil_rejected);

    std::ostringstream report;
    print_report(report, hybrid_search_problem, analyse_problem(hybrid_search_problem), hybrid_search_results, hybrid_options);
    const auto report_text = report.str();
    SPMV_REQUIRE(report_text.find("Hardware profile") != std::string::npos);
    SPMV_REQUIRE(report_text.find("Objective and gates") != std::string::npos);
    SPMV_REQUIRE(report_text.find("generated_wisdom") != std::string::npos);
    SPMV_REQUIRE(report_text.find("dia_csr_remainder") != std::string::npos);

    std::cout << "executor tests PASS\n";
}
