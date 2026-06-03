#include "spmv_design_space/plan.hpp"

#include <algorithm>
#include "test_support.hpp"
#include <iostream>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(8, 8);
    const auto facts = analyse_problem(problem);
    const expression_contract strict{};
    const auto plans = generate_candidate_plans(facts);

    SPMV_REQUIRE(plans.size() >= 20);

    const auto has_dia = std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) {
        return p.storage == storage_kind::dia && p.executor == executor_kind::dia_executor;
    });
    SPMV_REQUIRE(has_dia);

    const auto has_fused = std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) {
        return p.fusion == fusion_kind::row_local_fused;
    });
    SPMV_REQUIRE(has_fused);

    const auto has_all_widths =
        std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) { return p.simd == simd_kind::scalar; }) &&
        std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) { return p.simd == simd_kind::lanes4; }) &&
        std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) { return p.simd == simd_kind::lanes8; });
    SPMV_REQUIRE(has_all_widths);

    const auto has_recursive = std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) {
        return p.decomposition == decomposition_kind::recursive_grid_bisection &&
               p.storage == storage_kind::matrix_free_stencil;
    });
    SPMV_REQUIRE(has_recursive);

    auto illegal_unordered = plan_descriptor{
        .name = "bad reduction",
        .storage = storage_kind::csr,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .reduction = reduction_kind::thread_local_unordered_witness,
        .executor = executor_kind::csr_executor
    };
    const auto bad_reduction = analyse_legality(strict, facts, illegal_unordered);
    SPMV_REQUIRE(!bad_reduction.legal());
    SPMV_REQUIRE(!bad_reduction.observation_legal);

    auto illegal_smoother = plan_descriptor{
        .name = "bad smoother",
        .contract = contract_level::solver_family,
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::flat_rows,
        .ordering = ordering_kind::natural,
        .colouring = colouring_kind::none,
        .preconditioner = preconditioner_kind::coloured_smoother_solver_family,
        .executor = executor_kind::matrix_free_executor
    };
    auto solver_contract = contract_for_plan(strict, contract_level::solver_family);
    const auto bad_smoother = analyse_legality(solver_contract, facts, illegal_smoother);
    SPMV_REQUIRE(!bad_smoother.legal());
    SPMV_REQUIRE(!bad_smoother.structurally_legal);

    auto solver_none = plan_descriptor{
        .name = "solver family no preconditioner",
        .contract = contract_level::solver_family,
        .storage = storage_kind::matrix_free_stencil,
        .preconditioner = preconditioner_kind::none_solver_family,
        .executor = executor_kind::matrix_free_executor
    };
    const auto solver_ok = analyse_legality(solver_contract, facts, solver_none);
    SPMV_REQUIRE(solver_ok.legal());

    const auto strict_bad = analyse_legality(strict, facts, solver_none);
    SPMV_REQUIRE(!strict_bad.legal());
    SPMV_REQUIRE(!strict_bad.numerically_legal);

    std::cout << "plan tests PASS\n";
}
