#include "spmv_design_space/reporting.hpp"

#include "test_support.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(8, 8);
    const auto facts = analyse_problem(problem);

    search_options options{};
    options.scope = candidate_scope::strict_conforming_only;
    options.timing_observation = timing_observation_mode::solver_state_only;

    candidate_result selected{};
    selected.plan = plan_descriptor{
        .name = "strict/matrix-free/recursive/block4/[(R,Z,rho)][(Q,sigma)][alpha][U]/jacobi/canonical",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::recursive_grid_bisection,
        .threading = threading_kind::recursive_tasks,
        .simd = simd_kind::lanes4,
        .fusion = fusion_kind::rzp_u,
        .reduction = reduction_kind::canonical_pairwise,
        .executor = executor_kind::matrix_free_executor
    };
    selected.median_ns = 1234.0;
    selected.mean_ns = 1300.0;
    selected.rho_bits = 0x1111111111111111ull;
    selected.sigma_bits = 0x2222222222222222ull;
    selected.alpha_bits = 0x3333333333333333ull;
    selected.x_next_hash = 0x4444444444444444ull;
    selected.observation_hash = 0x5555555555555555ull;

    const auto row = make_selected_plan_summary(problem, facts, options, &selected, "test/os");
    SPMV_REQUIRE(row.found);
    SPMV_REQUIRE(row.platform == "test/os");
    SPMV_REQUIRE(row.problem_kind == "2d_stencil");
    SPMV_REQUIRE(row.layout == "matrix_free_stencil");
    SPMV_REQUIRE(row.decomposition == "recursive_grid_bisection");
    SPMV_REQUIRE(row.simd == "lanes4");
    SPMV_REQUIRE(row.observation_mode == "solver-state-only");
    SPMV_REQUIRE(row.search_mode == "strict_conforming_only");
    SPMV_REQUIRE(row.observation_hash == "0x5555555555555555");
    SPMV_REQUIRE(row.uses_threads);

    std::ostringstream csv;
    write_selected_plan_summary_csv(csv, {row});
    const auto csv_text = csv.str();
    SPMV_REQUIRE(csv_text.find("platform,problem_kind,width,height") != std::string::npos);
    SPMV_REQUIRE(csv_text.find("test/os,2d_stencil,8,8") != std::string::npos);
    // The fusion label contains a comma inside (Q,sigma), so it must be CSV-quoted.
    SPMV_REQUIRE(csv_text.find("\"[(R,Z,rho)][(Q,sigma)][alpha][U]\"") != std::string::npos);

    std::ostringstream md;
    write_selected_plan_summary_markdown(md, {row});
    const auto md_text = md.str();
    SPMV_REQUIRE(md_text.find("# SpMV selected strict plans by size") != std::string::npos);
    SPMV_REQUIRE(md_text.find("`0x5555555555555555`") != std::string::npos);

    auto row2 = row;
    row2.width = 16;
    row2.height = 16;
    row2.median_ns = 999.0;
    row2.simd = "lanes8";
    row2.selected = "strict/matrix-free/recursive/block8/[(R,Z,rho)][(Q,sigma)][alpha][U]/jacobi/canonical";
    row2.observation_hash = "0x6666666666666666";

    const auto families = summarize_selected_families({row, row2});
    SPMV_REQUIRE(families.size() == 1);
    SPMV_REQUIRE(families[0].wins == 2);
    SPMV_REQUIRE(families[0].best_size == "16x16");
    SPMV_REQUIRE(families[0].best_simd == "lanes8");

    std::ostringstream family_md;
    write_family_summary_markdown(family_md, {row, row2});
    SPMV_REQUIRE(family_md.str().find("SpMV selected-plan structural families") != std::string::npos);

    std::cout << "reporting tests PASS\n";
}
