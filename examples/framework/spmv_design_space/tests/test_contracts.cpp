#include "spmv_design_space/contracts.hpp"
#include "spmv_design_space/plan.hpp"

#include "test_support.hpp"
#include <iostream>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto c = capabilities_for(reduction_kind::canonical_pairwise);
    SPMV_REQUIRE(c.preserves_observed_reduction);
    SPMV_REQUIRE(c.supports_bitwise_check);

    const auto bad = capabilities_for(reduction_kind::thread_local_unordered_witness);
    SPMV_REQUIRE(!bad.preserves_observed_reduction);
    SPMV_REQUIRE(!bad.supports_bitwise_check);
    SPMV_REQUIRE(bad.may_use_unordered_parallelism);

    expression_contract expr{};
    SPMV_REQUIRE(expr.level == contract_level::strict_expression);
    SPMV_REQUIRE(expr.fixed_preconditioner_binding);
    SPMV_REQUIRE(expr.fixed_observed_reduction_tree);

    auto solver = contract_for_plan(expr, contract_level::solver_family);
    SPMV_REQUIRE(solver.level == contract_level::solver_family);
    SPMV_REQUIRE(!solver.fixed_preconditioner_binding);

    std::cout << "contract tests PASS\n";
}
