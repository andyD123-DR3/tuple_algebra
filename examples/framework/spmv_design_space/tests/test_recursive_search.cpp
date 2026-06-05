#include "spmv_design_space/recursive_search.hpp"
#include "spmv_design_space/search.hpp"

#include "test_support.hpp"

#include <iostream>
#include <string>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(64, 64);
    const auto facts = analyse_problem(problem);

    recursive_search_options split_options{};
    split_options.min_leaf_rows = 512;
    split_options.preferred_leaf_rows = 1024;
    split_options.max_depth = 3;
    split_options.split_overhead = 64.0;

    const auto split_tree = search_recursive_spmv_plan(facts, split_options);
    SPMV_REQUIRE(split_tree.selected_split);
    SPMV_REQUIRE(split_tree.children.size() == 2);

    const auto tree_text = recursive_tree_string(split_tree);
    const auto trace_text = recursive_trace_string(split_tree);
    SPMV_REQUIRE(tree_text.find("split<row_bisection>") != std::string::npos);
    SPMV_REQUIRE(tree_text.find("leaf<matrix_free_vectorised_rows>") != std::string::npos);
    SPMV_REQUIRE(trace_text.find("selected=split") != std::string::npos);

    recursive_search_options leaf_options = split_options;
    leaf_options.min_leaf_rows = 512;
    const auto small_leaf = search_region({0, 512}, facts, leaf_options);
    SPMV_REQUIRE(!small_leaf.selected_split);
    SPMV_REQUIRE(small_leaf.selected_reason.find("below split threshold") != std::string::npos);

    recursive_search_options depth_options = split_options;
    depth_options.max_depth = 0;
    const auto depth_leaf = search_recursive_spmv_plan(facts, depth_options);
    SPMV_REQUIRE(!depth_leaf.selected_split);
    SPMV_REQUIRE(depth_leaf.selected_reason.find("max recursive depth") != std::string::npos);

    auto no_stencil_facts = facts;
    no_stencil_facts.stencil_like = false;
    const auto no_stencil_leaf = search_recursive_spmv_plan(no_stencil_facts, split_options);
    SPMV_REQUIRE(!no_stencil_leaf.selected_split);
    SPMV_REQUIRE(no_stencil_leaf.selected_reason.find("no recognised stencil facts") != std::string::npos);

    expression_contract contract{};
    search_options search{};
    search.iterations = 3;
    search.warmup = 1;
    search.threads = 2;
    search.task_grain = 512;
    search.recursive = split_options;
    const auto results = run_design_space_search(problem, contract, 0.125, search);

    bool saw_recursive_candidate = false;
    for (const auto& result : results) {
        if (!result.recursive_search_tree.empty()) {
            saw_recursive_candidate = true;
            SPMV_REQUIRE(result.plan_tree.find("leaf<decomposition::recursive_grid_bisection>") != std::string::npos);
            SPMV_REQUIRE(result.recursive_search_tree.find("split<row_bisection>") != std::string::npos);
            SPMV_REQUIRE(result.recursive_search_trace.find("selected=split") != std::string::npos);
        }
    }
    SPMV_REQUIRE(saw_recursive_candidate);

    std::cout << "recursive search tests PASS\n";
}
