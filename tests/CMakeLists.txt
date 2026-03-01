# ── Google Test ───────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent GTest from overriding parent project settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ── Helper function ──────────────────────────────────────────────────────────
function(ctdp_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ctdp_all GTest::gtest GTest::gtest_main)
    gtest_discover_tests(${target})
endfunction()

# ── Core library tests ───────────────────────────────────────────────────
ctdp_add_test(core_test core/core_test.cc)

# ── Solver library tests (original Half A) ───────────────────────────────
ctdp_add_test(solver_test solver/solver_test.cc)

# ── Solver library tests (integrated Half B: algorithms, spaces, cost models, memo)
ctdp_add_test(test_solver_concepts     test_solver_concepts.cpp)
ctdp_add_test(test_per_element_space   test_per_element_space.cpp)
ctdp_add_test(test_interval_split_space test_interval_split_space.cpp)
ctdp_add_test(test_other_spaces        test_other_spaces.cpp)
ctdp_add_test(test_cost_models         test_cost_models.cpp)
ctdp_add_test(test_per_element_argmin  test_per_element_argmin.cpp)
ctdp_add_test(test_interval_dp         test_interval_dp.cpp)
ctdp_add_test(test_exhaustive_search   test_exhaustive_search.cpp)
ctdp_add_test(test_select_and_run      test_select_and_run.cpp)
ctdp_add_test(test_memo                test_memo.cpp)

# ── Space-parameterised cost and constraint tests ────────────────────────
ctdp_add_test(test_space_parameterised test_space_parameterised.cpp)
ctdp_add_test(test_candidate_traversal test_candidate_traversal.cpp)
ctdp_add_test(test_heterogeneous_space test_heterogeneous_space.cpp)

# ── Constrained search (beam_search, local_search) ──────────────────────
ctdp_add_test(test_constrained_search test_constrained_search.cpp)

# ── Constraint module (combinators, factories) ──────────────────────────
ctdp_add_test(test_constraints test_constraints.cpp)

# ── Algebra library tests ────────────────────────────────────────────────
ctdp_add_test(algebra_test algebra/algebra_test.cc)

# ── Graph library tests ──────────────────────────────────────────────────────
# Step 1: Graph representation, builder, implicit, equality
ctdp_add_test(graph_step1_test graph/graph_step1_test.cc)

# Step 2: Pipeline and stencil construction
ctdp_add_test(graph_step2_test graph/graph_step2_test.cc)

# Step 3: Topological sort, SCC, connected components
ctdp_add_test(graph_step3_test graph/graph_step3_test.cc)

# (Step 4 — annotations: property_map, kernel_info, I/O — was folded into
#  test_graph_io and the hardening tests rather than a standalone step file.)

# Step 5: Fusion legality, coarsening, fusion groups
ctdp_add_test(graph_step5_test graph/graph_step5_test.cc)

# Step 6: Schedule space, dependencies, resource constraints
ctdp_add_test(graph_step6_test graph/graph_step6_test.cc)

# Step 7: SpMV full pipeline end-to-end
ctdp_add_test(graph_step7_test graph/graph_step7_test.cc)

# Graph hardening: capacity guards, builder overflow, coarsen validation
ctdp_add_test(graph_hardening_test graph/graph_hardening_test.cc)

# Symmetric graph: undirected graph type
ctdp_add_test(test_symmetric_graph graph/test_symmetric_graph.cc)

# Graph hardening v2: add_edge validation, SCC performance, include hygiene
ctdp_add_test(test_graph_hardening_v2 graph/test_graph_hardening_v2.cc)

# Graph coloring: Welsh-Powell algorithm and coloring_to_groups bridge
ctdp_add_test(test_graph_coloring graph/test_graph_coloring.cc)

# Weighted view: edge_property_map, weighted_view, symmetric weights
ctdp_add_test(test_weighted_view graph/test_weighted_view.cpp)

# Graph transforms: transpose, induced_subgraph, contract
ctdp_add_test(test_transforms graph/test_transforms.cpp)

# Graph merge rules: named merge policies + group reduction
ctdp_add_test(test_merge_rules graph/test_merge_rules.cpp)

# Segmentation DP: composite DP over graph partitions
ctdp_add_test(test_segmentation_dp graph/test_segmentation_dp.cpp)

# Graph umbrella header: compilation + smoke test
ctdp_add_test(test_graph_umbrella graph/test_graph_umbrella.cpp)

# Bipartite matching: bipartite_graph, Hopcroft-Karp
ctdp_add_test(test_bipartite_matching graph/test_bipartite_matching.cpp)

# Instantiation tier: strategy_map, dispatch_table, plan_executor
ctdp_add_test(test_instantiation test_instantiation.cc)

ctdp_add_test(test_performance_models test_performance_models.cpp)

# ── Space layer v0.7.2 tests (descriptor_space, valid_view, tiers) ───────
ctdp_add_test(test_space_v072 test_space_v072.cpp)

# ── Aggregate target ─────────────────────────────────────────────────────────
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS core_test solver_test algebra_test
            graph_step1_test graph_step2_test graph_step3_test
            graph_step5_test graph_step6_test graph_step7_test
            graph_hardening_test
            test_symmetric_graph
            test_graph_hardening_v2
            test_graph_coloring
            test_weighted_view
            test_transforms
            test_merge_rules
            test_segmentation_dp
            test_graph_umbrella
            test_bipartite_matching
            test_instantiation
            test_solver_concepts test_per_element_space
            test_interval_split_space test_other_spaces
            test_cost_models test_per_element_argmin
            test_interval_dp test_exhaustive_search
            test_select_and_run test_memo
            test_space_parameterised
            test_candidate_traversal
            test_heterogeneous_space
            test_constrained_search
            test_constraints
            test_space_v072
    COMMENT "Running all CTDP tests"
)