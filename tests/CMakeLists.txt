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
ctdp_add_test(test_io_lexer core/test_io_lexer.cpp)

# ── Solver library tests (original Half A) ───────────────────────────────
ctdp_add_test(solver_test solver/solver_test.cc)

# ── Solver I/O tests (cost table) ────────────────────────────────────────
ctdp_add_test(test_cost_table_io solver/test_cost_table_io.cpp)

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
ctdp_add_test(test_graph_io graph/test_graph_io.cpp)

# Capacity policies: named tiers, concept, convenience aliases
ctdp_add_test(test_capacity_policy graph/test_capacity_policy.cpp)

# Deduction aliases: result-type deduction, is_symmetric_graph trait
ctdp_add_test(test_deduction_aliases graph/test_deduction_aliases.cpp)

# Graph I/O layering invariants: detail standalone, parse without iostream
ctdp_add_test(test_io_layering graph/test_io_layering.cpp)

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

# test_performance_models uses its own lightweight assert framework, not GTest.
# Register with add_test() directly — gtest_discover_tests() would time out
# because it runs the full binary (SVR training) during discovery.
add_executable(test_performance_models test_performance_models.cpp)
target_link_libraries(test_performance_models PRIVATE ctdp_all)
add_test(NAME test_performance_models COMMAND test_performance_models)

# ── Space layer v0.7.2 tests (descriptor_space, valid_view, tiers) ───────
ctdp_add_test(test_space_v072 test_space_v072.cpp)

# ── Space layer: conditional_dim (Sprint 2, property-driven dimension presence) ─
ctdp_add_test(test_conditional_dim test_conditional_dim.cpp)

# ── Space layer: reduction properties + space generator (Sprint 2, tasks 2.3-2.7) ─
ctdp_add_test(test_reduction_space test_reduction_space.cpp)

# ── Space layer: ordinal ordinate type + fix<I> alias (Sprint 1) ──────────
ctdp_add_test(test_ordinal_fix test_ordinal_fix.cpp)

# ── Space layer: partition ordinate type (Sprint 3) ───────────────────────
ctdp_add_test(test_partition test_partition.cpp)

# ── Space layer: tree_space (Sprint 4, partition-rooted hierarchy) ────────
ctdp_add_test(test_tree_space test_tree_space.cpp)
ctdp_add_test(test_reduction_tree_space test_reduction_tree_space.cpp)

# ── Space layer: counted_view + constraints (Sprint 5) ───────────────────
ctdp_add_test(test_counted_view test_counted_view.cpp)

# ── Integration tests (Sprint 6) ─────────────────────────────────────────
ctdp_add_test(test_integration_fix test_integration_fix.cpp)
ctdp_add_test(test_integration_gemm test_integration_gemm.cpp)
ctdp_add_test(test_integration_reduction test_integration_reduction.cpp)

# ── Space layer: permutation ordinate type (Sprint 7) ────────────────────
ctdp_add_test(test_permutation test_permutation.cpp)

# ── Bench / calibrator tests (percentile, FIX ET parser) ────────────────
# test_percentile fails to compile with GTest<=1.15.2 + GCC13/libstdc++13.
# Root cause: gtest.h compares const char* with std::string_view via operator==,
# which GCC13's C++20 rewritten comparisons make ambiguous.  Not a bug in our code.
# EXCLUDE_FROM_ALL keeps CI green.  Run manually once GTest is updated upstream:
#   cmake --build build --target test_percentile
add_executable(test_percentile bench/test_percentile.cpp)
target_link_libraries(test_percentile PRIVATE ctdp_all GTest::gtest GTest::gtest_main)
target_compile_features(test_percentile PRIVATE cxx_std_20)
set_target_properties(test_percentile PROPERTIES EXCLUDE_FROM_ALL TRUE)
# gtest_discover_tests omitted: EXCLUDE_FROM_ALL targets register as NOT_BUILT in CTest

# test_perf_counter: diagnostic tests for RDTSC / QueryThreadCycleTime /
# perf_counter_group.  Included in CTest so CI catches zero-cycles regressions.
# On Windows Tier 1 tests are skipped automatically (no perf_event_open).
add_executable(test_perf_counter bench/test_perf_counter.cpp)
target_link_libraries(test_perf_counter PRIVATE ctdp_all GTest::gtest GTest::gtest_main)
target_compile_features(test_perf_counter PRIVATE cxx_std_20)
add_test(NAME test_perf_counter COMMAND test_perf_counter)

# ── Aggregate target ─────────────────────────────────────────────────────────
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS core_test test_io_lexer solver_test algebra_test
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
            test_graph_io
            test_io_layering
            test_capacity_policy
            test_deduction_aliases
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
            test_cost_table_io
            test_space_v072
            test_conditional_dim
            test_reduction_space
            test_ordinal_fix
            test_partition
            test_tree_space
            test_reduction_tree_space
            test_counted_view
            test_integration_fix
            test_integration_gemm
            test_integration_reduction
            test_permutation
    COMMENT "Running all CTDP tests"
)

# ── Calibration tests (standalone, no GTest) ─────────────────────────────────
# These use their own lightweight assert-based test infrastructure.
set(CTDP_CALIBRATION_TEST_SOURCES
    calibration_test.cpp
    phase3_test.cpp
    # phase4_integration_test.cpp   # hardware-dependent, fails on CI
    # phase5_test.cpp               # hardware-dependent, fails on CI
    phase6_test.cpp
)

# ── fix/ parser-optimiser tests (standalone, no GTest, no HW) ────────────────
add_executable(test_fix_bench test_fix_bench.cpp)
target_include_directories(test_fix_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
target_compile_features(test_fix_bench PRIVATE cxx_std_20)
add_test(NAME test_fix_bench COMMAND test_fix_bench)

add_executable(test_fix_dp_search test_fix_dp_search.cpp)
target_include_directories(test_fix_dp_search PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(test_fix_dp_search PRIVATE cxx_std_20)
add_test(NAME test_fix_dp_search COMMAND test_fix_dp_search)

add_executable(test_phase2_strategies test_phase2_strategies.cpp)
target_include_directories(test_phase2_strategies PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(test_phase2_strategies PRIVATE cxx_std_20)
add_test(NAME test_phase2_strategies COMMAND test_phase2_strategies)

# Phase 3: calibration driver wiring tests (mock timing — no hardware required)
add_executable(test_phase3_fix_calibration test_phase3_fix_calibration.cpp)
target_include_directories(test_phase3_fix_calibration PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(test_phase3_fix_calibration PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(test_phase3_fix_calibration PRIVATE -O2 -Wall -Wextra)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(test_phase3_fix_calibration PRIVATE /O2 /W4)
endif()
add_test(NAME test_phase3_fix_calibration COMMAND test_phase3_fix_calibration)

foreach(src ${CTDP_CALIBRATION_TEST_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    set(target "ctdp_${name}")
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE ctdp::calibrator)

    # Propagate hardware-timing skip flag to calibration tests
    if(CTDP_SKIP_HW_TIMING_TESTS)
        target_compile_definitions(${target} PRIVATE CTDP_SKIP_HW_TIMING_TESTS)
    endif()

    # Phase 4+ tests need examples/ headers (calibration_scenarios.h)
    if(NOT ${name} STREQUAL "calibration_test" AND
       NOT ${name} STREQUAL "phase3_test")
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(${target} PRIVATE -O2 -Wall -Wextra -Wpedantic)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /O2 /W4)
    endif()

    add_test(NAME ${target} COMMAND ${target})
endforeach()

# Phase 4 and 5 validate calibration profiles against live measurements,
# so they require stable hardware timing (CPU pinning, no noisy neighbours).
# Exclude from default CI with:  ctest -LE hardware
# set_tests_properties(ctdp_phase4_integration_test ctdp_phase5_test
#     PROPERTIES LABELS "hardware"
# )
# -- SpMV framework demo tests -----------------------------------------------
add_subdirectory(spmv)
