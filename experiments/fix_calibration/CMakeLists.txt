# experiments/fix_calibration/CMakeLists.txt
#
# Build system for the FIX parser optimisation experiments.
# Shared headers, unit tests, and Program A.
#
# Options:
#   CTDP_FIX_EXPERIMENT_MOCK   Build with mock measurement (no RDTSC needed)
#
# Per-program targets (no global phase switch):
#   fix_experiment_A          Phase 1 (discovery) or mock
#   fix_experiment_A_verify   Phase 2 (verification, requires generated header)
#
# Usage (mock / CI):
#   cmake -B build -DCTDP_BUILD_EXPERIMENTS=ON -DCTDP_FIX_EXPERIMENT_MOCK=ON
#   cmake --build build --target fix_experiment_A
#
# Usage (real, two-phase):
#   cmake -B build -DCTDP_BUILD_EXPERIMENTS=ON
#   cmake --build build --target fix_experiment_A && ./build/.../fix_experiment_A
#   cmake --build build --target fix_experiment_A_verify && ./build/.../fix_experiment_A_verify

option(CTDP_FIX_EXPERIMENT_MOCK
    "Use deterministic mock measurement instead of real RDTSC" OFF)

# -- Common interface library ------------------------------------------
add_library(ctdp_fix_experiment_common INTERFACE)
target_include_directories(ctdp_fix_experiment_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ctdp_fix_experiment_common INTERFACE ctdp_all)

# Canonical source-tree path for generated headers.
# Programs use this to write Phase 1 output to the correct location.
target_compile_definitions(ctdp_fix_experiment_common INTERFACE
    CTDP_EXPERIMENT_GENERATED_DIR="${CMAKE_CURRENT_SOURCE_DIR}/generated")

if(CTDP_FIX_EXPERIMENT_MOCK)
    target_compile_definitions(ctdp_fix_experiment_common INTERFACE
        CTDP_FIX_EXPERIMENT_MOCK)
endif()

# -- Program A: p50 baseline ------------------------------------------
# Discovery target (mock or Phase 1 depending on CTDP_FIX_EXPERIMENT_MOCK)
add_executable(fix_experiment_A programs/A_p50_baseline.cpp)
target_link_libraries(fix_experiment_A PRIVATE ctdp_fix_experiment_common)

# Verification target (Phase 2 — only if generated header exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/A_candidates.h")
    add_executable(fix_experiment_A_verify programs/A_p50_baseline.cpp)
    target_link_libraries(fix_experiment_A_verify PRIVATE ctdp_fix_experiment_common)
    target_compile_definitions(fix_experiment_A_verify PRIVATE CTDP_PHASE2)
endif()

# -- Tests (only when CTDP_BUILD_TESTS is ON) -------------------------
if(CTDP_BUILD_TESTS)
    # GTest may already be available from the parent tests/ dir.
    # If not (e.g. experiments built standalone), fetch it here.
    if(NOT TARGET GTest::gtest_main)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    include(GoogleTest)

    # Unit tests: feature extractors
    add_executable(fix_experiment_test_extractors tests/test_extractors.cpp)
    target_link_libraries(fix_experiment_test_extractors PRIVATE
        ctdp_fix_experiment_common GTest::gtest GTest::gtest_main)
    gtest_discover_tests(fix_experiment_test_extractors)

    # Unit tests: beam search
    add_executable(fix_experiment_test_beam_search tests/test_beam_search.cpp)
    target_link_libraries(fix_experiment_test_beam_search PRIVATE
        ctdp_fix_experiment_common GTest::gtest GTest::gtest_main)
    gtest_discover_tests(fix_experiment_test_beam_search)

    # Unit tests: output layer (report_types, output_table, json_report)
    add_executable(fix_experiment_test_output tests/test_output.cpp)
    target_link_libraries(fix_experiment_test_output PRIVATE
        ctdp_fix_experiment_common GTest::gtest GTest::gtest_main)
    gtest_discover_tests(fix_experiment_test_output)

    # Integration tests: full mock pipeline (experiment_runner + mock_measurer)
    add_executable(fix_experiment_test_mock_pipeline tests/test_mock_pipeline.cpp)
    target_link_libraries(fix_experiment_test_mock_pipeline PRIVATE
        ctdp_fix_experiment_common GTest::gtest GTest::gtest_main)
    gtest_discover_tests(fix_experiment_test_mock_pipeline)

    # Unit tests: compiled_measurer dispatch table
    add_executable(fix_experiment_test_compiled_measurer tests/test_compiled_measurer.cpp)
    target_link_libraries(fix_experiment_test_compiled_measurer PRIVATE
        ctdp_fix_experiment_common GTest::gtest GTest::gtest_main)
    gtest_discover_tests(fix_experiment_test_compiled_measurer)
endif()
