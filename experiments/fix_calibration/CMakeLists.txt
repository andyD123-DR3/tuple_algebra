# experiments/fix_calibration/CMakeLists.txt
#
# Build system for the FIX parser optimisation experiments.
# Currently: shared headers + unit tests for feature extractors.
# Programs A-F will be added in subsequent PRs.
#
# Options:
#   CTDP_FIX_EXPERIMENT_MOCK   Build with mock measurement (no RDTSC needed)
#
# Usage:
#   cmake -B build -DCTDP_BUILD_EXPERIMENTS=ON -DCTDP_BUILD_TESTS=ON
#   cmake --build build
#   ctest --test-dir build -R OnehotExtractor

option(CTDP_FIX_EXPERIMENT_MOCK
    "Use deterministic mock measurement instead of real RDTSC" OFF)

# -- Common interface library ------------------------------------------
add_library(ctdp_fix_experiment_common INTERFACE)
target_include_directories(ctdp_fix_experiment_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/generated)
target_link_libraries(ctdp_fix_experiment_common INTERFACE ctdp_all)

if(CTDP_FIX_EXPERIMENT_MOCK)
    target_compile_definitions(ctdp_fix_experiment_common INTERFACE
        CTDP_FIX_EXPERIMENT_MOCK)
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
endif()
