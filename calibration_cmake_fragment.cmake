# ─── Calibration Libraries ───────────────────────────────────────
#
# Add these sections to the root CMakeLists.txt.
#
# ctdp-bench: measurement infrastructure (zero CT-DP knowledge)
# ctdp-calibrator: framework-facing calibration harness

# Header-only library: ctdp-bench
add_library(ctdp_bench INTERFACE)
target_include_directories(ctdp_bench INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Header-only library: ctdp-calibrator (depends on ctdp-bench)
add_library(ctdp_calibrator INTERFACE)
target_include_directories(ctdp_calibrator INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(ctdp_calibrator INTERFACE ctdp_bench)

# Calibration test
if(BUILD_TESTS)
    add_executable(calibration_test tests/calibration_test.cpp)
    target_link_libraries(calibration_test PRIVATE ctdp_calibrator)
    target_compile_options(calibration_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>)
    add_test(NAME calibration_test COMMAND calibration_test)
endif()

# Calibration example CLI
add_executable(calibration_main examples/calibration_main.cpp)
target_link_libraries(calibration_main PRIVATE ctdp_calibrator)
target_include_directories(calibration_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_compile_options(calibration_main PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>)
