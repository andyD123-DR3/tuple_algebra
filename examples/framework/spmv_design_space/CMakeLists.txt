cmake_minimum_required(VERSION 3.20)
project(spmv_design_space_demonstrator LANGUAGES CXX)

find_package(Threads REQUIRED)

add_library(spmv_design_space INTERFACE)
target_include_directories(spmv_design_space INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(spmv_design_space INTERFACE cxx_std_20)
target_link_libraries(spmv_design_space INTERFACE Threads::Threads)

enable_testing()

add_executable(spmv_design_space_demo spmv_design_space_demo.cpp)
target_link_libraries(spmv_design_space_demo PRIVATE spmv_design_space)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(spmv_design_space_demo PRIVATE -O3 -Wall -Wextra -Wpedantic)
else()
    target_compile_options(spmv_design_space_demo PRIVATE /O2 /W4)
endif()

set(SPMV_DESIGN_SPACE_TESTS
    spmv_design_space_contract_tests
    spmv_design_space_fact_tests
    spmv_design_space_plan_tests
    spmv_design_space_executor_tests
)

add_executable(spmv_design_space_contract_tests tests/test_contracts.cpp)
add_executable(spmv_design_space_fact_tests tests/test_facts.cpp)
add_executable(spmv_design_space_plan_tests tests/test_plans.cpp)
add_executable(spmv_design_space_executor_tests tests/test_executors.cpp)

foreach(test_target IN LISTS SPMV_DESIGN_SPACE_TESTS)
    target_link_libraries(${test_target} PRIVATE spmv_design_space)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${test_target} PRIVATE -O2 -Wall -Wextra -Wpedantic)
    else()
        target_compile_options(${test_target} PRIVATE /O2 /W4)
    endif()
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()


# Convenience targets for integration smoke testing and live demonstrations.
# They run the actual executable rather than a synthetic benchmark harness.
add_custom_target(spmv_design_space_benchmark
    COMMAND $<TARGET_FILE:spmv_design_space_demo> --size 512 --iterations 31 --warmup 3 --threads 4 --grain 2048
    DEPENDS spmv_design_space_demo
    COMMENT "Running SpMV design-space benchmark on a 512 x 512 stencil problem"
    VERBATIM)

add_custom_target(spmv_design_space_sweep
    COMMAND $<TARGET_FILE:spmv_design_space_demo> --sweep --iterations 11 --warmup 2 --threads 4 --grain 2048
    DEPENDS spmv_design_space_demo
    COMMENT "Running SpMV design-space benchmark sweep"
    VERBATIM)
