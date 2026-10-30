cmake_minimum_required(VERSION 3.20)
project(ctdp_fusion_boundary_redteam LANGUAGES CXX)

# Isolated C++20 reference experiment. No GoogleTest fetch, hardware timing,
# target-specific backend or production-DN dependency is required.
enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(CTDP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(ctdp_fusion_boundary_selftest
    "${CTDP_ROOT}/tests/solver/test_fusion_boundary_probe.cpp")
add_executable(ctdp_fusion_boundary_corpus
    "${CMAKE_CURRENT_LIST_DIR}/report.cpp")

foreach(target ctdp_fusion_boundary_selftest ctdp_fusion_boundary_corpus)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_include_directories(${target} PRIVATE "${CTDP_ROOT}/include")
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
endforeach()

add_test(NAME ctdp_fusion_boundary_selftest
    COMMAND ctdp_fusion_boundary_selftest)
