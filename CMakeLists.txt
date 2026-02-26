cmake_minimum_required(VERSION 3.20...3.30)

project(ctdp
    VERSION     0.6.0
    DESCRIPTION "Compile-Time Dynamic Programming Framework"
    LANGUAGES   CXX
)

# ── C++20 Standard ───────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ──────────────────────────────────────────────────────────────────
option(CTDP_BUILD_TESTS    "Build unit tests"              ON)
option(CTDP_BUILD_EXAMPLES "Build example programs"        ON)
option(CTDP_ASAN           "Enable AddressSanitizer"      OFF)

# ── Compiler-specific settings ───────────────────────────────────────────────
# Raise constexpr evaluation limits (compile-time DP needs this)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -fconstexpr-ops-limit=100000000
        -fconstexpr-loop-limit=1000000
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -fconstexpr-steps=100000000
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(
        /W4 /permissive-
        /constexpr:steps10000000
        /Zc:__cplusplus           # Report correct __cplusplus value
    )
endif()

if(CTDP_ASAN AND NOT MSVC)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ── Header-only libraries ────────────────────────────────────────────────────
# Core: foundational constexpr containers and concepts
add_library(ctdp_core INTERFACE)
target_include_directories(ctdp_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ctdp/core>
    $<INSTALL_INTERFACE:include/ctdp/core>
)

# Solver: DP engine (plan, algorithms, spaces, cost_models, memo)
add_library(ctdp_solver INTERFACE)
target_include_directories(ctdp_solver INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ctdp/solver>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ctdp/solver>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ctdp_solver INTERFACE ctdp_core)

# Algebra: tuple algebra for variadic reductions (P3666R2)
# Uses ct_dp::algebra namespace and #include "ct_dp/algebra/..." paths
add_library(ctdp_algebra INTERFACE)
target_include_directories(ctdp_algebra INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Graph: compile-time graph algorithms and analysis
add_library(ctdp_graph INTERFACE)
target_include_directories(ctdp_graph INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ctdp/graph>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ctdp/graph>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ctdp_graph INTERFACE ctdp_core)

# Engine: bridge between graph and solver tiers
add_library(ctdp_engine INTERFACE)
target_include_directories(ctdp_engine INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ctdp/engine/bridge>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ctdp/engine/bridge>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ctdp_engine INTERFACE ctdp_graph ctdp_solver)

# Domain: application-specific code (SpMV, FIX, etc.)
add_library(ctdp_domain INTERFACE)
target_include_directories(ctdp_domain INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ctdp/domain>
    $<INSTALL_INTERFACE:include/ctdp/domain>
)
target_link_libraries(ctdp_domain INTERFACE ctdp_engine ctdp_graph ctdp_solver)

# Space: descriptor-based search spaces (v0.7.2)
add_library(ctdp_space INTERFACE)
target_include_directories(ctdp_space INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ctdp_space INTERFACE ctdp_solver)

# Convenience target: all libraries
add_library(ctdp_all INTERFACE)
target_link_libraries(ctdp_all INTERFACE ctdp_core ctdp_solver ctdp_algebra ctdp_graph ctdp_engine ctdp_domain ctdp_space)

# ── Subdirectories ───────────────────────────────────────────────────────────
if(CTDP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CTDP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Install ──────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(DIRECTORY include/ctdp/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ctdp)
install(DIRECTORY include/ct_dp/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ct_dp)
install(TARGETS ctdp_core ctdp_solver ctdp_algebra ctdp_graph ctdp_engine ctdp_domain ctdp_space ctdp_all
    EXPORT ctdpTargets)
install(EXPORT ctdpTargets
    NAMESPACE ctdp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ctdp)
