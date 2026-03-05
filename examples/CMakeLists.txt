# -- FIX Cache-Line Layout ----------------------------------------------------
# Demonstrates consteval subset-DP for cache-line layout optimisation
# of a FIX protocol session struct (13 fields, 6 hot).
add_executable(fix_cacheline fix_cacheline.cpp)
target_link_libraries(fix_cacheline PRIVATE ctdp_all)

# -- FIX Parser p99 Calibration ------------------------------------------------
# Demonstrates p99-aware calibration of template-specialised FIX parsers.
# 241 ET parser configurations measured with 100K samples each.
add_executable(fix_p99_calibration fix_p99_calibration.cpp)
target_link_libraries(fix_p99_calibration PRIVATE ctdp_all)

# -- FIX Parser p99 SVR-guided search ------------------------------------------
# SVR cost model trained on 100 measurements, predicts 10K, verifies top 50.
add_executable(fix_p99_svr_search fix_p99_svr_search.cpp)
target_link_libraries(fix_p99_svr_search PRIVATE ctdp_all)

add_executable(fix_p99_beam_search fix_p99_beam_search.cpp)
target_link_libraries(fix_p99_beam_search PRIVATE ctdp_all)

# -- SpMV Benchmark -----------------------------------------------------------
# Demonstrates compile-time format selection matching runtime performance.
# Build with -O3 for meaningful timings.
add_executable(spmv_benchmark spmv_benchmark.cc)
target_link_libraries(spmv_benchmark PRIVATE ctdp_all)

# -- Variadic Reduce ---------------------------------------------------------
# Demonstrates tuple algebra: 7-lane single-pass statistics (P3666R2).
add_executable(variadic_reduce variadic_reduce.cpp)
target_link_libraries(variadic_reduce PRIVATE ctdp_algebra)

# -- Matrix Chain Demo --------------------------------------------------------
# End-to-end proof: Cormen matrix chain via interval_dp (compile-time + runtime).
add_executable(matrix_chain_demo matrix_chain_demo.cpp)
target_link_libraries(matrix_chain_demo PRIVATE ctdp_all)

# -- Matrix Chain ET (Expression Template Instantiation) ----------------------
# Companion to matrix_chain_demo.  Proves the plan can be instantiated as a
# TYPE: compile-time DP -> nested Mul<L,R> expression template -> zero-overhead
# execution.  Association order locked by type system.  Self-contained (no
# framework headers) for Godbolt portability.
add_executable(matrix_chain_et matrix_chain_et.cpp)

# -- Per-Element Demo ---------------------------------------------------------
# Factored optimisation: additive cost, dynamic constraints, select_and_run.
add_executable(per_element_demo per_element_demo.cpp)
target_link_libraries(per_element_demo PRIVATE ctdp_all)

# Enable optimisations for benchmarks regardless of global build type
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(fix_cacheline PRIVATE -O3 -march=native
        -fconstexpr-ops-limit=300000000)
    target_compile_options(fix_p99_calibration PRIVATE -O2 -march=native)
    target_compile_options(fix_p99_svr_search PRIVATE -O2 -march=native)
    target_compile_options(fix_p99_beam_search PRIVATE -O2 -march=native)
    target_compile_options(spmv_benchmark PRIVATE -O3 -march=native)
    target_compile_options(variadic_reduce PRIVATE -O3 -march=native)
    target_compile_options(matrix_chain_demo PRIVATE -O3 -march=native)
    target_compile_options(matrix_chain_et PRIVATE -O2)
    target_compile_options(per_element_demo PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(fix_cacheline PRIVATE -O3 -march=native
        -fconstexpr-steps=300000000)
    target_compile_options(fix_p99_calibration PRIVATE -O2 -march=native)
    target_compile_options(fix_p99_svr_search PRIVATE -O2 -march=native)
    target_compile_options(fix_p99_beam_search PRIVATE -O2 -march=native)
    target_compile_options(spmv_benchmark PRIVATE -O3 -march=native)
    target_compile_options(variadic_reduce PRIVATE -O3 -march=native)
    target_compile_options(matrix_chain_demo PRIVATE -O3 -march=native)
    target_compile_options(matrix_chain_et PRIVATE -O2)
    target_compile_options(per_element_demo PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(fix_cacheline PRIVATE /O2 /constexpr:steps300000000)
    target_compile_options(fix_p99_calibration PRIVATE /O2)
    target_compile_options(fix_p99_svr_search PRIVATE /O2)
    target_compile_options(fix_p99_beam_search PRIVATE /O2)
    target_compile_options(spmv_benchmark PRIVATE /O2)
    target_compile_options(variadic_reduce PRIVATE /O2)
    target_compile_options(matrix_chain_demo PRIVATE /O2)
    target_compile_options(matrix_chain_et PRIVATE /O2)
    target_compile_options(per_element_demo PRIVATE /O2)
endif()

# -- GEMM Tile Optimisation ---------------------------------------------------
# Demonstrates ML cost model calibration: preprocess -> train -> search.
add_executable(gemm_tile_optimisation gemm_tile_optimisation.cpp)
target_link_libraries(gemm_tile_optimisation PRIVATE ctdp_all)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(gemm_tile_optimisation PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(gemm_tile_optimisation PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(gemm_tile_optimisation PRIVATE /O2)
endif()

# ==============================================================================
# v0.7.2: Standalone + Framework examples
# ==============================================================================

# -- Standalone examples (no framework headers -- pure C++20) ------------------
# These demonstrate the CT-DP pattern (constexpr search -> NTTP dispatch)
# without importing framework types. They need AVX2+FMA for SIMD benchmarks.

add_executable(loop_nest_opt standalone/loop_nest_opt.cpp)
add_executable(dep_break_opt standalone/dep_break_opt.cpp)
add_executable(cache_size_opt standalone/cache_size_opt.cpp)

# -- Framework example (uses ctdp_space) --------------------------------------
# Policy search over cost model weights, wrapping a subset-DP inner solver.
add_executable(cache_line_layout framework/cache_line_layout.cpp)
target_link_libraries(cache_line_layout PRIVATE ctdp_space)

# -- Compiler options for v0.7.2 examples -------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(loop_nest_opt PRIVATE -O2 -mavx2 -mfma
        -Wno-conversion -Wno-sign-conversion)
    target_compile_options(dep_break_opt PRIVATE -O2 -mavx2 -mfma
        -Wno-conversion -Wno-sign-conversion)
    target_compile_options(cache_size_opt PRIVATE -O2
        -Wno-conversion -Wno-sign-conversion)
    target_compile_options(cache_line_layout PRIVATE -O2
        -fconstexpr-ops-limit=500000000
        -Wno-conversion -Wno-sign-conversion)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(loop_nest_opt PRIVATE -O2 -mavx2 -mfma
        -Wno-conversion -Wno-sign-conversion)
    target_compile_options(dep_break_opt PRIVATE -O2 -mavx2 -mfma
        -Wno-conversion -Wno-sign-conversion)
    target_compile_options(cache_size_opt PRIVATE -O2
        -Wno-conversion -Wno-sign-conversion)
    target_compile_options(cache_line_layout PRIVATE -O2
        -fconstexpr-steps=500000000
        -Wno-conversion -Wno-sign-conversion)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(loop_nest_opt PRIVATE /O2 /arch:AVX2)
    target_compile_options(dep_break_opt PRIVATE /O2 /arch:AVX2)
    target_compile_options(cache_size_opt PRIVATE /O2)
    target_compile_options(cache_line_layout PRIVATE /O2
        /constexpr:steps500000000)
endif()

# ==============================================================================
# Calibration examples (Phase 3 / Phase 7)
# ==============================================================================

# calibration_main  -- Phase 2 CLI runner (measure + CSV output)
add_executable(calibration_main calibration_main.cpp)
target_link_libraries(calibration_main PRIVATE ctdp::calibrator)
target_include_directories(calibration_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# calibration_demo  -- Phase 7 end-to-end demo (full pipeline -> wisdom)
add_executable(calibration_demo calibration_demo.cpp)
target_link_libraries(calibration_demo PRIVATE ctdp::calibrator)
target_include_directories(calibration_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# example_memory_regime -- Memory regime characterisation
add_executable(example_memory_regime example_memory_regime.cpp)
target_link_libraries(example_memory_regime PRIVATE ctdp::calibrator)
target_include_directories(example_memory_regime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# example_parser_pipeline -- Parser strategy comparison
add_executable(example_parser_pipeline example_parser_pipeline.cpp)
target_link_libraries(example_parser_pipeline PRIVATE ctdp::calibrator)
target_include_directories(example_parser_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compiler options for calibration examples
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    foreach(tgt calibration_main calibration_demo example_memory_regime example_parser_pipeline)
        target_compile_options(${tgt} PRIVATE -O2 -Wall -Wextra -Wpedantic)
    endforeach()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    foreach(tgt calibration_main calibration_demo example_memory_regime example_parser_pipeline)
        target_compile_options(${tgt} PRIVATE /O2 /W4)
    endforeach()
endif()
