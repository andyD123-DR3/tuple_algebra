# ── FIX Cache-Line Layout ────────────────────────────────────────────────────
# Demonstrates consteval subset-DP for cache-line layout optimisation
# of a FIX protocol session struct (13 fields, 6 hot).
add_executable(fix_cacheline fix_cacheline.cpp)
target_link_libraries(fix_cacheline PRIVATE ctdp_all)

# ── SpMV Benchmark ───────────────────────────────────────────────────────────
# Demonstrates compile-time format selection matching runtime performance.
# Build with -O3 for meaningful timings.
add_executable(spmv_benchmark spmv_benchmark.cc)
target_link_libraries(spmv_benchmark PRIVATE ctdp_all)

# ── Variadic Reduce ─────────────────────────────────────────────────────────
# Demonstrates tuple algebra: 7-lane single-pass statistics (P3666R2).
add_executable(variadic_reduce variadic_reduce.cpp)
target_link_libraries(variadic_reduce PRIVATE ctdp_algebra)

# ── Matrix Chain Demo ────────────────────────────────────────────────────────
# End-to-end proof: Cormen matrix chain via interval_dp (compile-time + runtime).
add_executable(matrix_chain_demo matrix_chain_demo.cpp)
target_link_libraries(matrix_chain_demo PRIVATE ctdp_all)

# ── Per-Element Demo ─────────────────────────────────────────────────────────
# Factored optimisation: additive cost, dynamic constraints, select_and_run.
add_executable(per_element_demo per_element_demo.cpp)
target_link_libraries(per_element_demo PRIVATE ctdp_all)

# Enable optimisations for benchmarks regardless of global build type
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(fix_cacheline PRIVATE -O3 -march=native
        -fconstexpr-ops-limit=300000000)
    target_compile_options(spmv_benchmark PRIVATE -O3 -march=native)
    target_compile_options(variadic_reduce PRIVATE -O3 -march=native)
    target_compile_options(matrix_chain_demo PRIVATE -O3 -march=native)
    target_compile_options(per_element_demo PRIVATE -O3 -march=native)
else()
    target_compile_options(fix_cacheline PRIVATE /O2 /constexpr:steps300000000)
    target_compile_options(spmv_benchmark PRIVATE /O2)
    target_compile_options(variadic_reduce PRIVATE /O2)
    target_compile_options(matrix_chain_demo PRIVATE /O2)
    target_compile_options(per_element_demo PRIVATE /O2)
endif()
