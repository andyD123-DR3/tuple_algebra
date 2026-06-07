# ── Graph Library Examples ────────────────────────────────────────────────────
# Each example demonstrates a graph algorithm at compile time with static_assert
# proofs, then prints results at runtime.

# Topological sort — build system ordering
add_executable(example_topo_sort example_topo_sort.cpp)
target_link_libraries(example_topo_sort PRIVATE ctdp_all)

# Connected components + SCC — module dependency analysis
add_executable(example_components_scc example_components_scc.cpp)
target_link_libraries(example_components_scc PRIVATE ctdp_all)

# Graph colouring — exam scheduling
add_executable(example_coloring example_coloring.cpp)
target_link_libraries(example_coloring PRIVATE ctdp_all)

# Dijkstra shortest path — data-centre routing
add_executable(example_shortest_path example_shortest_path.cpp)
target_link_libraries(example_shortest_path PRIVATE ctdp_all)

# Stoer-Wagner min cut — pipeline bottleneck
add_executable(example_min_cut example_min_cut.cpp)
target_link_libraries(example_min_cut PRIVATE ctdp_all)

# Hopcroft-Karp bipartite matching — task assignment
add_executable(example_bipartite example_bipartite.cpp)
target_link_libraries(example_bipartite PRIVATE ctdp_all)

# Reverse Cuthill-McKee — sparse matrix bandwidth reduction
add_executable(example_rcm example_rcm.cpp)
target_link_libraries(example_rcm PRIVATE ctdp_all)

# Reverse Cuthill-McKee — locality benchmark for an SpMV-like path stencil
add_executable(example_rcm_spmv_benchmark example_rcm_spmv_benchmark.cpp)
target_link_libraries(example_rcm_spmv_benchmark PRIVATE ctdp_all)

# Compiler options for graph examples
foreach(target
    example_topo_sort example_components_scc example_coloring
    example_shortest_path example_min_cut example_bipartite example_rcm
    example_rcm_spmv_benchmark)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE -O2
            -fconstexpr-ops-limit=100000000)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(${target} PRIVATE -O2
            -fconstexpr-steps=100000000)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /O2
            /constexpr:steps100000000)
    endif()
endforeach()

# The benchmark target is intended to measure locality, so build it with stronger optimisation.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(example_rcm_spmv_benchmark PRIVATE -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(example_rcm_spmv_benchmark PRIVATE /O2)
endif()
