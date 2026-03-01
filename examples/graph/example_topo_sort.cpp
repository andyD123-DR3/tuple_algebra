// examples/graph/example_topo_sort.cpp — Topological Sort: Build System Ordering
//
// A build system has targets with dependencies.  Topological sort finds
// a legal build order where every dependency is built before its dependents.
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_topo_sort examples/graph/example_topo_sort.cpp

#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/topological_sort.h>
#include <iostream>

using namespace ctdp::graph;

// =========================================================================
// Compile-time: prove a build order exists and is correct
// =========================================================================

// Build targets:
//   0: libcore     (no deps)
//   1: libmath     → libcore
//   2: libio       → libcore
//   3: libgraph    → libcore, libmath
//   4: app         → libgraph, libio
//   5: tests       → app, libmath
constexpr auto make_build_dag() {
    graph_builder<8, 16> b;
    auto core  = b.add_node();   // 0
    auto math  = b.add_node();   // 1
    auto io    = b.add_node();   // 2
    auto graph = b.add_node();   // 3
    auto app   = b.add_node();   // 4
    auto tests = b.add_node();   // 5

    b.add_edge(core, math);     // math depends on core
    b.add_edge(core, io);       // io depends on core
    b.add_edge(core, graph);    // graph depends on core
    b.add_edge(math, graph);    // graph depends on math
    b.add_edge(graph, app);     // app depends on graph
    b.add_edge(io, app);        // app depends on io
    b.add_edge(app, tests);     // tests depends on app
    b.add_edge(math, tests);    // tests depends on math

    return b.finalise();
}

constexpr auto build_dag = make_build_dag();
constexpr auto topo = topological_sort(build_dag);

// Compile-time proofs
static_assert(topo.is_dag,               "build graph must be acyclic");
static_assert(topo.order[0] == node_id{0}, "libcore must be built first");

// Prove: every dependency is built before its dependent
static_assert([]() {
    // For each edge u→v, position(u) < position(v) in topo order
    auto pos = [&](uint16_t node) -> std::size_t {
        for (std::size_t i = 0; i < 6; ++i)
            if (topo.order[i].value == node) return i;
        return 999;
    };
    return pos(0) < pos(1) &&   // core before math
           pos(0) < pos(2) &&   // core before io
           pos(0) < pos(3) &&   // core before graph
           pos(1) < pos(3) &&   // math before graph
           pos(3) < pos(4) &&   // graph before app
           pos(2) < pos(4) &&   // io before app
           pos(4) < pos(5);     // app before tests
}(), "topological order must respect all dependencies");

// =========================================================================
// Runtime: print the build order
// =========================================================================

int main() {
    const char* names[] = {"libcore", "libmath", "libio",
                           "libgraph", "app", "tests"};

    std::cout << "=== Build System: Topological Sort ===\n\n";
    std::cout << "Dependencies:\n";
    for (std::size_t u = 0; u < build_dag.node_count(); ++u) {
        auto uid = node_id{static_cast<uint16_t>(u)};
        for (auto v : build_dag.out_neighbors(uid))
            std::cout << "  " << names[u] << " → " << names[v.value] << "\n";
    }

    std::cout << "\nBuild order (topological sort):\n";
    for (std::size_t i = 0; i < build_dag.node_count(); ++i) {
        std::cout << "  " << (i + 1) << ". " << names[topo.order[i].value] << "\n";
    }

    std::cout << "\nIs DAG: " << (topo.is_dag ? "yes" : "no") << "\n";
    std::cout << "\nAll static_assert proofs passed at compile time.\n";
    return 0;
}
