// examples/graph/example_components_scc.cpp — Connected Components & SCC
//
// Analyse a software module dependency graph:
// - Connected components find independent subsystems
// - SCC finds circular dependencies that prevent incremental builds
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_components_scc examples/graph/example_components_scc.cpp

#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/connected_components.h>
#include <ctdp/graph/scc.h>
#include <iostream>

using namespace ctdp::graph;

// =========================================================================
// Module dependency graph
// =========================================================================
//
// Three subsystems with internal circular dependencies:
//
// Subsystem A (UI): modules 0,1,2 form a cycle (0→1→2→0)
// Subsystem B (Backend): modules 3,4 form a cycle (3→4→3)
// Subsystem C (Utils): modules 5,6 are acyclic (5→6)
//
// Cross-subsystem edges: A→B (module 1→3), B→C (module 4→5)
// No edge from C back to A or B, so the condensation DAG is: A→B→C.

constexpr auto make_module_graph() {
    graph_builder<8, 16> b;
    for (int i = 0; i < 7; ++i) (void)b.add_node();

    // Subsystem A: UI cycle
    b.add_edge(node_id{0}, node_id{1});
    b.add_edge(node_id{1}, node_id{2});
    b.add_edge(node_id{2}, node_id{0});   // cycle!

    // Subsystem B: Backend cycle
    b.add_edge(node_id{3}, node_id{4});
    b.add_edge(node_id{4}, node_id{3});   // cycle!

    // Subsystem C: Utils (acyclic)
    b.add_edge(node_id{5}, node_id{6});

    // Cross-subsystem
    b.add_edge(node_id{1}, node_id{3});   // UI → Backend
    b.add_edge(node_id{4}, node_id{5});   // Backend → Utils

    return b.finalise();
}

constexpr auto modules = make_module_graph();

// =========================================================================
// Connected components (weakly): how many independent subsystems?
// =========================================================================

constexpr auto cc = connected_components(modules);

// Everything is reachable (ignoring direction), so 1 component.
static_assert(cc.component_count == 1,
    "all modules are weakly connected through cross-subsystem edges");

// =========================================================================
// SCC: find circular dependencies
// =========================================================================

constexpr auto sccs = scc(modules);

// Three SCCs: {0,1,2}, {3,4}, and {5,6} are NOT all SCCs.
// Module 5→6 is one-way, so 5 and 6 are separate SCCs.
// Total: {0,1,2}, {3,4}, {5}, {6} = 4 SCCs.
static_assert(sccs.component_count == 4,
    "4 SCCs: UI cycle, backend cycle, and 2 singleton utils");

// Modules in a cycle share an SCC.
static_assert(sccs.component_of[0] == sccs.component_of[1]);
static_assert(sccs.component_of[1] == sccs.component_of[2]);
static_assert(sccs.component_of[3] == sccs.component_of[4]);

// Modules 5 and 6 are in different SCCs (no cycle between them).
static_assert(sccs.component_of[5] != sccs.component_of[6]);

// The UI and Backend cycles are separate SCCs.
static_assert(sccs.component_of[0] != sccs.component_of[3]);

// =========================================================================
// Runtime: print analysis
// =========================================================================

int main() {
    const char* names[] = {
        "ui_view", "ui_controller", "ui_model",   // 0, 1, 2
        "api_server", "api_handler",               // 3, 4
        "utils_core", "utils_log"                  // 5, 6
    };

    std::cout << "=== Module Dependency Analysis ===\n\n";

    std::cout << "Dependencies:\n";
    for (std::size_t u = 0; u < modules.node_count(); ++u) {
        auto uid = node_id{static_cast<uint16_t>(u)};
        for (auto v : modules.out_neighbors(uid))
            std::cout << "  " << names[u] << " → " << names[v.value] << "\n";
    }

    std::cout << "\nWeakly connected components: " << cc.component_count << "\n";

    std::cout << "\nStrongly connected components: " << sccs.component_count << "\n";

    // Group modules by SCC
    for (std::size_t c = 0; c < sccs.component_count; ++c) {
        std::cout << "  SCC " << c << ": {";
        bool first = true;
        for (std::size_t i = 0; i < modules.node_count(); ++i) {
            if (sccs.component_of[i] == c) {
                if (!first) std::cout << ", ";
                std::cout << names[i];
                first = false;
            }
        }
        std::cout << "}";
        // Count members to identify cycles
        std::size_t count = 0;
        for (std::size_t i = 0; i < modules.node_count(); ++i)
            if (sccs.component_of[i] == c) ++count;
        if (count > 1)
            std::cout << "  ← CIRCULAR DEPENDENCY (" << count << " modules)";
        std::cout << "\n";
    }

    std::cout << "\nRecommendation: break the " << sccs.component_count
              << " SCCs into DAG structure for incremental builds.\n";
    std::cout << "\nAll static_assert proofs passed at compile time.\n";
    return 0;
}
