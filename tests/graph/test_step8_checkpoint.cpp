// Step 8 Checkpoint: Retrofitted algorithms + constexpr preserved
#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/array_helpers.h>
#include <ctdp/graph/connected_components.h>
#include <ctdp/graph/scc.h>
#include <ctdp/graph/topological_sort.h>
#include <ctdp/graph/graph_coloring.h>
#include <ctdp/graph/property_map.h>
#include <ctdp/graph/implicit_graph.h>
// Engine-level headers that call retrofitted algorithms
#include <ctdp/graph/fuse_group.h>
#include <ctdp/graph/coarsen.h>
#include <ctdp/graph/fusion_legal.h>
#include <ctdp/graph/from_stencil.h>
#include <ctdp/graph/from_pipeline.h>

#include <cassert>
#include <iostream>

using namespace ctdp::graph;

// =========================================================================
// Test graphs
// =========================================================================

constexpr auto make_diamond() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2);
    b.add_edge(n1, n3); b.add_edge(n2, n3);
    return b.finalise();
}

constexpr auto make_cycle3() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n2, n0);
    return b.finalise();
}

constexpr auto make_disconnected() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n2, n3);
    return b.finalise();
}

constexpr auto make_sym_triangle() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n0, n2);
    return b.finalise();
}

constexpr auto make_sym_bipartite() {
    symmetric_graph_builder<8, 16> b;
    auto a0 = b.add_node(); auto a1 = b.add_node();
    auto b0 = b.add_node(); auto b1 = b.add_node();
    b.add_edge(a0, b0); b.add_edge(a0, b1);
    b.add_edge(a1, b0); b.add_edge(a1, b1);
    return b.finalise();
}

// =========================================================================
// 1. Topological sort — constexpr
// =========================================================================

static_assert(topological_sort(make_diamond()).is_dag);
static_assert(topological_sort(make_diamond()).order.size() == 4);
static_assert(topological_sort(make_diamond()).order[0] == node_id{0});

// Cycle detection
static_assert(!topological_sort(make_cycle3()).is_dag);

// =========================================================================
// 2. Connected components — constexpr
// =========================================================================

static_assert(connected_components(make_diamond()).component_count == 1);
static_assert(connected_components(make_disconnected()).component_count == 2);

// Verify same-component property
static_assert([]() {
    auto cc = connected_components(make_disconnected());
    return cc.component_of[0] == cc.component_of[1]
        && cc.component_of[2] == cc.component_of[3]
        && cc.component_of[0] != cc.component_of[2];
}());

// =========================================================================
// 3. SCC — constexpr
// =========================================================================

// DAG: each node is its own SCC
static_assert(scc(make_diamond()).component_count == 4);

// Cycle: all in one SCC
static_assert(scc(make_cycle3()).component_count == 1);

// =========================================================================
// 4. Graph coloring — constexpr, new API (no explicit MaxV)
// =========================================================================

static_assert(graph_coloring(make_sym_triangle()).verified);
static_assert(graph_coloring(make_sym_triangle()).color_count == 3);

// Bipartite graph: should use exactly 2 colours
static_assert(graph_coloring(make_sym_bipartite()).verified);
static_assert(graph_coloring(make_sym_bipartite()).color_count == 2);

// =========================================================================
// 5. Factory functions work
// =========================================================================

static_assert([]() {
    auto g = make_diamond();
    auto cr = make_components_result(g);
    return cr.component_count == 0;  // default-constructed
}());

static_assert([]() {
    auto g = make_diamond();
    auto sr = make_scc_result(g);
    return sr.component_count == 0;
}());

static_assert([]() {
    auto g = make_diamond();
    auto tr = make_topo_result(g);
    return tr.is_dag && tr.order.size() == 0;
}());

static_assert([]() {
    auto g = make_sym_triangle();
    auto cr = make_coloring_result(g);
    return cr.color_count == 0 && !cr.verified;
}());

// =========================================================================
// 6. Return types use correct MaxV from traits
// =========================================================================

static_assert([]() {
    auto g = make_diamond();  // constexpr_graph<8, 16>
    auto cc = connected_components(g);
    // Result should have array of size 8 (MaxV from graph)
    return sizeof(cc.component_of) == 8 * sizeof(std::uint16_t);
}());

static_assert([]() {
    auto g = make_sym_triangle();  // symmetric_graph<4, 8>
    auto cr = graph_coloring(g);
    // Result should have array of size 4 (MaxV from graph)
    return sizeof(cr.color_of) == 4 * sizeof(std::uint16_t);
}());

// =========================================================================
// 7. node_index_t and node_nil_v work in algorithm context
// =========================================================================

static_assert(std::is_same_v<node_index_t<decltype(make_diamond())>, std::uint16_t>);
static_assert(node_nil_v<decltype(make_diamond())> == 0xFFFF);

// =========================================================================
// Runtime tests
// =========================================================================

void test_runtime() {
    // Topological sort
    auto g = make_diamond();
    auto topo = topological_sort(g);
    assert(topo.is_dag);
    assert(topo.order.size() == 4);

    // Connected components
    auto cc = connected_components(make_disconnected());
    assert(cc.component_count == 2);

    // SCC
    auto sc = scc(make_cycle3());
    assert(sc.component_count == 1);

    // Coloring
    auto col = graph_coloring(make_sym_triangle());
    assert(col.verified);
    assert(col.color_count == 3);

    std::cout << "  runtime algorithms: PASS\n";
}

int main() {
    std::cout << "Step 8 Checkpoint\n";
    std::cout << "=================\n";
    std::cout << "  All static_asserts passed (constexpr preserved).\n";

    test_runtime();

    std::cout << "\nCHECKPOINT 2 (Step 8): PASS\n";
    return 0;
}
