// Baseline compilation test — verify all existing graph infrastructure compiles
// and key constexpr operations work.
#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_coloring.h>
#include <ctdp/graph/connected_components.h>
#include <ctdp/graph/scc.h>
#include <ctdp/graph/topological_sort.h>
#include <ctdp/graph/property_map.h>
#include <ctdp/graph/implicit_graph.h>
#include <ctdp/graph/coarsen.h>
#include <ctdp/graph/fuse_group.h>
#include <ctdp/graph/fusion_legal.h>
#include <ctdp/graph/from_stencil.h>

using namespace ctdp::graph;

constexpr auto make_diamond() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2);
    b.add_edge(n1, n3); b.add_edge(n2, n3);
    return b.finalise();
}
static_assert(make_diamond().node_count() == 4);
static_assert(make_diamond().edge_count() == 4);

constexpr auto topo = topological_sort(make_diamond());
static_assert(topo.is_dag);
static_assert(topo.order.size() == 4);

constexpr auto cc = connected_components(make_diamond());
static_assert(cc.component_count == 1);

constexpr auto sc = scc(make_diamond());
static_assert(sc.component_count == 4); // DAG: each node is its own SCC

// Symmetric graph
constexpr auto make_sym_triangle() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n0, n2);
    return b.finalise();
}
static_assert(make_sym_triangle().node_count() == 3);
static_assert(make_sym_triangle().undirected_edge_count() == 3);

constexpr auto col = graph_coloring(make_sym_triangle());
static_assert(col.color_count == 3);
static_assert(col.verified);

int main() {
    // All checks are compile-time. If we get here, everything passed.
    return 0;
}
