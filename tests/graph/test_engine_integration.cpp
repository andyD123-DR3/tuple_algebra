// Step 16: Engine integration test
// Verifies engine bridge types + all algorithms work through engine aliases

#include <ctdp/engine/bridge/graph_types.h>
#include <ctdp/engine/bridge/graph_to_space.h>
#include <ctdp/engine/bridge/graph_to_constraints.h>
#include <ctdp/engine/bridge/coloring_to_groups.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/fuse_group.h>
#include <ctdp/graph/array_helpers.h>

#include <cassert>
#include <iostream>
#include <type_traits>

using namespace ctdp::engine;
using namespace ctdp::graph;

// =========================================================================
// Compile-time checks on engine type aliases
// =========================================================================

static_assert(std::is_same_v<engine_graph, constexpr_graph<64, 256>>);
static_assert(std::is_same_v<engine_sym_graph, symmetric_graph<64, 256>>);
static_assert(std::is_same_v<engine_rt_graph, runtime_graph<64, 256>>);

static_assert(graph_queryable<engine_graph>);
static_assert(sized_graph<engine_graph>);
static_assert(graph_queryable<engine_rt_graph>);
static_assert(sized_graph<engine_rt_graph>);

static_assert(graph_traits<engine_graph>::max_nodes == 64);
static_assert(graph_traits<engine_graph>::max_edges == 256);
static_assert(graph_traits<engine_rt_graph>::max_nodes == 64);
static_assert(graph_traits<engine_rt_graph>::max_edges == 256);
static_assert(!graph_traits<engine_rt_graph>::is_constexpr_storage);
static_assert(graph_traits<engine_graph>::is_constexpr_storage);

// Result type aliases match
static_assert(std::is_same_v<engine_topo_result, topo_result<64>>);
static_assert(std::is_same_v<engine_cc_result, components_result<64>>);
static_assert(std::is_same_v<engine_scc_result, scc_result<64>>);
static_assert(std::is_same_v<engine_coloring_result, coloring_result<64>>);
static_assert(std::is_same_v<engine_sp_result, shortest_path_result<64>>);
static_assert(std::is_same_v<engine_min_cut_result, min_cut_result<64>>);

// =========================================================================
// Build a constexpr engine pipeline graph
// =========================================================================

// 4-node DAG: read -> compute_a -> compute_b -> write
constexpr auto make_engine_pipeline() {
    engine_graph_builder b;
    auto read = b.add_node();
    auto comp_a = b.add_node();
    auto comp_b = b.add_node();
    auto write = b.add_node();
    b.add_edge(read, comp_a);
    b.add_edge(read, comp_b);
    b.add_edge(comp_a, write);
    b.add_edge(comp_b, write);
    return b.finalise();
}

// Constexpr algorithms on engine-sized graphs
static_assert(topological_sort(make_engine_pipeline()).is_dag);
static_assert(topological_sort(make_engine_pipeline()).order.size() == 4);
static_assert(connected_components(make_engine_pipeline()).component_count == 1);
static_assert(scc(make_engine_pipeline()).component_count == 4);

// Dijkstra on engine graph
static_assert([]() {
    auto g = make_engine_pipeline();
    auto r = dijkstra(g, node_id{0},
        [](node_id, node_id) -> double { return 1.0; });
    return r.dist[0] == 0.0 && r.dist[3] == 2.0 && r.verified;
}());

// =========================================================================
// Build symmetric engine graph for coloring
// =========================================================================

constexpr auto make_engine_conflict() {
    engine_sym_builder b;
    auto a = b.add_node(); auto bb = b.add_node();
    auto c = b.add_node(); auto d = b.add_node();
    b.add_edge(a, bb); b.add_edge(bb, c);
    b.add_edge(c, d);  b.add_edge(a, d);
    return b.finalise();
}

static_assert(graph_coloring(make_engine_conflict()).verified);
static_assert(graph_coloring(make_engine_conflict()).color_count <= 3);

// Coloring -> groups bridge
static_assert([]() {
    auto cr = graph_coloring(make_engine_conflict());
    auto fg = coloring_to_groups(cr);
    return fg.group_count == cr.color_count && fg.is_valid_dag;
}());

// Stoer-Wagner on engine symmetric graph
static_assert([]() {
    auto g = make_engine_conflict();
    auto r = stoer_wagner(g, [](node_id, node_id) -> double { return 1.0; });
    return r.verified && r.cut_weight >= 1.0;
}());

// =========================================================================
// Runtime engine graph mirrors constexpr
// =========================================================================

void test_runtime_engine_graph() {
    engine_rt_builder b;
    auto read = b.add_node();
    auto comp_a = b.add_node();
    auto comp_b = b.add_node();
    auto write = b.add_node();
    b.add_edge(read, comp_a);
    b.add_edge(read, comp_b);
    b.add_edge(comp_a, write);
    b.add_edge(comp_b, write);
    auto g = b.finalise();

    auto topo = topological_sort(g);
    assert(topo.is_dag);
    assert(topo.order.size() == 4);

    auto cc = connected_components(g);
    assert(cc.component_count == 1);

    auto sc = scc(g);
    assert(sc.component_count == 4);

    auto sp = dijkstra(g, node_id{0},
        [](node_id, node_id) -> double { return 1.0; });
    assert(sp.dist[0] == 0.0);
    assert(sp.dist[3] == 2.0);
    assert(sp.verified);

    // Parity check: constexpr and runtime produce same results
    constexpr auto cg = make_engine_pipeline();
    constexpr auto ct_topo = topological_sort(cg);
    for (std::size_t i = 0; i < 4; ++i) {
        assert(ct_topo.order[i] == topo.order[i]);
    }

    std::cout << "  runtime engine graph: PASS\n";
}

// =========================================================================
// Schedule space bridge
// =========================================================================

void test_schedule_space_bridge() {
    constexpr auto g = make_engine_pipeline();
    constexpr auto kmap = make_uniform_kernel_map<64>(g,
        kernel_info{.tag = kernel_tag{1}, .flops = 100, .bytes_read = 64});

    constexpr auto space = build_schedule_space<64, 256>(g, kmap);
    static_assert(space.is_dag);
    static_assert(space.size() == 4);

    // Descriptors in topo order
    static_assert(space.descriptors[0].id == node_id{0});
    static_assert(space.descriptors[0].topo_rank == 0);

    // Extract dependencies
    constexpr auto deps = extract_dependencies<64, 256>(g, space);
    static_assert(deps.deps.size() == 4);  // 4 edges = 4 deps

    std::cout << "  schedule_space bridge: PASS\n";
}

// =========================================================================
// Array helpers work through engine types
// =========================================================================

void test_engine_array_helpers() {
    auto g = []() {
        engine_rt_builder b;
        (void)b.add_node(); (void)b.add_node(); (void)b.add_node();
        return b.finalise();
    }();

    auto dist = make_node_array<double>(g, 1e9);
    assert(dist.size() == 64);
    assert(dist[0] == 1e9);

    auto edges = make_edge_array<int>(g, -1);
    assert(edges.size() == 256);
    assert(edges[0] == -1);

    std::cout << "  engine array helpers: PASS\n";
}

// =========================================================================
// main
// =========================================================================

int main() {
    std::cout << "Step 16: Engine Integration\n";
    std::cout << "===========================\n";
    std::cout << "  All static_asserts passed (constexpr engine verified).\n";

    test_runtime_engine_graph();
    test_schedule_space_bridge();
    test_engine_array_helpers();

    std::cout << "\nStep 16: ALL PASS\n";
    return 0;
}
