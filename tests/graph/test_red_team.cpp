// Step 17: Red team / adversarial testing
// Boundary conditions, pathological inputs, numeric edge cases

#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/runtime_graph.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/array_helpers.h>
#include <ctdp/graph/connected_components.h>
#include <ctdp/graph/scc.h>
#include <ctdp/graph/topological_sort.h>
#include <ctdp/graph/graph_coloring.h>
#include <ctdp/graph/shortest_path.h>
#include <ctdp/graph/min_cut.h>
#include <ctdp/core/rt_array.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ctdp::graph;

// =========================================================================
// 1. Empty graph (0 nodes) — all algorithms handle gracefully
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    auto g = b.finalise();
    return g.node_count() == 0 && g.edge_count() == 0 && g.empty();
}());

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    auto g = b.finalise();
    auto topo = topological_sort(g);
    return topo.is_dag && topo.order.size() == 0;
}());

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    auto g = b.finalise();
    auto cc = connected_components(g);
    return cc.component_count == 0;
}());

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    auto g = b.finalise();
    auto sc = scc(g);
    return sc.component_count == 0;
}());

// =========================================================================
// 2. Singleton graph (1 node, no edges)
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    (void)b.add_node();
    auto g = b.finalise();
    return g.node_count() == 1 && g.edge_count() == 0;
}());

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    (void)b.add_node();
    auto g = b.finalise();
    auto topo = topological_sort(g);
    return topo.is_dag && topo.order.size() == 1 && topo.order[0] == node_id{0};
}());

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    (void)b.add_node();
    auto g = b.finalise();
    auto cc = connected_components(g);
    return cc.component_count == 1 && cc.component_of[0] == 0;
}());

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    (void)b.add_node();
    auto g = b.finalise();
    auto sp = dijkstra(g, node_id{0}, [](node_id, node_id) -> double { return 1.0; });
    return sp.dist[0] == 0.0 && sp.verified;
}());

// =========================================================================
// 3. Max-capacity graph: fill MaxV and MaxE completely
// =========================================================================

static_assert([]() {
    // 4 nodes, 6 edges = complete DAG: 0->1, 0->2, 0->3, 1->2, 1->3, 2->3
    graph_builder<cap_from<4, 6>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2); b.add_edge(n0, n3);
    b.add_edge(n1, n2); b.add_edge(n1, n3);
    b.add_edge(n2, n3);
    auto g = b.finalise();
    return g.node_count() == 4 && g.edge_count() == 6
        && g.node_capacity() == 4 && g.edge_capacity() == 6;
}());

static_assert([]() {
    graph_builder<cap_from<4, 6>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2); b.add_edge(n0, n3);
    b.add_edge(n1, n2); b.add_edge(n1, n3); b.add_edge(n2, n3);
    auto g = b.finalise();
    auto topo = topological_sort(g);
    return topo.is_dag && topo.order.size() == 4;
}());

// =========================================================================
// 4. Self-edge rejection (builder canonicalisation)
// =========================================================================

void test_self_edge_rejection() {
    // Runtime builder should filter self-edges
    runtime_graph_builder<cap_from<4, 8>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    b.add_edge(n0, n0);  // self-edge
    b.add_edge(n0, n1);
    b.add_edge(n1, n1);  // self-edge
    auto g = b.finalise();
    assert(g.edge_count() == 1);  // only 0->1 survives
    std::cout << "  self-edge rejection: PASS\n";
}

// =========================================================================
// 5. Duplicate edge dedup
// =========================================================================

void test_duplicate_dedup() {
    runtime_graph_builder<cap_from<4, 32>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    for (int i = 0; i < 10; ++i) {
        b.add_edge(n0, n1);
    }
    auto g = b.finalise();
    assert(g.edge_count() == 1);  // all dupes removed
    assert(g.out_degree(node_id{0}) == 1);
    std::cout << "  duplicate dedup: PASS\n";
}

// =========================================================================
// 6. Complete DAG (all forward edges): V=8
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<8, 64>> b;
    for (int i = 0; i < 8; ++i) (void)b.add_node();
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j)
            b.add_edge(node_id{static_cast<uint16_t>(i)},
                       node_id{static_cast<uint16_t>(j)});
    auto g = b.finalise();

    auto topo = topological_sort(g);
    if (!topo.is_dag) return false;
    if (topo.order.size() != 8) return false;
    // Must be 0,1,2,...,7 (only valid topo order for complete DAG)
    for (int i = 0; i < 8; ++i)
        if (topo.order[i] != node_id{static_cast<uint16_t>(i)})
            return false;

    auto cc = connected_components(g);
    if (cc.component_count != 1) return false;

    auto sc = scc(g);
    if (sc.component_count != 8) return false;  // DAG: each node is own SCC

    return true;
}());

// =========================================================================
// 7. Long chain: V=16
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<16, 16>> b;
    for (int i = 0; i < 16; ++i) (void)b.add_node();
    for (int i = 0; i < 15; ++i)
        b.add_edge(node_id{static_cast<uint16_t>(i)},
                   node_id{static_cast<uint16_t>(i + 1)});
    auto g = b.finalise();

    auto topo = topological_sort(g);
    if (!topo.is_dag || topo.order.size() != 16) return false;

    auto sp = dijkstra(g, node_id{0},
        [](node_id, node_id) -> double { return 1.0; });
    // dist[15] should be 15.0
    return sp.dist[15] == 15.0 && sp.verified;
}());

// =========================================================================
// 8. Many disconnected components: 8 isolated nodes
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<8, 8>> b;
    for (int i = 0; i < 8; ++i) (void)b.add_node();
    // No edges
    auto g = b.finalise();

    auto cc = connected_components(g);
    if (cc.component_count != 8) return false;
    // Each node in its own component
    for (int i = 0; i < 8; ++i)
        if (cc.component_of[i] != static_cast<uint16_t>(i))
            return false;

    auto sp = dijkstra(g, node_id{0},
        [](node_id, node_id) -> double { return 1.0; });
    // All unreachable except source
    for (int i = 1; i < 8; ++i)
        if (sp.dist[i] != std::numeric_limits<double>::infinity())
            return false;
    return sp.dist[0] == 0.0 && sp.verified;
}());

// =========================================================================
// 9. Strong cycle: all nodes in one SCC
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<8, 8>> b;
    for (int i = 0; i < 8; ++i) (void)b.add_node();
    for (int i = 0; i < 8; ++i)
        b.add_edge(node_id{static_cast<uint16_t>(i)},
                   node_id{static_cast<uint16_t>((i + 1) % 8)});
    auto g = b.finalise();

    auto sc = scc(g);
    if (sc.component_count != 1) return false;

    auto topo = topological_sort(g);
    return !topo.is_dag;  // cycle detected
}());

// =========================================================================
// 10. Dijkstra: zero-weight edges
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<4, 4>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2);
    auto g = b.finalise();

    auto r = dijkstra(g, node_id{0},
        [](node_id, node_id) -> double { return 0.0; });
    return r.dist[0] == 0.0 && r.dist[1] == 0.0 && r.dist[2] == 0.0
        && r.verified;
}());

// =========================================================================
// 11. Dijkstra: mixed weights, verify shortest path chosen
// =========================================================================

static_assert([]() {
    // Diamond: 0->1(10), 0->2(1), 2->1(1) — shortest 0->1 is via 0->2->1 = 2
    graph_builder<cap_from<4, 4>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2); b.add_edge(n2, n1);
    auto g = b.finalise();

    auto r = dijkstra(g, node_id{0}, [](node_id u, node_id v) -> double {
        if (u.value == 0 && v.value == 1) return 10.0;
        if (u.value == 0 && v.value == 2) return 1.0;
        if (u.value == 2 && v.value == 1) return 1.0;
        return 1e9;
    });
    return r.dist[1] == 2.0 && r.pred[1] == 2 && r.pred[2] == 0 && r.verified;
}());

// =========================================================================
// 12. Stoer-Wagner: complete graph K4, unit weights
// =========================================================================

static_assert([]() {
    symmetric_graph_builder<cap_from<4, 8>> b;
    for (int i = 0; i < 4; ++i) (void)b.add_node();
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            b.add_edge(node_id{static_cast<uint16_t>(i)},
                       node_id{static_cast<uint16_t>(j)});
    auto g = b.finalise();

    auto r = stoer_wagner(g, [](node_id, node_id) -> double { return 1.0; });
    // K4 min cut: isolate one vertex, cut 3 edges = weight 3
    return r.cut_weight == 3.0 && r.verified;
}());

// =========================================================================
// 13. Stoer-Wagner: bridge graph (min cut = 1 edge)
// =========================================================================

static_assert([]() {
    // Two triangles connected by a single bridge edge
    symmetric_graph_builder<cap_from<8, 16>> b;
    auto a = b.add_node(); auto bb = b.add_node(); auto c = b.add_node();
    auto d = b.add_node(); auto e = b.add_node(); auto f = b.add_node();
    // Triangle 1
    b.add_edge(a, bb); b.add_edge(bb, c); b.add_edge(a, c);
    // Triangle 2
    b.add_edge(d, e); b.add_edge(e, f); b.add_edge(d, f);
    // Bridge
    b.add_edge(c, d);  // weight 1
    auto g = b.finalise();

    auto r = stoer_wagner(g, [](node_id, node_id) -> double { return 1.0; });
    return r.cut_weight == 1.0 && r.verified;
}());

// =========================================================================
// 14. Graph coloring: complete graph K5 needs 5 colours
// =========================================================================

static_assert([]() {
    symmetric_graph_builder<cap_from<8, 16>> b;
    for (int i = 0; i < 5; ++i) (void)b.add_node();
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            b.add_edge(node_id{static_cast<uint16_t>(i)},
                       node_id{static_cast<uint16_t>(j)});
    auto g = b.finalise();
    auto cr = graph_coloring(g);
    return cr.color_count == 5 && cr.verified;
}());

// =========================================================================
// 15. Graph coloring: bipartite graph K_{3,3} needs exactly 2 colours
// =========================================================================

static_assert([]() {
    symmetric_graph_builder<cap_from<8, 16>> b;
    for (int i = 0; i < 6; ++i) (void)b.add_node();
    // K_{3,3}: {0,1,2} <-> {3,4,5}
    for (int i = 0; i < 3; ++i)
        for (int j = 3; j < 6; ++j)
            b.add_edge(node_id{static_cast<uint16_t>(i)},
                       node_id{static_cast<uint16_t>(j)});
    auto g = b.finalise();
    auto cr = graph_coloring(g);
    return cr.color_count == 2 && cr.verified;
}());

// =========================================================================
// 16. rt_array boundary conditions
// =========================================================================

void test_rt_array_boundaries() {
    // Size 0
    ctdp::rt_array<int> a0(0);
    assert(a0.size() == 0);
    assert(a0.empty());

    // Size 1
    ctdp::rt_array<int> a1(1, 42);
    assert(a1.size() == 1);
    assert(a1[0] == 42);
    assert(!a1.empty());

    // Large
    ctdp::rt_array<double> big(1000, 3.14);
    assert(big.size() == 1000);
    assert(big[999] == 3.14);
    big[500] = 2.71;
    assert(big[500] == 2.71);

    // Iterator range
    int count = 0;
    for ([[maybe_unused]] auto& x : big) ++count;
    assert(count == 1000);

    std::cout << "  rt_array boundaries: PASS\n";
}

// =========================================================================
// 17. Runtime builder error handling
// =========================================================================

void test_builder_errors() {
    // Exceeding MaxV
    bool caught_max_v = false;
    try {
        runtime_graph_builder<cap_from<2, 4>> b;
        (void)b.add_node(); (void)b.add_node();
        (void)b.add_node();  // should throw
    } catch (std::length_error const&) {
        caught_max_v = true;
    }
    assert(caught_max_v);

    // Edge to nonexistent source
    bool caught_src = false;
    try {
        runtime_graph_builder<cap_from<4, 4>> b;
        (void)b.add_node();
        b.add_edge(node_id{5}, node_id{0});  // node 5 doesn't exist
    } catch (std::out_of_range const&) {
        caught_src = true;
    }
    assert(caught_src);

    // Edge to nonexistent target
    bool caught_tgt = false;
    try {
        runtime_graph_builder<cap_from<4, 4>> b;
        (void)b.add_node();
        b.add_edge(node_id{0}, node_id{5});
    } catch (std::out_of_range const&) {
        caught_tgt = true;
    }
    assert(caught_tgt);

    // Edge on empty graph
    bool caught_empty = false;
    try {
        runtime_graph_builder<cap_from<4, 4>> b;
        b.add_edge(node_id{0}, node_id{1});
    } catch (std::logic_error const&) {
        caught_empty = true;
    }
    assert(caught_empty);

    std::cout << "  builder error handling: PASS\n";
}

// =========================================================================
// 18. Result type sizes match graph traits
// =========================================================================

static_assert([]() {
    using G8 = constexpr_graph<cap_from<8, 16>>;
    using G32 = constexpr_graph<cap_from<32, 64>>;
    using RG8 = runtime_graph<cap_from<8, 16>>;

    // Same MaxV => same result types
    bool same_cc = sizeof(components_result<8>) == sizeof(components_result<8>);
    bool diff_cc = sizeof(components_result<8>) != sizeof(components_result<32>);

    // Traits max_nodes matches
    bool g8_nodes = graph_traits<G8>::max_nodes == 8;
    bool g32_nodes = graph_traits<G32>::max_nodes == 32;
    bool rg8_nodes = graph_traits<RG8>::max_nodes == 8;

    return same_cc && diff_cc && g8_nodes && g32_nodes && rg8_nodes;
}());

// =========================================================================
// 19. node_nil_v sentinel never collides with valid nodes
// =========================================================================

static_assert([]() {
    using G = constexpr_graph<cap_from<8, 16>>;
    constexpr auto nil = node_nil_v<G>;
    // nil is 0xFFFF = 65535, MaxV is 8 — no collision possible
    return nil > 8;
}());

// =========================================================================
// 20. Star graph: hub with many spokes
// =========================================================================

static_assert([]() {
    graph_builder<cap_from<16, 16>> b;
    for (int i = 0; i < 16; ++i) (void)b.add_node();
    // Hub = node 0, spokes = 1..15
    for (int i = 1; i < 16; ++i)
        b.add_edge(node_id{0}, node_id{static_cast<uint16_t>(i)});
    auto g = b.finalise();

    auto cc = connected_components(g);
    if (cc.component_count != 1) return false;

    auto topo = topological_sort(g);
    if (!topo.is_dag) return false;
    if (topo.order[0] != node_id{0}) return false;  // hub first

    auto sp = dijkstra(g, node_id{0},
        [](node_id, node_id) -> double { return 1.0; });
    // All spokes at distance 1
    for (int i = 1; i < 16; ++i)
        if (sp.dist[i] != 1.0) return false;
    return sp.verified;
}());

// =========================================================================
// main
// =========================================================================

int main() {
    std::cout << "Step 17: Red Team Tests\n";
    std::cout << "=======================\n";
    std::cout << "  All static_asserts passed (20 constexpr adversarial tests).\n";

    test_self_edge_rejection();
    test_duplicate_dedup();
    test_rt_array_boundaries();
    test_builder_errors();

    std::cout << "\nStep 17: ALL PASS\n";
    return 0;
}
