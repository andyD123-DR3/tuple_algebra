// Tests for Dijkstra shortest path and Stoer-Wagner min cut
// Steps 10-11

#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/shortest_path.h>
#include <ctdp/graph/min_cut.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace ctdp::graph;

// =========================================================================
// Test graph builders
// =========================================================================

// Directed diamond: 0->1(w=1), 0->2(w=4), 1->3(w=2), 2->3(w=1)
// Shortest 0->3: 0->1->3 = 3
constexpr auto make_weighted_diamond() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2);
    b.add_edge(n1, n3); b.add_edge(n2, n3);
    return b.finalise();
}

// Weight function for the diamond
struct diamond_weight {
    constexpr double operator()(node_id u, node_id v) const {
        if (u.value == 0 && v.value == 1) return 1.0;
        if (u.value == 0 && v.value == 2) return 4.0;
        if (u.value == 1 && v.value == 3) return 2.0;
        if (u.value == 2 && v.value == 3) return 1.0;
        return 1e9;  // no edge
    }
};

// Linear chain: 0->1->2->3, each weight 1.0
constexpr auto make_chain4() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n2, n3);
    return b.finalise();
}

// Undirected triangle: edges {0,1}, {1,2}, {0,2}, all weight 1
constexpr auto make_sym_triangle() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n0, n2);
    return b.finalise();
}

// Undirected 4-node with weighted edges for min-cut:
//   0--1 (w=2), 0--2 (w=3), 1--3 (w=3), 2--3 (w=2), 1--2 (w=1)
// Min cut separates {0,2} from {1,3} or {0} from {1,2,3} etc.
// Let's compute: cut {0} vs {1,2,3} = w(0,1)+w(0,2) = 2+3 = 5
// cut {0,2} vs {1,3} = w(0,1)+w(2,3)+w(1,2) = 2+2+1 = 5 ... hmm
// Actually for a simpler test: bar graph 0-1-2-3
constexpr auto make_sym_bar() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1);  // weight 3
    b.add_edge(n1, n2);  // weight 1  <-- min cut
    b.add_edge(n2, n3);  // weight 3
    return b.finalise();
}

struct bar_weight {
    constexpr double operator()(node_id u, node_id v) const {
        auto a = u.value < v.value ? u.value : v.value;
        auto b = u.value < v.value ? v.value : u.value;
        if (a == 0 && b == 1) return 3.0;
        if (a == 1 && b == 2) return 1.0;
        if (a == 2 && b == 3) return 3.0;
        return 0.0;
    }
};

// Unit weight function
struct unit_weight {
    constexpr double operator()(node_id, node_id) const { return 1.0; }
};

// =========================================================================
// Dijkstra tests — constexpr
// =========================================================================

// Diamond: shortest 0->3 = 3.0 (via 0->1->3)
static_assert([]() {
    auto g = make_weighted_diamond();
    auto r = dijkstra(g, node_id{0}, diamond_weight{});
    return r.dist[0] == 0.0
        && r.dist[1] == 1.0
        && r.dist[3] == 3.0
        && r.verified;
}());

// Diamond: predecessor chain 3->1->0
static_assert([]() {
    auto g = make_weighted_diamond();
    auto r = dijkstra(g, node_id{0}, diamond_weight{});
    return r.pred[3] == 1 && r.pred[1] == 0;
}());

// Chain: distances are 0, 1, 2, 3
static_assert([]() {
    auto g = make_chain4();
    auto r = dijkstra(g, node_id{0}, unit_weight{});
    return r.dist[0] == 0.0
        && r.dist[1] == 1.0
        && r.dist[2] == 2.0
        && r.dist[3] == 3.0
        && r.verified;
}());

// Source in middle: chain, source = 1
// 1->2 = 1, 1->2->3 = 2, 0 is unreachable (directed)
static_assert([]() {
    auto g = make_chain4();
    auto r = dijkstra(g, node_id{1}, unit_weight{});
    return r.dist[1] == 0.0
        && r.dist[2] == 1.0
        && r.dist[3] == 2.0
        && r.dist[0] == std::numeric_limits<double>::infinity()
        && r.verified;
}());

// Single node graph
static_assert([]() {
    graph_builder<4, 4> b;
    (void)b.add_node();
    auto g = b.finalise();
    auto r = dijkstra(g, node_id{0}, unit_weight{});
    return r.dist[0] == 0.0 && r.verified;
}());

// Factory function
static_assert([]() {
    auto g = make_chain4();
    auto r = make_shortest_path_result(g);
    return r.dist[0] == std::numeric_limits<double>::infinity()
        && r.pred[0] == 0xFFFF;
}());

// =========================================================================
// Stoer-Wagner tests — constexpr
// =========================================================================

// Triangle with unit weights: min cut = 2 (cut one vertex off)
static_assert([]() {
    auto g = make_sym_triangle();
    auto r = stoer_wagner(g, unit_weight{});
    return r.cut_weight == 2.0 && r.verified;
}());

// Bar graph 0-1-2-3 with weights 3,1,3: min cut = 1 (cut edge 1-2)
static_assert([]() {
    auto g = make_sym_bar();
    auto r = stoer_wagner(g, bar_weight{});
    return r.cut_weight == 1.0 && r.verified;
}());

// Bar partition: nodes on opposite sides of the 1-2 edge
static_assert([]() {
    auto g = make_sym_bar();
    auto r = stoer_wagner(g, bar_weight{});
    // Partition should separate {0,1} from {2,3} or vice versa
    return r.partition[0] == r.partition[1]
        && r.partition[2] == r.partition[3]
        && r.partition[0] != r.partition[2];
}());

// Two-node graph: min cut is just the edge weight
static_assert([]() {
    symmetric_graph_builder<4, 4> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    b.add_edge(n0, n1);
    auto g = b.finalise();
    auto wt = [](node_id, node_id) -> double { return 5.0; };
    auto r = stoer_wagner(g, wt);
    return r.cut_weight == 5.0 && r.verified;
}());

// Single node: cut = 0
static_assert([]() {
    symmetric_graph_builder<4, 4> b;
    (void)b.add_node();
    auto g = b.finalise();
    auto r = stoer_wagner(g, unit_weight{});
    return r.cut_weight == 0.0 && r.verified;
}());

// Factory function
static_assert([]() {
    auto g = make_sym_triangle();
    auto r = make_min_cut_result(g);
    return r.cut_weight == std::numeric_limits<double>::infinity()
        && !r.verified;
}());

// =========================================================================
// Runtime tests
// =========================================================================

void test_dijkstra_runtime() {
    auto g = make_weighted_diamond();
    auto r = dijkstra(g, node_id{0}, diamond_weight{});
    assert(r.dist[0] == 0.0);
    assert(r.dist[1] == 1.0);
    assert(r.dist[3] == 3.0);
    assert(r.pred[3] == 1);
    assert(r.pred[1] == 0);
    assert(r.verified);
    std::cout << "  dijkstra runtime: PASS\n";
}

void test_stoer_wagner_runtime() {
    auto g = make_sym_bar();
    auto r = stoer_wagner(g, bar_weight{});
    assert(r.cut_weight == 1.0);
    assert(r.verified);
    assert(r.partition[0] == r.partition[1]);
    assert(r.partition[2] == r.partition[3]);
    assert(r.partition[0] != r.partition[2]);
    std::cout << "  stoer_wagner runtime: PASS\n";
}

int main() {
    std::cout << "Steps 10-11: New Algorithms\n";
    std::cout << "===========================\n";
    std::cout << "  All static_asserts passed (constexpr verified).\n";

    test_dijkstra_runtime();
    test_stoer_wagner_runtime();

    std::cout << "\nSteps 10-11: PASS\n";
    return 0;
}
