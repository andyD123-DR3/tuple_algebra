// Test: graph_traits infrastructure + capacity queries + array helpers
// Steps 1-6 checkpoint test
#include <ctdp/core/rt_array.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/array_helpers.h>
#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_concepts.h>

// Include all existing algorithm headers to verify nothing is broken
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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

using namespace ctdp::graph;

// =========================================================================
// 1. rt_array basic tests (runtime)
// =========================================================================

void test_rt_array() {
    // Default construction
    ctdp::rt_array<int> empty;
    assert(empty.size() == 0);
    assert(empty.empty());

    // Construction with size
    ctdp::rt_array<int> a(10);
    assert(a.size() == 10);
    assert(!a.empty());
    assert(a[0] == 0);  // value-initialised

    // Construction with fill
    ctdp::rt_array<double> b(5, 3.14);
    assert(b.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        assert(b[i] == 3.14);
    }

    // Mutation
    a[3] = 42;
    assert(a[3] == 42);

    // Iteration
    ctdp::rt_array<int> c(3, 7);
    int sum = 0;
    for (auto x : c) sum += x;
    assert(sum == 21);

    // Comparison
    ctdp::rt_array<int> d(3, 7);
    assert(c == d);
    d[0] = 99;
    assert(!(c == d));

    std::cout << "  rt_array: PASS\n";
}

// =========================================================================
// 2. graph_traits type checks (compile-time)
// =========================================================================

// -- constexpr_graph traits --
using CG = constexpr_graph<8, 16>;
using CG_Traits = graph_traits<CG>;

static_assert(CG_Traits::is_constexpr_storage);
static_assert(std::is_same_v<CG_Traits::node_index_type, std::uint16_t>);
static_assert(std::is_same_v<CG_Traits::node_array<int>, std::array<int, 8>>);
static_assert(std::is_same_v<CG_Traits::edge_array<int>, std::array<int, 16>>);

// node_index_t alias
static_assert(std::is_same_v<node_index_t<CG>, std::uint16_t>);

// node_nil_v
static_assert(node_nil_v<CG> == 0xFFFF);

// -- symmetric_graph traits --
using SG = symmetric_graph<8, 16>;
using SG_Traits = graph_traits<SG>;

static_assert(SG_Traits::is_constexpr_storage);
static_assert(std::is_same_v<SG_Traits::node_index_type, std::uint16_t>);
static_assert(std::is_same_v<SG_Traits::node_array<int>, std::array<int, 8>>);
static_assert(std::is_same_v<SG_Traits::edge_array<int>, std::array<int, 32>>); // 2*MaxE

// =========================================================================
// 3. make_node_array / make_edge_array at compile time
// =========================================================================

static_assert([]() {
    constexpr_graph<4, 8> g{};  // empty graph but with capacity 4,8
    auto arr = graph_traits<constexpr_graph<4, 8>>::make_node_array<int>(4, -1);
    return arr[0] == -1 && arr[3] == -1 && arr.size() == 4;
}());

static_assert([]() {
    auto arr = graph_traits<constexpr_graph<4, 8>>::make_edge_array<double>(8);
    return arr.size() == 8 && arr[0] == 0.0;
}());

// =========================================================================
// 4. Capacity queries on graph types
// =========================================================================

// constexpr_graph
static_assert([]() {
    graph_builder<8, 16> b;
    (void)b.add_node(); (void)b.add_node(); (void)b.add_node();
    auto g = b.finalise();
    return g.node_count() == 3 && g.node_capacity() == 8
        && g.edge_count() == 0 && g.edge_capacity() == 16;
}());

// symmetric_graph
static_assert([]() {
    symmetric_graph_builder<4, 8> b;
    (void)b.add_node(); (void)b.add_node();
    auto g = b.finalise();
    return g.node_count() == 2 && g.node_capacity() == 4
        && g.edge_count() == 0 && g.edge_capacity() == 16;  // 2*MaxE
}());

// =========================================================================
// 5. Concept satisfaction checks
// =========================================================================

// constexpr_graph satisfies graph_queryable AND sized_graph
static_assert(graph_queryable<constexpr_graph<8, 16>>);
static_assert(sized_graph<constexpr_graph<8, 16>>);

// symmetric_graph satisfies all three
static_assert(graph_queryable<symmetric_graph<8, 16>>);
static_assert(sized_graph<symmetric_graph<8, 16>>);
static_assert(symmetric_graph_queryable<symmetric_graph<8, 16>>);

// constexpr_graph does NOT satisfy symmetric_graph_queryable
static_assert(!symmetric_graph_queryable<constexpr_graph<8, 16>>);

// implicit_graph does NOT satisfy sized_graph (no edge_count, no capacity)
// (it satisfies graph_queryable only)
using IG = implicit_graph<decltype([](node_id) {
    return ctdp::constexpr_vector<node_id, 1>{};
})>;
static_assert(graph_queryable<IG>);
static_assert(!sized_graph<IG>);

// =========================================================================
// 6. Free helper functions with a real graph
// =========================================================================

static_assert([]() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2);
    auto g = b.finalise();

    auto node_arr = make_node_array<int>(g, -1);
    auto edge_arr = make_edge_array<bool>(g);

    return node_arr[0] == -1 && node_arr[7] == -1  // filled to capacity
        && edge_arr.size() == 16                     // sized to MaxE
        && !edge_arr[0];                             // value-initialised
}());

// =========================================================================
// 7. Existing algorithms still work after concept change
// =========================================================================

constexpr auto make_test_diamond() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n0, n2);
    b.add_edge(n1, n3); b.add_edge(n2, n3);
    return b.finalise();
}

constexpr auto make_test_sym() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n0, n2);
    return b.finalise();
}

// Topological sort
static_assert(topological_sort(make_test_diamond()).is_dag);
static_assert(topological_sort(make_test_diamond()).order.size() == 4);

// Connected components
static_assert(connected_components(make_test_diamond()).component_count == 1);

// SCC
static_assert(scc(make_test_diamond()).component_count == 4);

// Graph coloring
static_assert(graph_coloring(make_test_sym()).verified);
static_assert(graph_coloring(make_test_sym()).color_count == 3);

// Property map
static_assert([]() {
    auto g = make_test_diamond();
    auto pmap = make_uniform_property_map<int, 8>(g, 42);
    return pmap[node_id{0}] == 42 && pmap[node_id{3}] == 42;
}());

// =========================================================================
// 8. Graph traits make_* produce correct types
// =========================================================================

static_assert([]() {
    using T = decltype(graph_traits<constexpr_graph<4, 8>>::make_node_array<int>(4));
    return std::is_same_v<T, std::array<int, 4>>;
}());

static_assert([]() {
    using T = decltype(graph_traits<symmetric_graph<4, 8>>::make_edge_array<int>(16));
    return std::is_same_v<T, std::array<int, 16>>;
}());

// =========================================================================
// Runtime tests
// =========================================================================

void test_traits_runtime() {
    // Build a graph and use free helpers at runtime
    auto g = make_test_diamond();

    auto dist = make_node_array<double>(g, 1e9);
    assert(dist[0] == 1e9);
    assert(dist.size() == 8);  // capacity, not node_count

    auto visited = make_node_array<bool>(g);
    assert(!visited[0]);
    assert(visited.size() == 8);

    // Verify capacity queries
    assert(g.node_capacity() == 8);
    assert(g.edge_capacity() == 16);
    assert(g.node_count() == 4);
    assert(g.edge_count() == 4);

    std::cout << "  traits runtime: PASS\n";
}

void test_symmetric_traits_runtime() {
    auto g = make_test_sym();
    assert(g.node_capacity() == 4);
    assert(g.edge_capacity() == 16);  // 2*8
    assert(g.node_count() == 3);
    assert(g.undirected_edge_count() == 3);

    auto colors = make_node_array<int>(g, -1);
    assert(colors.size() == 4);
    assert(colors[0] == -1);

    std::cout << "  symmetric traits runtime: PASS\n";
}

// =========================================================================
// main
// =========================================================================

int main() {
    std::cout << "Steps 1-6 Checkpoint Tests\n";
    std::cout << "==========================\n";

    test_rt_array();
    test_traits_runtime();
    test_symmetric_traits_runtime();

    std::cout << "\nAll compile-time static_asserts passed.\n";
    std::cout << "All runtime tests passed.\n";
    std::cout << "\nCHECKPOINT 1: PASS\n";
    return 0;
}
