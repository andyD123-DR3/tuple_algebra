// tests/graph/test_rcm.cpp — Reverse Cuthill-McKee tests
//
// Tests: constexpr proofs (static_assert), runtime verification, edge cases,
// bandwidth reduction on known graph families, disconnected graphs, and
// round-trip consistency.

#include <ctdp/graph/rcm.h>
#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_traits.h>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <array>

using namespace ctdp::graph;

// =========================================================================
// Test graph builders
// =========================================================================

// Single node, no edges.
constexpr auto make_single() {
    symmetric_graph_builder<4, 8> b;
    b.add_node();
    return b.finalise();
}

// Two nodes, one edge.
constexpr auto make_pair() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    b.add_edge(n0, n1);
    return b.finalise();
}

// Triangle: 0-1, 1-2, 0-2.  Bandwidth = 2 (cannot improve).
constexpr auto make_triangle() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1); b.add_edge(n1, n2); b.add_edge(n0, n2);
    return b.finalise();
}

// Path graph: 0-1-2-3-4 (already optimal bandwidth = 1).
constexpr auto make_path5() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 5; ++i) b.add_node();
    for (int i = 0; i < 4; ++i)
        b.add_edge(node_id{static_cast<uint16_t>(i)},
                   node_id{static_cast<uint16_t>(i + 1)});
    return b.finalise();
}

// Star graph: node 0 connected to nodes 1,2,3,4.
// Identity bandwidth = 4 (edge 0-4).  Optimal bandwidth = 2.
constexpr auto make_star5() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 5; ++i) b.add_node();
    for (int i = 1; i < 5; ++i)
        b.add_edge(node_id{0}, node_id{static_cast<uint16_t>(i)});
    return b.finalise();
}

// Poorly-ordered path: nodes 0,1,2,3,4 but edges connect
// 0-4, 4-2, 2-3, 3-1 (path order is 0-4-2-3-1).
// Identity bandwidth = 4.  RCM should find bandwidth = 1.
constexpr auto make_scrambled_path() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 5; ++i) b.add_node();
    b.add_edge(node_id{0}, node_id{4});
    b.add_edge(node_id{4}, node_id{2});
    b.add_edge(node_id{2}, node_id{3});
    b.add_edge(node_id{3}, node_id{1});
    return b.finalise();
}

// Ring: 0-1-2-3-4-5-0.  Identity bandwidth = 5.  Optimal = 3.
constexpr auto make_ring6() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 6; ++i) b.add_node();
    for (int i = 0; i < 6; ++i)
        b.add_edge(node_id{static_cast<uint16_t>(i)},
                   node_id{static_cast<uint16_t>((i + 1) % 6)});
    return b.finalise();
}

// Two disconnected triangles: {0,1,2} and {3,4,5}.
constexpr auto make_two_triangles() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 6; ++i) b.add_node();
    // Triangle A
    b.add_edge(node_id{0}, node_id{1});
    b.add_edge(node_id{1}, node_id{2});
    b.add_edge(node_id{0}, node_id{2});
    // Triangle B
    b.add_edge(node_id{3}, node_id{4});
    b.add_edge(node_id{4}, node_id{5});
    b.add_edge(node_id{3}, node_id{5});
    return b.finalise();
}

// Grid 3x3 (9 nodes, 12 edges).  Good test for bandwidth reduction.
// Natural labelling (row-major): bandwidth = 3.
// Optimal bandwidth for 3x3 grid = 3 (cannot do better).
constexpr auto make_grid3x3() {
    symmetric_graph_builder<16, 32> b;
    for (int i = 0; i < 9; ++i) b.add_node();
    // Horizontal edges
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            b.add_edge(node_id{static_cast<uint16_t>(r * 3 + c)},
                       node_id{static_cast<uint16_t>(r * 3 + c + 1)});
    // Vertical edges
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            b.add_edge(node_id{static_cast<uint16_t>(r * 3 + c)},
                       node_id{static_cast<uint16_t>((r + 1) * 3 + c)});
    return b.finalise();
}

// Scrambled grid: same 3x3 grid but nodes relabelled badly.
// Node mapping: 0→0, 1→6, 2→3, 3→1, 4→7, 5→4, 6→2, 7→8, 8→5
// This creates high bandwidth under identity ordering.
constexpr auto make_scrambled_grid() {
    symmetric_graph_builder<16, 32> b;
    for (int i = 0; i < 9; ++i) b.add_node();
    // Map original grid node (r,c) → scrambled id
    constexpr int m[] = {0, 6, 3, 1, 7, 4, 2, 8, 5};
    // Horizontal
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c)
            b.add_edge(node_id{static_cast<uint16_t>(m[r * 3 + c])},
                       node_id{static_cast<uint16_t>(m[r * 3 + c + 1])});
    // Vertical
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            b.add_edge(node_id{static_cast<uint16_t>(m[r * 3 + c])},
                       node_id{static_cast<uint16_t>(m[(r + 1) * 3 + c])});
    return b.finalise();
}

// Complete graph K4 (every pair connected).  Bandwidth = 3, optimal = 3.
constexpr auto make_K4() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 4; ++i) b.add_node();
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            b.add_edge(node_id{static_cast<uint16_t>(i)},
                       node_id{static_cast<uint16_t>(j)});
    return b.finalise();
}

// Caterpillar graph: spine 0-1-2-3, leaves hanging off each spine node.
// Spine: 0-1-2-3.  Leaves: 4-0, 5-1, 6-1, 7-2, 8-3, 9-3.
// 10 nodes, 9 edges.
constexpr auto make_caterpillar() {
    symmetric_graph_builder<16, 32> b;
    for (int i = 0; i < 10; ++i) b.add_node();
    // Spine
    b.add_edge(node_id{0}, node_id{1});
    b.add_edge(node_id{1}, node_id{2});
    b.add_edge(node_id{2}, node_id{3});
    // Leaves
    b.add_edge(node_id{4}, node_id{0});
    b.add_edge(node_id{5}, node_id{1});
    b.add_edge(node_id{6}, node_id{1});
    b.add_edge(node_id{7}, node_id{2});
    b.add_edge(node_id{8}, node_id{3});
    b.add_edge(node_id{9}, node_id{3});
    return b.finalise();
}

// =========================================================================
// Constexpr tests (compile-time proofs)
// =========================================================================

// --- Single node ---
static_assert([]() {
    auto g = make_single();
    auto r = rcm(g);
    return r.verified && r.node_count == 1 &&
           r.bandwidth_before == 0 && r.bandwidth_after == 0 &&
           r.permutation[0] == 0 && r.inverse[0] == 0;
}());

// --- Pair ---
static_assert([]() {
    auto g = make_pair();
    auto r = rcm(g);
    return r.verified && r.node_count == 2 &&
           r.bandwidth_after == 1;
}());

// --- Triangle: bandwidth must remain 2 (complete graph on 3 nodes) ---
static_assert([]() {
    auto g = make_triangle();
    auto r = rcm(g);
    return r.verified && r.node_count == 3 &&
           r.bandwidth_after == 2;
}());

// --- Path 5: already optimal, bandwidth = 1 ---
static_assert([]() {
    auto g = make_path5();
    auto r = rcm(g);
    return r.verified && r.node_count == 5 &&
           r.bandwidth_after <= 1;
}());

// --- Scrambled path: RCM must find bandwidth = 1 ---
static_assert([]() {
    auto g = make_scrambled_path();
    auto r = rcm(g);
    return r.verified && r.node_count == 5 &&
           r.bandwidth_before == 4 &&
           r.bandwidth_after == 1;
}());

// --- Star 5: RCM should reduce bandwidth ---
static_assert([]() {
    auto g = make_star5();
    auto r = rcm(g);
    return r.verified && r.node_count == 5 &&
           r.bandwidth_after <= r.bandwidth_before;
}());

// --- K4: bandwidth = 3, cannot improve ---
static_assert([]() {
    auto g = make_K4();
    auto r = rcm(g);
    return r.verified && r.node_count == 4 &&
           r.bandwidth_after == 3;
}());

// --- Disconnected graph: two triangles ---
static_assert([]() {
    auto g = make_two_triangles();
    auto r = rcm(g);
    return r.verified && r.node_count == 6 &&
           r.bandwidth_after <= r.bandwidth_before;
}());

// --- Ring 6: identity bandwidth = 5, RCM should reduce it ---
static_assert([]() {
    auto g = make_ring6();
    auto r = rcm(g);
    return r.verified && r.node_count == 6 &&
           r.bandwidth_after < r.bandwidth_before;
}());

// --- Grid 3x3: RCM should maintain or reduce bandwidth ---
static_assert([]() {
    auto g = make_grid3x3();
    auto r = rcm(g);
    return r.verified && r.node_count == 9 &&
           r.bandwidth_after <= r.bandwidth_before;
}());

// --- Scrambled grid: RCM should significantly reduce bandwidth ---
static_assert([]() {
    auto g = make_scrambled_grid();
    auto r = rcm(g);
    return r.verified && r.node_count == 9 &&
           r.bandwidth_after < r.bandwidth_before;
}());

// --- Caterpillar: RCM should improve locality ---
static_assert([]() {
    auto g = make_caterpillar();
    auto r = rcm(g);
    return r.verified && r.node_count == 10 &&
           r.bandwidth_after <= r.bandwidth_before;
}());

// --- Permutation is a valid bijection ---
static_assert([]() {
    auto g = make_ring6();
    auto r = rcm(g);
    // Check every position appears exactly once
    std::array<bool, 8> seen{};
    for (std::size_t i = 0; i < r.node_count; ++i)
        seen[r.permutation[i]] = true;
    for (std::size_t i = 0; i < r.node_count; ++i)
        if (!seen[i]) return false;
    // Check inverse consistency
    for (std::size_t i = 0; i < r.node_count; ++i)
        if (r.inverse[r.permutation[i]] != i) return false;
    return true;
}());

// =========================================================================
// Runtime tests
// =========================================================================

void test_single() {
    auto g = make_single();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_before == 0);
    assert(r.bandwidth_after == 0);
    std::cout << "  PASS: single node\n";
}

void test_pair() {
    auto g = make_pair();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after == 1);
    std::cout << "  PASS: pair\n";
}

void test_triangle() {
    auto g = make_triangle();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after == 2);
    std::cout << "  PASS: triangle (bandwidth = 2, irreducible)\n";
}

void test_path5() {
    auto g = make_path5();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after <= 1);
    std::cout << "  PASS: path(5), bandwidth_after = " << r.bandwidth_after << "\n";
}

void test_scrambled_path() {
    auto g = make_scrambled_path();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_before == 4);
    assert(r.bandwidth_after == 1);
    std::cout << "  PASS: scrambled path, " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_star5() {
    auto g = make_star5();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after <= r.bandwidth_before);
    std::cout << "  PASS: star(5), " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_ring6() {
    auto g = make_ring6();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after < r.bandwidth_before);
    std::cout << "  PASS: ring(6), " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_two_triangles() {
    auto g = make_two_triangles();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after <= r.bandwidth_before);
    std::cout << "  PASS: two triangles (disconnected), " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_K4() {
    auto g = make_K4();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after == 3);
    std::cout << "  PASS: K4 (complete), bandwidth = 3 (irreducible)\n";
}

void test_grid3x3() {
    auto g = make_grid3x3();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after <= r.bandwidth_before);
    std::cout << "  PASS: grid 3x3, " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_scrambled_grid() {
    auto g = make_scrambled_grid();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after < r.bandwidth_before);
    std::cout << "  PASS: scrambled grid 3x3, " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_caterpillar() {
    auto g = make_caterpillar();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_after <= r.bandwidth_before);
    std::cout << "  PASS: caterpillar(10), " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
}

void test_permutation_bijection() {
    auto g = make_caterpillar();
    auto r = rcm(g);
    assert(r.verified);
    std::array<bool, 16> seen{};
    for (std::size_t i = 0; i < r.node_count; ++i) {
        assert(r.permutation[i] < r.node_count);
        assert(!seen[r.permutation[i]]);
        seen[r.permutation[i]] = true;
    }
    for (std::size_t i = 0; i < r.node_count; ++i) {
        assert(r.inverse[r.permutation[i]] == i);
        assert(r.permutation[r.inverse[i]] == i);
    }
    std::cout << "  PASS: permutation bijection check\n";
}

void test_print_reordering() {
    auto g = make_scrambled_path();
    auto r = rcm(g);
    assert(r.verified);
    std::cout << "  Scrambled path RCM reordering:\n";
    std::cout << "    Bandwidth: " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n";
    std::cout << "    Permutation (old → new): ";
    for (std::size_t i = 0; i < r.node_count; ++i)
        std::cout << i << "→" << r.permutation[i] << " ";
    std::cout << "\n    Inverse (new → old):     ";
    for (std::size_t k = 0; k < r.node_count; ++k)
        std::cout << k << "→" << r.inverse[k] << " ";
    std::cout << "\n";
}

// =========================================================================
// Empty graph (edge case)
// =========================================================================

void test_empty_graph() {
    symmetric_graph_builder<4, 8> b;
    auto g = b.finalise();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.node_count == 0);
    assert(r.bandwidth_before == 0);
    assert(r.bandwidth_after == 0);
    std::cout << "  PASS: empty graph\n";
}

// =========================================================================
// Isolated nodes (no edges)
// =========================================================================

void test_isolated_nodes() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 5; ++i) b.add_node();
    // No edges
    auto g = b.finalise();
    auto r = rcm(g);
    assert(r.verified);
    assert(r.bandwidth_before == 0);
    assert(r.bandwidth_after == 0);
    std::cout << "  PASS: 5 isolated nodes\n";
}

// =========================================================================
// Main
// =========================================================================

int main() {
    std::cout << "=== RCM Constexpr Tests ===\n";
    std::cout << "All static_assert tests passed at compile time.\n\n";

    std::cout << "=== RCM Runtime Tests ===\n";
    test_empty_graph();
    test_single();
    test_pair();
    test_triangle();
    test_path5();
    test_scrambled_path();
    test_star5();
    test_ring6();
    test_two_triangles();
    test_K4();
    test_grid3x3();
    test_scrambled_grid();
    test_caterpillar();
    test_permutation_bijection();
    test_isolated_nodes();
    std::cout << "\n";
    test_print_reordering();

    std::cout << "\n=== All RCM tests passed ===\n";
    return 0;
}
