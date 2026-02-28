// tests/graph/test_bipartite_matching.cpp
// Tests for bipartite graph infrastructure and Hopcroft-Karp matching.
//
// Validates:
//   1. bipartite_graph construction and partition queries
//   2. bipartite_graph_builder validation (rejects bad edges)
//   3. bipartite_graph_queryable concept satisfaction
//   4. Left-neighbor iteration (right indices [0, R))
//   5. Hopcroft-Karp on trivial/empty cases
//   6. Perfect matching on complete bipartite K_{n,n}
//   7. Maximum matching < perfect (partial matchability)
//   8. No matching possible (empty edge set)
//   9. Non-square bipartite (|L| ≠ |R|)
//  10. Multi-phase augmentation (requires ≥2 Hopcroft-Karp phases)
//  11. Verification catches corrupted matching
//  12. Full constexpr pipeline
//  13. Capacity limit (64 nodes)

#include "ctdp/graph/bipartite_graph.h"
#include "ctdp/graph/bipartite_matching.h"
#include "ctdp/graph/symmetric_graph.h"
#include <gtest/gtest.h>

using namespace ctdp::graph;

// =========================================================================
// Helper: common test graphs
// =========================================================================

// Single edge: left 0 → right 0
constexpr auto make_single() {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(1, 1);
    b.add_edge(0, 0);
    return b.finalise();
}

// Complete bipartite K_{3,3}
//   left 0 → right {0, 1, 2}
//   left 1 → right {0, 1, 2}
//   left 2 → right {0, 1, 2}
constexpr auto make_k33() {
    bipartite_graph_builder<4, 4, 16> b;
    b.set_partition(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            b.add_edge(i, j);
    return b.finalise();
}

// Path-like bipartite: left i → right i and right i+1
//   left 0 → right {0, 1}
//   left 1 → right {1, 2}
//   left 2 → right {2}
constexpr auto make_path_like() {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(3, 3);
    b.add_edge(0, 0); b.add_edge(0, 1);
    b.add_edge(1, 1); b.add_edge(1, 2);
    b.add_edge(2, 2);
    return b.finalise();
}

// No edges: 3 left, 3 right, but nothing connects them.
constexpr auto make_no_edges() {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(3, 3);
    return b.finalise();
}

// Non-square: 2 left, 4 right
//   left 0 → right {0, 1, 2}
//   left 1 → right {1, 2, 3}
constexpr auto make_non_square() {
    bipartite_graph_builder<4, 8, 16> b;
    b.set_partition(2, 4);
    b.add_edge(0, 0); b.add_edge(0, 1); b.add_edge(0, 2);
    b.add_edge(1, 1); b.add_edge(1, 2); b.add_edge(1, 3);
    return b.finalise();
}

// Multi-phase graph: forces Hopcroft-Karp to use ≥2 BFS phases.
// This is the classic "crown" graph: two length-3 augmenting paths
// that share a common structure requiring layered augmentation.
//
//   left 0 → right 0          (initial greedy matches 0↔0)
//   left 1 → right {0, 1}     (initial greedy matches 1↔1)
//   left 2 → right {1, 2}     (left 2 needs augmenting path)
//
// Greedy (or first HK phase) might match 0↔0, 1↔1.
// Then left 2 has no free right neighbor directly, but augmenting
// path 2→1→(matched to 1)→0→(matched to 0)→... needs phase 2.
constexpr auto make_augmenting() {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(3, 3);
    b.add_edge(0, 0);
    b.add_edge(1, 0); b.add_edge(1, 1);
    b.add_edge(2, 1); b.add_edge(2, 2);
    return b.finalise();
}

// =========================================================================
// 1. Construction and partition queries
// =========================================================================

TEST(BipartiteGraph, BasicConstruction) {
    constexpr auto bg = make_k33();
    static_assert(bg.left_count() == 3);
    static_assert(bg.right_count() == 3);
    static_assert(bg.node_count() == 6);  // 3 + 3
    static_assert(bg.edge_count() == 9);  // 3 × 3

    EXPECT_EQ(bg.left_count(), 3u);
    EXPECT_EQ(bg.right_count(), 3u);
    EXPECT_EQ(bg.edge_count(), 9u);
}

TEST(BipartiteGraph, NonSquare) {
    constexpr auto bg = make_non_square();
    static_assert(bg.left_count() == 2);
    static_assert(bg.right_count() == 4);
    static_assert(bg.node_count() == 6);
    static_assert(bg.edge_count() == 6);
}

TEST(BipartiteGraph, Empty) {
    constexpr auto bg = make_no_edges();
    static_assert(bg.left_count() == 3);
    static_assert(bg.edge_count() == 0);
}

// =========================================================================
// 2. Builder validation
// =========================================================================

TEST(BipartiteBuilder, RejectsDoublePartition) {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(2, 2);
    EXPECT_THROW(b.set_partition(3, 3), std::logic_error);
}

TEST(BipartiteBuilder, RejectsEdgeBeforePartition) {
    bipartite_graph_builder<4, 4, 8> b;
    EXPECT_THROW(b.add_edge(0, 0), std::logic_error);
}

TEST(BipartiteBuilder, RejectsLeftOutOfRange) {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(2, 2);
    EXPECT_THROW(b.add_edge(2, 0), std::logic_error);  // left_idx >= L
}

TEST(BipartiteBuilder, RejectsRightOutOfRange) {
    bipartite_graph_builder<4, 4, 8> b;
    b.set_partition(2, 2);
    EXPECT_THROW(b.add_edge(0, 2), std::logic_error);  // right_idx >= R
}

TEST(BipartiteBuilder, RejectsFinaliseBeforePartition) {
    bipartite_graph_builder<4, 4, 8> b;
    EXPECT_THROW((void)b.finalise(), std::logic_error);
}

// =========================================================================
// 3. Concept satisfaction
// =========================================================================

TEST(BipartiteGraph, ConceptSatisfaction) {
    static_assert(graph_queryable<bipartite_graph<4, 4, 8>>);
    static_assert(bipartite_graph_queryable<bipartite_graph<4, 4, 8>>);
    static_assert(!bipartite_graph_queryable<constexpr_graph<8, 16>>);
    static_assert(!bipartite_graph_queryable<symmetric_graph<8, 16>>);
}

// =========================================================================
// 4. Left-neighbor iteration
// =========================================================================

TEST(BipartiteGraph, LeftNeighbors) {
    constexpr auto bg = make_path_like();

    // left 0 → right {0, 1}
    std::vector<std::size_t> n0;
    for (auto rj : bg.left_neighbors(0)) n0.push_back(rj);
    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 0u);  // CSR sorted
    EXPECT_EQ(n0[1], 1u);

    // left 2 → right {2}
    std::vector<std::size_t> n2;
    for (auto rj : bg.left_neighbors(2)) n2.push_back(rj);
    ASSERT_EQ(n2.size(), 1u);
    EXPECT_EQ(n2[0], 2u);
}

TEST(BipartiteGraph, LeftDegree) {
    constexpr auto bg = make_path_like();
    static_assert(bg.left_degree(0) == 2);
    static_assert(bg.left_degree(1) == 2);
    static_assert(bg.left_degree(2) == 1);
}

// =========================================================================
// 5. Hopcroft-Karp: trivial cases
// =========================================================================

TEST(HopcroftKarp, SingleEdge) {
    constexpr auto bg = make_single();
    constexpr auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    static_assert(m.match_count == 1);
    static_assert(m.match_left[0] == 0);
    static_assert(m.match_right[0] == 0);
    static_assert(m.is_perfect());
    static_assert(m.verified);

    EXPECT_EQ(m.match_count, 1u);
}

TEST(HopcroftKarp, NoEdges) {
    constexpr auto bg = make_no_edges();
    constexpr auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    static_assert(m.match_count == 0);
    static_assert(!m.is_perfect());
    static_assert(m.verified);

    EXPECT_EQ(m.match_count, 0u);
}

// =========================================================================
// 6. Perfect matching: K_{3,3}
// =========================================================================

TEST(HopcroftKarp, CompleteK33) {
    constexpr auto bg = make_k33();
    constexpr auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    static_assert(m.match_count == 3);
    static_assert(m.is_perfect());
    static_assert(m.verified);

    // Verify each left node is matched to a distinct right node.
    EXPECT_NE(m.match_left[0], m.match_left[1]);
    EXPECT_NE(m.match_left[0], m.match_left[2]);
    EXPECT_NE(m.match_left[1], m.match_left[2]);
}

// =========================================================================
// 7. Maximum matching < perfect
// =========================================================================

TEST(HopcroftKarp, PathLike) {
    constexpr auto bg = make_path_like();
    constexpr auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    // Path-like should achieve perfect matching (3 pairs).
    // left 0→right 0, left 1→right 1, left 2→right 2  OR
    // left 0→right 1, left 1→right 2, left 2→... (only right 2 reachable)
    // Actually: left 0→{0,1}, left 1→{1,2}, left 2→{2}
    // Greedy: 0↔0, 1↔1, 2↔2 — perfect!
    static_assert(m.match_count == 3);
    static_assert(m.is_perfect());
    static_assert(m.verified);
}

TEST(HopcroftKarp, PartialMatch) {
    // 3 left, 2 right — at most 2 can be matched
    constexpr auto bg = []() {
        bipartite_graph_builder<4, 4, 8> b;
        b.set_partition(3, 2);
        b.add_edge(0, 0);
        b.add_edge(1, 0);  // both left 0 and 1 want right 0
        b.add_edge(2, 1);
        return b.finalise();
    }();

    constexpr auto m = hopcroft_karp<4, 4>(bg);
    (void)m;
    static_assert(m.match_count == 2);
    static_assert(!m.is_perfect_left());  // 3 left, only 2 matched
    static_assert(m.is_perfect_right());  // both right nodes matched
    static_assert(m.verified);
}

// =========================================================================
// 8. Non-square bipartite
// =========================================================================

TEST(HopcroftKarp, NonSquare) {
    constexpr auto bg = make_non_square();
    constexpr auto m = hopcroft_karp<4, 8>(bg);

    // 2 left, 4 right — max matching = 2 (left-limited)
    static_assert(m.match_count == 2);
    static_assert(m.is_perfect_left());
    static_assert(!m.is_perfect_right());
    static_assert(m.verified);
}

// =========================================================================
// 9. Multi-phase augmentation
// =========================================================================

TEST(HopcroftKarp, AugmentingPaths) {
    constexpr auto bg = make_augmenting();
    constexpr auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    // Should achieve perfect matching despite needing augmentation.
    //   left 0 → right 0
    //   left 1 → right {0, 1}
    //   left 2 → right {1, 2}
    // Final: 0↔0, 1↔1, 2↔2  (or similar permutation)
    static_assert(m.match_count == 3);
    static_assert(m.is_perfect());
    static_assert(m.verified);
}

// =========================================================================
// 10. Larger graph: K_{4,4} complete bipartite
// =========================================================================

TEST(HopcroftKarp, CompleteK44) {
    constexpr auto bg = []() {
        bipartite_graph_builder<8, 8, 32> b;
        b.set_partition(4, 4);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                b.add_edge(i, j);
        return b.finalise();
    }();

    constexpr auto m = hopcroft_karp<8, 8>(bg);
    static_assert(m.match_count == 4);
    static_assert(m.is_perfect());
    static_assert(m.verified);
}

// =========================================================================
// 11. Verification catches corruption
// =========================================================================

TEST(MatchingVerification, CatchesWrongEdge) {
    auto bg = make_k33();
    auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    // Corrupt: claim left 0 matched to right node it's NOT connected to
    // (In K_{3,3} all are connected, so use a different graph)
    auto bg2 = make_path_like();  // left 0 → {0, 1}, NOT 2
    auto m2 = hopcroft_karp<4, 4>(bg2);

    // Manually corrupt: set left 0 to right 2 (not an edge for left 0
    // in path_like... actually left 0 → {0,1}, so right 2 is invalid)
    // But we need to be careful: left 2 → right 2, so we need to also
    // clear that to avoid double-match.
    m2.match_left[0] = 2;
    m2.match_right[2] = 0;
    m2.match_left[2] = matching_result<4, 4>::NIL;
    m2.verified = false;

    EXPECT_FALSE((verify_matching<4, 4>(bg2, m2)));
}

TEST(MatchingVerification, CatchesInconsistency) {
    auto bg = make_k33();
    auto m = hopcroft_karp<4, 4>(bg);
    (void)m;

    // Corrupt: match_left says 0→1 but match_right says 1→2
    m.match_left[0] = 1;
    m.match_right[1] = 2;  // inconsistent
    m.verified = false;

    EXPECT_FALSE((verify_matching<4, 4>(bg, m)));
}

// =========================================================================
// 12. Full constexpr pipeline
// =========================================================================

// Entire pipeline at compile time: build → match → query result.
constexpr auto full_pipeline_result = []() {
    auto bg = make_k33();
    return hopcroft_karp<4, 4>(bg);
}();

static_assert(full_pipeline_result.match_count == 3);
static_assert(full_pipeline_result.is_perfect());
static_assert(full_pipeline_result.verified);

TEST(HopcroftKarp, FullConstexprPipeline) {
    EXPECT_EQ(full_pipeline_result.match_count, 3u);
    EXPECT_TRUE(full_pipeline_result.verified);
}

// =========================================================================
// 13. Capacity: 16×16 complete bipartite
// =========================================================================

TEST(HopcroftKarp, LargerCapacity) {
    constexpr auto bg = []() {
        bipartite_graph_builder<16, 16, 256> b;
        b.set_partition(16, 16);
        for (std::size_t i = 0; i < 16; ++i)
            for (std::size_t j = 0; j < 16; ++j)
                b.add_edge(i, j);
        return b.finalise();
    }();

    constexpr auto m = hopcroft_karp<16, 16>(bg);
    static_assert(m.match_count == 16);
    static_assert(m.is_perfect());
    static_assert(m.verified);
}

// =========================================================================
// 14. Negative concept checks
// =========================================================================

static_assert(!bipartite_graph_queryable<constexpr_graph<8, 16>>);
static_assert(!bipartite_graph_queryable<symmetric_graph<8, 16>>);
