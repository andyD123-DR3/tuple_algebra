// graph/test/graph_step5_test.cc - Tests for fusion_legal, coarsen, fuse_group
// Part of the compile-time DP library (C++20)

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "property_map.h"
#include "kernel_info.h"
#include "fusion_legal.h"
#include "coarsen.h"
#include "fuse_group.h"
#include "topological_sort.h"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// Test graph factories
// =============================================================================

constexpr auto make_empty() {
    graph_builder<cap_from<8, 16>> b;
    return b.finalise();
}

constexpr auto make_singleton() {
    graph_builder<cap_from<8, 16>> b;
    [[maybe_unused]] auto n = b.add_node();
    return b.finalise();
}

// 0→1→2→3
constexpr auto make_chain() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

// 0→1, 0→2, 1→3, 2→3
constexpr auto make_diamond() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n1, n3);
    b.add_edge(n2, n3);
    return b.finalise();
}

// 0→1, 2→3 (disconnected)
constexpr auto make_disconnected() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n2, n3);
    return b.finalise();
}

// 0→1, 0→2, 0→3, 0→4
constexpr auto make_star() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    auto n4 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n0, n3);
    b.add_edge(n0, n4);
    return b.finalise();
}

// 0→1, 0→2, 1→2  (triangle DAG)
constexpr auto make_triangle() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n1, n2);
    return b.finalise();
}

// Two sources converging: 0→2, 1→2, 2→3
constexpr auto make_converge() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n2);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

// =============================================================================
// Kernel map helpers
// =============================================================================

inline constexpr kernel_tag tag_A{1};
inline constexpr kernel_tag tag_B{2};

/// All nodes same tag, fusable.
template<std::size_t MaxV, graph_queryable G>
constexpr kernel_map<MaxV> make_uniform_fusable(G const& g, kernel_tag t = tag_A) {
    return make_kernel_map<MaxV>(g, [t](node_id) {
        return kernel_info{.tag = t, .flops = 100, .bytes_read = 400,
                           .bytes_written = 8, .is_fusable = true};
    });
}

/// All nodes same tag, NOT fusable.
template<std::size_t MaxV, graph_queryable G>
constexpr kernel_map<MaxV> make_uniform_nonfusable(G const& g) {
    return make_kernel_map<MaxV>(g, [](node_id) {
        return kernel_info{.tag = tag_A, .flops = 100, .bytes_read = 400,
                           .bytes_written = 8, .is_fusable = false};
    });
}

// =============================================================================
// Compile-time verification: fusion_legal
// =============================================================================

// --- Empty graph: no pairs ---
constexpr auto fl_empty = fusion_pairs<8, 16>(make_empty(),
    kernel_map<8>{});
static_assert(fl_empty.pairs.size() == 0);
static_assert(fl_empty.total_edges == 0);

// --- Singleton: no edges ---
constexpr auto g_single = make_singleton();
constexpr auto km_single = make_uniform_fusable<8>(g_single);
constexpr auto fl_single = fusion_pairs<8, 16>(g_single, km_single);
static_assert(fl_single.pairs.size() == 0);
static_assert(fl_single.total_edges == 0);

// --- Chain, all same tag + fusable: all 3 edges legal ---
constexpr auto g_chain = make_chain();
constexpr auto km_chain = make_uniform_fusable<8>(g_chain);
constexpr auto fl_chain = fusion_pairs<8, 16>(g_chain, km_chain);
static_assert(fl_chain.pairs.size() == 3);
static_assert(fl_chain.total_edges == 3);
static_assert(fl_chain.fusion_ratio() == 1.0);

// --- Chain, all non-fusable: 0 pairs ---
constexpr auto km_chain_nf = make_uniform_nonfusable<8>(g_chain);
constexpr auto fl_chain_nf = fusion_pairs<8, 16>(g_chain, km_chain_nf);
static_assert(fl_chain_nf.pairs.size() == 0);
static_assert(fl_chain_nf.total_edges == 3);
static_assert(fl_chain_nf.fusion_ratio() == 0.0);

// --- Chain, mixed tags: tag_A, tag_A, tag_B, tag_B → edge 0→1 and 2→3 legal ---
constexpr auto km_chain_mixed = [] {
    kernel_map<8> km(4, default_kernel_info);
    km[node_id{0}] = kernel_info{.tag = tag_A, .flops = 10, .is_fusable = true};
    km[node_id{1}] = kernel_info{.tag = tag_A, .flops = 20, .is_fusable = true};
    km[node_id{2}] = kernel_info{.tag = tag_B, .flops = 30, .is_fusable = true};
    km[node_id{3}] = kernel_info{.tag = tag_B, .flops = 40, .is_fusable = true};
    return km;
}();
constexpr auto fl_chain_mixed = fusion_pairs<8, 16>(g_chain, km_chain_mixed);
static_assert(fl_chain_mixed.pairs.size() == 2);  // 0→1 and 2→3
static_assert(fl_chain_mixed.total_edges == 3);

// --- any_tag_policy: all fusable regardless of tag ---
constexpr auto fl_any = fusion_pairs<8, 16>(g_chain, km_chain_mixed,
    any_tag_policy{});
static_assert(fl_any.pairs.size() == 3);

// --- can_fuse_local point queries ---
static_assert(can_fuse_local(km_chain, node_id{0}, node_id{1}));
static_assert(!can_fuse_local(km_chain_nf, node_id{0}, node_id{1}));
static_assert(can_fuse_local(km_chain_mixed, node_id{0}, node_id{1}));
static_assert(!can_fuse_local(km_chain_mixed, node_id{1}, node_id{2}));  // A vs B

// --- Diamond: 4 edges, all same tag → 4 legal ---
constexpr auto g_diamond = make_diamond();
constexpr auto km_diamond = make_uniform_fusable<8>(g_diamond);
constexpr auto fl_diamond = fusion_pairs<8, 16>(g_diamond, km_diamond);
static_assert(fl_diamond.pairs.size() == 4);
static_assert(fl_diamond.total_edges == 4);

// --- Partially fusable: node 2 not fusable ---
constexpr auto km_diamond_partial = [] {
    kernel_map<8> km(4, default_kernel_info);
    km[node_id{0}] = kernel_info{.tag = tag_A, .is_fusable = true};
    km[node_id{1}] = kernel_info{.tag = tag_A, .is_fusable = true};
    km[node_id{2}] = kernel_info{.tag = tag_A, .is_fusable = false}; // barrier
    km[node_id{3}] = kernel_info{.tag = tag_A, .is_fusable = true};
    return km;
}();
constexpr auto fl_dp = fusion_pairs<8, 16>(g_diamond, km_diamond_partial);
// Edges: 0→1 (ok), 0→2 (2 not fusable), 1→3 (ok), 2→3 (2 not fusable)
static_assert(fl_dp.pairs.size() == 2);

// =============================================================================
// Compile-time verification: fusion_neighbor_map
// =============================================================================

constexpr auto fnm = fusion_neighbor_map<8, 8>(g_chain, km_chain);
static_assert(fnm[node_id{0}].size() == 1);  // 0 can fuse with 1
static_assert(fnm[node_id{0}][0] == node_id{1});
static_assert(fnm[node_id{1}].size() == 1);  // 1 can fuse with 2
static_assert(fnm[node_id{2}].size() == 1);  // 2 can fuse with 3
static_assert(fnm[node_id{3}].size() == 0);  // 3 has no out-neighbors

// =============================================================================
// Compile-time verification: coarsen
// =============================================================================

// --- Chain grouped into 2: {0,1} → group 0, {2,3} → group 1 ---
constexpr auto cr_chain2 = [] {
    auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{0}] = 0;
    grp[node_id{1}] = 0;
    grp[node_id{2}] = 1;
    grp[node_id{3}] = 1;
    return coarsen<8, 16>(g, km, grp, 2);
}();
static_assert(cr_chain2.graph.node_count() == 2);
static_assert(cr_chain2.graph.edge_count() == 1);  // group 0 → group 1
static_assert(cr_chain2.kernels[node_id{0}].flops == 200);  // merged 100+100
static_assert(cr_chain2.kernels[node_id{1}].flops == 200);

// --- Chain all in one group ---
constexpr auto cr_chain1 = [] {
    auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    return coarsen<8, 16>(g, km, grp, 1);
}();
static_assert(cr_chain1.graph.node_count() == 1);
static_assert(cr_chain1.graph.edge_count() == 0);
static_assert(cr_chain1.kernels[node_id{0}].flops == 400);  // 4×100

// --- Chain all singletons → same structure ---
constexpr auto cr_chain_sing = [] {
    auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{0}] = 0;
    grp[node_id{1}] = 1;
    grp[node_id{2}] = 2;
    grp[node_id{3}] = 3;
    return coarsen<8, 16>(g, km, grp, 4);
}();
static_assert(cr_chain_sing.graph.node_count() == 4);
static_assert(cr_chain_sing.graph.edge_count() == 3);
static_assert(cr_chain_sing.kernels[node_id{0}].flops == 100);

// --- Diamond: merge {1,2} into group 1, keep 0 and 3 separate ---
constexpr auto cr_diamond = [] {
    auto g = make_diamond();
    auto km = make_kernel_map<8>(g, [](node_id n) {
        return kernel_info{.tag = tag_A,
            .flops = static_cast<std::size_t>(n.value + 1) * 10,
            .bytes_read = 100, .bytes_written = 8, .is_fusable = true};
    });
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{0}] = 0;
    grp[node_id{1}] = 1;
    grp[node_id{2}] = 1;  // merge 1 and 2
    grp[node_id{3}] = 2;
    return coarsen<8, 16>(g, km, grp, 3);
}();
static_assert(cr_diamond.graph.node_count() == 3);
// Edges: 0→merged(1,2), merged(1,2)→3
static_assert(cr_diamond.graph.edge_count() == 2);
// Group 1 merges node 1 (20 flops) + node 2 (30 flops) = 50
static_assert(cr_diamond.kernels[node_id{1}].flops == 50);
// Group 0 = node 0 (10 flops)
static_assert(cr_diamond.kernels[node_id{0}].flops == 10);
// Group 2 = node 3 (40 flops)
static_assert(cr_diamond.kernels[node_id{2}].flops == 40);

// --- Empty graph coarsen ---
constexpr auto cr_empty = coarsen<8, 16>(make_empty(), kernel_map<8>{},
    property_map<std::uint16_t, 8>{}, 0);
static_assert(cr_empty.graph.node_count() == 0);
static_assert(cr_empty.graph.edge_count() == 0);

// =============================================================================
// Compile-time verification: fuse_group
// =============================================================================

// --- Empty graph ---
constexpr auto fg_empty = find_fusion_groups<8, 16>(make_empty(),
    kernel_map<8>{});
static_assert(fg_empty.group_count == 0);

// --- Singleton: 1 group ---
constexpr auto fg_single = find_fusion_groups<8, 16>(g_single, km_single);
// Singleton has no edges → each node is its own group
static_assert(fg_single.group_count == 1);

// --- Chain, all same tag + fusable: all 4 fuse into 1 group ---
constexpr auto fg_chain = find_fusion_groups<8, 16>(g_chain, km_chain);
static_assert(fg_chain.group_count == 1);
static_assert(fg_chain.is_valid_dag);
static_assert(fg_chain.group_of[node_id{0}] == fg_chain.group_of[node_id{1}]);
static_assert(fg_chain.group_of[node_id{1}] == fg_chain.group_of[node_id{2}]);
static_assert(fg_chain.group_of[node_id{2}] == fg_chain.group_of[node_id{3}]);
static_assert(fg_chain.fused_edge_count == 3);

// --- Chain, all non-fusable: each node alone ---
constexpr auto fg_chain_nf = find_fusion_groups<8, 16>(g_chain, km_chain_nf);
static_assert(fg_chain_nf.group_count == 4);
static_assert(fg_chain_nf.fused_edge_count == 0);

// --- Chain, mixed tags A,A,B,B → 2 groups ---
constexpr auto fg_chain_mixed = find_fusion_groups<8, 16>(g_chain,
    km_chain_mixed);
static_assert(fg_chain_mixed.group_count == 2);
static_assert(fg_chain_mixed.is_valid_dag);
static_assert(fg_chain_mixed.group_of[node_id{0}] == fg_chain_mixed.group_of[node_id{1}]);
static_assert(fg_chain_mixed.group_of[node_id{2}] == fg_chain_mixed.group_of[node_id{3}]);
static_assert(fg_chain_mixed.group_of[node_id{0}] != fg_chain_mixed.group_of[node_id{2}]);

// --- Diamond, all fusable: all fuse into 1 group (DAG safe) ---
constexpr auto fg_diamond = find_fusion_groups<8, 16>(g_diamond, km_diamond);
static_assert(fg_diamond.group_count == 1);
static_assert(fg_diamond.is_valid_dag);
static_assert(fg_diamond.fused_edge_count == 4);

// --- Diamond, node 2 not fusable: {0,1,3} would form a group but ---
// --- coarsened graph is cyclic (group→2→group), so fallback to singletons ---
constexpr auto fg_diamond_partial = find_fusion_groups<8, 16>(g_diamond,
    km_diamond_partial);
static_assert(fg_diamond_partial.group_count == 4);  // fallback: singletons
static_assert(!fg_diamond_partial.is_valid_dag);      // detected cycle
static_assert(fg_diamond_partial.fused_edge_count == 0);

// --- Disconnected: 0→1, 2→3, same tag → 2 groups ---
constexpr auto g_disc = make_disconnected();
constexpr auto km_disc = make_uniform_fusable<8>(g_disc);
constexpr auto fg_disc = find_fusion_groups<8, 16>(g_disc, km_disc);
static_assert(fg_disc.group_count == 2);
static_assert(fg_disc.is_valid_dag);
static_assert(fg_disc.group_of[node_id{0}] == fg_disc.group_of[node_id{1}]);
static_assert(fg_disc.group_of[node_id{2}] == fg_disc.group_of[node_id{3}]);
static_assert(fg_disc.group_of[node_id{0}] != fg_disc.group_of[node_id{2}]);

// --- Star: all same tag → 1 group ---
constexpr auto g_star = make_star();
constexpr auto km_star = make_uniform_fusable<8>(g_star);
constexpr auto fg_star = find_fusion_groups<8, 16>(g_star, km_star);
static_assert(fg_star.group_count == 1);
static_assert(fg_star.is_valid_dag);

// =============================================================================
// Compile-time verification: group_sizes, max_group_size, fused_group_count
// =============================================================================

constexpr auto sizes_chain = group_sizes(fg_chain);
static_assert(sizes_chain[std::size_t{0}] == 4);

constexpr auto sizes_mixed = group_sizes(fg_chain_mixed);
static_assert(sizes_mixed[std::size_t{0}] == 2);
static_assert(sizes_mixed[std::size_t{1}] == 2);

static_assert(max_group_size(fg_chain) == 4);
static_assert(max_group_size(fg_chain_mixed) == 2);
static_assert(max_group_size(fg_chain_nf) == 1);

static_assert(fused_group_count(fg_chain) == 1);         // 1 group with size > 1
static_assert(fused_group_count(fg_chain_mixed) == 2);    // 2 groups with size > 1
static_assert(fused_group_count(fg_chain_nf) == 0);       // no fusions
static_assert(fused_group_count(fg_disc) == 2);           // 2 fused groups

// =============================================================================
// Runtime tests: fusion_legal
// =============================================================================

class FusionLegalTest : public ::testing::Test {};

TEST_F(FusionLegalTest, EmptyGraph) {
    constexpr auto g = make_empty();
    auto fl = fusion_pairs<8, 16>(g, kernel_map<8>{});
    EXPECT_EQ(fl.pairs.size(), 0u);
    EXPECT_EQ(fl.total_edges, 0u);
}

TEST_F(FusionLegalTest, AllFusableChain) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    auto fl = fusion_pairs<8, 16>(g, km);
    EXPECT_EQ(fl.pairs.size(), 3u);
    EXPECT_EQ(fl.total_edges, 3u);
    EXPECT_DOUBLE_EQ(fl.fusion_ratio(), 1.0);
}

TEST_F(FusionLegalTest, NonesFusableChain) {
    constexpr auto g = make_chain();
    auto km = make_uniform_nonfusable<8>(g);
    auto fl = fusion_pairs<8, 16>(g, km);
    EXPECT_EQ(fl.pairs.size(), 0u);
    EXPECT_DOUBLE_EQ(fl.fusion_ratio(), 0.0);
}

TEST_F(FusionLegalTest, MixedTagsChain) {
    constexpr auto g = make_chain();
    auto fl = fusion_pairs<8, 16>(g, km_chain_mixed);
    EXPECT_EQ(fl.pairs.size(), 2u);
    // Check which pairs: 0→1 (A→A) and 2→3 (B→B)
    EXPECT_EQ(fl.pairs[0].u, node_id{0});
    EXPECT_EQ(fl.pairs[0].v, node_id{1});
    EXPECT_EQ(fl.pairs[1].u, node_id{2});
    EXPECT_EQ(fl.pairs[1].v, node_id{3});
}

TEST_F(FusionLegalTest, AnyTagPolicy) {
    constexpr auto g = make_chain();
    auto fl = fusion_pairs<8, 16>(g, km_chain_mixed, any_tag_policy{});
    EXPECT_EQ(fl.pairs.size(), 3u);  // all 3 edges legal
}

TEST_F(FusionLegalTest, PartiallyFusableDiamond) {
    constexpr auto g = make_diamond();
    auto fl = fusion_pairs<8, 16>(g, km_diamond_partial);
    EXPECT_EQ(fl.pairs.size(), 2u);  // 0→1, 1→3
}

TEST_F(FusionLegalTest, FusionNeighborMap) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    auto fnm = fusion_neighbor_map<8, 8>(g, km);
    EXPECT_EQ(fnm[node_id{0}].size(), 1u);
    EXPECT_EQ(fnm[node_id{0}][0], node_id{1});
    EXPECT_EQ(fnm[node_id{1}].size(), 1u);
    EXPECT_EQ(fnm[node_id{1}][0], node_id{2});
    EXPECT_EQ(fnm[node_id{2}].size(), 1u);
    EXPECT_EQ(fnm[node_id{2}][0], node_id{3});
    EXPECT_EQ(fnm[node_id{3}].size(), 0u);
}

// =============================================================================
// Runtime tests: coarsen
// =============================================================================

class CoarsenTest : public ::testing::Test {};

TEST_F(CoarsenTest, ChainTwoGroups) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{2}] = 1;
    grp[node_id{3}] = 1;
    auto cr = coarsen<8, 16>(g, km, grp, 2);
    EXPECT_EQ(cr.graph.node_count(), 2u);
    EXPECT_EQ(cr.graph.edge_count(), 1u);
    EXPECT_EQ(cr.kernels[node_id{0}].flops, 200u);  // nodes 0,1
    EXPECT_EQ(cr.kernels[node_id{1}].flops, 200u);  // nodes 2,3
}

TEST_F(CoarsenTest, ChainOneGroup) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    auto cr = coarsen<8, 16>(g, km, grp, 1);
    EXPECT_EQ(cr.graph.node_count(), 1u);
    EXPECT_EQ(cr.graph.edge_count(), 0u);
    EXPECT_EQ(cr.kernels[node_id{0}].flops, 400u);
}

TEST_F(CoarsenTest, ChainSingletons) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{0}] = 0;
    grp[node_id{1}] = 1;
    grp[node_id{2}] = 2;
    grp[node_id{3}] = 3;
    auto cr = coarsen<8, 16>(g, km, grp, 4);
    EXPECT_EQ(cr.graph.node_count(), 4u);
    EXPECT_EQ(cr.graph.edge_count(), 3u);
}

TEST_F(CoarsenTest, DiamondMergeMiddle) {
    constexpr auto g = make_diamond();
    auto km = make_kernel_map<8>(g, [](node_id n) {
        return kernel_info{.tag = tag_A,
            .flops = static_cast<std::size_t>(n.value + 1) * 10,
            .bytes_read = 100, .bytes_written = 8};
    });
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{0}] = 0;
    grp[node_id{1}] = 1;
    grp[node_id{2}] = 1;
    grp[node_id{3}] = 2;
    auto cr = coarsen<8, 16>(g, km, grp, 3);
    EXPECT_EQ(cr.graph.node_count(), 3u);
    EXPECT_EQ(cr.graph.edge_count(), 2u);
    EXPECT_EQ(cr.kernels[node_id{0}].flops, 10u);   // node 0
    EXPECT_EQ(cr.kernels[node_id{1}].flops, 50u);   // nodes 1+2 = 20+30
    EXPECT_EQ(cr.kernels[node_id{2}].flops, 40u);   // node 3
}

TEST_F(CoarsenTest, CoarsenedIsDag) {
    // Verify that coarsening a DAG with valid groups produces a DAG
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    property_map<std::uint16_t, 8> grp(4, 0);
    grp[node_id{2}] = 1;
    grp[node_id{3}] = 1;
    auto cr = coarsen<8, 16>(g, km, grp, 2);
    auto topo = topological_sort(cr.graph);
    EXPECT_TRUE(topo.is_dag);
}

TEST_F(CoarsenTest, EmptyGraph) {
    auto cr = coarsen<8, 16>(make_empty(), kernel_map<8>{},
        property_map<std::uint16_t, 8>{}, 0);
    EXPECT_EQ(cr.graph.node_count(), 0u);
}

TEST_F(CoarsenTest, KernelInfoMergedCorrectly) {
    // Verify bytes_read/written aggregate
    constexpr auto g = make_chain();
    auto km = make_kernel_map<8>(g, [](node_id n) {
        auto const v = static_cast<std::size_t>(n.value);
        return kernel_info{.tag = tag_A, .flops = v * 10,
                           .bytes_read = v * 100,
                           .bytes_written = v * 8};
    });
    property_map<std::uint16_t, 8> grp(4, 0);  // all one group
    auto cr = coarsen<8, 16>(g, km, grp, 1);
    // flops: 0+10+20+30 = 60
    EXPECT_EQ(cr.kernels[node_id{0}].flops, 60u);
    // bytes_read: 0+100+200+300 = 600
    EXPECT_EQ(cr.kernels[node_id{0}].bytes_read, 600u);
    // bytes_written: 0+8+16+24 = 48
    EXPECT_EQ(cr.kernels[node_id{0}].bytes_written, 48u);
}

// =============================================================================
// Runtime tests: fuse_group
// =============================================================================

class FuseGroupTest : public ::testing::Test {};

TEST_F(FuseGroupTest, EmptyGraph) {
    auto fg = find_fusion_groups<8, 16>(make_empty(), kernel_map<8>{});
    EXPECT_EQ(fg.group_count, 0u);
}

TEST_F(FuseGroupTest, SingletonNode) {
    constexpr auto g = make_singleton();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 1u);
}

TEST_F(FuseGroupTest, ChainAllFusable) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 1u);
    EXPECT_TRUE(fg.is_valid_dag);
    EXPECT_EQ(fg.fused_edge_count, 3u);
    // All nodes in same group
    EXPECT_EQ(fg.group_of[node_id{0}], fg.group_of[node_id{1}]);
    EXPECT_EQ(fg.group_of[node_id{1}], fg.group_of[node_id{2}]);
    EXPECT_EQ(fg.group_of[node_id{2}], fg.group_of[node_id{3}]);
}

TEST_F(FuseGroupTest, ChainNonFusable) {
    constexpr auto g = make_chain();
    auto km = make_uniform_nonfusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 4u);
    EXPECT_EQ(fg.fused_edge_count, 0u);
}

TEST_F(FuseGroupTest, ChainMixedTags) {
    constexpr auto g = make_chain();
    auto fg = find_fusion_groups<8, 16>(g, km_chain_mixed);
    EXPECT_EQ(fg.group_count, 2u);
    EXPECT_TRUE(fg.is_valid_dag);
    EXPECT_EQ(fg.group_of[node_id{0}], fg.group_of[node_id{1}]);
    EXPECT_EQ(fg.group_of[node_id{2}], fg.group_of[node_id{3}]);
    EXPECT_NE(fg.group_of[node_id{0}], fg.group_of[node_id{2}]);
}

TEST_F(FuseGroupTest, DiamondAllFusable) {
    constexpr auto g = make_diamond();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 1u);
    EXPECT_TRUE(fg.is_valid_dag);
}

TEST_F(FuseGroupTest, DiamondPartiallyFusable) {
    // Node 2 not fusable → {0,1,3} would fuse, but coarsened graph
    // has cycle (group({0,1,3}) → {2} → group({0,1,3})), so fallback.
    constexpr auto g = make_diamond();
    auto fg = find_fusion_groups<8, 16>(g, km_diamond_partial);
    EXPECT_EQ(fg.group_count, 4u);  // singletons (conservative fallback)
    EXPECT_FALSE(fg.is_valid_dag);
}

TEST_F(FuseGroupTest, DisconnectedTwoGroups) {
    constexpr auto g = make_disconnected();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 2u);
    EXPECT_TRUE(fg.is_valid_dag);
}

TEST_F(FuseGroupTest, StarAllFusable) {
    constexpr auto g = make_star();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 1u);
    EXPECT_TRUE(fg.is_valid_dag);
}

TEST_F(FuseGroupTest, TriangleAllFusable) {
    constexpr auto g = make_triangle();
    auto km = make_uniform_fusable<8>(g, tag_A);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 1u);
    EXPECT_TRUE(fg.is_valid_dag);
}

TEST_F(FuseGroupTest, ConvergeDAG) {
    constexpr auto g = make_converge();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fg.group_count, 1u);
    EXPECT_TRUE(fg.is_valid_dag);
}

TEST_F(FuseGroupTest, GroupSizes) {
    constexpr auto g = make_chain();
    auto fg = find_fusion_groups<8, 16>(g, km_chain_mixed);
    auto sizes = group_sizes(fg);
    // Two groups of size 2
    EXPECT_EQ(sizes[std::size_t{0}], 2u);
    EXPECT_EQ(sizes[std::size_t{1}], 2u);
}

TEST_F(FuseGroupTest, MaxGroupSize) {
    constexpr auto g = make_chain();
    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(max_group_size(fg), 4u);
}

TEST_F(FuseGroupTest, FusedGroupCount) {
    constexpr auto g = make_chain();
    auto km_nf = make_uniform_nonfusable<8>(g);
    auto fg_nf = find_fusion_groups<8, 16>(g, km_nf);
    EXPECT_EQ(fused_group_count(fg_nf), 0u);

    auto km = make_uniform_fusable<8>(g);
    auto fg = find_fusion_groups<8, 16>(g, km);
    EXPECT_EQ(fused_group_count(fg), 1u);

    auto fg_m = find_fusion_groups<8, 16>(g, km_chain_mixed);
    EXPECT_EQ(fused_group_count(fg_m), 2u);
}

TEST_F(FuseGroupTest, EndToEndCoarsen) {
    // Full pipeline: find groups → coarsen → verify DAG
    constexpr auto g = make_chain();

    // Mixed tags: A,A,B,B → 2 fusion groups
    kernel_map<8> km_mixed(4, default_kernel_info);
    km_mixed[node_id{0}] = kernel_info{.tag = tag_A, .flops = 50,
                                        .bytes_read = 400, .bytes_written = 8,
                                        .is_fusable = true};
    km_mixed[node_id{1}] = kernel_info{.tag = tag_A, .flops = 100,
                                        .bytes_read = 400, .bytes_written = 8,
                                        .is_fusable = true};
    km_mixed[node_id{2}] = kernel_info{.tag = tag_B, .flops = 150,
                                        .bytes_read = 400, .bytes_written = 8,
                                        .is_fusable = true};
    km_mixed[node_id{3}] = kernel_info{.tag = tag_B, .flops = 200,
                                        .bytes_read = 400, .bytes_written = 8,
                                        .is_fusable = true};

    auto fg = find_fusion_groups<8, 16>(g, km_mixed);
    ASSERT_EQ(fg.group_count, 2u);

    auto cr = coarsen<8, 16>(g, km_mixed, fg.group_of, fg.group_count);
    EXPECT_EQ(cr.graph.node_count(), 2u);
    EXPECT_EQ(cr.graph.edge_count(), 1u);

    // Group 0 (nodes 0,1): 50+100 = 150 flops
    EXPECT_EQ(cr.kernels[node_id{0}].flops, 150u);
    // Group 1 (nodes 2,3): 150+200 = 350 flops
    EXPECT_EQ(cr.kernels[node_id{1}].flops, 350u);

    // Coarsened graph is a DAG
    auto topo = topological_sort(cr.graph);
    EXPECT_TRUE(topo.is_dag);
}
