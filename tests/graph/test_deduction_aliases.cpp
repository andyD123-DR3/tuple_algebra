// tests/graph/test_deduction_aliases.cpp
// Tests for result-type deduction aliases and is_symmetric_graph trait.
//
// Validates:
//   1. topo_result_for, components_result_for, etc. produce correct types
//   2. Aliases work with both constexpr_graph and symmetric_graph
//   3. is_symmetric_graph_v correctly detects symmetric vs directed

#include "ctdp/graph/graph_traits.h"
#include "ctdp/graph/constexpr_graph.h"
#include "ctdp/graph/symmetric_graph.h"
#include "ctdp/graph/topological_sort.h"
#include "ctdp/graph/connected_components.h"
#include "ctdp/graph/scc.h"
#include "ctdp/graph/graph_coloring.h"
#include "ctdp/graph/shortest_path.h"
#include "ctdp/graph/min_cut.h"
#include "ctdp/graph/fuse_group.h"
#include <gtest/gtest.h>

#include <type_traits>

namespace cg = ctdp::graph;
using cg::cap_from;

// ─── 1. is_symmetric_graph_v ────────────────────────────────────────────────

static_assert(!cg::is_symmetric_graph_v<cg::constexpr_graph<cap_from<8, 16>>>);
static_assert(!cg::is_symmetric_graph_v<cg::constexpr_graph<cap_from<64, 256>>>);
static_assert(cg::is_symmetric_graph_v<cg::symmetric_graph<cap_from<8, 16>>>);
static_assert(cg::is_symmetric_graph_v<cg::symmetric_graph<cap_from<64, 256>>>);

// ─── 2. Deduction aliases for constexpr_graph ───────────────────────────────

using G8 = cg::constexpr_graph<cap_from<8, 16>>;

static_assert(std::is_same_v<
    cg::topo_result_for<G8>,
    cg::topo_result<8>>);

static_assert(std::is_same_v<
    cg::components_result_for<G8>,
    cg::components_result<8>>);

static_assert(std::is_same_v<
    cg::scc_result_for<G8>,
    cg::scc_result<8>>);

static_assert(std::is_same_v<
    cg::coloring_result_for<G8>,
    cg::coloring_result<8>>);

static_assert(std::is_same_v<
    cg::shortest_path_result_for<G8>,
    cg::shortest_path_result<8>>);

static_assert(std::is_same_v<
    cg::min_cut_result_for<G8>,
    cg::min_cut_result<8>>);

static_assert(std::is_same_v<
    cg::fuse_group_result_for<G8>,
    cg::fuse_group_result<8>>);

// ─── 3. Deduction aliases for symmetric_graph ───────────────────────────────

using SG16 = cg::symmetric_graph<cap_from<16, 32>>;

// symmetric_graph<cap_from<16, 32>> has graph_traits::max_nodes == 16.
static_assert(std::is_same_v<
    cg::topo_result_for<SG16>,
    cg::topo_result<16>>);

static_assert(std::is_same_v<
    cg::components_result_for<SG16>,
    cg::components_result<16>>);

static_assert(std::is_same_v<
    cg::scc_result_for<SG16>,
    cg::scc_result<16>>);

static_assert(std::is_same_v<
    cg::coloring_result_for<SG16>,
    cg::coloring_result<16>>);

static_assert(std::is_same_v<
    cg::shortest_path_result_for<SG16>,
    cg::shortest_path_result<16>>);

static_assert(std::is_same_v<
    cg::min_cut_result_for<SG16>,
    cg::min_cut_result<16>>);

static_assert(std::is_same_v<
    cg::fuse_group_result_for<SG16>,
    cg::fuse_group_result<16>>);

// ─── 4. Larger graph — verify max_nodes propagates ──────────────────────────

using G256 = cg::constexpr_graph<cap_from<256, 1024>>;

static_assert(std::is_same_v<
    cg::topo_result_for<G256>,
    cg::topo_result<256>>);

static_assert(std::is_same_v<
    cg::components_result_for<G256>,
    cg::components_result<256>>);

// ─── 5. const-qualified graphs work through traits ──────────────────────────

static_assert(cg::graph_traits<const G8>::max_nodes == 8);
static_assert(cg::graph_traits<const G8>::max_edges == 16);

static_assert(std::is_same_v<
    cg::topo_result_for<const G8>,
    cg::topo_result<8>>);

static_assert(cg::graph_traits<const SG16>::max_nodes == 16);
static_assert(cg::graph_traits<const SG16>::max_undirected_edges == 32);
static_assert(cg::graph_traits<const SG16>::max_directed_edges == 64);

static_assert(std::is_same_v<
    cg::fuse_group_result_for<const SG16>,
    cg::fuse_group_result<16>>);

// ─── 6. symmetric_graph traits expose undirected/directed edge counts ───────

static_assert(cg::graph_traits<SG16>::max_undirected_edges == 32);
static_assert(cg::graph_traits<SG16>::max_directed_edges == 64);
static_assert(cg::graph_traits<SG16>::max_edges == 64);  // == max_directed_edges

// ─── Gtest anchor (all tests are static_assert) ────────────────────────────

TEST(DeductionAliases, StaticAssertsPassed) {
    SUCCEED() << "All deduction alias static_asserts passed";
}
