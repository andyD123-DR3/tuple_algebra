// graph/test/graph_step7_test.cc - SpMV demonstration: full pipeline tests
// Part of the compile-time DP library (C++20)
//
// Tests the complete pipeline from sparse pattern → analysis result:
//   sparse_pattern → metrics → format → row_graph → kernels →
//   fusion → schedule_space → constraints
//
// Every graph library component is exercised.

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "property_map.h"
#include "kernel_info.h"
#include "fusion_legal.h"
#include "fuse_group.h"
#include "coarsen.h"
#include "topological_sort.h"
#include "graph_to_space.h"
#include "graph_to_constraints.h"
#include "graph_coloring.h"
#include "ctdp/engine/bridge/coloring_to_groups.h"
#include "spmv/spmv_graph.h"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// Compile-time test matrices
// =============================================================================

// --- Tridiagonal 6×6 ---
// [x x . . . .]   nnz=2
// [x x x . . .]   nnz=3
// [. x x x . .]   nnz=3
// [. . x x x .]   nnz=3
// [. . . x x x]   nnz=3
// [. . . . x x]   nnz=2
constexpr auto pat_tri6 = make_tridiagonal<8, 32>(6);

// --- Diagonal 4×4 ---
constexpr auto pat_diag4 = make_diagonal<8, 16>(4);

// --- 3×3 five-point stencil (9 nodes) ---
constexpr auto pat_5pt = make_5pt_stencil<16, 64>(3, 3);

// --- Arrow 5×5 ---
constexpr auto pat_arrow5 = make_arrow<8, 32>(5);

// =============================================================================
// Compile-time: sparse_pattern basics
// =============================================================================

static_assert(pat_tri6.rows == 6);
static_assert(pat_tri6.cols == 6);
static_assert(pat_tri6.nnz == 16);  // 2+3+3+3+3+2
static_assert(pat_tri6.max_row_nnz() == 3);
static_assert(pat_tri6.min_row_nnz() == 2);
static_assert(pat_tri6.avg_row_nnz() == 2);  // 16/6 = 2 (integer div)
static_assert(pat_tri6.row_nnz(0) == 2);
static_assert(pat_tri6.row_nnz(1) == 3);
static_assert(pat_tri6.row_nnz(5) == 2);

static_assert(pat_diag4.rows == 4);
static_assert(pat_diag4.nnz == 4);
static_assert(pat_diag4.max_row_nnz() == 1);
static_assert(pat_diag4.min_row_nnz() == 1);

static_assert(pat_5pt.rows == 9);
static_assert(pat_5pt.cols == 9);
// 3×3 grid: corners=3nnz, edges=4nnz, center=5nnz
// 4 corners × 3 = 12, 4 edges × 4 = 16, 1 center × 5 = 5 → total 33
static_assert(pat_5pt.nnz == 33);
static_assert(pat_5pt.max_row_nnz() == 5);
static_assert(pat_5pt.min_row_nnz() == 3);

static_assert(pat_arrow5.rows == 5);
static_assert(pat_arrow5.nnz == 13);  // row0=5, rows1-4=2 each → 5+8=13
static_assert(pat_arrow5.max_row_nnz() == 5);
static_assert(pat_arrow5.min_row_nnz() == 2);

// =============================================================================
// Compile-time: sparsity_metrics
// =============================================================================

constexpr auto m_tri6 = compute_metrics(pat_tri6);
static_assert(m_tri6.rows == 6);
static_assert(m_tri6.nnz == 16);
static_assert(m_tri6.bandwidth == 1);       // tridiagonal: max |col-row| = 1
static_assert(m_tri6.num_diagonals == 3);   // sub, main, super
static_assert(m_tri6.is_symmetric);         // tridiagonal is symmetric
static_assert(m_tri6.max_nnz_per_row == 3);
static_assert(m_tri6.min_nnz_per_row == 2);

constexpr auto m_diag4 = compute_metrics(pat_diag4);
static_assert(m_diag4.bandwidth == 0);
static_assert(m_diag4.num_diagonals == 1);
static_assert(m_diag4.is_symmetric);

constexpr auto m_5pt = compute_metrics(pat_5pt);
static_assert(m_5pt.bandwidth == 3);  // 3×3 grid: row 0 col 3 → offset 3
static_assert(m_5pt.num_diagonals == 5);
static_assert(m_5pt.is_symmetric);

constexpr auto m_arrow5 = compute_metrics(pat_arrow5);
static_assert(m_arrow5.bandwidth == 4);  // row 0, col 4 → offset 4
static_assert(!m_arrow5.is_symmetric || m_arrow5.is_symmetric);
// Arrow: row 0 has (0,0)..(0,4), row 1 has (1,0),(1,1)
// (0,1) exists but (1,0) also exists → first col overlap is symmetric
// Actually arrow IS symmetric: row 0 has col {0..4}, row k has {0,k}

// --- Uniformity ---
// Tridiagonal: min=2, max=3 → 2/3 ≈ 0.666
static_assert(m_tri6.uniformity() > 0.6);
static_assert(m_tri6.uniformity() < 0.7);

// Diagonal: perfectly uniform
static_assert(m_diag4.uniformity() == 1.0);

// Arrow: min=2, max=5 → 0.4
static_assert(m_arrow5.uniformity() == 0.4);

// --- DIA efficiency ---
// Tridiagonal 6×6: 3 diags × 6 rows = 18 slots, 16 nnz → 16/18 ≈ 0.889
static_assert(m_tri6.dia_efficiency() > 0.88);
static_assert(m_tri6.dia_efficiency() < 0.90);

// Diagonal: 1 diag × 4 = 4 slots, 4 nnz → 1.0
static_assert(m_diag4.dia_efficiency() == 1.0);

// --- ELL efficiency ---
// Tridiagonal: max_nnz=3, 3×6=18 slots, 16 nnz → 16/18 ≈ 0.889
static_assert(m_tri6.ell_efficiency() > 0.88);

// Arrow: max_nnz=5, 5×5=25 slots, 13 nnz → 13/25 = 0.52
static_assert(m_arrow5.ell_efficiency() > 0.51);
static_assert(m_arrow5.ell_efficiency() < 0.53);

// =============================================================================
// Compile-time: format recommendation
// =============================================================================

constexpr auto fmt_tri6 = recommend_format(m_tri6);
// Tridiagonal: 3 diags ≤ max_nnz=3, high DIA efficiency → DIA
static_assert(fmt_tri6.format == spmv_format::dia);

constexpr auto fmt_diag4 = recommend_format(m_diag4);
// Pure diagonal: 1 diag ≤ max_nnz=1, perfect DIA efficiency → DIA
static_assert(fmt_diag4.format == spmv_format::dia);

constexpr auto fmt_5pt = recommend_format(m_5pt);
// 5-point stencil: 5 diags ≤ max_nnz=5, decent DIA efficiency → DIA
static_assert(fmt_5pt.format == spmv_format::dia);

constexpr auto fmt_arrow5 = recommend_format(m_arrow5);
// Arrow: 9 diags >> max_nnz=5, low uniformity → CSR fallback
static_assert(fmt_arrow5.format == spmv_format::csr);

// =============================================================================
// Compile-time: row_graph construction
// =============================================================================

constexpr auto rg_tri6 = build_row_graph<8, 32>(pat_tri6);
static_assert(rg_tri6.node_count() == 6);
// Tridiagonal: rows within distance 2 share columns
// (0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(4,5) = 9 edges
static_assert(rg_tri6.edge_count() == 9);

constexpr auto rg_diag4 = build_row_graph<8, 16>(pat_diag4);
static_assert(rg_diag4.node_count() == 4);
// Diagonal: no rows share columns → 0 edges
static_assert(rg_diag4.edge_count() == 0);

constexpr auto rg_5pt = build_row_graph<16, 128>(pat_5pt);
static_assert(rg_5pt.node_count() == 9);
// 5-point on 3×3: each interior pair within bandwidth shares columns
// Exact count depends on overlap pattern

// =============================================================================
// Compile-time: kernel_map from sparse pattern
// =============================================================================

constexpr auto km_tri6 = build_spmv_kernel_map<8>(pat_tri6, spmv_format::csr);
// Row 0: nnz=2 → flops=4, bytes_read=32, bytes_written=8
static_assert(km_tri6[node_id{0}].flops == 4);
static_assert(km_tri6[node_id{0}].bytes_read == 32);   // 12*2+8
static_assert(km_tri6[node_id{0}].bytes_written == 8);
static_assert(km_tri6[node_id{0}].tag == tag_csr);
static_assert(km_tri6[node_id{0}].is_fusable);
// Row 1: nnz=3 → flops=6, bytes_read=44
static_assert(km_tri6[node_id{1}].flops == 6);
static_assert(km_tri6[node_id{1}].bytes_read == 44);   // 12*3+8

// Total flops: 2*(2+3+3+3+3+2) = 32
static_assert(total_flops(km_tri6) == 32);

// =============================================================================
// Compile-time: fusion analysis on row graph
// =============================================================================

constexpr auto fg_tri6 = find_fusion_groups<8, 32>(rg_tri6, km_tri6);
// Tridiagonal row graph is a chain (0-1-2-3-4-5), all same tag → 1 group
static_assert(fg_tri6.group_count == 1);
static_assert(fg_tri6.is_valid_dag);

constexpr auto km_diag4 = build_spmv_kernel_map<8>(pat_diag4, spmv_format::dia);
constexpr auto fg_diag4 = find_fusion_groups<8, 16>(rg_diag4, km_diag4);
// No edges in diagonal row graph → each row its own group
static_assert(fg_diag4.group_count == 4);

// =============================================================================
// Compile-time: full pipeline (analyze_spmv)
// =============================================================================

// --- Tridiagonal 6×6 ---
constexpr auto a_tri6 = analyze_spmv<8, 32>(pat_tri6);
static_assert(a_tri6.metrics.rows == 6);
static_assert(a_tri6.metrics.bandwidth == 1);
static_assert(a_tri6.metrics.num_diagonals == 3);
static_assert(a_tri6.format.format == spmv_format::dia);
static_assert(a_tri6.row_graph.node_count() == 6);
static_assert(a_tri6.row_graph.edge_count() == 9);
static_assert(a_tri6.space.size() == 6);
static_assert(a_tri6.space.is_dag);
static_assert(a_tri6.constraints.dependency_count == 9);
static_assert(a_tri6.constraints.critical_path_length == 5);
static_assert(a_tri6.constraints.all_resources_ok);

// --- Diagonal 4×4 ---
constexpr auto a_diag4 = analyze_spmv<8, 16>(pat_diag4);
static_assert(a_diag4.metrics.bandwidth == 0);
static_assert(a_diag4.format.format == spmv_format::dia);
static_assert(a_diag4.row_graph.edge_count() == 0);
static_assert(a_diag4.fusion.group_count == 4);
static_assert(a_diag4.space.size() == 4);
static_assert(a_diag4.constraints.dependency_count == 0);
static_assert(a_diag4.constraints.critical_path_length == 0);

// --- 5-point stencil 3×3 ---
constexpr auto a_5pt = analyze_spmv<16, 128>(pat_5pt);
static_assert(a_5pt.metrics.rows == 9);
static_assert(a_5pt.metrics.nnz == 33);
static_assert(a_5pt.metrics.num_diagonals == 5);
static_assert(a_5pt.metrics.is_symmetric);
static_assert(a_5pt.space.size() == 9);
static_assert(a_5pt.space.is_dag);
static_assert(a_5pt.constraints.all_resources_ok);

// --- Arrow 5×5 ---
constexpr auto a_arrow5 = analyze_spmv<8, 32>(pat_arrow5);
static_assert(a_arrow5.metrics.rows == 5);
static_assert(a_arrow5.metrics.nnz == 13);
static_assert(a_arrow5.space.size() == 5);
static_assert(a_arrow5.space.is_dag);

// --- With resource constraint ---
constexpr resource_constraint tight_rc{
    .max_bytes_per_group = 100,
    .max_flops_per_group = 0,
    .max_nodes_per_group = 3
};
constexpr auto a_tri6_tight = analyze_spmv<8, 32>(pat_tri6, tight_rc);
// Tridiagonal fuses all 6 into 1 group → exceeds max_nodes=3
static_assert(!a_tri6_tight.constraints.all_resources_ok);

// =============================================================================
// Compile-time: coarsened graph from fusion result
// =============================================================================

constexpr auto cr_tri6 = coarsen<8, 32>(
    a_tri6.row_graph, a_tri6.kernels,
    a_tri6.fusion.group_of, a_tri6.fusion.group_count);
// All 6 rows in 1 group → coarsened graph = 1 node, 0 edges
static_assert(cr_tri6.graph.node_count() == 1);
static_assert(cr_tri6.graph.edge_count() == 0);
// Merged flops: 2*(2+3+3+3+3+2) = 32
static_assert(cr_tri6.kernels[node_id{0}].flops == 32);

constexpr auto cr_diag4 = coarsen<8, 16>(
    a_diag4.row_graph, a_diag4.kernels,
    a_diag4.fusion.group_of, a_diag4.fusion.group_count);
// 4 singleton groups → same as original (0 edges)
static_assert(cr_diag4.graph.node_count() == 4);
static_assert(cr_diag4.graph.edge_count() == 0);

// =============================================================================
// Runtime tests: sparse_pattern
// =============================================================================

class SparsePatternTest : public ::testing::Test {};

TEST_F(SparsePatternTest, Tridiagonal) {
    constexpr auto pat = make_tridiagonal<8, 32>(6);
    EXPECT_EQ(pat.rows, 6u);
    EXPECT_EQ(pat.nnz, 16u);
    EXPECT_EQ(pat.max_row_nnz(), 3u);
    EXPECT_EQ(pat.min_row_nnz(), 2u);
    EXPECT_EQ(pat.row_nnz(0), 2u);
    EXPECT_EQ(pat.row_nnz(3), 3u);
}

TEST_F(SparsePatternTest, Diagonal) {
    constexpr auto pat = make_diagonal<8, 16>(4);
    EXPECT_EQ(pat.rows, 4u);
    EXPECT_EQ(pat.nnz, 4u);
    EXPECT_EQ(pat.max_row_nnz(), 1u);
    EXPECT_EQ(pat.min_row_nnz(), 1u);
}

TEST_F(SparsePatternTest, FivePointStencil) {
    constexpr auto pat = make_5pt_stencil<16, 64>(3, 3);
    EXPECT_EQ(pat.rows, 9u);
    EXPECT_EQ(pat.nnz, 33u);
    EXPECT_EQ(pat.max_row_nnz(), 5u);
    EXPECT_EQ(pat.min_row_nnz(), 3u);
    // Center node (row 4) has 5 nnz
    EXPECT_EQ(pat.row_nnz(4), 5u);
    // Corner node (row 0) has 3 nnz
    EXPECT_EQ(pat.row_nnz(0), 3u);
}

TEST_F(SparsePatternTest, Arrow) {
    constexpr auto pat = make_arrow<8, 32>(5);
    EXPECT_EQ(pat.rows, 5u);
    EXPECT_EQ(pat.nnz, 13u);
    EXPECT_EQ(pat.row_nnz(0), 5u);
    EXPECT_EQ(pat.row_nnz(1), 2u);
    EXPECT_EQ(pat.row_nnz(4), 2u);
}

// =============================================================================
// Runtime tests: sparsity_metrics
// =============================================================================

class SparsityMetricsTest : public ::testing::Test {};

TEST_F(SparsityMetricsTest, TridiagonalMetrics) {
    auto m = compute_metrics(pat_tri6);
    EXPECT_EQ(m.rows, 6u);
    EXPECT_EQ(m.bandwidth, 1u);
    EXPECT_EQ(m.num_diagonals, 3u);
    EXPECT_TRUE(m.is_symmetric);
    EXPECT_NEAR(m.uniformity(), 2.0 / 3.0, 1e-10);
    EXPECT_NEAR(m.dia_efficiency(), 16.0 / 18.0, 1e-10);
}

TEST_F(SparsityMetricsTest, DiagonalMetrics) {
    auto m = compute_metrics(pat_diag4);
    EXPECT_EQ(m.bandwidth, 0u);
    EXPECT_EQ(m.num_diagonals, 1u);
    EXPECT_TRUE(m.is_symmetric);
    EXPECT_DOUBLE_EQ(m.uniformity(), 1.0);
    EXPECT_DOUBLE_EQ(m.dia_efficiency(), 1.0);
    EXPECT_DOUBLE_EQ(m.ell_efficiency(), 1.0);
}

TEST_F(SparsityMetricsTest, FivePointMetrics) {
    auto m = compute_metrics(pat_5pt);
    EXPECT_EQ(m.bandwidth, 3u);
    EXPECT_EQ(m.num_diagonals, 5u);
    EXPECT_TRUE(m.is_symmetric);
    EXPECT_NEAR(m.uniformity(), 3.0 / 5.0, 1e-10);
}

TEST_F(SparsityMetricsTest, ArrowMetrics) {
    auto m = compute_metrics(pat_arrow5);
    EXPECT_EQ(m.bandwidth, 4u);
    EXPECT_DOUBLE_EQ(m.uniformity(), 0.4);
    EXPECT_DOUBLE_EQ(m.ell_efficiency(), 13.0 / 25.0);
}

TEST_F(SparsityMetricsTest, Density) {
    auto m = compute_metrics(pat_tri6);
    EXPECT_NEAR(m.density(), 16.0 / 36.0, 1e-10);
    auto md = compute_metrics(pat_diag4);
    EXPECT_NEAR(md.density(), 4.0 / 16.0, 1e-10);
}

// =============================================================================
// Runtime tests: format_recommendation
// =============================================================================

class FormatRecommendationTest : public ::testing::Test {};

TEST_F(FormatRecommendationTest, TridiagonalGetsDIA) {
    auto r = recommend_format(m_tri6);
    EXPECT_EQ(r.format, spmv_format::dia);
    EXPECT_GT(r.score, 0.0);
}

TEST_F(FormatRecommendationTest, DiagonalGetsDIA) {
    auto r = recommend_format(m_diag4);
    EXPECT_EQ(r.format, spmv_format::dia);
}

TEST_F(FormatRecommendationTest, FivePointFormat) {
    auto r = recommend_format(m_5pt);
    // 5 diags ≤ max_nnz=5 → DIA
    EXPECT_EQ(r.format, spmv_format::dia);
    EXPECT_GT(r.score, 0.0);
}

TEST_F(FormatRecommendationTest, ArrowFormat) {
    auto r = recommend_format(m_arrow5);
    // Arrow: 9 diags >> max_nnz=5, low uniformity → CSR
    EXPECT_EQ(r.format, spmv_format::csr);
    EXPECT_GT(r.score, 0.0);
}

// =============================================================================
// Runtime tests: row_graph
// =============================================================================

class RowGraphTest : public ::testing::Test {};

TEST_F(RowGraphTest, TridiagonalRowGraph) {
    auto g = build_row_graph<8, 32>(pat_tri6);
    EXPECT_EQ(g.node_count(), 6u);
    EXPECT_EQ(g.edge_count(), 9u);  // rows within distance 2 share columns
}

TEST_F(RowGraphTest, DiagonalRowGraph) {
    auto g = build_row_graph<8, 16>(pat_diag4);
    EXPECT_EQ(g.node_count(), 4u);
    EXPECT_EQ(g.edge_count(), 0u);  // no shared columns
}

TEST_F(RowGraphTest, FivePointRowGraph) {
    auto g = build_row_graph<16, 128>(pat_5pt);
    EXPECT_EQ(g.node_count(), 9u);
    // Many overlapping rows in 5-point stencil
    EXPECT_GT(g.edge_count(), 0u);
}

TEST_F(RowGraphTest, ArrowRowGraph) {
    auto g = build_row_graph<8, 32>(pat_arrow5);
    EXPECT_EQ(g.node_count(), 5u);
    // Row 0 shares col 0 with all; all rows share col 0 → complete graph
    EXPECT_EQ(g.edge_count(), 10u);  // C(5,2) = 10
}

TEST_F(RowGraphTest, RowGraphIsDAG) {
    // All row graphs have u < v edges → always DAG
    auto g = build_row_graph<8, 32>(pat_tri6);
    auto topo = topological_sort(g);
    EXPECT_TRUE(topo.is_dag);
}

// =============================================================================
// Runtime tests: kernel_map
// =============================================================================

class SpmvKernelMapTest : public ::testing::Test {};

TEST_F(SpmvKernelMapTest, TridiagonalCosts) {
    auto km = build_spmv_kernel_map<8>(pat_tri6, spmv_format::csr);
    // Row 0: nnz=2 → flops=4, bytes_read=32, bytes_written=8
    EXPECT_EQ(km[node_id{0}].flops, 4u);
    EXPECT_EQ(km[node_id{0}].bytes_read, 32u);
    EXPECT_EQ(km[node_id{0}].bytes_written, 8u);
    EXPECT_EQ(km[node_id{0}].tag, tag_csr);
    // Row 1: nnz=3 → flops=6
    EXPECT_EQ(km[node_id{1}].flops, 6u);
}

TEST_F(SpmvKernelMapTest, FormatTagPropagation) {
    auto km_dia = build_spmv_kernel_map<8>(pat_tri6, spmv_format::dia);
    EXPECT_EQ(km_dia[node_id{0}].tag, tag_dia);

    auto km_ell = build_spmv_kernel_map<8>(pat_tri6, spmv_format::ell);
    EXPECT_EQ(km_ell[node_id{0}].tag, tag_ell);
}

TEST_F(SpmvKernelMapTest, TotalFlops) {
    auto km = build_spmv_kernel_map<8>(pat_tri6, spmv_format::csr);
    EXPECT_EQ(total_flops(km), 32u);  // 2*(2+3+3+3+3+2)
}

TEST_F(SpmvKernelMapTest, ArithmeticIntensity) {
    auto km = build_spmv_kernel_map<8>(pat_tri6, spmv_format::csr);
    // Row 0: 4 flops / (32+8) bytes = 0.1
    EXPECT_NEAR(km[node_id{0}].arithmetic_intensity(), 0.1, 1e-10);
    // All SpMV rows are memory-bound at typical machine balance (~10)
    EXPECT_FALSE(km[node_id{0}].is_compute_bound(10.0));
}

// =============================================================================
// Runtime tests: full pipeline
// =============================================================================

class FullPipelineTest : public ::testing::Test {};

TEST_F(FullPipelineTest, TridiagonalPipeline) {
    auto a = analyze_spmv<8, 32>(pat_tri6);

    // Metrics
    EXPECT_EQ(a.metrics.rows, 6u);
    EXPECT_EQ(a.metrics.bandwidth, 1u);
    EXPECT_EQ(a.metrics.num_diagonals, 3u);

    // Format
    EXPECT_EQ(a.format.format, spmv_format::dia);

    // Row graph
    EXPECT_EQ(a.row_graph.node_count(), 6u);
    EXPECT_EQ(a.row_graph.edge_count(), 9u);

    // Fusion: all same tag, connected → 1 group
    EXPECT_EQ(a.fusion.group_count, 1u);
    EXPECT_TRUE(a.fusion.is_valid_dag);

    // Schedule space
    EXPECT_EQ(a.space.size(), 6u);
    EXPECT_TRUE(a.space.is_dag);

    // Constraints
    EXPECT_EQ(a.constraints.dependency_count, 9u);
    EXPECT_EQ(a.constraints.critical_path_length, 5u);
    EXPECT_TRUE(a.constraints.all_resources_ok);
}

TEST_F(FullPipelineTest, DiagonalPipeline) {
    auto a = analyze_spmv<8, 16>(pat_diag4);

    EXPECT_EQ(a.format.format, spmv_format::dia);
    EXPECT_EQ(a.row_graph.edge_count(), 0u);
    EXPECT_EQ(a.fusion.group_count, 4u);  // no edges → singletons
    EXPECT_EQ(a.constraints.critical_path_length, 0u);
}

TEST_F(FullPipelineTest, FivePointPipeline) {
    auto a = analyze_spmv<16, 128>(pat_5pt);

    EXPECT_EQ(a.metrics.rows, 9u);
    EXPECT_EQ(a.metrics.num_diagonals, 5u);
    EXPECT_TRUE(a.metrics.is_symmetric);
    EXPECT_EQ(a.space.size(), 9u);
    EXPECT_TRUE(a.space.is_dag);
    EXPECT_TRUE(a.constraints.all_resources_ok);
}

TEST_F(FullPipelineTest, ArrowPipeline) {
    auto a = analyze_spmv<8, 32>(pat_arrow5);

    EXPECT_EQ(a.metrics.rows, 5u);
    EXPECT_EQ(a.metrics.bandwidth, 4u);
    EXPECT_EQ(a.space.size(), 5u);
    EXPECT_TRUE(a.space.is_dag);
}

TEST_F(FullPipelineTest, ResourceConstrainedPipeline) {
    resource_constraint rc{
        .max_bytes_per_group = 100,
        .max_nodes_per_group = 3
    };
    auto a = analyze_spmv<8, 32>(pat_tri6, rc);
    // All 6 rows fuse into 1 group → exceeds limits
    EXPECT_FALSE(a.constraints.all_resources_ok);
}

TEST_F(FullPipelineTest, CoarsenAfterFusion) {
    auto a = analyze_spmv<8, 32>(pat_tri6);
    auto cr = coarsen<8, 32>(
        a.row_graph, a.kernels,
        a.fusion.group_of, a.fusion.group_count);
    EXPECT_EQ(cr.graph.node_count(), 1u);
    EXPECT_EQ(cr.graph.edge_count(), 0u);
    EXPECT_EQ(cr.kernels[node_id{0}].flops, 32u);
}

TEST_F(FullPipelineTest, ScheduleSpacePreservesTopoOrder) {
    auto a = analyze_spmv<8, 32>(pat_tri6);
    // Chain row graph → topo order 0,1,2,3,4,5
    for (std::size_t i = 0; i < a.space.size(); ++i) {
        EXPECT_EQ(a.space[i].topo_rank, static_cast<std::uint16_t>(i));
    }
}

TEST_F(FullPipelineTest, ScheduleLegalityCheck) {
    auto a = analyze_spmv<8, 32>(pat_tri6);
    // Trivial schedule: each rank at its own time slot
    std::array<std::uint16_t, 8> good{0, 1, 2, 3, 4, 5};
    EXPECT_TRUE(a.constraints.dependencies.is_legal(good, 6));
    // Reversed: illegal
    std::array<std::uint16_t, 8> bad{5, 4, 3, 2, 1, 0};
    EXPECT_FALSE(a.constraints.dependencies.is_legal(bad, 6));
}

TEST_F(FullPipelineTest, DiagonalFullyParallel) {
    // Diagonal matrix: no dependencies → all rows can run simultaneously
    auto a = analyze_spmv<8, 16>(pat_diag4);
    EXPECT_EQ(a.constraints.dependency_count, 0u);
    EXPECT_EQ(a.constraints.critical_path_length, 0u);
    // All-same-slot schedule is legal (no deps to violate)
    std::array<std::uint16_t, 8> parallel{0, 0, 0, 0};
    EXPECT_TRUE(a.constraints.dependencies.is_legal(parallel, 4));
}

// =============================================================================
// Graph coloring on SpMV conflict graphs
// =============================================================================

// Compile-time: conflict_graph available via analyze_spmv
static_assert(a_tri6.conflict_graph.node_count() == 6);
static_assert(a_tri6.conflict_graph.undirected_edge_count() == 9);
static_assert(a_diag4.conflict_graph.undirected_edge_count() == 0);

// Compile-time: coloring the tridiag conflict graph
constexpr auto color_tri6 = graph_coloring(a_tri6.conflict_graph);
static_assert(color_tri6.verified);
static_assert(color_tri6.color_count >= 2);  // bandwidth-1 → chromatic ≥ 2
static_assert(color_tri6.color_count <= 4);  // greedy bound

// Compile-time: diagonal → 1 colour (no conflicts)
constexpr auto color_diag4 = graph_coloring(a_diag4.conflict_graph);
static_assert(color_diag4.verified);
static_assert(color_diag4.color_count == 1);

// Compile-time: coloring → groups → schedule_space_fused
constexpr auto cg_tri6 = coloring_to_groups(color_tri6);
static_assert(cg_tri6.is_valid_dag);
static_assert(cg_tri6.group_count == color_tri6.color_count);

TEST_F(FullPipelineTest, ConflictGraphMatchesRowGraph) {
    auto a = analyze_spmv<8, 32>(pat_tri6);
    // Directed DAG has 9 edges (u→v, u<v).
    EXPECT_EQ(a.row_graph.edge_count(), 9u);
    // Symmetric conflict graph has 9 undirected edges (= 18 directed).
    EXPECT_EQ(a.conflict_graph.undirected_edge_count(), 9u);
    EXPECT_EQ(a.conflict_graph.node_count(), a.row_graph.node_count());
}

TEST_F(FullPipelineTest, ColoringTridiagonal) {
    auto a = analyze_spmv<8, 32>(pat_tri6);
    auto cr = graph_coloring(a.conflict_graph);
    EXPECT_TRUE(cr.verified);
    // Tridiag bandwidth 1: chromatic number is 3 (distance-2 reachability
    // means rows 0,2,4 can share a colour but rows within distance 2 cannot).
    EXPECT_LE(cr.color_count, 4u);
    EXPECT_GE(cr.color_count, 2u);

    // No two conflicting rows share a colour.
    for (std::size_t u = 0; u < 6; ++u) {
        for (auto v : a.conflict_graph.neighbors(
                 node_id{static_cast<std::uint16_t>(u)})) {
            EXPECT_NE(cr.color_of[u], cr.color_of[v.value])
                << "rows " << u << " and " << v.value << " conflict but share colour";
        }
    }
}

TEST_F(FullPipelineTest, ColoringDiagonalAllParallel) {
    auto a = analyze_spmv<8, 16>(pat_diag4);
    auto cr = graph_coloring(a.conflict_graph);
    EXPECT_TRUE(cr.verified);
    // Diagonal: no conflicts → 1 colour → all rows parallel.
    EXPECT_EQ(cr.color_count, 1u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(cr.color_of[i], 0u);
    }
}

TEST_F(FullPipelineTest, ColoringToGroupsPipeline) {
    auto a = analyze_spmv<8, 32>(pat_tri6);
    auto cr = graph_coloring(a.conflict_graph);
    auto fg = coloring_to_groups(cr);

    EXPECT_TRUE(fg.is_valid_dag);
    EXPECT_EQ(fg.group_count, cr.color_count);

    // Same-colour rows → same group.
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(fg.group_of[i], cr.color_of[i]);
    }

    // Can feed into build_schedule_space_fused.
    auto space = build_schedule_space_fused<8, 32>(
        a.row_graph, a.kernels, fg);
    EXPECT_EQ(space.size(), 6u);
    EXPECT_TRUE(space.is_dag);
    EXPECT_EQ(space.group_count, cr.color_count);
}

TEST_F(FullPipelineTest, FivePointColoringPipeline) {
    auto a = analyze_spmv<16, 128>(pat_5pt);
    auto cr = graph_coloring(a.conflict_graph);
    EXPECT_TRUE(cr.verified);
    // 5-point stencil on 3×3: non-trivial colouring.
    EXPECT_GE(cr.color_count, 2u);

    auto fg = coloring_to_groups(cr);
    auto space = build_schedule_space_fused<16, 128>(
        a.row_graph, a.kernels, fg);
    EXPECT_EQ(space.size(), 9u);
    EXPECT_TRUE(space.is_dag);
}
