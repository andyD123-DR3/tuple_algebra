// graph/demo/spmv_graph.h - SpMV demonstration: full graph pipeline
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// This is the capstone header tying the entire graph library together.
// It demonstrates the pipeline:
//
//   Sparse matrix pattern (compile-time CSR)
//       → Sparsity metrics (bandwidth, uniformity, diagonal count)
//       → Row-conflict graph (rows sharing columns → undirected edges)
//       → kernel_info per row (NNZ-derived costs)
//       → Fusion analysis (same-format rows can fuse)
//       → Schedule space (topo-ordered rows with constraints)
//       → Format recommendation (CSR/DIA/ELL based on metrics)
//
// The sparse_pattern represents a fixed sparsity structure known at
// compile time — the typical case for stencil-derived matrices in PDE
// solvers. Format selection is the DP "strategy choice" per row group.
//
// DOMAIN CONTEXT:
// In a 5-point Laplacian on a 100×100 grid, the matrix is 10000×10000
// with 5 non-zeros per interior row. The sparsity pattern is entirely
// determined by the stencil and grid dimensions — known at compile time.
// This header works on smaller compile-time-feasible examples that
// demonstrate the same analysis at full scale.

#ifndef CTDP_GRAPH_SPMV_GRAPH_H
#define CTDP_GRAPH_SPMV_GRAPH_H

#include "../../graph/graph_concepts.h"
#include "../../graph/constexpr_graph.h"
#include "../../graph/symmetric_graph.h"
#include "../../graph/graph_builder.h"
#include "../../graph/property_map.h"
#include "../../graph/kernel_info.h"
#include "../../graph/fusion_legal.h"
#include "../../graph/fuse_group.h"
#include "../../graph/coarsen.h"
#include "../../graph/graph_coloring.h"
#include "../../graph/topological_sort.h"
#include "../../engine/bridge/graph_to_space.h"
#include "../../engine/bridge/graph_to_constraints.h"
#include "../../core/constexpr_vector.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// Sparse format enum
// =============================================================================

/// SpMV storage format candidates.
///
/// The DP solver selects between these based on sparsity metrics.
/// Each format has different trade-offs:
/// - CSR:  General purpose, indirect column access, poor SIMD
/// - ELL:  Padded to max row length, good SIMD, wasted space if uneven
/// - DIA:  Diagonal offsets, excellent SIMD for banded matrices
/// - BCSR: Block structure, good for dense sub-blocks
enum class spmv_format : std::uint8_t {
    csr  = 0,  // Compressed Sparse Row
    ell  = 1,  // ELLPACK (padded rows)
    dia  = 2,  // Diagonal
    bcsr = 3   // Block CSR
};

/// kernel_tags for each format (used in fusion legality).
inline constexpr kernel_tag tag_csr{10};
inline constexpr kernel_tag tag_ell{11};
inline constexpr kernel_tag tag_dia{12};
inline constexpr kernel_tag tag_bcsr{13};

[[nodiscard]] constexpr kernel_tag format_to_tag(spmv_format f) noexcept {
    switch (f) {
        case spmv_format::csr:  return tag_csr;
        case spmv_format::ell:  return tag_ell;
        case spmv_format::dia:  return tag_dia;
        case spmv_format::bcsr: return tag_bcsr;
    }
    return tag_csr;
}

// =============================================================================
// Compile-time sparse pattern (CSR-like)
// =============================================================================

/// Fixed-capacity compile-time sparsity pattern in CSR format.
///
/// Stores the column indices and row pointers of a sparse matrix.
/// No values — only structure matters for format selection and graph
/// analysis.
///
/// Template parameters:
/// - MaxRows: maximum number of rows
/// - MaxNNZ:  maximum total non-zeros
///
/// Example:
/// ```cpp
/// // 4×4 tridiagonal:
/// // [x . . .]   row 0: cols {0,1}
/// // [x x . .]   row 1: cols {0,1,2}
/// // [. x x x]   row 2: cols {1,2,3}
/// // [. . x x]   row 3: cols {2,3}
/// constexpr auto pat = sparse_pattern<4, 10>::build(4, 4, ...);
/// ```
template<std::size_t MaxRows, std::size_t MaxNNZ>
struct sparse_pattern {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::array<std::size_t, MaxRows + 1> row_ptr{};     // CSR row pointers
    std::array<std::uint16_t, MaxNNZ> col_idx{};         // CSR column indices
    std::size_t nnz = 0;

    constexpr sparse_pattern() = default;

    /// Construct from row/col counts. Rows added via add_entry().
    constexpr sparse_pattern(std::size_t r, std::size_t c)
        : rows(r), cols(c) {
        row_ptr[0] = 0;
    }

    /// Add a non-zero entry at (row, col). Entries must be added in
    /// row-major order (all entries for row 0, then row 1, etc.).
    constexpr void add_entry(std::size_t row, std::size_t col) {
        col_idx[nnz] = static_cast<std::uint16_t>(col);
        nnz++;
        row_ptr[row + 1] = nnz;
    }

    /// Number of non-zeros in a given row.
    [[nodiscard]] constexpr std::size_t
    row_nnz(std::size_t row) const noexcept {
        return row_ptr[row + 1] - row_ptr[row];
    }

    /// Maximum non-zeros in any row.
    [[nodiscard]] constexpr std::size_t max_row_nnz() const noexcept {
        std::size_t mx = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            auto const rn = row_nnz(r);
            if (rn > mx) mx = rn;
        }
        return mx;
    }

    /// Minimum non-zeros in any row.
    [[nodiscard]] constexpr std::size_t min_row_nnz() const noexcept {
        if (rows == 0) return 0;
        std::size_t mn = nnz;
        for (std::size_t r = 0; r < rows; ++r) {
            auto const rn = row_nnz(r);
            if (rn < mn) mn = rn;
        }
        return mn;
    }

    /// Average non-zeros per row (integer division).
    [[nodiscard]] constexpr std::size_t avg_row_nnz() const noexcept {
        if (rows == 0) return 0;
        return nnz / rows;
    }

    /// Column index at position k in the CSR arrays.
    [[nodiscard]] constexpr std::uint16_t
    col_at(std::size_t k) const noexcept {
        return col_idx[k];
    }
};

// =============================================================================
// Sparsity metrics: the input to format selection
// =============================================================================

/// Metrics derived from a sparse pattern for format recommendation.
///
/// These directly drive the DP format selection:
/// - Bandwidth: determines DIA feasibility
/// - Row uniformity: determines ELL efficiency
/// - Diagonal count: determines DIA storage
/// - Density: overall sparsity ratio
struct sparsity_metrics {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz = 0;
    std::size_t max_nnz_per_row = 0;
    std::size_t min_nnz_per_row = 0;
    std::size_t avg_nnz_per_row = 0;
    std::size_t bandwidth = 0;         // max |col - row| across all entries
    std::size_t num_diagonals = 0;     // number of occupied diagonals
    bool is_symmetric = false;         // symmetric sparsity pattern

    /// Row length uniformity: 1.0 = all rows same length, 0.0 = very uneven.
    /// Computed as min/max ratio.
    [[nodiscard]] constexpr double uniformity() const noexcept {
        if (max_nnz_per_row == 0) return 1.0;
        return static_cast<double>(min_nnz_per_row) /
               static_cast<double>(max_nnz_per_row);
    }

    /// Density: nnz / (rows × cols).
    [[nodiscard]] constexpr double density() const noexcept {
        if (rows == 0 || cols == 0) return 0.0;
        return static_cast<double>(nnz) /
               (static_cast<double>(rows) * static_cast<double>(cols));
    }

    /// DIA efficiency: how much of DIA storage would be used.
    /// DIA stores num_diagonals × rows entries; efficiency = nnz / that.
    [[nodiscard]] constexpr double dia_efficiency() const noexcept {
        if (num_diagonals == 0 || rows == 0) return 0.0;
        return static_cast<double>(nnz) /
               static_cast<double>(num_diagonals * rows);
    }

    /// ELL efficiency: nnz / (max_nnz_per_row × rows).
    [[nodiscard]] constexpr double ell_efficiency() const noexcept {
        if (max_nnz_per_row == 0 || rows == 0) return 0.0;
        return static_cast<double>(nnz) /
               static_cast<double>(max_nnz_per_row * rows);
    }

    friend constexpr bool
    operator==(sparsity_metrics const&, sparsity_metrics const&) = default;
};

/// Compute sparsity metrics from a sparse pattern.
template<std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr sparsity_metrics
compute_metrics(sparse_pattern<MaxRows, MaxNNZ> const& pat) {
    sparsity_metrics m;
    m.rows = pat.rows;
    m.cols = pat.cols;
    m.nnz = pat.nnz;
    m.max_nnz_per_row = pat.max_row_nnz();
    m.min_nnz_per_row = pat.min_row_nnz();
    m.avg_nnz_per_row = pat.avg_row_nnz();

    // Bandwidth and diagonal census.
    std::array<bool, MaxRows + MaxNNZ> diag_seen{};  // oversized but safe
    std::size_t max_offset = 0;

    for (std::size_t r = 0; r < pat.rows; ++r) {
        for (std::size_t k = pat.row_ptr[r]; k < pat.row_ptr[r + 1]; ++k) {
            auto const c = static_cast<std::size_t>(pat.col_idx[k]);
            std::size_t offset;
            if (c >= r) {
                offset = c - r;
            } else {
                offset = r - c;
            }
            if (offset > max_offset) max_offset = offset;

            // Diagonal index: col - row + (rows-1) to make non-negative
            auto const diag_idx = static_cast<std::size_t>(
                static_cast<int>(c) - static_cast<int>(r)
                + static_cast<int>(pat.rows - 1));
            if (diag_idx < MaxRows + MaxNNZ) {
                diag_seen[diag_idx] = true;
            }
        }
    }

    m.bandwidth = max_offset;

    std::size_t diag_count = 0;
    for (std::size_t d = 0; d < 2 * pat.rows; ++d) {
        if (diag_seen[d]) ++diag_count;
    }
    m.num_diagonals = diag_count;

    // Symmetry check (structural only).
    // For small matrices, check if (i,j) exists implies (j,i) exists.
    m.is_symmetric = true;
    for (std::size_t r = 0; r < pat.rows && m.is_symmetric; ++r) {
        for (std::size_t k = pat.row_ptr[r]; k < pat.row_ptr[r + 1]; ++k) {
            auto const c = static_cast<std::size_t>(pat.col_idx[k]);
            if (c >= pat.rows) { m.is_symmetric = false; break; }
            // Check if (c, r) exists
            bool found = false;
            for (std::size_t k2 = pat.row_ptr[c];
                 k2 < pat.row_ptr[c + 1]; ++k2) {
                if (pat.col_idx[k2] == static_cast<std::uint16_t>(r)) {
                    found = true;
                    break;
                }
            }
            if (!found) m.is_symmetric = false;
        }
    }

    return m;
}

// =============================================================================
// Format recommendation
// =============================================================================

/// Format recommendation with rationale.
struct format_recommendation {
    spmv_format format = spmv_format::csr;
    double score = 0.0;  // higher = better fit

    friend constexpr bool
    operator==(format_recommendation const&,
               format_recommendation const&) = default;
};

/// Recommend optimal SpMV format based on sparsity metrics.
///
/// Decision logic (simplified roofline-aware heuristic):
/// - DIA: preferred for banded matrices (few diagonals, high fill)
///   DIA has stride-1 access → best SIMD. Storage = num_diags × rows.
/// - ELL: preferred for uniform-but-unbanded patterns.
///   Padded to max_nnz → good SIMD. Storage = max_nnz × rows.
/// - CSR: fallback for irregular patterns (low uniformity).
///
/// Key insight: when DIA and ELL have similar storage, DIA wins because
/// strided access beats gathered column indices.
[[nodiscard]] constexpr format_recommendation
recommend_format(sparsity_metrics const& m) noexcept {
    if (m.rows == 0 || m.nnz == 0) {
        return {spmv_format::csr, 0.0};
    }

    double const dia_eff = m.dia_efficiency();
    double const ell_eff = m.ell_efficiency();

    // DIA candidate: few diagonals relative to max_nnz, high fill.
    // DIA beats ELL when num_diags ≤ max_nnz AND dia_efficiency is good.
    // DIA storage = num_diags × rows, ELL storage = max_nnz × rows.
    // So DIA is better storage-wise when num_diags ≤ max_nnz.
    double dia_score = 0.0;
    if (m.num_diagonals > 0 && m.num_diagonals <= m.max_nnz_per_row) {
        // DIA uses less-or-equal storage than ELL, PLUS has stride-1 access.
        // Score = DIA fill efficiency (how much of the rectangle is used).
        dia_score = dia_eff;
    } else if (m.num_diagonals > 0 &&
               m.num_diagonals <= 2 * m.max_nnz_per_row) {
        // DIA uses more storage but stride-1 may compensate.
        // Penalize proportionally.
        dia_score = dia_eff * static_cast<double>(m.max_nnz_per_row) /
                    static_cast<double>(m.num_diagonals);
    }

    // ELL candidate: uniform row lengths → low padding waste.
    // Score = fill efficiency × uniformity.
    double ell_score = ell_eff * m.uniformity();

    // CSR: always feasible baseline.
    double csr_score = 0.3;

    if (dia_score >= ell_score && dia_score >= csr_score) {
        return {spmv_format::dia, dia_score};
    }
    if (ell_score >= csr_score) {
        return {spmv_format::ell, ell_score};
    }
    return {spmv_format::csr, csr_score};
}

// =============================================================================
// Row-conflict graph: rows sharing columns → edges
// =============================================================================

/// Build a directed row-dependency graph from a sparse pattern.
///
/// Build a row-conflict graph using the given builder type.
///
/// The column-overlap relation is defined once: row u conflicts with
/// row v (u < v) iff they share at least one column index.
/// The Builder determines the data structure produced:
///   graph_builder          → constexpr_graph  (directed, for scheduling)
///   symmetric_graph_builder → symmetric_graph  (undirected, for coloring)
template<typename Builder,
         std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr auto
build_row_graph_with(sparse_pattern<MaxRows, MaxNNZ> const& pat) {
    Builder b;
    for (std::size_t r = 0; r < pat.rows; ++r) {
        [[maybe_unused]] auto n = b.add_node();
    }

    for (std::size_t u = 0; u < pat.rows; ++u) {
        for (std::size_t v = u + 1; v < pat.rows; ++v) {
            bool shared = false;
            for (std::size_t ku = pat.row_ptr[u];
                 ku < pat.row_ptr[u + 1] && !shared; ++ku) {
                for (std::size_t kv = pat.row_ptr[v];
                     kv < pat.row_ptr[v + 1] && !shared; ++kv) {
                    if (pat.col_idx[ku] == pat.col_idx[kv]) {
                        shared = true;
                    }
                }
            }
            if (shared) {
                b.add_edge(
                    node_id{static_cast<std::uint16_t>(u)},
                    node_id{static_cast<std::uint16_t>(v)});
            }
        }
    }

    return b.finalise();
}

/// Directed row-conflict graph for scheduling algorithms.
///
/// Returns constexpr_graph (u→v where u<v) — suitable for
/// topological_sort, build_schedule_space, build_constraints.
template<std::size_t MaxV, std::size_t MaxE,
         std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr constexpr_graph<cap_from<MaxV, MaxE>>
build_row_graph(sparse_pattern<MaxRows, MaxNNZ> const& pat) {
    return build_row_graph_with<graph_builder<cap_from<MaxV, MaxE>>>(pat);
}

/// Undirected row-conflict graph for coloring algorithms.
///
/// Returns symmetric_graph (u↔v) — suitable for graph_coloring.
template<std::size_t MaxV, std::size_t MaxE,
         std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr symmetric_graph<cap_from<MaxV, MaxE>>
build_row_conflict_graph(sparse_pattern<MaxRows, MaxNNZ> const& pat) {
    return build_row_graph_with<symmetric_graph_builder<cap_from<MaxV, MaxE>>>(pat);
}

// =============================================================================
// Kernel info from sparse pattern: NNZ-derived costs per row
// =============================================================================

/// Build kernel_map for an SpMV row graph.
///
/// Each row is a kernel. Cost model:
/// - flops = 2 × NNZ (one multiply + one add per non-zero)
/// - bytes_read = 12 × NNZ + 8 (val:8 + col_idx:4 per NNZ, plus row_ptr)
/// - bytes_written = 8 (one output double)
/// - tag = format tag (same for all rows in uniform-format SpMV)
template<std::size_t MaxV,
         std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr kernel_map<MaxV>
build_spmv_kernel_map(sparse_pattern<MaxRows, MaxNNZ> const& pat,
                      spmv_format fmt = spmv_format::csr) {
    kernel_map<MaxV> km(pat.rows, default_kernel_info);
    auto const tag = format_to_tag(fmt);

    for (std::size_t r = 0; r < pat.rows; ++r) {
        auto const rnnz = pat.row_nnz(r);
        km[r] = kernel_info{
            .tag = tag,
            .flops = 2 * rnnz,
            .bytes_read = 12 * rnnz + 8,
            .bytes_written = 8,
            .is_fusable = true
        };
    }

    return km;
}

// =============================================================================
// Full pipeline: sparse_pattern → analysis result
// =============================================================================

/// Complete SpMV analysis result from the full pipeline.
template<std::size_t MaxV, std::size_t MaxE>
struct spmv_analysis {
    sparsity_metrics metrics{};
    format_recommendation format{};
    constexpr_graph<cap_from<MaxV, MaxE>> row_graph{};
    symmetric_graph<cap_from<MaxV, MaxE>> conflict_graph{};  // undirected, for coloring
    kernel_map<MaxV> kernels{};
    fuse_group_result<MaxV> fusion{};
    schedule_space<MaxV> space{};
    constraint_summary<MaxV, MaxE> constraints{};
};

/// Run the complete SpMV analysis pipeline.
///
/// This is the grand finale: every graph library component is used.
///
/// Pipeline:
/// 1. compute_metrics()              → sparsity analysis
/// 2. recommend_format()             → CSR/DIA/ELL selection
/// 3. build_row_graph()              → directed row DAG (scheduling)
/// 4. build_row_conflict_graph()     → undirected conflict graph (coloring)
/// 5. build_spmv_kernel_map()        → NNZ-derived costs per row
/// 6. find_fusion_groups()           → which rows can fuse
/// 7. build_schedule_space_fused()   → topo-ordered descriptors
/// 8. build_constraints()            → dependencies + resources + critical path
///
/// Example:
/// ```cpp
/// constexpr auto pat = make_tridiagonal(8);
/// constexpr auto result = analyze_spmv<16, 64>(pat);
/// static_assert(result.format.format == spmv_format::dia);
/// static_assert(result.space.size() == 8);
/// ```
template<std::size_t MaxV, std::size_t MaxE,
         std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr spmv_analysis<MaxV, MaxE>
analyze_spmv(sparse_pattern<MaxRows, MaxNNZ> const& pat,
             resource_constraint const& rc = unconstrained_resources) {
    spmv_analysis<MaxV, MaxE> result;

    // 1. Metrics
    result.metrics = compute_metrics(pat);

    // 2. Format recommendation
    result.format = recommend_format(result.metrics);

    // 3. Row conflict graph (directed DAG for scheduling)
    result.row_graph = build_row_graph<MaxV, MaxE>(pat);

    // 4. Undirected conflict graph (for graph_coloring)
    result.conflict_graph = build_row_conflict_graph<MaxV, MaxE>(pat);

    // 5. Kernel map
    result.kernels = build_spmv_kernel_map<MaxV>(pat, result.format.format);

    // 6. Fusion groups
    result.fusion = find_fusion_groups<MaxV, MaxE>(
        result.row_graph, result.kernels);

    // 7. Schedule space
    result.space = build_schedule_space_fused<MaxV, MaxE>(
        result.row_graph, result.kernels, result.fusion);

    // 8. Constraints
    result.constraints = build_constraints<MaxV, MaxE>(
        result.row_graph, result.space, rc);

    return result;
}

// =============================================================================
// Standard test matrix factories
// =============================================================================

/// Build a tridiagonal matrix pattern (N×N, 3 diagonals).
///
/// Structure: main diagonal + super/sub diagonals.
/// NNZ = 3N - 2 (boundary rows have 2 entries, interior rows have 3).
template<std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr sparse_pattern<MaxRows, MaxNNZ>
make_tridiagonal(std::size_t n) {
    sparse_pattern<MaxRows, MaxNNZ> pat(n, n);
    for (std::size_t r = 0; r < n; ++r) {
        if (r > 0) pat.add_entry(r, r - 1);
        pat.add_entry(r, r);
        if (r + 1 < n) pat.add_entry(r, r + 1);
    }
    return pat;
}

/// Build a 5-point stencil pattern on a grid_w × grid_h grid.
///
/// Matrix dimension: (grid_w × grid_h) × (grid_w × grid_h).
/// Each interior point has 5 non-zeros (self + 4 neighbors).
template<std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr sparse_pattern<MaxRows, MaxNNZ>
make_5pt_stencil(std::size_t grid_w, std::size_t grid_h) {
    auto const N = grid_w * grid_h;
    sparse_pattern<MaxRows, MaxNNZ> pat(N, N);

    for (std::size_t r = 0; r < N; ++r) {
        auto const row = r / grid_w;
        auto const col = r % grid_w;

        // Entries added in column order for valid CSR
        if (row > 0)           pat.add_entry(r, r - grid_w);  // up
        if (col > 0)           pat.add_entry(r, r - 1);       // left
        pat.add_entry(r, r);                                    // center
        if (col + 1 < grid_w)  pat.add_entry(r, r + 1);       // right
        if (row + 1 < grid_h)  pat.add_entry(r, r + grid_w);  // down
    }

    return pat;
}

/// Build a diagonal matrix pattern (N×N, only main diagonal).
template<std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr sparse_pattern<MaxRows, MaxNNZ>
make_diagonal(std::size_t n) {
    sparse_pattern<MaxRows, MaxNNZ> pat(n, n);
    for (std::size_t r = 0; r < n; ++r) {
        pat.add_entry(r, r);
    }
    return pat;
}

/// Build an arrow matrix pattern: dense first row/col + diagonal.
///
/// Row 0 has N entries, rows 1..N-1 have 2 entries each.
/// Extremely non-uniform → CSR preferred over ELL.
template<std::size_t MaxRows, std::size_t MaxNNZ>
[[nodiscard]] constexpr sparse_pattern<MaxRows, MaxNNZ>
make_arrow(std::size_t n) {
    sparse_pattern<MaxRows, MaxNNZ> pat(n, n);
    // Row 0: all columns
    for (std::size_t c = 0; c < n; ++c) {
        pat.add_entry(0, c);
    }
    // Rows 1..N-1: col 0 + diagonal
    for (std::size_t r = 1; r < n; ++r) {
        pat.add_entry(r, 0);
        pat.add_entry(r, r);
    }
    return pat;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SPMV_GRAPH_H
