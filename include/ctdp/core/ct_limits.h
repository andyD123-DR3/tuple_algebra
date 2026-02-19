// core/ct_limits.h - Compile-time resource limits for DP algorithms
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Every analytics algorithm has a ceiling to prevent compile-time budget exhaustion.
// Limits are enforced by static_assert with actionable diagnostic messages.
//
// OVERRIDE MECHANISM: Template Parameters (Policy-based)
// Each algorithm accepts a template parameter defaulting to the global limit:
//
//   template<size_t MaxNodes = ct_limits::topo_sort_max>
//   constexpr auto topological_sort(graph const& g);
//
// Users can override per-call:
//   auto result = topological_sort<1024>(large_graph);  // Override to 1024
//
// Or use defaults:
//   auto result = topological_sort(small_graph);        // Uses 512
//
// This is superior to preprocessor because:
// - Type-safe
// - Scoped to individual calls
// - Works with constexpr
// - No global macro pollution
// - Can have different limits in same translation unit

#ifndef CTDP_CORE_CT_LIMITS_H
#define CTDP_CORE_CT_LIMITS_H

#include <cstddef>

namespace ctdp {

/// Compile-time resource limits for dynamic programming algorithms.
///
/// These limits prevent excessive compile-time computation and provide
/// clear error messages when problems exceed reasonable bounds.
///
/// Each limit is a guideline based on practical constexpr budgets.
/// Algorithms use these as default template parameters and enforce
/// them via static_assert.
///
/// Usage in algorithms:
/// ```cpp
/// template<size_t MaxN = ct_limits::topo_sort_max>
/// constexpr auto topological_sort(graph<MaxN> const& g) {
///     static_assert(MaxN <= ct_limits::topo_sort_max,
///         "Graph size exceeds ct_limits::topo_sort_max. "
///         "Override with template parameter or use build-step mode.");
///     // ...
/// }
/// ```
///
/// User override:
/// ```cpp
/// // Use default limit (512)
/// auto result = topological_sort(graph);
///
/// // Override to 2048 for large problem
/// auto result = topological_sort<2048>(large_graph);
/// ```
namespace ct_limits {

// =============================================================================
// Graph Algorithms
// =============================================================================

/// Maximum nodes for topological sort.
/// Complexity: O(V + E)
/// Typical usage: Dependency ordering, build systems
inline constexpr size_t topo_sort_max = 512;

/// Maximum nodes for graph coloring (greedy/exact).
/// Complexity: O(V²) for greedy, exponential for exact
/// Typical usage: Register allocation, scheduling
inline constexpr size_t coloring_max = 128;

/// Maximum nodes for strongly connected components (Tarjan's algorithm).
/// Complexity: O(V + E)
/// Typical usage: Dependency cycles, reachability
inline constexpr size_t scc_max = 512;

/// Maximum nodes for connected components (DFS/BFS).
/// Complexity: O(V + E)
/// Typical usage: Graph partitioning, clustering
inline constexpr size_t connected_components_max = 512;

/// Maximum nodes for minimum cut (Stoer-Wagner).
/// Complexity: O(V³)
/// Typical usage: Network partitioning, clustering
inline constexpr size_t min_cut_max = 64;

/// Maximum nodes for shortest path (Dijkstra/Bellman-Ford).
/// Complexity: O(V² + E) for Dijkstra, O(VE) for Bellman-Ford
/// Typical usage: Distance computations, routing
inline constexpr size_t shortest_path_max = 256;

/// Maximum nodes for bipartite matching (Hopcroft-Karp).
/// Complexity: O(E√V)
/// Typical usage: Assignment problems, resource allocation
inline constexpr size_t bipartite_matching_max = 64;

/// Maximum nodes for Reverse Cuthill-McKee (bandwidth reduction).
/// Complexity: O(V + E)
/// Typical usage: Matrix reordering, cache optimization
inline constexpr size_t rcm_max = 256;

// =============================================================================
// Search Algorithms
// =============================================================================

/// Maximum Pareto frontier size for multi-objective optimization.
/// Complexity: O(n log n) per update
/// Typical usage: Plan sets, multi-objective DP
inline constexpr size_t pareto_front_max = 50;

/// Maximum candidates for exhaustive search.
/// Complexity: O(n)
/// Typical usage: Small discrete optimization problems
inline constexpr size_t exhaustive_max = 10'000;

/// Maximum beam width for beam search.
/// Complexity: O(w × d) where w=width, d=depth
/// Typical usage: Pruned state space search
inline constexpr size_t beam_width_max = 256;

// =============================================================================
// Dynamic Programming
// =============================================================================

/// Maximum subproblems for sequence DP.
/// Complexity: O(n²) typical
/// Typical usage: Edit distance, longest common subsequence
inline constexpr size_t sequence_dp_max = 1000;

/// Maximum intervals for interval DP.
/// Complexity: O(n³) typical
/// Typical usage: Matrix chain multiplication, optimal BST
inline constexpr size_t interval_dp_max = 500;

/// Maximum elements for permutation space.
/// Complexity: O(n!) for exhaustive, O(n² log n) for DP
/// Typical usage: TSP, assignment problems
inline constexpr size_t permutation_max = 20;

/// Maximum tiles for tiling DP.
/// Complexity: O(n²) to O(n³)
/// Typical usage: Loop tiling, cache optimization
inline constexpr size_t tile_max = 100;

/// Maximum segments for segmentation DP.
/// Complexity: O(n²k) where k=max segments
/// Typical usage: Text segmentation, clustering
inline constexpr size_t segmentation_max = 1000;

// =============================================================================
// Memoization
// =============================================================================

/// Maximum memoization table size (constexpr_map capacity).
/// Complexity: O(n log n) for lookup in sorted vector
/// Typical usage: DP subproblem caching
inline constexpr size_t memo_table_max = 10'000;

// =============================================================================
// Container Limits
// =============================================================================

/// Maximum constexpr_vector capacity.
/// Used when no domain-specific limit applies.
inline constexpr size_t vector_max = 10'000;

/// Maximum constexpr_map capacity.
/// Used when no domain-specific limit applies.
inline constexpr size_t map_max = 10'000;

// =============================================================================
// Iteration Limits
// =============================================================================

/// Maximum loop iterations for iterative algorithms.
/// Protects against infinite loops in constexpr context.
inline constexpr size_t max_iterations = 100'000;

/// Maximum recursion depth for recursive algorithms.
/// Stack depth tracking for debugging.
inline constexpr size_t max_recursion_depth = 256;

// =============================================================================
// Validation Helpers
// =============================================================================

/// Check if a size is within a given limit.
/// Returns true if size <= limit, false otherwise.
/// Use in static_assert for clear error messages.
///
/// Example:
/// ```cpp
/// static_assert(ct_limits::within_limit(N, ct_limits::topo_sort_max),
///     "Graph size exceeds compile-time limit for topological sort");
/// ```
constexpr bool within_limit(size_t size, size_t limit) {
    return size <= limit;
}

/// Generate a diagnostic message for limit exceeded.
/// This is a helper for creating informative static_assert messages.
///
/// Note: C++20 doesn't support string literals in static_assert,
/// so this is primarily for documentation. Actual messages are string literals.
struct limit_exceeded {
    char const* algorithm;
    size_t actual;
    size_t limit;
    char const* suggestion;
};

} // namespace ct_limits

} // namespace ctdp

#endif // CTDP_CORE_CT_LIMITS_H
