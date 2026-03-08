// examples/pareto_tiling_demo.cpp
// Pareto-optimal tile configuration for variadic reduction
//
// Demonstrates:
//   1. descriptor_space with 6 dimensions (3,456 candidates)
//   2. valid_view for hardware feasibility (register + cache budget)
//   3. Multi-objective cost evaluation (6 objectives)
//   4. Pareto frontier extraction at compile time
//   5. Lexicographic and weighted policy selection
//
// Build:
//   g++ -std=c++20 -O2 -I include examples/pareto_tiling_demo.cpp -o pareto_demo
//
// Expected output:
//   Tile configuration space: 3456 points
//   Feasible after hardware constraints: ~200
//   Pareto frontier size: ~10-30
//   Lexicographic winner (determinism, error, throughput): ...
//   Weighted winner (balanced): ...
//
// This example uses NO runtime measurement — all costs are analytical.
// The entire search can be constexpr (shown via static_assert).

#include "ctdp/space/descriptor.h"
#include "ctdp/space/space.h"
#include "ctdp/core/cost_vector.h"
#include "ctdp/core/pareto.h"
#include "ctdp/core/constexpr_vector.h"

#include <cstddef>
#include <cstdio>

using namespace ctdp;
using namespace ctdp::space;

// =============================================================================
// Domain enums
// =============================================================================

enum class traversal_order : int { sequential, balanced_tree, cache_oblivious };
enum class simd_strategy   : int { horizontal, vertical, hybrid };

constexpr const char* traversal_name(traversal_order t) {
    switch (t) {
        case traversal_order::sequential:      return "sequential";
        case traversal_order::balanced_tree:    return "balanced_tree";
        case traversal_order::cache_oblivious:  return "cache_oblivious";
    }
    return "?";
}

constexpr const char* simd_name(simd_strategy s) {
    switch (s) {
        case simd_strategy::horizontal: return "horizontal";
        case simd_strategy::vertical:   return "vertical";
        case simd_strategy::hybrid:     return "hybrid";
    }
    return "?";
}

// =============================================================================
// Hardware model (instance data)
// =============================================================================

struct hardware_model {
    int l1_bytes;
    int l2_bytes;
    int cache_line;
    int gp_registers;
    int simd_width;
};

constexpr hardware_model hw{
    .l1_bytes     = 48 * 1024,   // 48 KB L1
    .l2_bytes     = 512 * 1024,  // 512 KB L2
    .cache_line   = 64,
    .gp_registers = 16,
    .simd_width   = 8            // AVX2: 8 doubles
};

constexpr int K = 4;  // 4 variadic reductions: sum, sum², weighted_sum, count

// =============================================================================
// Tile configuration space (6 dimensions, 3456 points)
// =============================================================================

constexpr auto make_tile_space() {
    return descriptor_space("tile",
        power_2("tile_size", 32, 1024),                                    // 6 values
        make_int_set("unroll", {1, 2, 4, 8}),                             // 4 values
        make_enum_vals("traversal", {traversal_order::sequential,
                                     traversal_order::balanced_tree,
                                     traversal_order::cache_oblivious}),   // 3 values
        make_int_set("prefetch", {0, 1, 2, 4}),                           // 4 values
        make_enum_vals("simd", {simd_strategy::horizontal,
                                simd_strategy::vertical,
                                simd_strategy::hybrid}),                   // 3 values
        make_int_set("reg_block", {2, 4, 8, 16}));                        // 4 values
    // Total: 6 × 4 × 3 × 4 × 3 × 4 = 3,456
}

// =============================================================================
// Feasibility predicate (hardware constraints)
// =============================================================================

constexpr auto make_feasibility() {
    return [](auto const& pt) constexpr -> bool {
        auto [tile, unroll, trav, pf, simd, rb] = pt;
        // Register budget: reg_block * K accumulators + unroll spill must fit
        if (rb * K + unroll * 2 > hw.gp_registers) return false;
        // Cache budget: tile working set (input + K partials) must fit L1/2
        // (half L1 for data, half for code/stack)
        if (tile * (K + 1) * static_cast<int>(sizeof(double)) > hw.l1_bytes / 2) return false;
        // Unroll factor must not exceed tile size
        if (unroll > tile) return false;
        // Hybrid SIMD requires tile >= 4 * simd_width
        if (simd == simd_strategy::hybrid && tile < 4 * hw.simd_width) return false;
        // Cache-oblivious traversal only makes sense for larger tiles
        if (trav == traversal_order::cache_oblivious && tile < 128) return false;
        return true;
    };
}

// =============================================================================
// Six cost objectives
// =============================================================================

// Objective 0: Cache miss rate (lower = better)
constexpr double cache_miss_cost(int tile, int pf) {
    int working_set = tile * K * static_cast<int>(sizeof(double));
    double base_rate = (working_set <= hw.l1_bytes / 2) ? 0.01
                     : (working_set <= hw.l1_bytes)     ? 0.05
                     : (working_set <= hw.l2_bytes)     ? 0.15
                     : 0.50;
    double pf_benefit = (pf > 0) ? (1.0 - 0.1 * pf) : 1.0;
    return base_rate * pf_benefit * tile;
}

// Objective 1: Register pressure (lower = better)
constexpr double reg_pressure_cost(int rb, int unroll) {
    return static_cast<double>(rb * K + unroll * 2);
}

// Objective 2: Error bound (lower = better)
constexpr double error_bound_cost(traversal_order trav, int tile) {
    switch (trav) {
        case traversal_order::sequential:     return 1.0;   // best: compensated
        case traversal_order::balanced_tree:   return static_cast<double>(tile) * 0.01;
        case traversal_order::cache_oblivious: return static_cast<double>(tile) * 0.005;
    }
    return 100.0;
}

// Objective 3: Throughput estimate (lower = better, inverted)
constexpr double throughput_cost(int tile, int unroll,
                                 simd_strategy simd, traversal_order trav) {
    double base = 1000.0 / (tile * unroll);  // larger tile+unroll = lower cost
    double simd_factor = (simd == simd_strategy::horizontal) ? 0.5
                       : (simd == simd_strategy::hybrid)     ? 0.6
                       : 1.0;
    double tree_factor = (trav == traversal_order::balanced_tree) ? 0.8
                       : (trav == traversal_order::sequential)    ? 1.5
                       : 1.0;
    return base * simd_factor * tree_factor;
}

// Objective 4: Determinism score (lower = better, 0 = fully deterministic)
constexpr double determinism_cost(traversal_order trav) {
    switch (trav) {
        case traversal_order::sequential:     return 0.0;  // fully deterministic
        case traversal_order::balanced_tree:   return 0.5;  // fixed tree, deterministic
        case traversal_order::cache_oblivious: return 1.0;  // may vary with cache state
    }
    return 2.0;
}

// Objective 5: Latency to first result (lower = better)
constexpr double latency_cost(int tile) {
    return static_cast<double>(tile);  // larger tile = longer to first result
}

// Combined multi-objective cost function
constexpr auto multi_cost = [](auto const& pt) constexpr -> cost_vector<6> {
    auto [tile, unroll, trav, pf, simd, rb] = pt;
    return {{
        cache_miss_cost(tile, pf),
        reg_pressure_cost(rb, unroll),
        error_bound_cost(trav, tile),
        throughput_cost(tile, unroll, simd, trav),
        determinism_cost(trav),
        latency_cost(tile)
    }};
};

// =============================================================================
// Compile-time search
// =============================================================================

constexpr auto run_pareto_search() {
    auto space = make_tile_space();
    auto feasible = filter_valid(space, make_feasibility());
    return pareto_search<6, 500>(feasible, multi_cost);
}

// Uncomment the following for full constexpr verification (may hit
// compiler step limits on some implementations; works on GCC 13+):
//
// constexpr auto frontier = run_pareto_search();
// static_assert(frontier.size() >= 3);   // non-trivial frontier
// static_assert(frontier.size() <= 100); // meaningful pruning

// =============================================================================
// Main — runtime execution with pretty output
// =============================================================================

int main() {
    auto space = make_tile_space();
    std::printf("Tile configuration space: %zu points\n", space.cardinality());

    // Count feasible
    auto feasible = filter_valid(space, make_feasibility());
    std::size_t n_feasible = 0;
    feasible.enumerate([&](auto const&) { ++n_feasible; });
    std::printf("Feasible after hardware constraints: %zu\n", n_feasible);

    // Pareto search
    auto frontier = pareto_search<6, 500>(feasible, multi_cost);
    std::printf("Pareto frontier size: %zu\n\n", frontier.size());

    // Print frontier
    std::printf("%-6s %-8s %-16s %-4s %-12s %-4s | %-8s %-8s %-8s %-8s %-6s %-8s\n",
        "Tile", "Unroll", "Traversal", "PF", "SIMD", "RB",
        "Cache", "RegPres", "Error", "Thru", "Det", "Latency");
    std::printf("%-6s %-8s %-16s %-4s %-12s %-4s | %-8s %-8s %-8s %-8s %-6s %-8s\n",
        "----", "------", "---------", "--", "----", "--",
        "-----", "-------", "-----", "----", "---", "-------");

    for (std::size_t i = 0; i < frontier.size(); ++i) {
        auto const& [cand, cv] = frontier[i];
        auto [tile, unroll, trav, pf, simd, rb] = cand;
        std::printf("%-6d %-8d %-16s %-4d %-12s %-4d | %-8.2f %-8.1f %-8.3f %-8.3f %-6.1f %-8.1f\n",
            tile, unroll, traversal_name(trav), pf, simd_name(simd), rb,
            cv[0], cv[1], cv[2], cv[3], cv[4], cv[5]);
    }

    // Lexicographic: determinism first, error second, throughput third
    auto det_winner = lex_select<4, 2, 3>(frontier);
    std::printf("\nLexicographic winner (determinism > error > throughput):\n");
    {
        auto [tile, unroll, trav, pf, simd, rb] = det_winner.candidate;
        std::printf("  tile=%d unroll=%d traversal=%s prefetch=%d simd=%s reg_block=%d\n",
            tile, unroll, traversal_name(trav), pf, simd_name(simd), rb);
        std::printf("  cost: cache=%.2f reg=%.1f error=%.3f thru=%.3f det=%.1f lat=%.1f\n",
            det_winner.cost[0], det_winner.cost[1], det_winner.cost[2],
            det_winner.cost[3], det_winner.cost[4], det_winner.cost[5]);
    }

    // Lexicographic: throughput first, cache second
    auto perf_winner = lex_select<3, 0>(frontier);
    std::printf("\nLexicographic winner (throughput > cache):\n");
    {
        auto [tile, unroll, trav, pf, simd, rb] = perf_winner.candidate;
        std::printf("  tile=%d unroll=%d traversal=%s prefetch=%d simd=%s reg_block=%d\n",
            tile, unroll, traversal_name(trav), pf, simd_name(simd), rb);
        std::printf("  cost: cache=%.2f reg=%.1f error=%.3f thru=%.3f det=%.1f lat=%.1f\n",
            perf_winner.cost[0], perf_winner.cost[1], perf_winner.cost[2],
            perf_winner.cost[3], perf_winner.cost[4], perf_winner.cost[5]);
    }

    // Weighted: balanced across all objectives
    auto balanced_winner = weighted_select(frontier,
        cost_vector<6>{{0.15, 0.15, 0.20, 0.25, 0.15, 0.10}});
    std::printf("\nWeighted winner (balanced):\n");
    {
        auto [tile, unroll, trav, pf, simd, rb] = balanced_winner.candidate;
        std::printf("  tile=%d unroll=%d traversal=%s prefetch=%d simd=%s reg_block=%d\n",
            tile, unroll, traversal_name(trav), pf, simd_name(simd), rb);
        std::printf("  cost: cache=%.2f reg=%.1f error=%.3f thru=%.3f det=%.1f lat=%.1f\n",
            balanced_winner.cost[0], balanced_winner.cost[1], balanced_winner.cost[2],
            balanced_winner.cost[3], balanced_winner.cost[4], balanced_winner.cost[5]);
    }

    std::printf("\nDone. Different policies → different winners from the same frontier.\n");
    return 0;
}
