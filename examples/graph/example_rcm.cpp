// examples/example_rcm.cpp — Reverse Cuthill-McKee bandwidth reduction
//
// Demonstrates RCM reordering at both compile time and runtime.
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_rcm examples/example_rcm.cpp
//
// This example shows:
//   1. Compile-time: prove bandwidth reduction on a scrambled grid
//   2. Compile-time: sparse matrix structure before and after reordering
//   3. Runtime: reorder a dynamically-built conflict graph

#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/rcm.h>
#include <iostream>
#include <iomanip>

using namespace ctdp::graph;

// =========================================================================
// Example 1: Compile-time bandwidth reduction on a scrambled grid
// =========================================================================

// Build a 4x4 grid graph with deliberately bad node numbering.
// The grid has 16 nodes and 24 edges.
// Natural row-major numbering gives bandwidth = 4.
// Scrambled numbering gives bandwidth = 13.
// RCM should recover near-optimal bandwidth.
constexpr auto make_scrambled_grid4x4() {
    symmetric_graph_builder<32, 64> b;
    for (int i = 0; i < 16; ++i) b.add_node();

    // Map from grid position (row-major) to scrambled node id
    // This deliberately scatters adjacent grid positions far apart.
    constexpr int m[] = {0, 12, 5, 9, 14, 2, 11, 7, 3, 15, 6, 10, 8, 1, 13, 4};

    // Horizontal edges within each row
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            b.add_edge(node_id{static_cast<uint16_t>(m[r * 4 + c])},
                       node_id{static_cast<uint16_t>(m[r * 4 + c + 1])});

    // Vertical edges between rows
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            b.add_edge(node_id{static_cast<uint16_t>(m[r * 4 + c])},
                       node_id{static_cast<uint16_t>(m[(r + 1) * 4 + c])});

    return b.finalise();
}

constexpr auto grid4x4 = make_scrambled_grid4x4();
constexpr auto rcm_grid = rcm(grid4x4);

// Compile-time proof: RCM reduces bandwidth
static_assert(rcm_grid.verified);
static_assert(rcm_grid.node_count == 16);
static_assert(rcm_grid.bandwidth_after < rcm_grid.bandwidth_before);

// =========================================================================
// Example 2: Visualise sparse matrix structure
// =========================================================================

/// Print the adjacency matrix pattern under a given permutation.
/// 'X' marks an edge, '.' marks empty, '*' marks the diagonal.
template<typename G>
void print_matrix(G const& g,
                  std::uint16_t const* inverse,
                  std::size_t V,
                  const char* title) {
    std::cout << title << " (bandwidth shown in brackets):\n";

    // Build adjacency under the permutation
    std::size_t bw = 0;
    for (std::size_t i = 0; i < V; ++i) {
        std::cout << "  ";
        auto old_i = inverse[i];
        for (std::size_t j = 0; j < V; ++j) {
            auto old_j = inverse[j];
            if (i == j) {
                std::cout << '*';
            } else {
                // Check if edge (old_i, old_j) exists
                bool found = false;
                for (auto nb : g.neighbors(node_id{old_i})) {
                    if (nb.value == old_j) { found = true; break; }
                }
                std::cout << (found ? 'X' : '.');
                if (found) {
                    auto diff = i > j ? i - j : j - i;
                    if (diff > bw) bw = diff;
                }
            }
        }
        std::cout << '\n';
    }
    std::cout << "  [bandwidth = " << bw << "]\n\n";
}

// =========================================================================
// Example 3: Runtime — reorder a SpMV conflict graph
// =========================================================================

void runtime_example() {
    // Build a conflict graph for 8 rows of a sparse matrix.
    // Rows that share non-zero columns are connected.
    // Deliberately scattered numbering creates high bandwidth.
    symmetric_graph_builder<16, 64> b;
    for (int i = 0; i < 8; ++i) b.add_node();

    // Row conflicts (simulate an arrowhead pattern)
    b.add_edge(node_id{0}, node_id{7});   // row 0 ↔ row 7 (far apart)
    b.add_edge(node_id{0}, node_id{1});
    b.add_edge(node_id{1}, node_id{2});
    b.add_edge(node_id{2}, node_id{3});
    b.add_edge(node_id{3}, node_id{4});
    b.add_edge(node_id{4}, node_id{5});
    b.add_edge(node_id{5}, node_id{6});
    b.add_edge(node_id{6}, node_id{7});

    auto g = b.finalise();
    auto r = rcm(g);

    std::cout << "=== Runtime SpMV Conflict Graph ===\n";
    std::cout << "Nodes: " << r.node_count << "\n";
    std::cout << "Bandwidth: " << r.bandwidth_before
              << " → " << r.bandwidth_after << "\n\n";

    std::cout << "RCM permutation (old → new):\n  ";
    for (std::size_t i = 0; i < r.node_count; ++i)
        std::cout << "row" << i << "→pos" << r.permutation[i] << "  ";
    std::cout << "\n\n";

    std::cout << "New row processing order:\n  ";
    for (std::size_t k = 0; k < r.node_count; ++k)
        std::cout << "pos" << k << "=row" << r.inverse[k] << "  ";
    std::cout << "\n\n";

    std::cout << "Verified: " << (r.verified ? "yes" : "no") << "\n\n";
}

// =========================================================================
// Main
// =========================================================================

int main() {
    std::cout << "=== CT-DP RCM Example ===\n\n";

    // --- Compile-time results ---
    std::cout << "=== Scrambled 4x4 Grid (compile-time) ===\n";
    std::cout << "Nodes: " << rcm_grid.node_count << "\n";
    std::cout << "Bandwidth: " << rcm_grid.bandwidth_before
              << " → " << rcm_grid.bandwidth_after << "\n\n";

    // Show matrix structure before reordering (identity permutation)
    std::array<std::uint16_t, 32> identity{};
    for (std::size_t i = 0; i < 16; ++i)
        identity[i] = static_cast<std::uint16_t>(i);
    print_matrix(grid4x4, identity.data(), 16, "Before RCM (identity ordering)");

    // Show matrix structure after RCM reordering
    print_matrix(grid4x4, rcm_grid.inverse.data(), 16, "After RCM reordering");

    // Show the permutation
    std::cout << "Permutation (old → new):\n  ";
    for (std::size_t i = 0; i < 16; ++i)
        std::cout << std::setw(2) << i << "→"
                  << std::setw(2) << rcm_grid.permutation[i] << "  ";
    std::cout << "\n\n";

    // --- Runtime example ---
    runtime_example();

    std::cout << "=== Done ===\n";
    return 0;
}
