// examples/graph/example_min_cut.cpp — Min Cut: Pipeline Bottleneck
//
// A data processing pipeline has stages connected by communication links.
// Edge weights represent bandwidth (MB/s).  The minimum cut identifies
// the bottleneck — the weakest link that limits end-to-end throughput.
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_min_cut examples/graph/example_min_cut.cpp

#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/min_cut.h>
#include <iostream>

using namespace ctdp::graph;

// =========================================================================
// Pipeline topology
// =========================================================================
//
// Two processing clusters connected by a narrow link:
//
//   Cluster A (ingest):       Cluster B (analytics):
//   [parser_1]──100──┐       ┌──100──[aggregator]
//   [parser_2]──100──┤       │
//                [merger]──20──[splitter]
//   [parser_3]──100──┘       └──100──[reporter]
//
// Intra-cluster links: 100 MB/s each
// Inter-cluster link (merger→splitter): 20 MB/s ← bottleneck
//
// Min cut should be 20 MB/s, separating the two clusters.

constexpr auto make_pipeline() {
    symmetric_graph_builder<8, 16> b;
    auto parser1    = b.add_node();   // 0
    auto parser2    = b.add_node();   // 1
    auto parser3    = b.add_node();   // 2
    auto merger     = b.add_node();   // 3
    auto splitter   = b.add_node();   // 4
    auto aggregator = b.add_node();   // 5
    auto reporter   = b.add_node();   // 6

    // Cluster A: parsers → merger
    b.add_edge(parser1, merger);
    b.add_edge(parser2, merger);
    b.add_edge(parser3, merger);

    // Cross-cluster bottleneck
    b.add_edge(merger, splitter);

    // Cluster B: splitter → outputs
    b.add_edge(splitter, aggregator);
    b.add_edge(splitter, reporter);

    return b.finalise();
}

struct bandwidth_weight {
    constexpr double operator()(node_id u, node_id v) const {
        auto a = u.value < v.value ? u.value : v.value;
        auto b = u.value < v.value ? v.value : u.value;
        // Intra-cluster: 100 MB/s
        if (a <= 2 && b == 3) return 100.0;   // parser → merger
        if (a == 4 && b >= 5) return 100.0;    // splitter → outputs
        // Inter-cluster bottleneck: 20 MB/s
        if (a == 3 && b == 4) return 20.0;     // merger → splitter
        return 0.0;
    }
};

constexpr auto pipeline = make_pipeline();
constexpr auto mc = stoer_wagner(pipeline, bandwidth_weight{});

// =========================================================================
// Compile-time proofs
// =========================================================================

static_assert(mc.verified);
static_assert(mc.cut_weight == 20.0,
    "bottleneck is the 20 MB/s merger→splitter link");

// The partition separates clusters:
static_assert(mc.partition[3] != mc.partition[4],
    "merger and splitter must be on opposite sides of the cut");

// =========================================================================
// Runtime: print analysis
// =========================================================================

int main() {
    const char* names[] = {"parser_1", "parser_2", "parser_3",
                           "merger", "splitter", "aggregator", "reporter"};

    std::cout << "=== Pipeline Bottleneck: Min Cut ===\n\n";

    std::cout << "Topology (bandwidth in MB/s):\n";
    for (std::size_t u = 0; u < pipeline.node_count(); ++u) {
        auto uid = node_id{static_cast<uint16_t>(u)};
        for (auto v : pipeline.neighbors(uid)) {
            if (v.value > u) {
                auto w = bandwidth_weight{}(uid, v);
                std::cout << "  " << names[u] << " ↔ " << names[v.value]
                          << "  " << w << " MB/s\n";
            }
        }
    }

    std::cout << "\nMin cut weight: " << mc.cut_weight << " MB/s\n\n";

    std::cout << "Partition:\n  Side A: ";
    for (std::size_t i = 0; i < pipeline.node_count(); ++i)
        if (mc.partition[i] == 0) std::cout << names[i] << "  ";
    std::cout << "\n  Side B: ";
    for (std::size_t i = 0; i < pipeline.node_count(); ++i)
        if (mc.partition[i] == 1) std::cout << names[i] << "  ";
    std::cout << "\n";

    std::cout << "\nBottleneck: the " << mc.cut_weight
              << " MB/s link between merger and splitter\n"
              << "limits end-to-end throughput regardless of cluster capacity.\n";
    std::cout << "\nProven optimal at compile time.\n";
    return 0;
}
