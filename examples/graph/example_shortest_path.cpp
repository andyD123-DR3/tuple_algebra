// examples/graph/example_shortest_path.cpp — Dijkstra: Network Routing
//
// Find the cheapest route through a data-centre network.  Edge weights
// represent latency in microseconds.  Dijkstra finds the minimum-latency
// path from the source server to every other node.
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_shortest_path examples/graph/example_shortest_path.cpp

#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/shortest_path.h>
#include <iostream>
#include <iomanip>

using namespace ctdp::graph;

// =========================================================================
// Data-centre network topology
// =========================================================================
//
// Nodes:
//   0: web_server        4: cache
//   1: load_balancer      5: db_primary
//   2: app_server_1       6: db_replica
//   3: app_server_2
//
// Edges (bidirectional latencies in microseconds — modelled as directed):
//   web → lb (2μs), lb → app1 (3μs), lb → app2 (5μs),
//   app1 → cache (1μs), app2 → cache (2μs),
//   app1 → db_primary (10μs), cache → db_primary (4μs),
//   db_primary → db_replica (1μs)

constexpr auto make_network() {
    graph_builder<8, 32> b;
    for (int i = 0; i < 7; ++i) (void)b.add_node();

    b.add_edge(node_id{0}, node_id{1});   // web → lb
    b.add_edge(node_id{1}, node_id{2});   // lb → app1
    b.add_edge(node_id{1}, node_id{3});   // lb → app2
    b.add_edge(node_id{2}, node_id{4});   // app1 → cache
    b.add_edge(node_id{3}, node_id{4});   // app2 → cache
    b.add_edge(node_id{2}, node_id{5});   // app1 → db_primary
    b.add_edge(node_id{4}, node_id{5});   // cache → db_primary
    b.add_edge(node_id{5}, node_id{6});   // db_primary → db_replica

    return b.finalise();
}

struct network_latency {
    constexpr double operator()(node_id u, node_id v) const {
        // Latencies in microseconds
        if (u.value == 0 && v.value == 1) return 2.0;    // web → lb
        if (u.value == 1 && v.value == 2) return 3.0;    // lb → app1
        if (u.value == 1 && v.value == 3) return 5.0;    // lb → app2
        if (u.value == 2 && v.value == 4) return 1.0;    // app1 → cache
        if (u.value == 3 && v.value == 4) return 2.0;    // app2 → cache
        if (u.value == 2 && v.value == 5) return 10.0;   // app1 → db (direct)
        if (u.value == 4 && v.value == 5) return 4.0;    // cache → db
        if (u.value == 5 && v.value == 6) return 1.0;    // db → replica
        return 1e9;
    }
};

constexpr auto net = make_network();
constexpr auto sp = dijkstra(net, node_id{0}, network_latency{});

// =========================================================================
// Compile-time proofs
// =========================================================================

static_assert(sp.verified);

// Shortest latency from web_server to each node:
static_assert(sp.dist[0] == 0.0,     "web → web = 0");
static_assert(sp.dist[1] == 2.0,     "web → lb = 2μs");
static_assert(sp.dist[2] == 5.0,     "web → app1 = 2+3 = 5μs");
static_assert(sp.dist[3] == 7.0,     "web → app2 = 2+5 = 7μs");
static_assert(sp.dist[4] == 6.0,     "web → cache = 2+3+1 = 6μs");
static_assert(sp.dist[5] == 10.0,    "web → db = 2+3+1+4 = 10μs (via cache)");
static_assert(sp.dist[6] == 11.0,    "web → replica = 10+1 = 11μs");

// The direct route app1→db (10μs) is worse than via cache (1+4 = 5μs).
// So the optimal path to db is: web → lb → app1 → cache → db (10μs total).

// =========================================================================
// Runtime: print routes
// =========================================================================

int main() {
    const char* names[] = {"web_server", "load_balancer", "app_server_1",
                           "app_server_2", "cache", "db_primary", "db_replica"};

    std::cout << "=== Data-Centre Routing: Dijkstra ===\n\n";
    std::cout << "Source: " << names[0] << "\n\n";

    std::cout << std::left;
    std::cout << std::setw(18) << "Destination"
              << std::setw(12) << "Latency"
              << "Path\n";
    std::cout << std::string(50, '-') << "\n";

    for (std::size_t dest = 0; dest < net.node_count(); ++dest) {
        std::cout << std::setw(18) << names[dest]
                  << std::setw(8) << sp.dist[dest] << " μs   ";

        // Reconstruct path by following predecessors
        uint16_t path[8];
        int len = 0;
        auto cur = static_cast<uint16_t>(dest);
        while (cur != 0xFFFF && len < 8) {
            path[len++] = cur;
            cur = sp.pred[cur];
        }
        // Print in reverse
        for (int i = len - 1; i >= 0; --i) {
            std::cout << names[path[i]];
            if (i > 0) std::cout << " → ";
        }
        std::cout << "\n";
    }

    std::cout << "\nKey insight: direct route to db_primary (15μs via app1)"
              << "\n             is slower than cache route (10μs via app1→cache→db).\n";
    std::cout << "\nAll routes proven optimal at compile time.\n";
    return 0;
}
