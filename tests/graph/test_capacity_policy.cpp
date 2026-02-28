// tests/graph/test_capacity_policy.cpp
// Tests for capacity policies, convenience aliases, and capacity_policy concept.
//
// Validates:
//   1. Named tiers satisfy capacity_policy concept
//   2. Convenience aliases produce correct graph/builder types
//   3. Building a graph through cap aliases works end-to-end

#include "ctdp/graph/graph_capacity.h"
#include "ctdp/graph/constexpr_graph.h"
#include "ctdp/graph/symmetric_graph.h"
#include "ctdp/graph/graph_builder.h"
#include <gtest/gtest.h>

#include <type_traits>

namespace cg = ctdp::graph;
using cg::cap_from;

// ─── 1. Named tiers satisfy capacity_policy ─────────────────────────────────

static_assert(cg::capacity_policy<cg::cap::tiny>);
static_assert(cg::capacity_policy<cg::cap::small>);
static_assert(cg::capacity_policy<cg::cap::medium>);
static_assert(cg::capacity_policy<cg::cap::large>);
static_assert(cg::capacity_policy<cg::cap::xlarge>);

// ─── 2. Tier values are correct ─────────────────────────────────────────────

static_assert(cg::cap::tiny::max_v == 8);
static_assert(cg::cap::tiny::max_e == 24);
static_assert(cg::cap::small::max_v == 16);
static_assert(cg::cap::small::max_e == 64);
static_assert(cg::cap::medium::max_v == 64);
static_assert(cg::cap::medium::max_e == 256);
static_assert(cg::cap::large::max_v == 256);
static_assert(cg::cap::large::max_e == 1024);
static_assert(cg::cap::xlarge::max_v == 1024);
static_assert(cg::cap::xlarge::max_e == 4096);

// ─── 3. Convenience aliases produce correct types ───────────────────────────

// directed<Cap> == constexpr_graph<Cap>
static_assert(std::is_same_v<
    cg::directed<cg::cap::small>,
    cg::constexpr_graph<cg::cap::small>>);

static_assert(std::is_same_v<
    cg::directed_builder<cg::cap::small>,
    cg::graph_builder<cg::cap::small>>);

static_assert(std::is_same_v<
    cg::undirected<cg::cap::small>,
    cg::symmetric_graph<cg::cap::small>>);

static_assert(std::is_same_v<
    cg::undirected_builder<cg::cap::small>,
    cg::symmetric_graph_builder<cg::cap::small>>);

// Default is medium.
static_assert(std::is_same_v<
    cg::directed<>,
    cg::constexpr_graph<cg::cap::medium>>);

static_assert(std::is_same_v<
    cg::directed_builder<>,
    cg::graph_builder<cg::cap::medium>>);

// ─── 4. Custom user-defined capacity policy ─────────────────────────────────

struct my_cap {
    static constexpr std::size_t max_v = 32;
    static constexpr std::size_t max_e = 128;
};
static_assert(cg::capacity_policy<my_cap>);
static_assert(std::is_same_v<
    cg::directed<my_cap>,
    cg::constexpr_graph<my_cap>>);

// ─── 5. Types that do NOT satisfy capacity_policy ───────────────────────────

struct no_max_v { static constexpr std::size_t max_e = 10; };
struct no_max_e { static constexpr std::size_t max_v = 10; };
struct zero_v   { static constexpr std::size_t max_v = 0; static constexpr std::size_t max_e = 10; };
struct zero_e   { static constexpr std::size_t max_v = 10; static constexpr std::size_t max_e = 0; };
struct empty_struct {};

static_assert(!cg::capacity_policy<no_max_v>);
static_assert(!cg::capacity_policy<no_max_e>);
static_assert(!cg::capacity_policy<zero_v>);
static_assert(!cg::capacity_policy<zero_e>);
static_assert(!cg::capacity_policy<empty_struct>);
static_assert(!cg::capacity_policy<int>);

// ─── 6. Build a graph using cap aliases ─────────────────────────────────────

TEST(CapacityPolicy, DirectedBuild) {
    constexpr auto g = []() {
        cg::directed_builder<cg::cap::tiny> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        return b.finalise();
    }();

    static_assert(g.node_count() == 3);
    static_assert(g.edge_count() == 2);
}

TEST(CapacityPolicy, UndirectedBuild) {
    constexpr auto g = []() {
        cg::undirected_builder<cg::cap::tiny> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        return b.finalise();
    }();

    static_assert(g.node_count() == 3);
    static_assert(g.undirected_edge_count() == 2);
}

// ─── 7. cap_from defaulting: cap_from<V> == cap_from<V, 4*V> ───────────────

static_assert(std::is_same_v<
    cg::constexpr_graph<cap_from<8>>,
    cg::constexpr_graph<cap_from<8, 32>>>);

static_assert(std::is_same_v<
    cg::graph_builder<cap_from<16>>,
    cg::graph_builder<cap_from<16, 64>>>);

static_assert(std::is_same_v<
    cg::symmetric_graph<cap_from<4>>,
    cg::symmetric_graph<cap_from<4, 16>>>);

static_assert(std::is_same_v<
    cg::symmetric_graph_builder<cap_from<4>>,
    cg::symmetric_graph_builder<cap_from<4, 16>>>);

// ─── 8. Runtime-intent aliases ──────────────────────────────────────────────

static_assert(std::is_same_v<
    cg::rt_graph<cg::cap::small>,
    cg::constexpr_graph<cg::cap::small>>);

static_assert(std::is_same_v<
    cg::rt_builder<cg::cap::medium>,
    cg::graph_builder<cg::cap::medium>>);

static_assert(std::is_same_v<
    cg::symmetric_rt_builder<cg::cap::large>,
    cg::symmetric_graph_builder<cg::cap::large>>);
