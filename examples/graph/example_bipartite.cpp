// examples/graph/example_bipartite.cpp — Bipartite Matching: Task Assignment
//
// Assign engineers to projects based on their skills.  Each engineer
// can be assigned to at most one project; each project needs exactly one
// engineer.  Hopcroft-Karp finds the maximum number of assignments.
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_bipartite examples/graph/example_bipartite.cpp

#include <ctdp/graph/bipartite_graph.h>
#include <ctdp/graph/bipartite_matching.h>
#include <iostream>

using namespace ctdp::graph;

// =========================================================================
// Skill matrix
// =========================================================================
//
// Left (engineers):    Right (projects):
//   0: Alice             0: Frontend
//   1: Bob               1: Backend
//   2: Carol             2: ML Pipeline
//   3: Dave              3: DevOps
//
// Compatibility (engineer can work on project):
//   Alice  → Frontend, Backend
//   Bob    → Backend, ML Pipeline
//   Carol  → Frontend, DevOps
//   Dave   → ML Pipeline, DevOps

constexpr auto make_assignment() {
    bipartite_graph_builder<4, 4, 16> b;
    b.set_partition(4, 4);   // 4 engineers, 4 projects

    b.add_edge(0, 0);   // Alice → Frontend
    b.add_edge(0, 1);   // Alice → Backend
    b.add_edge(1, 1);   // Bob → Backend
    b.add_edge(1, 2);   // Bob → ML Pipeline
    b.add_edge(2, 0);   // Carol → Frontend
    b.add_edge(2, 3);   // Carol → DevOps
    b.add_edge(3, 2);   // Dave → ML Pipeline
    b.add_edge(3, 3);   // Dave → DevOps

    return b.finalise();
}

constexpr auto bg = make_assignment();
constexpr auto m = hopcroft_karp<4, 4>(bg);

// =========================================================================
// Compile-time proofs
// =========================================================================

static_assert(m.verified);
static_assert(m.match_count == 4,    "all 4 engineers can be assigned");
static_assert(m.is_perfect_left(),   "every engineer gets a project");
static_assert(m.is_perfect_right(),  "every project gets an engineer");

// Verify no conflicts (each engineer appears at most once,
// each project appears at most once — guaranteed by matching).

// =========================================================================
// Runtime: print assignments
// =========================================================================

int main() {
    const char* engineers[] = {"Alice", "Bob", "Carol", "Dave"};
    const char* projects[]  = {"Frontend", "Backend", "ML Pipeline", "DevOps"};

    std::cout << "=== Task Assignment: Bipartite Matching ===\n\n";

    std::cout << "Skills:\n";
    for (std::size_t i = 0; i < bg.left_count(); ++i) {
        std::cout << "  " << engineers[i] << ": ";
        // Iterate edges from left node i
        auto lid = node_id{static_cast<uint16_t>(i)};
        bool first = true;
        for (auto nb : bg.out_neighbors(lid)) {
            auto right_idx = nb.value - static_cast<uint16_t>(bg.left_count());
            if (!first) std::cout << ", ";
            std::cout << projects[right_idx];
            first = false;
        }
        std::cout << "\n";
    }

    std::cout << "\nMaximum matching: " << m.match_count
              << " / " << m.left_count << " engineers assigned\n\n";

    std::cout << "Assignments:\n";
    for (std::size_t i = 0; i < m.left_count; ++i) {
        if (m.left_matched(i)) {
            std::cout << "  " << engineers[i] << " → "
                      << projects[m.match_left[i]] << "\n";
        } else {
            std::cout << "  " << engineers[i] << " → (unassigned)\n";
        }
    }

    std::cout << "\nPerfect matching: "
              << (m.is_perfect_left() ? "yes" : "no")
              << " — proven at compile time.\n";
    return 0;
}
