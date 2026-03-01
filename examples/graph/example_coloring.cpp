// examples/graph/example_coloring.cpp — Graph Colouring: Parallel Scheduling
//
// Given tasks that conflict (cannot run simultaneously), graph colouring
// assigns each task to a time slot so that no two conflicting tasks
// share a slot.  The number of colours = number of parallel phases.
//
// Compile:
//   g++ -std=c++20 -O2 -I include -o example_coloring examples/graph/example_coloring.cpp

#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/graph_coloring.h>
#include <iostream>

using namespace ctdp::graph;

// =========================================================================
// Compile-time: exam scheduling
// =========================================================================
//
// 6 exams: Math(0), Physics(1), CS(2), English(3), Chemistry(4), History(5)
//
// Conflicts (students taking both exams):
//   Math–Physics, Math–CS, Physics–Chemistry, CS–English,
//   English–History, Chemistry–History
//
// Question: minimum number of exam slots so no student has two exams
// in the same slot?

constexpr auto make_exam_conflicts() {
    symmetric_graph_builder<8, 16> b;
    for (int i = 0; i < 6; ++i) (void)b.add_node();

    b.add_edge(node_id{0}, node_id{1});   // Math–Physics
    b.add_edge(node_id{0}, node_id{2});   // Math–CS
    b.add_edge(node_id{1}, node_id{4});   // Physics–Chemistry
    b.add_edge(node_id{2}, node_id{3});   // CS–English
    b.add_edge(node_id{3}, node_id{5});   // English–History
    b.add_edge(node_id{4}, node_id{5});   // Chemistry–History

    return b.finalise();
}

constexpr auto exams = make_exam_conflicts();
constexpr auto cr = graph_coloring(exams);

// Compile-time proofs
static_assert(cr.verified,              "colouring must be valid");
static_assert(cr.color_count <= 3,      "3 exam slots suffice");
static_assert(cr.color_count >= 2,      "need at least 2 (conflicts exist)");

// No two conflicting exams share a colour (verified by the algorithm,
// but let's also prove specific cases):
static_assert(cr.color_of[0] != cr.color_of[1], "Math ≠ Physics");
static_assert(cr.color_of[0] != cr.color_of[2], "Math ≠ CS");
static_assert(cr.color_of[4] != cr.color_of[5], "Chemistry ≠ History");

// =========================================================================
// Runtime: print the schedule
// =========================================================================

int main() {
    const char* subjects[] = {"Math", "Physics", "CS",
                              "English", "Chemistry", "History"};
    const char* slots[] = {"Morning", "Afternoon", "Evening"};

    std::cout << "=== Exam Scheduling: Graph Colouring ===\n\n";

    std::cout << "Conflicts:\n";
    for (std::size_t u = 0; u < exams.node_count(); ++u) {
        auto uid = node_id{static_cast<uint16_t>(u)};
        for (auto v : exams.neighbors(uid))
            if (v.value > u)
                std::cout << "  " << subjects[u] << " – " << subjects[v.value] << "\n";
    }

    std::cout << "\nSlots needed: " << cr.color_count << "\n\n";

    std::cout << "Schedule:\n";
    for (std::size_t c = 0; c < cr.color_count; ++c) {
        std::cout << "  " << slots[c] << ": ";
        bool first = true;
        for (std::size_t i = 0; i < exams.node_count(); ++i) {
            if (cr.color_of[i] == c) {
                if (!first) std::cout << ", ";
                std::cout << subjects[i];
                first = false;
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nVerified at compile time: no two conflicting exams share a slot.\n";
    return 0;
}
