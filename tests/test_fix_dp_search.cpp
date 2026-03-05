// test_fix_dp_search.cpp  —  15 tests for fix_dp_search.h
//
// Group A: DpResult accessors and defaults         T01–T03
// Group B: exhaustive_search correctness           T04–T08
// Group C: beam_dp correctness and pruning         T09–T13
// Group D: BeamDpOnline and verify utility         T14–T15

#include <ctdp/calibrator/fix/fix_dp_search.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace ctdp::calibrator::fix;

// ── helpers ───────────────────────────────────────────────────────────────────

// Build a fully-populated CostTable<4,double> for trivial_schema.
// Cost for dense_id i is (base + i * step).
static CostTable<4, double> make_table(double base = 40.0, double step = 0.5)
{
    CostTable<4, double> t(trivial_index, 0.0);
    for (int32_t id = 0; id < trivial_index.size(); ++id)
        t[id] = base + static_cast<double>(id) * step;
    return t;
}

// Build a table with one designated winner at dense_id == winner_id.
static CostTable<4, double> make_table_with_winner(int32_t winner_id,
                                                     double winner_cost = 1.0,
                                                     double other_cost  = 100.0)
{
    CostTable<4, double> t(trivial_index, other_cost);
    t[winner_id] = winner_cost;
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP A: DpResult accessors and defaults
// ─────────────────────────────────────────────────────────────────────────────

// T01: default-constructed DpResult has found() == false
static void t01_dp_result_default_not_found() {
    DpResult<4> r{};
    assert(!r.found());
    assert(r.optimal_id    == -1);
    assert(r.optimal_cost  == 0.0);
    assert(r.rank_in_corpus == 0);
    assert(r.n_measured     == 0);
    assert(r.n_evaluated    == 0);
    std::cout << "T01 PASS: DpResult default — found()=false, id=-1\n";
}

// T02: DpResult::plan_str() returns correct string for known plan
static void t02_dp_result_plan_str() {
    DpResult<4> r{};
    r.optimal_plan = {Strategy::Unrolled, Strategy::SWAR,
                      Strategy::Loop,     Strategy::Generic};
    r.optimal_id   = 0;  // mark as found
    assert(r.plan_str() == "USLG");
    std::cout << "T02 PASS: DpResult::plan_str() = \"USLG\"\n";
}

// T03: DpResult solver string set by algorithms
static void t03_solver_string() {
    auto tbl = make_table();
    auto ex   = exhaustive_search<4>(tbl);
    auto beam = beam_dp<4>(tbl, 3);
    assert(ex.solver   == "exhaustive");
    assert(beam.solver == "beam_dp(w=3)");
    std::cout << "T03 PASS: solver strings (\"exhaustive\", \"beam_dp(w=3)\")\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP B: exhaustive_search
// ─────────────────────────────────────────────────────────────────────────────

// T04: exhaustive finds the global minimum in a fully-populated table
static void t04_exhaustive_finds_minimum() {
    // Minimum is at dense_id=0 (cost=40.0)
    auto tbl = make_table(40.0, 0.5);
    auto r   = exhaustive_search<4>(tbl);
    assert(r.found());
    assert(r.optimal_id   == 0);
    assert(r.optimal_cost == 40.0);
    assert(r.rank_in_corpus == 1);
    assert(r.n_measured    == 64);
    assert(r.n_evaluated   == 64);
    std::cout << "T04 PASS: exhaustive finds minimum at dense_id=0\n";
}

// T05: exhaustive finds a minimum in the middle of the table
static void t05_exhaustive_minimum_middle() {
    // Winner at dense_id=31
    auto tbl = make_table_with_winner(31);
    auto r   = exhaustive_search<4>(tbl);
    assert(r.found());
    assert(r.optimal_id   == 31);
    assert(r.optimal_cost == 1.0);
    assert(r.rank_in_corpus == 1);
    std::cout << "T05 PASS: exhaustive finds winner at dense_id=31\n";
}

// T06: exhaustive returns full ranked_corpus sorted ascending
static void t06_exhaustive_ranked_corpus_sorted() {
    auto tbl = make_table(10.0, 1.0);   // costs 10,11,...,73
    auto r   = exhaustive_search<4>(tbl);
    assert(static_cast<int>(r.ranked_corpus.size()) == 64);
    for (int i = 1; i < 64; ++i)
        assert(r.ranked_corpus[i].cost >= r.ranked_corpus[i-1].cost);
    assert(r.ranked_corpus.front().cost == 10.0);
    assert(r.ranked_corpus.back().cost  == 73.0);
    std::cout << "T06 PASS: exhaustive ranked_corpus is sorted ascending\n";
}

// T07: exhaustive on empty table (all zeros) returns not-found
static void t07_exhaustive_empty_table() {
    CostTable<4, double> empty(trivial_index, 0.0);
    auto r = exhaustive_search<4>(empty);
    assert(!r.found());
    assert(r.n_measured   == 0);
    assert(r.n_evaluated  == 0);
    assert(r.ranked_corpus.empty());
    std::cout << "T07 PASS: exhaustive on empty table → not-found\n";
}

// T08: exhaustive on partially-filled table only considers non-zero entries
static void t08_exhaustive_partial_table() {
    CostTable<4, double> t(trivial_index, 0.0);
    // Only populate 10 entries
    for (int32_t id = 0; id < 10; ++id)
        t[id] = 50.0 + static_cast<double>(id);
    t[5] = 1.0;  // winner
    auto r = exhaustive_search<4>(t);
    assert(r.found());
    assert(r.optimal_id   == 5);
    assert(r.n_measured   == 10);
    assert(r.rank_in_corpus == 1);
    std::cout << "T08 PASS: exhaustive partial table — only non-zero entries\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP C: beam_dp
// ─────────────────────────────────────────────────────────────────────────────

// T09: beam_dp with width=64 matches exhaustive on full table (correctness)
static void t09_beam_matches_exhaustive_full_beam() {
    auto tbl  = make_table_with_winner(17);
    auto ex   = exhaustive_search<4>(tbl);
    auto beam = beam_dp<4>(tbl, 64);
    assert(beam.found());
    assert(beam.optimal_id == ex.optimal_id);
    assert(beam.optimal_cost == ex.optimal_cost);
    std::cout << "T09 PASS: beam_dp(w=64) matches exhaustive on full table\n";
}

// T10: beam_dp with width=4 finds the winner when it is at dense_id=0
//      (always in the seed beam)
static void t10_beam_finds_seed_winner() {
    auto tbl = make_table_with_winner(0);  // winner at id=0
    auto r   = beam_dp<4>(tbl, 4);
    assert(r.found());
    assert(r.optimal_id == 0);
    std::cout << "T10 PASS: beam_dp(w=4) finds winner at dense_id=0\n";
}

// T11: beam_dp n_evaluated <= beam_width * N * NUM_STRATEGIES
static void t11_beam_evaluation_count_bounded() {
    auto tbl = make_table();
    const int w = 4;
    auto r = beam_dp<4>(tbl, w);
    // Upper bound: w * N * 4  (seed pass + N-1 expansions)
    const int upper = w * 4 /* N */ * NUM_STRATEGIES;
    assert(r.n_evaluated <= upper);
    assert(r.n_evaluated > 0);
    std::cout << "T11 PASS: beam_dp evaluation count bounded by w×N×S = "
              << upper << " (got " << r.n_evaluated << ")\n";
}

// T12: beam_dp on empty table returns not-found
static void t12_beam_empty_table() {
    CostTable<4, double> empty(trivial_index, 0.0);
    auto r = beam_dp<4>(empty, 4);
    assert(!r.found());
    std::cout << "T12 PASS: beam_dp on empty table → not-found\n";
}

// T13: verify_beam_matches_exhaustive:
//   - beam_width ≥ plan_space_size always agrees with exhaustive
//   - for small beam_width, only guarantees agreement when cost surface
//     guides the beam toward the winner at every stage.
//   We test with winner=0 (always in seed) and width=4,8,64.
static void t13_verify_utility() {
    // Winner at dense_id=0; costs ascend so beam is always guided to it
    auto tbl = make_table(1.0, 1.0);   // costs 1,2,...,64; minimum at 0
    assert(verify_beam_matches_exhaustive<4>(tbl, 4));
    assert(verify_beam_matches_exhaustive<4>(tbl, 8));
    assert(verify_beam_matches_exhaustive<4>(tbl, 64));
    // With beam_width=64 any table is equivalent to exhaustive
    auto tbl2 = make_table_with_winner(37, 5.0, 100.0);
    assert(verify_beam_matches_exhaustive<4>(tbl2, 64));
    std::cout << "T13 PASS: verify_beam_matches_exhaustive (w=4,8,64)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP D: BeamDpOnline and integration
// ─────────────────────────────────────────────────────────────────────────────

// T14: BeamDpOnline calls measure callback exactly once per unique plan
static void t14_online_calls_measure_once_per_plan() {
    CostTable<4, double> t(trivial_index, 0.0);
    BeamDpOnline<4> searcher(trivial_index, /*beam_width=*/4);

    std::vector<std::array<Strategy,4>> measured_plans;

    auto measure_fn = [&](const std::array<Strategy,4>& plan) -> double {
        measured_plans.push_back(plan);
        // Return a simple cost: dense_id + 1
        int32_t id = trivial_index.encode(plan);
        return static_cast<double>(id) + 1.0;
    };

    auto r = searcher.run(t, measure_fn);

    // No plan should have been measured twice
    std::sort(measured_plans.begin(), measured_plans.end());
    auto dup = std::adjacent_find(measured_plans.begin(), measured_plans.end());
    assert(dup == measured_plans.end());

    assert(r.found());
    assert(r.n_evaluated > 0);
    std::cout << "T14 PASS: BeamDpOnline measures each plan at most once ("
              << measured_plans.size() << " unique)\n";
}

// T15: BeamDpOnline result agrees with exhaustive when table is fully populated
//      afterwards (online fills the sparse table; exhaustive reads it)
static void t15_online_agrees_with_exhaustive_on_full_corpus() {
    // Use make_table data as our "ground truth" oracle
    auto ground_truth = make_table_with_winner(7, 1.0, 50.0);

    // Run online search; measure_fn reads from ground_truth
    CostTable<4, double> sparse(trivial_index, 0.0);
    BeamDpOnline<4> searcher(trivial_index, /*beam_width=*/64);

    auto measure_fn = [&](const std::array<Strategy,4>& plan) -> double {
        int32_t id = trivial_index.encode(plan);
        return ground_truth[id];
    };

    auto online_result = searcher.run(sparse, measure_fn);

    // Exhaustive on the ground truth should agree
    auto ex_result = exhaustive_search<4>(ground_truth);

    assert(online_result.found());
    assert(ex_result.found());
    assert(online_result.optimal_id == ex_result.optimal_id);
    assert(online_result.optimal_cost == ex_result.optimal_cost);
    std::cout << "T15 PASS: BeamDpOnline(w=64) agrees with exhaustive "
              << "(optimal dense_id=" << online_result.optimal_id << ")\n";
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    t01_dp_result_default_not_found();
    t02_dp_result_plan_str();
    t03_solver_string();

    t04_exhaustive_finds_minimum();
    t05_exhaustive_minimum_middle();
    t06_exhaustive_ranked_corpus_sorted();
    t07_exhaustive_empty_table();
    t08_exhaustive_partial_table();

    t09_beam_matches_exhaustive_full_beam();
    t10_beam_finds_seed_winner();
    t11_beam_evaluation_count_bounded();
    t12_beam_empty_table();
    t13_verify_utility();

    t14_online_calls_measure_once_per_plan();
    t15_online_agrees_with_exhaustive_on_full_corpus();

    std::cout << "\n15/15 tests passed.\n";
}
