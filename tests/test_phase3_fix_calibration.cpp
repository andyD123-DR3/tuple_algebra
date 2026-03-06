// test_phase3_fix_calibration.cpp — Phase 3 wiring tests (15 tests)
//
// Validates fix_calibration_driver.h using synthetic (mock) timing.
// NO hardware measurement calls are made — fully CI-safe.
//
// Groups:
//   A (t01–t04)  corpus skeleton construction
//   B (t05–t08)  corpus_to_cost_table population
//   C (t09–t12)  exhaustive_search integration
//   D (t13–t15)  beam_dp integration and agreement

#include <ctdp/calibrator/fix/fix_calibration_driver.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace ctdp::calibrator::fix;

// ─── test helpers ─────────────────────────────────────────────────────────────

// Flat: every plan reports the same p99
static CalibrationResult make_flat(double p99 = 50.0) {
    return mock_calibration_result([=](int32_t) { return p99; });
}

// Ascending: plan 0 cheapest (30 ns), plan 63 most expensive (93 ns)
static CalibrationResult make_ascending() {
    return mock_calibration_result(
        [](int32_t id) { return 30.0 + static_cast<double>(id); });
}

// Descending: plan 63 cheapest (30 ns), plan 0 most expensive (93 ns)
static CalibrationResult make_descending() {
    return mock_calibration_result(
        [](int32_t id) { return 93.0 - static_cast<double>(id); });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Group A — corpus skeleton construction  (t01–t04)
// ─────────────────────────────────────────────────────────────────────────────

// t01: corpus has exactly 64 DataPoints
static void t01_corpus_has_64_plans() {
    auto r = make_flat();
    assert(static_cast<int>(r.corpus.size()) == 64);
    std::printf("  [01] corpus_has_64_plans                         PASS\n");
}

// t02: every DataPoint encodes to a valid dense_id in [0,64)
static void t02_all_plans_have_valid_dense_ids() {
    auto r = make_flat();
    for (const auto& dp : r.corpus) {
        int32_t id = trivial_index.encode(dp.plan);
        assert(id >= 0 && id < 64);
    }
    std::printf("  [02] all_plans_have_valid_dense_ids               PASS\n");
}

// t03: small jitter (< 5% threshold) → all 64 DataPoints are Ok
static void t03_small_jitter_all_ok() {
    // a2_jitter=0.5 on p99=50 → ratio = 0.5/50 = 1% << SOFT_THRESH(5%)
    auto r = make_flat(50.0);
    assert(r.n_ok == 64);
    assert(r.n_suspicious == 0);
    assert(r.n_hard_deleted == 0);
    std::printf("  [03] small_jitter_all_ok                         PASS\n");
}

// t04: large jitter (> 20% HARD_THRESH) → some HardDeleted
static void t04_large_jitter_produces_hard_deleted() {
    // ratio = 15/30 = 50% >> HARD_THRESH(20%)
    auto r = mock_calibration_result(
        [](int32_t) { return 30.0; },
        /*a2_jitter=*/15.0,
        /*b_offset=*/0.0);
    assert(r.n_hard_deleted > 0);
    assert(r.n_ok < 64);
    std::printf("  [04] large_jitter_produces_hard_deleted           PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Group B — corpus_to_cost_table  (t05–t08)
// ─────────────────────────────────────────────────────────────────────────────

// t05: all 64 entries populated when all plans are Ok
static void t05_cost_table_fully_populated() {
    auto r = make_flat(40.0);
    assert(r.exhaustive.n_measured == 64);
    std::printf("  [05] cost_table_fully_populated                   PASS\n");
}

// t06: cost table entry equals mean_ab_p99 = (A1+A2)/2
static void t06_cost_value_equals_mean_ab() {
    // A1=60, A2=60+0.5=60.5 → mean=60.25
    auto r = mock_calibration_result(
        [](int32_t) { return 60.0; }, /*jitter=*/0.5);
    for (int32_t id = 0; id < 64; ++id) {
        double cost = r.cost_table[id];
        assert(std::abs(cost - 60.25) < 0.01);
    }
    std::printf("  [06] cost_value_equals_mean_ab                   PASS\n");
}

// t07: HardDeleted DataPoints produce zero (excluded) cost table entries
static void t07_hard_deleted_excluded_from_cost_table() {
    // Plan 0: ratio = 15/30 = 50% → HardDeleted; everyone else Ok
    std::vector<DataPoint<4>> corpus = make_data_points_from_schema(trivial_schema);
    for (auto& dp : corpus) {
        int32_t id = trivial_index.encode(dp.plan);
        PerfSample s1{}, s2{}, sb{};
        if (id == 0) {
            s1.p99_ns = 30.0; s2.p99_ns = 55.0; sb.p99_ns = 35.0; // 83% ratio
        } else {
            s1.p99_ns = 40.0; s2.p99_ns = 40.5; sb.p99_ns = 45.0;
        }
        dp.runs = RunTriple{ s1, s2, sb };
        dp.update_status();
    }
    auto table = corpus_to_cost_table(corpus);
    assert(table[0] == 0.0);   // excluded
    assert(table[1]  > 0.0);   // present
    std::printf("  [07] hard_deleted_excluded_from_cost_table        PASS\n");
}

// t08: B run does NOT contribute to cost — only A1 and A2
static void t08_b_run_excluded_from_cost() {
    // A1=40, A2=40, B=200 (extreme cold spike)
    // cost should be mean_ab = 40, not 200 or 120
    auto r = mock_calibration_result(
        [](int32_t) { return 40.0; },
        /*a2_jitter=*/0.0,
        /*b_offset=*/160.0);
    for (int32_t id = 0; id < 64; ++id) {
        double cost = r.cost_table[id];
        if (cost > 0.0)
            assert(std::abs(cost - 40.0) < 0.1);
    }
    std::printf("  [08] b_run_excluded_from_cost                    PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Group C — exhaustive_search integration  (t09–t12)
// ─────────────────────────────────────────────────────────────────────────────

// t09: exhaustive finds plan 0 on ascending cost surface
static void t09_exhaustive_finds_cheapest_plan() {
    auto r = make_ascending();  // plan 0 = 30ns, plan 63 = 93ns
    assert(r.exhaustive.found());
    assert(r.exhaustive.optimal_id == 0);
    // cost = mean_ab = 30 + 0.25 = 30.25
    assert(std::abs(r.exhaustive.optimal_cost - 30.25) < 0.5);
    std::printf("  [09] exhaustive_finds_cheapest_plan               PASS\n");
}

// t10: exhaustive finds plan 63 on descending cost surface
static void t10_exhaustive_finds_last_when_descending() {
    auto r = make_descending();
    assert(r.exhaustive.found());
    assert(r.exhaustive.optimal_id == 63);
    std::printf("  [10] exhaustive_finds_last_when_descending        PASS\n");
}

// t11: ranked_corpus is sorted ascending by cost (64 entries)
static void t11_ranked_corpus_sorted_ascending() {
    auto r = make_ascending();
    assert(static_cast<int>(r.exhaustive.ranked_corpus.size()) == 64);
    for (std::size_t i = 1; i < r.exhaustive.ranked_corpus.size(); ++i) {
        assert(r.exhaustive.ranked_corpus[i].cost >=
               r.exhaustive.ranked_corpus[i-1].cost);
    }
    std::printf("  [11] ranked_corpus_sorted_ascending               PASS\n");
}

// t12: exhaustive returns not-found when cost table is empty (all zeros)
static void t12_exhaustive_not_found_on_empty_table() {
    CostTable<4, double> empty(trivial_index, 0.0);
    auto result = exhaustive_search<4>(empty);
    assert(!result.found());
    assert(result.n_measured == 0);
    std::printf("  [12] exhaustive_not_found_on_empty_table          PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Group D — beam_dp integration and agreement  (t13–t15)
// ─────────────────────────────────────────────────────────────────────────────

// t13: beam_dp agrees with exhaustive on ascending surface
static void t13_beam_agrees_ascending() {
    auto r = make_ascending();
    assert(r.beam_agrees());
    assert(r.beam.optimal_id == r.exhaustive.optimal_id);
    std::printf("  [13] beam_agrees_ascending                        PASS\n");
}

// t14: beam_dp agrees with exhaustive on descending surface
static void t14_beam_agrees_descending() {
    auto r = make_descending();
    assert(r.beam_agrees());
    assert(r.beam.optimal_id == r.exhaustive.optimal_id);
    std::printf("  [14] beam_agrees_descending                       PASS\n");
}

// t15: verify_beam_matches_exhaustive utility returns true on both surfaces
static void t15_verify_utility_both_surfaces() {
    auto r_asc  = make_ascending();
    auto r_desc = make_descending();
    assert(verify_beam_matches_exhaustive<4>(r_asc.cost_table,  4));
    assert(verify_beam_matches_exhaustive<4>(r_desc.cost_table, 4));
    // Flat surface: all costs equal → both solvers find some plan at rank 1.
    // verify may or may not agree (degenerate), but must not crash.
    auto r_flat = make_flat(42.0);
    (void)verify_beam_matches_exhaustive<4>(r_flat.cost_table, 4);
    std::printf("  [15] verify_utility_both_surfaces                 PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::printf("CT-DP FIX Parser — Phase 3 Calibration Driver Tests\n");
    std::printf("====================================================\n\n");

    std::printf("Group A — corpus skeleton construction\n");
    t01_corpus_has_64_plans();
    t02_all_plans_have_valid_dense_ids();
    t03_small_jitter_all_ok();
    t04_large_jitter_produces_hard_deleted();

    std::printf("\nGroup B — corpus_to_cost_table population\n");
    t05_cost_table_fully_populated();
    t06_cost_value_equals_mean_ab();
    t07_hard_deleted_excluded_from_cost_table();
    t08_b_run_excluded_from_cost();

    std::printf("\nGroup C — exhaustive_search integration\n");
    t09_exhaustive_finds_cheapest_plan();
    t10_exhaustive_finds_last_when_descending();
    t11_ranked_corpus_sorted_ascending();
    t12_exhaustive_not_found_on_empty_table();

    std::printf("\nGroup D — beam_dp integration and agreement\n");
    t13_beam_agrees_ascending();
    t14_beam_agrees_descending();
    t15_verify_utility_both_surfaces();

    std::printf("\n====================================================\n");
    std::printf("15/15 tests passed.\n");
    return 0;
}
