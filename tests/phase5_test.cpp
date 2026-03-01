// ctdp-calibrator Phase 5 integration test — Solver integration
//
// End-to-end: profile → cost_model → solver → plan
//
// Tests:
//   plan.h           — plan structure, queries, filtering
//   cost_model.h     — CostModel concept, profile adapter, composite
//   solver.h         — exhaustive, filtered, pareto, beam search
//   plan_builder.h   — convenience pipeline
//   Full pipeline    — scenario → dataset → profile → plan → validate

#include "calibration_scenarios.h"

#include <ctdp/bench/cache_thrasher.h>
#include <ctdp/bench/compiler_barrier.h>
#include <ctdp/bench/environment.h>
#include <ctdp/bench/measurement_kernel.h>
#include <ctdp/bench/metric.h>
#include <ctdp/bench/statistics.h>

#include <ctdp/calibrator/calibration_dataset.h>
#include <ctdp/calibrator/calibration_harness.h>
#include <ctdp/calibrator/calibration_profile.h>
#include <ctdp/calibrator/cost_model.h>
#include <ctdp/calibrator/csv_writer.h>
#include <ctdp/calibrator/data_point.h>
#include <ctdp/calibrator/feature_encoder.h>
#include <ctdp/calibrator/plan.h>
#include <ctdp/calibrator/plan_builder.h>
#include <ctdp/calibrator/plan_validate.h>
#include <ctdp/calibrator/provenance.h>
#include <ctdp/calibrator/scenario.h>
#include <ctdp/calibrator/solver.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// ═══════════════════════════════════════════════════════════════════
//  Test infrastructure
// ═══════════════════════════════════════════════════════════════════

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { ++tests_run; \
    std::printf("  [%d] %-55s", tests_run, #name); \
    try { test_##name(); std::printf("PASS\n"); ++tests_passed; } \
    catch (std::exception const& e) { std::printf("FAIL: %s\n", e.what()); } \
    catch (...) { std::printf("FAIL: unknown\n"); } } while(0)

#define ASSERT_TRUE(expr) \
    do { if(!(expr)) throw std::runtime_error("ASSERT_TRUE: " #expr); } while(0)
#define ASSERT_EQ(a,b) \
    do { if((a)!=(b)) { std::ostringstream o; o<<(a)<<" != "<<(b); \
    throw std::runtime_error(o.str()); } } while(0)
#define ASSERT_NEAR(a,b,t) \
    do { if(std::abs(double(a)-double(b))>(t)) { std::ostringstream o; \
    o<<(a)<<" vs "<<(b); throw std::runtime_error(o.str()); } } while(0)

// ═══════════════════════════════════════════════════════════════════
//  Type aliases & helpers
// ═══════════════════════════════════════════════════════════════════

namespace ex = ctdp::calibrator::examples;
namespace cal = ctdp::calibrator;
namespace bench = ctdp::bench;

using null_snap = bench::null_metric::null_snapshot;

struct parser_space {
    using point_type = ex::parser_point;
};

struct memory_space {
    using point_type = ex::memory_point;
};

/// Build a synthetic dataset with known costs for deterministic testing.
/// Cost model: cost = 10.0 * digits + strategy_offset
///   generic=0, loop=5, swar=2, unrolled=8
auto make_synthetic_dataset(int max_digits = 4)
    -> cal::calibration_dataset<parser_space, ex::integer_parser_tag, null_snap>
{
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> pts;

    auto strategy_offset = [](ex::parse_strategy s) -> double {
        switch (s) {
            case ex::parse_strategy::generic:  return 0.0;
            case ex::parse_strategy::loop:     return 5.0;
            case ex::parse_strategy::swar:     return 2.0;
            case ex::parse_strategy::unrolled: return 8.0;
        }
        return 0.0;
    };

    for (int d = 1; d <= max_digits; ++d) {
        for (auto s : {ex::parse_strategy::generic, ex::parse_strategy::loop,
                       ex::parse_strategy::swar, ex::parse_strategy::unrolled}) {
            double cost = 10.0 * d + strategy_offset(s);
            pts.push_back({
                .space_point   = {d, s},
                .median_ns     = cost,
                .mad_ns        = 0.1,
                .raw_timings   = {cost},
                .raw_snapshots = {null_snap{}},
                .env           = {}
            });
        }
    }

    cal::dataset_provenance prov;
    prov.scenario_name = "synthetic_parser";
    return cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(pts), prov);
}


// ═══════════════════════════════════════════════════════════════════
//  plan.h tests
// ═══════════════════════════════════════════════════════════════════

void test_plan_structure() {
    cal::plan<parser_space, ex::integer_parser_tag> p;
    ASSERT_TRUE(p.empty());
    ASSERT_EQ(p.size(), 0u);

    // Add entries
    p.entries.push_back({
        .point = {3, ex::parse_strategy::swar},
        .cost_ns = 32.0, .objectives = {32.0}, .pareto_optimal = true});
    p.entries.push_back({
        .point = {1, ex::parse_strategy::generic},
        .cost_ns = 10.0, .objectives = {10.0}, .pareto_optimal = false});
    p.entries.push_back({
        .point = {2, ex::parse_strategy::loop},
        .cost_ns = 25.0, .objectives = {25.0}, .pareto_optimal = false});

    p.sort_by_cost();

    ASSERT_EQ(p.size(), 3u);
    ASSERT_NEAR(p.optimal_cost(), 10.0, 1e-9);
    ASSERT_EQ(p.optimal_point().digits, 1);
}

void test_plan_queries() {
    cal::plan<parser_space, ex::integer_parser_tag> p;
    p.entries.push_back({{1, ex::parse_strategy::generic}, 10.0, {10.0}, true});
    p.entries.push_back({{2, ex::parse_strategy::swar},    22.0, {22.0}, false});
    p.entries.push_back({{3, ex::parse_strategy::loop},    35.0, {35.0}, false});
    p.entries.push_back({{4, ex::parse_strategy::unrolled},48.0, {48.0}, false});

    auto top2 = p.top_n(2);
    ASSERT_EQ(top2.size(), 2u);
    ASSERT_NEAR(top2[0].cost_ns, 10.0, 1e-9);
    ASSERT_NEAR(top2[1].cost_ns, 22.0, 1e-9);

    ASSERT_NEAR(p.predict({2, ex::parse_strategy::swar}), 22.0, 1e-9);
    ASSERT_NEAR(p.predict({99, ex::parse_strategy::generic}), -1.0, 1e-9);
}

void test_plan_pareto_queries() {
    cal::plan<parser_space, ex::integer_parser_tag> p;
    p.entries.push_back({{1, ex::parse_strategy::generic}, 10.0, {10.0, 100.0}, true});
    p.entries.push_back({{2, ex::parse_strategy::swar},    20.0, {20.0, 50.0}, true});
    p.entries.push_back({{3, ex::parse_strategy::loop},    15.0, {15.0, 80.0}, false});

    ASSERT_EQ(p.pareto_size(), 2u);
    auto frontier = p.pareto_frontier();
    ASSERT_EQ(frontier.size(), 2u);
}

void test_plan_filter() {
    cal::plan<parser_space, ex::integer_parser_tag> p;
    p.entries.push_back({{1, ex::parse_strategy::generic}, 10.0, {10.0}, true});
    p.entries.push_back({{2, ex::parse_strategy::swar},    22.0, {22.0}, false});
    p.entries.push_back({{3, ex::parse_strategy::loop},    35.0, {35.0}, false});

    auto swar_only = p.filtered([](auto const& pt) {
        return pt.strategy == ex::parse_strategy::swar;
    });
    ASSERT_EQ(swar_only.size(), 1u);
    ASSERT_EQ(swar_only.entries[0].point.digits, 2);
}

void test_plan_type_safety() {
    // Compile-time check: different Callables → different plan types
    static_assert(!std::is_same_v<
        cal::plan<parser_space, ex::integer_parser_tag>,
        cal::plan<memory_space, ex::memory_traverse_tag>>);

    // Same space, different callable → different plan
    struct alt_parser_tag {};
    static_assert(!std::is_same_v<
        cal::plan<parser_space, ex::integer_parser_tag>,
        cal::plan<parser_space, alt_parser_tag>>);
    ASSERT_TRUE(true);
}


// ═══════════════════════════════════════════════════════════════════
//  cost_model.h tests
// ═══════════════════════════════════════════════════════════════════

void test_profile_cost_model() {
    auto ds = make_synthetic_dataset(3);  // 12 points
    auto prof = cal::fit_lookup_profile(ds);

    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    // CostModel concept check
    static_assert(cal::CostModel<decltype(model), ex::parser_point>);

    // Known values: cost = 10*digits + strategy_offset
    ASSERT_NEAR(model.cost({1, ex::parse_strategy::generic}),  10.0, 1e-9);
    ASSERT_NEAR(model.cost({2, ex::parse_strategy::swar}),     22.0, 1e-9);
    ASSERT_NEAR(model.cost({3, ex::parse_strategy::unrolled}), 38.0, 1e-9);

    // Feasibility
    ASSERT_TRUE(model.feasible({1, ex::parse_strategy::generic}));
}

// File-scope encoder for linear model tests
struct digits_encoder {
    static constexpr std::size_t width = 1;
    static constexpr std::array<std::string_view, 1> column_names = {"digits"};
    [[nodiscard]] auto encode(ex::parser_point const& pt) const
        -> std::array<float, 1> { return {static_cast<float>(pt.digits)}; }
};

void test_linear_cost_model() {
    // Fit a linear model: cost = 10*digits + 2 (swar-only)
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> pts;
    for (int d = 1; d <= 10; ++d) {
        double cost = 10.0 * d + 2.0;
        pts.push_back({
            .space_point   = {d, ex::parse_strategy::swar},
            .median_ns     = cost,
            .mad_ns        = 0.01,
            .raw_timings   = {cost},
            .raw_snapshots = {null_snap{}},
            .env           = {}
        });
    }
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(pts));

    auto prof = cal::fit_linear_profile(ds, digits_encoder{});

    cal::linear_cost_model<parser_space, ex::integer_parser_tag, digits_encoder>
        model{prof, digits_encoder{}};

    static_assert(cal::CostModel<decltype(model), ex::parser_point>);

    ASSERT_NEAR(model.cost({5, ex::parse_strategy::swar}), 52.0, 0.5);
    ASSERT_NEAR(model.cost({15, ex::parse_strategy::swar}), 152.0, 0.5);
}

void test_constrained_cost_model() {
    auto ds = make_synthetic_dataset(4);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> base{prof};

    auto model = cal::make_constrained<ex::parser_point>(base,
        [](ex::parser_point const& pt) { return pt.digits <= 2; });

    static_assert(cal::ConstrainedCostModel<decltype(model), ex::parser_point>);

    ASSERT_TRUE(model.feasible({1, ex::parse_strategy::generic}));
    ASSERT_TRUE(model.feasible({2, ex::parse_strategy::swar}));
    ASSERT_TRUE(!model.feasible({3, ex::parse_strategy::generic}));
    ASSERT_NEAR(model.cost({2, ex::parse_strategy::generic}), 20.0, 1e-9);
}

void test_composite_cost_model() {
    auto ds = make_synthetic_dataset(3);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> latency{prof};

    // Second objective: "memory" = digits * 100
    cal::composite_cost_model<ex::parser_point> composite;
    composite.add_objective("latency", latency);
    composite.add_objective("memory",
        [](ex::parser_point const& pt) -> double {
            return static_cast<double>(pt.digits) * 100.0;
        });

    ASSERT_EQ(composite.dimension(), 2u);

    auto costs = composite.evaluate({2, ex::parse_strategy::generic});
    ASSERT_EQ(costs.size(), 2u);
    ASSERT_NEAR(costs[0], 20.0, 1e-9);  // latency
    ASSERT_NEAR(costs[1], 200.0, 1e-9); // memory

    auto names = composite.names();
    ASSERT_EQ(names[0], std::string("latency"));
    ASSERT_EQ(names[1], std::string("memory"));
}


// ═══════════════════════════════════════════════════════════════════
//  solver.h tests
// ═══════════════════════════════════════════════════════════════════

void test_exhaustive_search() {
    auto ds = make_synthetic_dataset(3);  // 12 points
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    ex::parser_strategy_scenario scenario(1, 3);
    auto p = cal::exhaustive_search<parser_space, ex::integer_parser_tag>(
        model, scenario, prof.provenance);

    ASSERT_EQ(p.size(), 12u);
    ASSERT_EQ(p.evaluated_points, 12u);
    ASSERT_EQ(p.solver_name, std::string("exhaustive_search"));

    // Cheapest should be digits=1, generic (cost = 10 + 0 = 10)
    ASSERT_NEAR(p.optimal_cost(), 10.0, 1e-9);
    ASSERT_EQ(p.optimal_point().digits, 1);
    ASSERT_TRUE(p.optimal_point().strategy == ex::parse_strategy::generic);

    // Most expensive: digits=3, unrolled (cost = 30 + 8 = 38)
    ASSERT_NEAR(p.entries.back().cost_ns, 38.0, 1e-9);
}

void test_filtered_search() {
    auto ds = make_synthetic_dataset(4);  // 16 points
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    auto const& pts = ds.points;
    std::vector<ex::parser_point> all_pts;
    all_pts.reserve(pts.size());
    for (auto const& dp : pts) all_pts.push_back(dp.space_point);

    // Only allow digits <= 2
    auto p = cal::filtered_search<parser_space, ex::integer_parser_tag>(
        model, all_pts,
        [](ex::parser_point const& pt) { return pt.digits <= 2; },
        prof.provenance);

    ASSERT_EQ(p.size(), 8u);  // 2 digits × 4 strategies
    ASSERT_EQ(p.solver_name, std::string("filtered_search"));
    ASSERT_NEAR(p.optimal_cost(), 10.0, 1e-9);  // digits=1, generic

    // No digits > 2 in plan
    for (auto const& e : p.entries) {
        ASSERT_TRUE(e.point.digits <= 2);
    }
}

void test_filtered_search_constrained_model() {
    auto ds = make_synthetic_dataset(4);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> base{prof};

    auto model = cal::make_constrained<ex::parser_point>(base,
        [](ex::parser_point const& pt) {
            return pt.strategy == ex::parse_strategy::swar;
        });

    std::vector<ex::parser_point> all_pts;
    for (auto const& dp : ds.points) all_pts.push_back(dp.space_point);

    auto p = cal::filtered_search<parser_space, ex::integer_parser_tag>(
        model, all_pts, prof.provenance);

    // Only swar points should survive
    ASSERT_EQ(p.size(), 4u);
    for (auto const& e : p.entries) {
        ASSERT_TRUE(e.point.strategy == ex::parse_strategy::swar);
    }
}

void test_pareto_search() {
    // Create a 2-objective scenario where latency and memory trade off
    auto ds = make_synthetic_dataset(3);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> latency{prof};

    cal::composite_cost_model<ex::parser_point> composite;
    composite.add_objective("latency", latency);
    // Memory: lower digits = more memory (inverse relationship for trade-off)
    composite.add_objective("memory",
        [](ex::parser_point const& pt) -> double {
            return 1000.0 / static_cast<double>(pt.digits);
        });

    std::vector<ex::parser_point> all_pts;
    for (auto const& dp : ds.points) all_pts.push_back(dp.space_point);

    auto p = cal::pareto_search<parser_space, ex::integer_parser_tag>(
        composite, all_pts, prof.provenance);

    ASSERT_EQ(p.size(), 12u);
    ASSERT_EQ(p.solver_name, std::string("pareto_search"));

    // Should have Pareto frontier members
    auto frontier = p.pareto_frontier();
    ASSERT_TRUE(frontier.size() >= 2u);  // at least 2 non-dominated

    // Each frontier point should have 2 objectives
    for (auto const& e : frontier) {
        ASSERT_EQ(e.objectives.size(), 2u);
        ASSERT_TRUE(e.pareto_optimal);
    }

    // The lowest-latency point (digits=1, generic) should be on frontier
    bool found_cheapest = false;
    for (auto const& e : frontier) {
        if (e.point.digits == 1
            && e.point.strategy == ex::parse_strategy::generic) {
            found_cheapest = true;
            ASSERT_NEAR(e.objectives[0], 10.0, 1e-9);
        }
    }
    ASSERT_TRUE(found_cheapest);
}

void test_beam_search() {
    auto ds = make_synthetic_dataset(4);  // 16 points
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    std::vector<ex::parser_point> all_pts;
    for (auto const& dp : ds.points) all_pts.push_back(dp.space_point);

    cal::beam_config cfg;
    cfg.beam_width = 5;

    auto p = cal::beam_search<parser_space, ex::integer_parser_tag>(
        model, all_pts, cfg, prof.provenance);

    ASSERT_EQ(p.size(), 5u);  // capped to beam_width
    ASSERT_EQ(p.solver_name, std::string("beam_search"));
    ASSERT_EQ(p.evaluated_points, 16u);  // all evaluated, just 5 kept

    // Still returns the optimal
    ASSERT_NEAR(p.optimal_cost(), 10.0, 1e-9);

    // All costs should be ≤ the 5th-cheapest cost
    double max_cost = p.entries.back().cost_ns;
    for (auto const& e : p.entries) {
        ASSERT_TRUE(e.cost_ns <= max_cost + 1e-9);
    }
}

void test_staged_beam_search() {
    // A simple 2-stage expander:
    // Stage 0: generates (digits, generic) for all digits
    // Stage 1: generates all strategies for each digits value
    struct parser_expander {
        int max_digits = 3;

        auto expand(std::vector<ex::parser_point> const& beam)
            -> std::vector<ex::parser_point>
        {
            std::vector<ex::parser_point> result;
            for (auto const& pt : beam) {
                if (pt.strategy == ex::parse_strategy::generic
                    && pt.digits == 0) {
                    // Stage 0: seed → all digits with generic
                    for (int d = 1; d <= max_digits; ++d) {
                        result.push_back({d, ex::parse_strategy::generic});
                    }
                } else {
                    // Stage 1: expand each digit to all strategies
                    for (auto s : {ex::parse_strategy::generic,
                                   ex::parse_strategy::loop,
                                   ex::parse_strategy::swar,
                                   ex::parse_strategy::unrolled}) {
                        result.push_back({pt.digits, s});
                    }
                }
            }
            return result;
        }

        bool complete(ex::parser_point const& pt) const {
            return pt.digits > 0
                && pt.strategy != ex::parse_strategy::generic;
        }
    };

    auto ds = make_synthetic_dataset(3);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    parser_expander expander;
    expander.max_digits = 3;

    // Seed: a single "empty" point
    std::vector<ex::parser_point> seed = {{0, ex::parse_strategy::generic}};

    cal::beam_config cfg;
    cfg.beam_width = 10;
    cfg.max_stages = 5;

    auto p = cal::staged_beam_search<parser_space, ex::integer_parser_tag>(
        model, expander, seed, cfg);

    ASSERT_EQ(p.solver_name, std::string("staged_beam_search"));
    ASSERT_TRUE(p.size() > 0u);
    // All entries should be "complete" (non-generic strategy)
    for (auto const& e : p.entries) {
        ASSERT_TRUE(e.point.strategy != ex::parse_strategy::generic);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  plan_builder.h tests
// ═══════════════════════════════════════════════════════════════════

void test_build_plan_from_scenario() {
    auto ds = make_synthetic_dataset(3);
    auto prof = cal::fit_lookup_profile(ds);
    ex::parser_strategy_scenario scenario(1, 3);

    auto p = cal::build_plan<parser_space, ex::integer_parser_tag>(
        prof, scenario);

    ASSERT_EQ(p.size(), 12u);
    ASSERT_NEAR(p.optimal_cost(), 10.0, 1e-9);
}

void test_build_plan_from_points() {
    auto ds = make_synthetic_dataset(2);
    auto prof = cal::fit_lookup_profile(ds);

    std::vector<ex::parser_point> pts;
    for (auto const& dp : ds.points) pts.push_back(dp.space_point);

    auto p = cal::build_plan<parser_space, ex::integer_parser_tag>(prof, pts);
    ASSERT_EQ(p.size(), 8u);
}

void test_build_plan_with_constraint() {
    auto ds = make_synthetic_dataset(4);
    auto prof = cal::fit_lookup_profile(ds);
    ex::parser_strategy_scenario scenario(1, 4);

    auto p = cal::build_plan<parser_space, ex::integer_parser_tag>(
        prof, scenario,
        [](ex::parser_point const& pt) {
            return pt.strategy == ex::parse_strategy::swar;
        });

    ASSERT_EQ(p.size(), 4u);
    for (auto const& e : p.entries) {
        ASSERT_TRUE(e.point.strategy == ex::parse_strategy::swar);
    }
}

void test_build_beam_plan() {
    auto ds = make_synthetic_dataset(4);
    auto prof = cal::fit_lookup_profile(ds);

    std::vector<ex::parser_point> pts;
    for (auto const& dp : ds.points) pts.push_back(dp.space_point);

    auto p = cal::build_beam_plan<parser_space, ex::integer_parser_tag>(
        prof, pts, {.beam_width = 3});

    ASSERT_EQ(p.size(), 3u);
    ASSERT_NEAR(p.optimal_cost(), 10.0, 1e-9);
}

void test_build_linear_plan() {
    // Build a linear profile and use it in the plan builder
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> pts;
    for (int d = 1; d <= 6; ++d) {
        double cost = 10.0 * d + 5.0;
        pts.push_back({
            .space_point   = {d, ex::parse_strategy::swar},
            .median_ns     = cost,
            .mad_ns        = 0.01,
            .raw_timings   = {cost},
            .raw_snapshots = {null_snap{}},
            .env           = {}
        });
    }

    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(pts));

    auto prof = cal::fit_linear_profile(ds, digits_encoder{});

    std::vector<ex::parser_point> eval_pts;
    for (int d = 1; d <= 6; ++d) {
        eval_pts.push_back({d, ex::parse_strategy::swar});
    }

    auto p = cal::build_linear_plan<parser_space, ex::integer_parser_tag>(
        prof, eval_pts, digits_encoder{});

    ASSERT_EQ(p.size(), 6u);
    // Cheapest: digits=1 (cost ≈ 15)
    ASSERT_NEAR(p.optimal_cost(), 15.0, 1.0);
    ASSERT_EQ(p.optimal_point().digits, 1);
}

void test_build_pareto_plan() {
    auto ds = make_synthetic_dataset(3);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> latency{prof};

    std::vector<ex::parser_point> all_pts;
    for (auto const& dp : ds.points) all_pts.push_back(dp.space_point);

    // Memory objective: inverse of digits (smaller digits = more memory)
    struct memory_model {
        double cost(ex::parser_point const& pt) const {
            return 1000.0 / static_cast<double>(pt.digits);
        }
    };

    auto p = cal::build_pareto_plan<parser_space, ex::integer_parser_tag>(
        all_pts, latency, memory_model{},
        "latency_ns", "memory_bytes");

    ASSERT_EQ(p.size(), 12u);
    ASSERT_TRUE(p.pareto_size() >= 2u);
}


// ═══════════════════════════════════════════════════════════════════
//  Full pipeline: scenario → dataset → profile → plan → validate
// ═══════════════════════════════════════════════════════════════════

/// Helper: run a scenario through harness and return data_points
template <cal::Scenario S>
auto measure_scenario(S& scenario, std::size_t reps = 5)
    -> std::vector<cal::data_point<typename S::point_type, null_snap>>
{
    using pt_t = typename S::point_type;
    using dp_t = cal::data_point<pt_t, null_snap>;

    bench::null_metric nm;
    std::vector<dp_t> results;
    results.reserve(scenario.points().size());

    for (auto const& pt : scenario.points()) {
        scenario.prepare(pt);
        auto fn = [&]() -> bench::result_token {
            return scenario.execute(pt);
        };
        auto meas = bench::measure_repeated(fn, []{}, nm, reps, 10, 1);
        results.push_back(dp_t{
            .space_point    = pt,
            .median_ns      = meas.median_ns,
            .mad_ns         = meas.mad_ns,
            .raw_timings    = std::move(meas.all_ns),
            .raw_snapshots  = std::move(meas.all_snapshots),
            .env            = bench::capture_environment()
        });
    }
    return results;
}

void test_full_pipeline_parser() {
    // 1. Create scenario (3 digits × 4 strategies = 12 points)
    ex::parser_strategy_scenario scenario(1, 3);
    ASSERT_EQ(scenario.points().size(), 12u);

    // 2. Measure
    auto data = measure_scenario(scenario, 3);

    // 3. Dataset + provenance
    auto env = bench::capture_environment();
    auto prov = cal::capture_provenance(scenario, env, 3);
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(data), prov);

    // 4. Profile
    auto prof = cal::fit_lookup_profile(ds);
    ASSERT_EQ(prof.training_points, 12u);

    // 5. Build plan
    auto p = cal::build_plan<parser_space, ex::integer_parser_tag>(
        prof, scenario);

    ASSERT_EQ(p.size(), 12u);
    ASSERT_TRUE(p.optimal_cost() > 0.0);

    // 6. The optimal point should be predictable
    auto best = p.optimal_point();
    double predicted = p.predict(best);
    ASSERT_TRUE(predicted > 0.0);
    ASSERT_NEAR(predicted, p.optimal_cost(), 1e-9);

    // 7. Validate plan predictions against fresh measurements
    cal::validation_config vcfg;
    vcfg.reps = 3;
    vcfg.tolerance = 0.80;  // generous for container noise
    vcfg.flush_cache = false;

    auto vr = cal::validate_profile<parser_space, ex::integer_parser_tag>(
        prof, scenario, vcfg);
    ASSERT_EQ(vr.total_points, 12u);
    ASSERT_TRUE(vr.pass_rate() > 0.2);  // at least some pass
}

void test_full_pipeline_memory() {
    ex::memory_regime_scenario scenario(16, 128, 4);
    auto data = measure_scenario(scenario, 3);
    auto ds = cal::make_dataset<memory_space, ex::memory_traverse_tag, null_snap>(
        std::move(data));
    auto prof = cal::fit_lookup_profile(ds);

    auto const& pts = scenario.points();
    std::vector<ex::memory_point> vec(pts.begin(), pts.end());
    auto p = cal::build_plan<memory_space, ex::memory_traverse_tag>(prof, vec);

    ASSERT_EQ(p.size(), 4u);
    ASSERT_TRUE(p.optimal_cost() > 0.0);
    // Smallest working set should be cheapest
    ASSERT_TRUE(p.optimal_point().bytes <= 32 * 1024);
}


// ═══════════════════════════════════════════════════════════════════
//  Edge cases
// ═══════════════════════════════════════════════════════════════════

void test_empty_space() {
    auto ds = make_synthetic_dataset(2);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    std::vector<ex::parser_point> empty_pts;
    auto p = cal::exhaustive_search<parser_space, ex::integer_parser_tag>(
        model, empty_pts);

    ASSERT_TRUE(p.empty());
    ASSERT_EQ(p.size(), 0u);
    ASSERT_EQ(p.evaluated_points, 0u);
}

void test_single_point() {
    auto ds = make_synthetic_dataset(1);  // 4 strategies for digits=1
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    std::vector<ex::parser_point> one_pt = {{1, ex::parse_strategy::swar}};
    auto p = cal::exhaustive_search<parser_space, ex::integer_parser_tag>(
        model, one_pt);

    ASSERT_EQ(p.size(), 1u);
    ASSERT_NEAR(p.optimal_cost(), 12.0, 1e-9);  // 10*1 + 2
    ASSERT_TRUE(p.entries[0].pareto_optimal);
}

void test_all_filtered_out() {
    auto ds = make_synthetic_dataset(2);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> model{prof};

    std::vector<ex::parser_point> pts;
    for (auto const& dp : ds.points) pts.push_back(dp.space_point);

    auto p = cal::filtered_search<parser_space, ex::integer_parser_tag>(
        model, pts,
        [](ex::parser_point const&) { return false; });  // reject all

    ASSERT_TRUE(p.empty());
    ASSERT_EQ(p.evaluated_points, 0u);
}


// ═══════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("ctdp-calibrator Phase 5 solver integration test\n");
    std::printf("═════════════════════════════════════════════════════════\n\n");

    std::printf("plan.h:\n");
    TEST(plan_structure);
    TEST(plan_queries);
    TEST(plan_pareto_queries);
    TEST(plan_filter);
    TEST(plan_type_safety);

    std::printf("\ncost_model.h:\n");
    TEST(profile_cost_model);
    TEST(linear_cost_model);
    TEST(constrained_cost_model);
    TEST(composite_cost_model);

    std::printf("\nsolver.h:\n");
    TEST(exhaustive_search);
    TEST(filtered_search);
    TEST(filtered_search_constrained_model);
    TEST(pareto_search);
    TEST(beam_search);
    TEST(staged_beam_search);

    std::printf("\nplan_builder.h:\n");
    TEST(build_plan_from_scenario);
    TEST(build_plan_from_points);
    TEST(build_plan_with_constraint);
    TEST(build_beam_plan);
    TEST(build_linear_plan);
    TEST(build_pareto_plan);

    std::printf("\nFull pipeline:\n");
    TEST(full_pipeline_parser);
    TEST(full_pipeline_memory);

    std::printf("\nEdge cases:\n");
    TEST(empty_space);
    TEST(single_point);
    TEST(all_filtered_out);

    std::printf("\n═════════════════════════════════════════════════════════\n");
    std::printf("%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
