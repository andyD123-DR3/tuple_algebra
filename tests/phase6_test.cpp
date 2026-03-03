// ctdp-calibrator Phase 6 test — Code Instantiation
//
// Tests:
//   dispatch_table.h  — dispatch table construction, lookup, dispatch
//   plan_emit.h       — constexpr header emission from plans
//   wisdom.h          — FFTW-style serialise/deserialise plans
//   Full pipeline     — scenario → plan → dispatch table → wisdom round-trip

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
#include <ctdp/calibrator/data_point.h>
#include <ctdp/calibrator/dispatch_table.h>
#include <ctdp/calibrator/plan.h>
#include <ctdp/calibrator/plan_builder.h>
#include <ctdp/calibrator/plan_emit.h>
#include <ctdp/calibrator/scenario.h>
#include <ctdp/calibrator/solver.h>
#include <ctdp/calibrator/wisdom.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <optional>
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
//  Aliases & helpers
// ═══════════════════════════════════════════════════════════════════

namespace ex = ctdp::calibrator::examples;
namespace cal = ctdp::calibrator;
namespace bench = ctdp::bench;

using null_snap = bench::null_metric::null_snapshot;

struct parser_space {
    using point_type = ex::parser_point;
};

/// Deterministic synthetic dataset: cost = 10*digits + strategy_offset
auto make_synthetic_dataset(int max_digits = 4)
    -> cal::calibration_dataset<parser_space, ex::integer_parser_tag, null_snap>
{
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> pts;

    auto offset = [](ex::parse_strategy s) -> double {
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
            double cost = 10.0 * d + offset(s);
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
    prov.hostname = "testhost";
    prov.compiler = "g++ 13";
    return cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(pts), prov);
}

/// Build a test plan from synthetic data.
auto make_test_plan(int max_digits = 3)
    -> cal::plan<parser_space, ex::integer_parser_tag>
{
    auto ds = make_synthetic_dataset(max_digits);
    auto prof = cal::fit_lookup_profile(ds);
    ex::parser_strategy_scenario scenario(1, max_digits);
    return cal::build_plan<parser_space, ex::integer_parser_tag>(prof, scenario);
}

/// Parser point serialiser for wisdom/emit.
struct parser_point_serialiser {
    std::string operator()(ex::parser_point const& pt) const {
        return "{" + std::to_string(pt.digits) + ","
               + std::to_string(static_cast<int>(pt.strategy)) + "}";
    }
};

/// Parser for wisdom round-trip.
auto parse_parser_point(std::string const& s)
    -> std::optional<ex::parser_point>
{
    // Format: {digits,strategy_int}
    if (s.size() < 3 || s.front() != '{' || s.back() != '}') {
        return std::nullopt;
    }
    auto inner = s.substr(1, s.size() - 2);
    auto comma = inner.find(',');
    if (comma == std::string::npos) return std::nullopt;

    int digits = std::stoi(inner.substr(0, comma));
    int strat  = std::stoi(inner.substr(comma + 1));
    return ex::parser_point{digits, static_cast<ex::parse_strategy>(strat)};
}


// ═══════════════════════════════════════════════════════════════════
//  dispatch_table.h tests
// ═══════════════════════════════════════════════════════════════════

void test_dispatch_table_construction() {
    auto plan = make_test_plan(3);  // 12 entries
    auto table = cal::make_dispatch_table(plan);

    ASSERT_EQ(table.size(), 12u);
    ASSERT_TRUE(!table.empty());
    ASSERT_NEAR(table.optimal_cost(), 10.0, 1e-9);
    ASSERT_EQ(table.optimal_point().digits, 1);
}

void test_dispatch_table_lookup() {
    auto plan = make_test_plan(3);
    auto table = cal::make_dispatch_table(plan);

    // Known point
    auto* e = table.lookup({2, ex::parse_strategy::swar});
    ASSERT_TRUE(e != nullptr);
    ASSERT_NEAR(e->cost_ns, 22.0, 1e-9);
    ASSERT_EQ(e->rank, 5u);  // rank depends on sort order

    // Unknown point
    auto* missing = table.lookup({99, ex::parse_strategy::generic});
    ASSERT_TRUE(missing == nullptr);

    // predict
    ASSERT_NEAR(table.predict({1, ex::parse_strategy::generic}), 10.0, 1e-9);
    ASSERT_NEAR(table.predict({99, ex::parse_strategy::generic}), -1.0, 1e-9);
}

void test_dispatch_table_ranked() {
    auto plan = make_test_plan(2);  // 8 entries
    auto table = cal::make_dispatch_table(plan);

    auto const& ranked = table.ranked();
    ASSERT_EQ(ranked.size(), 8u);

    // Verify monotonically increasing cost
    for (std::size_t i = 1; i < ranked.size(); ++i) {
        ASSERT_TRUE(ranked[i].cost_ns >= ranked[i-1].cost_ns);
    }

    // Verify rank assignments
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        ASSERT_EQ(ranked[i].rank, i);
    }

    // top_n
    auto top3 = table.top_n(3);
    ASSERT_EQ(top3.size(), 3u);
    ASSERT_NEAR(top3[0].cost_ns, 10.0, 1e-9);  // digits=1, generic

    // at_rank
    ASSERT_NEAR(table.at_rank(0).cost_ns, 10.0, 1e-9);
}

void test_dispatch_table_dispatch() {
    auto plan = make_test_plan(2);
    auto table = cal::make_dispatch_table(plan);

    // dispatch optimal
    bool called = false;
    table.dispatch([&](ex::parser_point const& pt, double cost) {
        called = true;
        ASSERT_EQ(pt.digits, 1);
        ASSERT_NEAR(cost, 10.0, 1e-9);
    });
    ASSERT_TRUE(called);

    // dispatch_rank
    called = false;
    table.dispatch_rank(1, [&](ex::parser_point const&, double cost) {
        called = true;
        ASSERT_NEAR(cost, 12.0, 1e-9);  // digits=1, swar (offset=2)
    });
    ASSERT_TRUE(called);

    // dispatch_if
    int swar_count = 0;
    table.dispatch_if(
        [](ex::parser_point const& pt) {
            return pt.strategy == ex::parse_strategy::swar;
        },
        [&](ex::parser_point const&, double) { ++swar_count; });
    ASSERT_EQ(swar_count, 2);  // digits 1 and 2
}

void test_dispatch_table_filter() {
    auto plan = make_test_plan(3);
    auto table = cal::make_dispatch_table(plan);

    auto swar_table = table.filtered([](ex::parser_point const& pt) {
        return pt.strategy == ex::parse_strategy::swar;
    });

    ASSERT_EQ(swar_table.size(), 3u);  // digits 1, 2, 3
    ASSERT_NEAR(swar_table.optimal_cost(), 12.0, 1e-9);

    // Ranks should be re-assigned
    for (std::size_t i = 0; i < swar_table.size(); ++i) {
        ASSERT_EQ(swar_table.ranked()[i].rank, i);
    }
}

void test_dispatch_table_pareto() {
    // Build a multi-objective plan with Pareto entries
    auto ds = make_synthetic_dataset(3);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> latency{prof};

    cal::composite_cost_model<ex::parser_point> composite;
    composite.add_objective("latency", latency);
    composite.add_objective("memory",
        [](ex::parser_point const& pt) -> double {
            return 1000.0 / static_cast<double>(pt.digits);
        });

    std::vector<ex::parser_point> all_pts;
    for (auto const& dp : ds.points) all_pts.push_back(dp.space_point);

    auto plan = cal::pareto_search<parser_space, ex::integer_parser_tag>(
        composite, all_pts);
    auto table = cal::make_dispatch_table(plan);

    auto frontier = table.pareto_frontier();
    ASSERT_TRUE(frontier.size() >= 2u);
    for (auto const& e : frontier) {
        ASSERT_TRUE(e.pareto_optimal);
    }
}

void test_dispatch_table_summary() {
    auto plan = make_test_plan(2);
    auto table = cal::make_dispatch_table(plan);

    auto s = table.summary();
    ASSERT_TRUE(s.find("8 entries") != std::string::npos);
    ASSERT_TRUE(s.find("10.00") != std::string::npos);
    ASSERT_TRUE(s.find("dispatch_table") != std::string::npos);
}

void test_dispatch_table_empty() {
    cal::plan<parser_space, ex::integer_parser_tag> empty_plan;
    auto table = cal::make_dispatch_table(empty_plan);

    ASSERT_TRUE(table.empty());
    ASSERT_EQ(table.size(), 0u);
    ASSERT_NEAR(table.predict({1, ex::parse_strategy::generic}), -1.0, 1e-9);
}


// ═══════════════════════════════════════════════════════════════════
//  plan_emit.h tests
// ═══════════════════════════════════════════════════════════════════

void test_plan_emit_header() {
    auto plan = make_test_plan(2);  // 8 entries
    std::ostringstream out;
    cal::emit_plan_header(out, plan,
        "parser_plan", "parser_space", "integer_parser_tag",
        parser_point_serialiser{});

    auto code = out.str();

    // Header guard
    ASSERT_TRUE(code.find("#ifndef CTDP_WISDOM_PARSER_PLAN_H") != std::string::npos);
    ASSERT_TRUE(code.find("#define CTDP_WISDOM_PARSER_PLAN_H") != std::string::npos);

    // Namespace
    ASSERT_TRUE(code.find("namespace ctdp::wisdom") != std::string::npos);

    // Entry struct
    ASSERT_TRUE(code.find("struct parser_plan_entry") != std::string::npos);

    // Array
    ASSERT_TRUE(code.find("parser_plan_N = 8") != std::string::npos);
    ASSERT_TRUE(code.find("parser_plan_entries") != std::string::npos);
    ASSERT_TRUE(code.find("constexpr") != std::string::npos);

    // Optimal cost
    ASSERT_TRUE(code.find("parser_plan_optimal_cost") != std::string::npos);
    ASSERT_TRUE(code.find("10.000000") != std::string::npos);

    // Solver name
    ASSERT_TRUE(code.find("exhaustive_search") != std::string::npos);

    // Point serialisation in comments
    ASSERT_TRUE(code.find("{1,0}") != std::string::npos);

    // Lookup helper
    ASSERT_TRUE(code.find("parser_plan_predict") != std::string::npos);
}

void test_plan_emit_default_serialiser() {
    auto plan = make_test_plan(1);  // 4 entries
    std::ostringstream out;
    cal::emit_plan_header(out, plan,
        "test", "parser_space", "integer_parser_tag");

    auto code = out.str();
    ASSERT_TRUE(code.find("provide PointSerialiser") != std::string::npos);
    ASSERT_TRUE(code.find("test_N = 4") != std::string::npos);
}

void test_plan_emit_provenance() {
    auto plan = make_test_plan(1);
    plan.provenance.hostname = "buildhost42";
    plan.provenance.compiler = "g++ 14.2";

    std::ostringstream out;
    cal::emit_plan_header(out, plan,
        "x", "S", "C", parser_point_serialiser{});

    auto code = out.str();
    ASSERT_TRUE(code.find("buildhost42") != std::string::npos);
    ASSERT_TRUE(code.find("g++ 14.2") != std::string::npos);
}

void test_plan_emit_pareto() {
    // Build a plan with Pareto entries
    auto ds = make_synthetic_dataset(2);
    auto prof = cal::fit_lookup_profile(ds);
    cal::profile_cost_model<parser_space, ex::integer_parser_tag> latency{prof};

    cal::composite_cost_model<ex::parser_point> composite;
    composite.add_objective("latency", latency);
    composite.add_objective("memory",
        [](ex::parser_point const& pt) -> double {
            return 1000.0 / static_cast<double>(pt.digits);
        });

    std::vector<ex::parser_point> all_pts;
    for (auto const& dp : ds.points) all_pts.push_back(dp.space_point);

    auto plan = cal::pareto_search<parser_space, ex::integer_parser_tag>(
        composite, all_pts);

    std::ostringstream out;
    cal::emit_plan_header(out, plan,
        "pareto_test", "S", "C", parser_point_serialiser{});

    auto code = out.str();
    ASSERT_TRUE(code.find("pareto_test_pareto_count") != std::string::npos);
    ASSERT_TRUE(code.find("true") != std::string::npos);  // at least one Pareto
}


// ═══════════════════════════════════════════════════════════════════
//  wisdom.h tests
// ═══════════════════════════════════════════════════════════════════

void test_wisdom_write() {
    auto plan = make_test_plan(2);  // 8 entries

    std::ostringstream out;
    cal::write_wisdom(out, plan, parser_point_serialiser{});

    auto text = out.str();
    ASSERT_TRUE(text.find("# ctdp wisdom v1") != std::string::npos);
    ASSERT_TRUE(text.find("# solver: exhaustive_search") != std::string::npos);
    ASSERT_TRUE(text.find("# entries: 8") != std::string::npos);
    ASSERT_TRUE(text.find("# optimal_cost: 10.000000") != std::string::npos);
    ASSERT_TRUE(text.find("---") != std::string::npos);

    // Data lines
    ASSERT_TRUE(text.find("0 10.000000 1 {1,0}") != std::string::npos);
}

void test_wisdom_read() {
    auto plan = make_test_plan(2);

    std::ostringstream out;
    cal::write_wisdom(out, plan, parser_point_serialiser{});

    std::istringstream in(out.str());
    auto wf = cal::read_wisdom(in);

    ASSERT_TRUE(wf.valid());
    ASSERT_EQ(wf.size(), 8u);
    ASSERT_EQ(wf.metadata.version, 1);
    ASSERT_EQ(wf.metadata.solver, std::string("exhaustive_search"));
    ASSERT_EQ(wf.metadata.entry_count, 8u);
    ASSERT_NEAR(wf.metadata.optimal_cost, 10.0, 1e-5);

    // First entry
    ASSERT_EQ(wf.entries[0].rank, 0u);
    ASSERT_NEAR(wf.entries[0].cost_ns, 10.0, 1e-5);
    ASSERT_TRUE(wf.entries[0].pareto_optimal);
    ASSERT_TRUE(wf.entries[0].point_str.find("{1,0}") != std::string::npos);
}

void test_wisdom_round_trip() {
    auto original = make_test_plan(3);  // 12 entries

    // Write
    std::ostringstream out;
    cal::write_wisdom(out, original, parser_point_serialiser{});

    // Read
    std::istringstream in(out.str());
    auto wf = cal::read_wisdom(in);
    ASSERT_TRUE(wf.valid());

    // Reconstruct
    auto restored = cal::reconstruct_plan<parser_space, ex::integer_parser_tag>(
        wf, [](std::string const& s)
            -> std::optional<ex::parser_point>
        {
            return parse_parser_point(s);
        });

    ASSERT_EQ(restored.entries.size(), original.entries.size());
    ASSERT_EQ(restored.solver_name, original.solver_name);
    ASSERT_EQ(restored.provenance.scenario_name,
              original.provenance.scenario_name);

    // Verify costs match
    for (std::size_t i = 0; i < original.entries.size(); ++i) {
        ASSERT_NEAR(restored.entries[i].cost_ns,
                    original.entries[i].cost_ns, 1e-4);
        ASSERT_TRUE(restored.entries[i].point == original.entries[i].point);
        ASSERT_EQ(restored.entries[i].pareto_optimal,
                  original.entries[i].pareto_optimal);
    }
}

void test_wisdom_read_string() {
    std::string text =
        "# ctdp wisdom v1\n"
        "# solver: test_solver\n"
        "# scenario: test_scenario\n"
        "# entries: 2\n"
        "# optimal_cost: 5.000000\n"
        "---\n"
        "0 5.000000 1 {1,0}\n"
        "1 15.000000 0 {2,1}\n";

    auto wf = cal::read_wisdom_string(text);
    ASSERT_TRUE(wf.valid());
    ASSERT_EQ(wf.size(), 2u);
    ASSERT_EQ(wf.metadata.solver, std::string("test_solver"));
    ASSERT_EQ(wf.metadata.scenario, std::string("test_scenario"));
    ASSERT_NEAR(wf.entries[0].cost_ns, 5.0, 1e-5);
    ASSERT_NEAR(wf.entries[1].cost_ns, 15.0, 1e-5);
}

void test_wisdom_invalid() {
    std::string text = "garbage data\nnot a wisdom file\n";
    auto wf = cal::read_wisdom_string(text);
    ASSERT_TRUE(!wf.valid());
}

void test_wisdom_provenance_preservation() {
    auto plan = make_test_plan(1);
    plan.provenance.hostname = "prod-server-42";
    plan.provenance.compiler = "clang 18.1";
    plan.provenance.scenario_name = "fix_parser_benchmark";

    std::ostringstream out;
    cal::write_wisdom(out, plan, parser_point_serialiser{});

    auto wf = cal::read_wisdom_string(out.str());
    ASSERT_EQ(wf.metadata.hostname, std::string("prod-server-42"));
    ASSERT_EQ(wf.metadata.compiler, std::string("clang 18.1"));
    ASSERT_EQ(wf.metadata.scenario, std::string("fix_parser_benchmark"));
}


// ═══════════════════════════════════════════════════════════════════
//  Full pipeline: scenario → plan → dispatch → wisdom → reconstruct
// ═══════════════════════════════════════════════════════════════════

template <cal::Scenario S>
auto measure_scenario(S& scenario, std::size_t reps = 3)
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
        auto meas = bench::measure_repeated(fn, []{}, nm, reps, 10, 100);
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

void test_full_pipeline_dispatch() {
    // 1. Scenario
    ex::parser_strategy_scenario scenario(1, 3);

    // 2. Measure → dataset → profile → plan
    auto data = measure_scenario(scenario, 3);
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(data));
    auto prof = cal::fit_lookup_profile(ds);
    auto plan = cal::build_plan<parser_space, ex::integer_parser_tag>(
        prof, scenario);

    ASSERT_EQ(plan.size(), 12u);

    // 3. Dispatch table
    auto table = cal::make_dispatch_table(plan);
    ASSERT_EQ(table.size(), 12u);
    ASSERT_TRUE(table.optimal_cost() > 0.0);

    // 4. Dispatch — call visitor with optimal point
    bool dispatched = false;
    table.dispatch([&](ex::parser_point const& pt, double cost) {
        dispatched = true;
        ASSERT_TRUE(pt.digits >= 1);
        ASSERT_TRUE(cost > 0.0);
    });
    ASSERT_TRUE(dispatched);

    // 5. Emit constexpr header
    std::ostringstream emit_out;
    cal::emit_plan_header(emit_out, plan,
        "parser_wisdom", "parser_space", "integer_parser_tag",
        parser_point_serialiser{});
    auto code = emit_out.str();
    ASSERT_TRUE(code.find("parser_wisdom_N = 12") != std::string::npos);

    // 6. Write wisdom
    std::ostringstream wisdom_out;
    cal::write_wisdom(wisdom_out, plan, parser_point_serialiser{});
    auto wisdom_text = wisdom_out.str();
    ASSERT_TRUE(wisdom_text.find("# ctdp wisdom v1") != std::string::npos);

    // 7. Round-trip: read wisdom → reconstruct → dispatch table
    auto wf = cal::read_wisdom_string(wisdom_text);
    ASSERT_TRUE(wf.valid());
    auto restored = cal::reconstruct_plan<parser_space, ex::integer_parser_tag>(
        wf, [](std::string const& s) { return parse_parser_point(s); });

    auto table2 = cal::make_dispatch_table(restored);
    ASSERT_EQ(table2.size(), 12u);
    ASSERT_NEAR(table2.optimal_cost(), table.optimal_cost(), 1e-4);
}

void test_full_pipeline_constrained_dispatch() {
    auto ds = make_synthetic_dataset(4);
    auto prof = cal::fit_lookup_profile(ds);
    ex::parser_strategy_scenario scenario(1, 4);

    // Build constrained plan (only swar)
    auto plan = cal::build_plan<parser_space, ex::integer_parser_tag>(
        prof, scenario,
        [](ex::parser_point const& pt) {
            return pt.strategy == ex::parse_strategy::swar;
        });

    ASSERT_EQ(plan.size(), 4u);

    auto table = cal::make_dispatch_table(plan);
    ASSERT_EQ(table.size(), 4u);
    ASSERT_NEAR(table.optimal_cost(), 12.0, 1e-9);  // digits=1, swar=10+2

    // Filtered dispatch table further
    auto small = table.filtered([](ex::parser_point const& pt) {
        return pt.digits <= 2;
    });
    ASSERT_EQ(small.size(), 2u);
}

void test_full_pipeline_type_chain() {
    // Verify the complete type chain from scenario to dispatch_table
    using scenario_t  = ex::parser_strategy_scenario;
    using callable_t  = scenario_t::callable_type;
    using dataset_t   = cal::calibration_dataset<parser_space, callable_t, null_snap>;
    using profile_t   = cal::calibration_profile<parser_space, callable_t>;
    using plan_t      = cal::plan<parser_space, callable_t>;
    using table_t     = cal::dispatch_table<parser_space, callable_t>;

    // All share the same callable_type
    static_assert(std::is_same_v<callable_t, ex::integer_parser_tag>);
    static_assert(std::is_same_v<dataset_t::callable_type, callable_t>);
    static_assert(std::is_same_v<profile_t::callable_type, callable_t>);
    static_assert(std::is_same_v<plan_t::callable_type, callable_t>);
    static_assert(std::is_same_v<table_t::callable_type, callable_t>);

    ASSERT_TRUE(true);
}


// ═══════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("ctdp-calibrator Phase 6 code instantiation test\n");
    std::printf("═════════════════════════════════════════════════════════\n\n");

    std::printf("dispatch_table.h:\n");
    TEST(dispatch_table_construction);
    TEST(dispatch_table_lookup);
    TEST(dispatch_table_ranked);
    TEST(dispatch_table_dispatch);
    TEST(dispatch_table_filter);
    TEST(dispatch_table_pareto);
    TEST(dispatch_table_summary);
    TEST(dispatch_table_empty);

    std::printf("\nplan_emit.h:\n");
    TEST(plan_emit_header);
    TEST(plan_emit_default_serialiser);
    TEST(plan_emit_provenance);
    TEST(plan_emit_pareto);

    std::printf("\nwisdom.h:\n");
    TEST(wisdom_write);
    TEST(wisdom_read);
    TEST(wisdom_round_trip);
    TEST(wisdom_read_string);
    TEST(wisdom_invalid);
    TEST(wisdom_provenance_preservation);

    std::printf("\nFull pipeline:\n");
    TEST(full_pipeline_dispatch);
    TEST(full_pipeline_constrained_dispatch);
    TEST(full_pipeline_type_chain);

    std::printf("\n═════════════════════════════════════════════════════════\n");
    std::printf("%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
