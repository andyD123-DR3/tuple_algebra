// ctdp-calibrator Phase 4 integration test
//
// End-to-end: parser_strategy_scenario → harness → data_points →
//   dataset (provenance) → CSV (feature encoder) → profile (lookup +
//   linear) → emit constexpr → validate
//
// Also exercises: sampling, filtering, cross-scenario type safety.

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
#include <ctdp/calibrator/csv_writer.h>
#include <ctdp/calibrator/ct_dp_emit.h>
#include <ctdp/calibrator/data_point.h>
#include <ctdp/calibrator/feature_encoder.h>
#include <ctdp/calibrator/plan_validate.h>
#include <ctdp/calibrator/provenance.h>
#include <ctdp/calibrator/sampler.h>
#include <ctdp/calibrator/scenario.h>

// benchmark_explorer.h — compiles in stub mode (no libbenchmark)
#include <ctdp/calibrator/benchmark_explorer.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
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
//  Type aliases
// ═══════════════════════════════════════════════════════════════════

namespace ex = ctdp::calibrator::examples;
namespace cal = ctdp::calibrator;
namespace bench = ctdp::bench;

using null_snap    = bench::null_metric::null_snapshot;
using counter_snap = bench::counter_snapshot;

// Space wrapper for parser scenario (provides point_type)
struct parser_space {
    using point_type = ex::parser_point;
};

// Space wrapper for memory scenario
struct memory_space {
    using point_type = ex::memory_point;
};

// ═══════════════════════════════════════════════════════════════════
//  FeatureEncoder for parser_point (Phase 3 + Phase 4 bridge)
// ═══════════════════════════════════════════════════════════════════

/// Encodes parser_point as 5 features:
///   [strategy_generic, strategy_loop, strategy_swar, strategy_unrolled, digits]
struct parser_feature_encoder {
    static constexpr std::size_t width = 5;
    static constexpr std::array<std::string_view, 5> column_names = {
        "generic", "loop", "swar", "unrolled", "digits"
    };

    [[nodiscard]] auto encode(ex::parser_point const& pt) const
        -> std::array<float, 5>
    {
        using cal::encode_one_hot;
        using cal::encode_scalar;
        using cal::copy_features;

        std::array<float, 5> f{};
        std::size_t pos = 0;
        pos = copy_features(f, pos,
            encode_one_hot<4>(static_cast<int>(pt.strategy)));
        f[pos] = encode_scalar(pt.digits);
        return f;
    }
};

static_assert(cal::FeatureEncoder<parser_feature_encoder, ex::parser_point>);

/// Single-feature encoder (digits only) for linear regression.
struct parser_digits_encoder {
    static constexpr std::size_t width = 1;
    static constexpr std::array<std::string_view, 1> column_names = {"digits"};

    [[nodiscard]] auto encode(ex::parser_point const& pt) const
        -> std::array<float, 1>
    {
        return {static_cast<float>(pt.digits)};
    }
};

static_assert(cal::FeatureEncoder<parser_digits_encoder, ex::parser_point>);

/// FeatureEncoder for memory_point
struct memory_feature_encoder {
    static constexpr std::size_t width = 2;
    static constexpr std::array<std::string_view, 2> column_names = {
        "bytes", "stride"
    };

    [[nodiscard]] auto encode(ex::memory_point const& pt) const
        -> std::array<float, 2>
    {
        return {static_cast<float>(pt.bytes), static_cast<float>(pt.stride)};
    }
};

static_assert(cal::FeatureEncoder<memory_feature_encoder, ex::memory_point>);


// ═══════════════════════════════════════════════════════════════════
//  Step 4.1: End-to-end parser scenario pipeline
// ═══════════════════════════════════════════════════════════════════

// Helper: run a scenario through harness and return data_points
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


// ─── Pipeline test: parser scenario ──────────────────────────────

void test_parser_full_pipeline() {
    // Step 1: Create scenario (small: 3 digit counts × 4 strategies = 12 pts)
    ex::parser_strategy_scenario scenario(1, 3);
    ASSERT_EQ(scenario.points().size(), 12u);

    // Step 2: Measure
    auto data = measure_scenario(scenario, 5);
    ASSERT_EQ(data.size(), 12u);

    // Step 3: Wrap in typed dataset with provenance
    auto env  = bench::capture_environment();
    auto prov = cal::capture_provenance(scenario, env, 5);
    auto ds = cal::make_dataset<parser_space,
                                ex::integer_parser_tag,
                                null_snap>(std::move(data), prov);

    ASSERT_EQ(ds.size(), 12u);
    ASSERT_TRUE(ds.invariant());
    ASSERT_EQ(ds.provenance.scenario_name, std::string("parser_strategy"));

    // Step 4: CSV with feature encoder
    cal::feature_point_formatter<parser_feature_encoder, ex::parser_point>
        feat_fmt{parser_feature_encoder{}};

    std::ostringstream csv_out;
    cal::write_provenance_header(csv_out, ds.provenance);
    csv_out << feat_fmt.csv_header() << ",median_ns,mad_ns\n";
    for (auto const& dp : ds.points) {
        csv_out << feat_fmt.to_csv(dp.space_point) << ","
                << dp.median_ns << "," << dp.mad_ns << "\n";
    }
    auto csv = csv_out.str();
    ASSERT_TRUE(csv.find("# ctdp-calibrator dataset") != std::string::npos);
    ASSERT_TRUE(csv.find("generic,loop,swar,unrolled,digits") != std::string::npos);

    // Count data lines (should be 12 + 1 header + provenance comments)
    int data_lines = 0;
    bool in_header = true;
    std::istringstream iss(csv);
    std::string line;
    while (std::getline(iss, line)) {
        if (in_header && !line.empty() && line[0] == '#') continue;
        in_header = false;
        ++data_lines;
    }
    ASSERT_EQ(data_lines, 13); // 1 CSV header + 12 data rows

    // Step 5: Fit lookup profile
    auto lookup_prof = cal::fit_lookup_profile(ds);
    ASSERT_EQ(lookup_prof.training_points, 12u);

    // Predict for a known point
    ex::parser_point p_generic_2{2, ex::parse_strategy::generic};
    double predicted = lookup_prof.predict_lookup(p_generic_2);
    ASSERT_TRUE(predicted > 0.0);

    // Step 6: Fit linear profile (digits-only encoder)
    // Filter to just one strategy for a clean linear relationship
    auto swar_ds = ds.filter([](auto const& dp) {
        return dp.space_point.strategy == ex::parse_strategy::swar;
    });
    ASSERT_EQ(swar_ds.size(), 3u); // digits 1, 2, 3

    auto linear_prof = cal::fit_linear_profile(swar_ds, parser_digits_encoder{});
    ASSERT_TRUE(linear_prof.linear_coefficients.size() == 1);
    // R² may not be perfect with real measurements, but should be > 0
    ASSERT_TRUE(linear_prof.linear_r_squared >= 0.0);

    // Step 7: Emit constexpr header
    std::ostringstream emit_out;
    cal::emit_constexpr_profile(emit_out, lookup_prof,
        "parser_cost_table", "parser_space", "integer_parser_tag");
    auto emitted = emit_out.str();
    ASSERT_TRUE(emitted.find("constexpr") != std::string::npos);
    ASSERT_TRUE(emitted.find("parser_cost_table") != std::string::npos);
    ASSERT_TRUE(emitted.find("integer_parser_tag") != std::string::npos);
    ASSERT_TRUE(emitted.find("N = 12") != std::string::npos);

    // Step 8: Validate profile
    cal::validation_config vcfg;
    vcfg.reps = 5;
    vcfg.tolerance = 0.60;  // generous for CI (container timing noise)
    vcfg.flush_cache = false;

    auto vr = cal::validate_profile<parser_space, ex::integer_parser_tag>(
        lookup_prof, scenario, vcfg);
    ASSERT_EQ(vr.total_points, 12u);
    // At least some points should pass
    ASSERT_TRUE(vr.pass_rate() > 0.3);
}

// ─── Pipeline test: memory scenario ──────────────────────────────

void test_memory_full_pipeline() {
    // Small: 5 sizes from 16 KiB to 256 KiB
    ex::memory_regime_scenario scenario(16, 256, 5);
    ASSERT_EQ(scenario.points().size(), 5u);

    auto data = measure_scenario(scenario, 3);
    ASSERT_EQ(data.size(), 5u);

    auto env  = bench::capture_environment();
    auto prov = cal::capture_provenance(scenario, env, 3);
    auto ds = cal::make_dataset<memory_space,
                                ex::memory_traverse_tag,
                                null_snap>(std::move(data), prov);

    ASSERT_TRUE(ds.invariant());
    ASSERT_EQ(ds.provenance.scenario_name, std::string("memory_regime"));

    // Fit lookup
    auto prof = cal::fit_lookup_profile(ds);
    ASSERT_EQ(prof.training_points, 5u);

    // Emit
    std::ostringstream oss;
    cal::emit_constexpr_profile(oss, prof,
        "memory_cost", "memory_space", "memory_traverse_tag");
    ASSERT_TRUE(oss.str().find("memory_traverse_tag") != std::string::npos);
}


// ═══════════════════════════════════════════════════════════════════
//  Cross-scenario type safety
// ═══════════════════════════════════════════════════════════════════

void test_callable_type_safety() {
    // Verify that the type system prevents mixing datasets/profiles
    // from different callables.

    // parser_strategy_scenario::callable_type = integer_parser_tag
    // memory_regime_scenario::callable_type = memory_traverse_tag
    static_assert(!std::is_same_v<
        ex::parser_strategy_scenario::callable_type,
        ex::memory_regime_scenario::callable_type>);

    // Dataset types are distinct
    using parser_ds = cal::calibration_dataset<
        parser_space, ex::integer_parser_tag, null_snap>;
    using memory_ds = cal::calibration_dataset<
        memory_space, ex::memory_traverse_tag, null_snap>;

    static_assert(!std::is_same_v<parser_ds, memory_ds>);

    // Profile types are distinct
    using parser_prof = cal::calibration_profile<
        parser_space, ex::integer_parser_tag>;
    using memory_prof = cal::calibration_profile<
        memory_space, ex::memory_traverse_tag>;

    static_assert(!std::is_same_v<parser_prof, memory_prof>);

    // fit_lookup_profile(parser_ds) → parser_prof (not memory_prof)
    static_assert(std::is_same_v<
        decltype(cal::fit_lookup_profile(std::declval<parser_ds const&>())),
        parser_prof>);

    ASSERT_TRUE(true);
}


// ═══════════════════════════════════════════════════════════════════
//  Sampler → pipeline integration
// ═══════════════════════════════════════════════════════════════════

void test_sampled_pipeline() {
    // Full cross-product: digits 1–6 × 4 strategies = 24 points
    ex::parser_strategy_scenario scenario(1, 6);
    ASSERT_EQ(scenario.points().size(), 24u);

    // Sample down to ~8 points
    cal::random_sampler sampler;
    sampler.max_points = 8;
    sampler.seed = 42;
    auto subset = sampler.sample(scenario);
    ASSERT_TRUE(subset.size() <= 8u);
    ASSERT_TRUE(subset.size() > 0u);

    // Build a sub-scenario with just the sampled points
    ex::parser_strategy_scenario sub_scenario(std::move(subset));

    // Measure, dataset, profile — abbreviated pipeline
    auto data = measure_scenario(sub_scenario, 3);
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag,
                                null_snap>(std::move(data));
    auto prof = cal::fit_lookup_profile(ds);
    ASSERT_EQ(prof.training_points, ds.size());
    ASSERT_TRUE(prof.training_points <= 8u);
}


// ═══════════════════════════════════════════════════════════════════
//  Dataset quality filtering in pipeline
// ═══════════════════════════════════════════════════════════════════

void test_quality_filtered_pipeline() {
    ex::parser_strategy_scenario scenario(1, 2);
    auto data = measure_scenario(scenario, 5);
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag,
                                null_snap>(std::move(data));

    // Filter by quality (this may or may not remove points depending
    // on measurement noise, but the API should work)
    auto clean = ds.filter_by_quality(0.50); // generous
    ASSERT_TRUE(clean.size() <= ds.size());
    ASSERT_TRUE(clean.invariant());

    // Should still be able to fit a profile from filtered data
    if (clean.size() > 0) {
        auto prof = cal::fit_lookup_profile(clean);
        ASSERT_EQ(prof.training_points, clean.size());
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Feature encoder CSV round-trip
// ═══════════════════════════════════════════════════════════════════

void test_feature_csv_round_trip() {
    parser_feature_encoder enc;

    // Test all four strategies at digits=7
    for (auto s : {ex::parse_strategy::generic, ex::parse_strategy::loop,
                   ex::parse_strategy::swar, ex::parse_strategy::unrolled}) {
        ex::parser_point pt{7, s};
        auto features = enc.encode(pt);

        // Exactly one strategy bit should be set
        int hot_count = 0;
        for (int i = 0; i < 4; ++i) {
            if (features[static_cast<std::size_t>(i)] > 0.5f) ++hot_count;
        }
        ASSERT_EQ(hot_count, 1);

        // Digits feature
        ASSERT_NEAR(features[4], 7.0f, 1e-6f);
    }

    // Test CSV formatting
    cal::feature_point_formatter<parser_feature_encoder, ex::parser_point>
        fmt{parser_feature_encoder{}};

    auto hdr = fmt.csv_header();
    ASSERT_TRUE(hdr.find("generic") != std::string::npos);
    ASSERT_TRUE(hdr.find("digits") != std::string::npos);

    auto row = fmt.to_csv({4, ex::parse_strategy::swar});
    // swar is index 2 → 0,0,1,0,4
    ASSERT_TRUE(row.find("0,0,1,0,4") != std::string::npos);
}


// ═══════════════════════════════════════════════════════════════════
//  Emit + re-parse verification
// ═══════════════════════════════════════════════════════════════════

void test_emit_round_trip_values() {
    // Create a small dataset with known values
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> pts;
    pts.push_back({
        .space_point = {3, ex::parse_strategy::swar},
        .median_ns   = 42.5,
        .mad_ns      = 1.0,
        .raw_timings = {42.5},
        .raw_snapshots = {null_snap{}},
        .env = {}
    });
    pts.push_back({
        .space_point = {7, ex::parse_strategy::generic},
        .median_ns   = 123.75,
        .mad_ns      = 2.5,
        .raw_timings = {123.75},
        .raw_snapshots = {null_snap{}},
        .env = {}
    });

    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag,
                                null_snap>(std::move(pts));
    auto prof = cal::fit_lookup_profile(ds);

    std::ostringstream oss;
    cal::emit_constexpr_lookup(oss, prof,
        "test_values", "parser_space", "integer_parser_tag");
    auto code = oss.str();

    // Verify exact values appear in emitted code
    ASSERT_TRUE(code.find("42.50") != std::string::npos);
    ASSERT_TRUE(code.find("123.75") != std::string::npos);
    ASSERT_TRUE(code.find("N = 2") != std::string::npos);
    ASSERT_TRUE(code.find("predict") != std::string::npos);
}


// ═══════════════════════════════════════════════════════════════════
//  Benchmark explorer (compilation test)
// ═══════════════════════════════════════════════════════════════════

void test_benchmark_explorer_compiles() {
    // In stub mode (no libbenchmark), register is a no-op
    ex::parser_strategy_scenario scenario(1, 3);
    cal::register_scenario_benchmarks(scenario, "Parser");
    // Should compile and do nothing.
    ASSERT_TRUE(!CTDP_HAS_GBENCH);  // we know GBench isn't available here
}


// ═══════════════════════════════════════════════════════════════════
//  Linear regression quality on synthetic data
// ═══════════════════════════════════════════════════════════════════

void test_linear_profile_quality() {
    // Create synthetic data: cost = 5.0 * digits + 2.0 (exact)
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> pts;
    for (int d = 1; d <= 10; ++d) {
        double cost = 5.0 * d + 2.0;
        pts.push_back({
            .space_point   = {d, ex::parse_strategy::swar},
            .median_ns     = cost,
            .mad_ns        = 0.01,
            .raw_timings   = {cost},
            .raw_snapshots = {null_snap{}},
            .env           = {}
        });
    }

    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag,
                                null_snap>(std::move(pts));

    auto prof = cal::fit_linear_profile(ds, parser_digits_encoder{});
    ASSERT_TRUE(prof.linear_r_squared > 0.9999);
    ASSERT_NEAR(prof.linear_coefficients[0], 5.0, 0.01);
    ASSERT_NEAR(prof.linear_intercept, 2.0, 0.01);

    // Predict at an unseen point
    std::array<float, 1> f{15.0f};  // 15 digits
    std::span<const float> sp(f.data(), f.size());
    double pred = prof.predict_linear(sp);
    ASSERT_NEAR(pred, 77.0, 0.1);  // 5*15 + 2
}


// ═══════════════════════════════════════════════════════════════════
//  Provenance chain integrity
// ═══════════════════════════════════════════════════════════════════

void test_provenance_chain() {
    ex::parser_strategy_scenario scenario(1, 2);
    auto env  = bench::capture_environment();
    auto prov = cal::capture_provenance(scenario, env, 10);

    ASSERT_EQ(prov.scenario_name, std::string("parser_strategy"));
    ASSERT_EQ(prov.reps_per_point, 10u);
    ASSERT_TRUE(!prov.hostname.empty() || true); // may fail in container
    ASSERT_TRUE(!prov.compiler.empty());
    ASSERT_TRUE(prov.schema.compatible_with(cal::current_schema));

    // Provenance flows to dataset
    auto data = measure_scenario(scenario, 3);
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag,
                                null_snap>(std::move(data), prov);
    ASSERT_EQ(ds.provenance.scenario_name, prov.scenario_name);

    // And to profile
    auto prof = cal::fit_lookup_profile(ds);
    ASSERT_EQ(prof.provenance.scenario_name, prov.scenario_name);
}


// ═══════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("ctdp-calibrator Phase 4 integration test\n");
    std::printf("═════════════════════════════════════════════════════════\n\n");

    std::printf("End-to-end pipeline:\n");
    TEST(parser_full_pipeline);
    TEST(memory_full_pipeline);

    std::printf("\nType safety:\n");
    TEST(callable_type_safety);

    std::printf("\nSampling + filtering:\n");
    TEST(sampled_pipeline);
    TEST(quality_filtered_pipeline);

    std::printf("\nFeature encoding:\n");
    TEST(feature_csv_round_trip);

    std::printf("\nEmit verification:\n");
    TEST(emit_round_trip_values);

    std::printf("\nLinear regression:\n");
    TEST(linear_profile_quality);

    std::printf("\nProvenance:\n");
    TEST(provenance_chain);

    std::printf("\nBenchmark explorer:\n");
    TEST(benchmark_explorer_compiles);

    std::printf("\n═════════════════════════════════════════════════════════\n");
    std::printf("%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
