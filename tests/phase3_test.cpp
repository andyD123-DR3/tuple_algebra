// ctdp-calibrator Phase 3 test suite — 38 tests

#include <ctdp/bench/compiler_barrier.h>
#include <ctdp/bench/environment.h>
#include <ctdp/bench/measurement_kernel.h>
#include <ctdp/bench/metric.h>
#include <ctdp/bench/statistics.h>
#include <ctdp/bench/cache_thrasher.h>

#include <ctdp/calibrator/data_point.h>
#include <ctdp/calibrator/scenario.h>
#include <ctdp/calibrator/calibration_harness.h>
#include <ctdp/calibrator/csv_writer.h>
#include <ctdp/calibrator/feature_encoder.h>
#include <ctdp/calibrator/sampler.h>
#include <ctdp/calibrator/provenance.h>
#include <ctdp/calibrator/calibration_dataset.h>
#include <ctdp/calibrator/calibration_profile.h>
#include <ctdp/calibrator/ct_dp_emit.h>
#include <ctdp/calibrator/plan_validate.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { ++tests_run; std::printf("  [%d] %-45s", tests_run, #name); \
    try { test_##name(); std::printf("PASS\n"); ++tests_passed; } \
    catch (std::exception const& e) { std::printf("FAIL: %s\n", e.what()); } \
    catch (...) { std::printf("FAIL: unknown\n"); } } while(0)

#define ASSERT_TRUE(expr) do { if(!(expr)) throw std::runtime_error("ASSERT_TRUE: " #expr); } while(0)
#define ASSERT_EQ(a,b) do { if((a)!=(b)) { std::ostringstream o; o<<(a)<<" != "<<(b); throw std::runtime_error(o.str()); } } while(0)
#define ASSERT_NEAR(a,b,t) do { if(std::abs(double(a)-double(b))>(t)) { std::ostringstream o; o<<(a)<<" vs "<<(b); throw std::runtime_error(o.str()); } } while(0)

// ─── Mock types ──────────────────────────────────────────────────

using null_snap = ctdp::bench::null_metric::null_snapshot;

enum class TestStrategy : int { Alpha=0, Beta=1, Gamma=2 };

struct test_point {
    TestStrategy strategy; int size; double threshold;
    bool operator==(test_point const&) const = default;
};

struct test_callable {};
struct test_space { using point_type = test_point; };

struct test_scenario {
    using point_type = test_point;
    using callable_type = test_callable;
    auto name() const -> std::string_view { return "test_scenario"; }
    auto points() const -> std::vector<test_point> const& { return pts_; }
    void prepare(test_point const&) {}
    auto execute(test_point const& pt) -> ctdp::bench::result_token {
        std::uint64_t acc = 0;
        for (unsigned i = 0, n = static_cast<unsigned>(pt.size*100); i < n; ++i) {
            acc += i*7+13; acc ^= acc>>3;
        }
        return ctdp::bench::result_token{acc};
    }
    test_scenario() {
        for (int s=0; s<3; ++s)
            for (int sz : {1,2,4,8})
                for (double t : {0.1,0.5})
                    pts_.push_back({static_cast<TestStrategy>(s), sz, t});
    }
private:
    std::vector<test_point> pts_;
};

static_assert(ctdp::calibrator::Scenario<test_scenario>);

using dp_t = ctdp::calibrator::data_point<test_point, null_snap>;

static dp_t make_dp(TestStrategy s, int sz, double thr, double ns) {
    return dp_t{ .space_point={s,sz,thr}, .median_ns=ns, .mad_ns=ns*0.02,
        .raw_timings={ns}, .raw_snapshots={null_snap{}}, .env={} };
}

// ─── Encoders (file scope for C++20) ─────────────────────────────

struct test_encoder {
    static constexpr std::size_t width = 5;
    static constexpr std::array<std::string_view, 5> column_names = {
        "strategy_alpha","strategy_beta","strategy_gamma","size","threshold"};
    auto encode(test_point const& pt) const -> std::array<float,5> {
        using namespace ctdp::calibrator;
        std::array<float,5> f{};
        std::size_t p=0;
        p = copy_features(f, p, encode_one_hot<3>(static_cast<int>(pt.strategy)));
        f[p++] = encode_scalar(pt.size);
        f[p++] = encode_scalar(pt.threshold);
        return f;
    }
};
static_assert(ctdp::calibrator::FeatureEncoder<test_encoder, test_point>);

struct size_encoder {
    static constexpr std::size_t width = 1;
    static constexpr std::array<std::string_view, 1> column_names = {"size"};
    auto encode(test_point const& pt) const -> std::array<float,1> {
        return {static_cast<float>(pt.size)};
    }
};
static_assert(ctdp::calibrator::FeatureEncoder<size_encoder, test_point>);

// ═══════════ 3.1 feature_encoder ═══════════

void test_encode_scalar_int()    { ASSERT_NEAR(ctdp::calibrator::encode_scalar(42), 42.0f, 1e-6f); }
void test_encode_scalar_double() { ASSERT_NEAR(ctdp::calibrator::encode_scalar(3.14), 3.14f, 1e-3f); }

void test_encode_one_hot() {
    auto oh = ctdp::calibrator::encode_one_hot<3>(0);
    ASSERT_NEAR(oh[0],1.0f,1e-6f); ASSERT_NEAR(oh[1],0.0f,1e-6f); ASSERT_NEAR(oh[2],0.0f,1e-6f);
    auto oh2 = ctdp::calibrator::encode_one_hot<3>(2);
    ASSERT_NEAR(oh2[2],1.0f,1e-6f);
}

void test_encode_one_hot_out_of_range() {
    auto oh = ctdp::calibrator::encode_one_hot<3>(5);
    ASSERT_NEAR(oh[0],0.0f,1e-6f); ASSERT_NEAR(oh[1],0.0f,1e-6f); ASSERT_NEAR(oh[2],0.0f,1e-6f);
}

void test_copy_features() {
    std::array<float,5> dst{};
    std::size_t pos = ctdp::calibrator::copy_features(dst, 0, std::array<float,2>{10.f,20.f});
    ASSERT_EQ(pos, 2u);
    pos = ctdp::calibrator::copy_features(dst, pos, std::array<float,1>{30.f});
    ASSERT_EQ(pos, 3u);
    ASSERT_NEAR(dst[0],10.f,1e-6f); ASSERT_NEAR(dst[2],30.f,1e-6f);
}

void test_encoder_full_point() {
    test_encoder enc;
    auto f = enc.encode({TestStrategy::Beta, 8, 0.5});
    ASSERT_NEAR(f[0],0.f,1e-6f); ASSERT_NEAR(f[1],1.f,1e-6f);
    ASSERT_NEAR(f[3],8.f,1e-6f); ASSERT_NEAR(f[4],0.5f,1e-3f);
}

void test_feature_point_formatter() {
    ctdp::calibrator::feature_point_formatter<test_encoder,test_point> fmt{test_encoder{}};
    auto hdr = fmt.csv_header();
    ASSERT_TRUE(hdr.find("strategy_alpha") != std::string::npos);
    auto row = fmt.to_csv({TestStrategy::Alpha, 4, 0.1});
    ASSERT_TRUE(row.find("4") != std::string::npos);
}

// ═══════════ 3.2 sampler ═══════════

void test_full_sampler() {
    test_scenario s; ctdp::calibrator::full_sampler sm;
    ASSERT_EQ(sm.sample(s).size(), s.points().size());
}

void test_stride_sampler_all() {
    test_scenario s; ctdp::calibrator::stride_sampler sm{.max_points=100};
    ASSERT_EQ(sm.sample(s).size(), 24u);
}

void test_stride_sampler_subset() {
    test_scenario s; ctdp::calibrator::stride_sampler sm{.max_points=8};
    auto pts = sm.sample(s);
    ASSERT_TRUE(pts.size() <= 8u && pts.size() > 0u);
}

void test_random_sampler_deterministic() {
    test_scenario s;
    ctdp::calibrator::random_sampler sm{.max_points=10,.seed=42};
    auto a = sm.sample(s), b = sm.sample(s);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i=0; i<a.size(); ++i) ASSERT_TRUE(a[i]==b[i]);
}

void test_random_sampler_different_seed() {
    test_scenario s;
    auto a = ctdp::calibrator::random_sampler{.max_points=10,.seed=42}.sample(s);
    auto b = ctdp::calibrator::random_sampler{.max_points=10,.seed=99}.sample(s);
    bool diff = false;
    for (std::size_t i=0; i<std::min(a.size(),b.size()); ++i) if(!(a[i]==b[i])) { diff=true; break; }
    ASSERT_TRUE(diff);
}

void test_sampler_concept() {
    static_assert(ctdp::calibrator::Sampler<ctdp::calibrator::full_sampler, test_scenario>);
    static_assert(ctdp::calibrator::Sampler<ctdp::calibrator::stride_sampler, test_scenario>);
    static_assert(ctdp::calibrator::Sampler<ctdp::calibrator::random_sampler, test_scenario>);
    ASSERT_TRUE(true);
}

// ═══════════ 3.3 provenance ═══════════

void test_schema_version_compatible() {
    ctdp::calibrator::schema_version v1{1, 0};
    ctdp::calibrator::schema_version v1_1{1, 1};
    ctdp::calibrator::schema_version v2{2, 0};
    ASSERT_TRUE(v1.compatible_with(v1_1)); ASSERT_TRUE(!v1.compatible_with(v2));
}

void test_schema_version_to_string() {
    ctdp::calibrator::schema_version v{1, 2};
    ASSERT_EQ(v.to_string(), std::string("1.2"));
}

void test_capture_provenance() {
    test_scenario scen; auto env = ctdp::bench::capture_environment();
    auto p = ctdp::calibrator::capture_provenance(scen, env, 10);
    ASSERT_TRUE(!p.timestamp.empty());
    ASSERT_EQ(p.scenario_name, std::string("test_scenario"));
    ASSERT_EQ(p.reps_per_point, 10u);
    ASSERT_TRUE(!p.compiler.empty());
}

void test_write_provenance_header() {
    ctdp::calibrator::dataset_provenance p;
    p.scenario_name="test"; p.timestamp="2026-03-01T12:00:00Z"; p.cpu_model="X";
    p.schema = ctdp::calibrator::schema_version{1, 0};
    std::ostringstream oss; ctdp::calibrator::write_provenance_header(oss, p);
    ASSERT_TRUE(oss.str().find("# ctdp-calibrator dataset v1.0") != std::string::npos);
}

// ═══════════ 3.4 calibration_dataset ═══════════

void test_make_dataset() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,4,0.1,10.0), make_dp(TestStrategy::Beta,8,0.5,20.0)};
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    ASSERT_EQ(ds.size(), 2u); ASSERT_EQ(ds.provenance.total_points, 2u);
}

void test_dataset_queries() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,1,0.1,5.0),
        make_dp(TestStrategy::Beta,4,0.5,15.0), make_dp(TestStrategy::Gamma,8,0.1,25.0)};
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    ASSERT_NEAR(ds.fastest()->median_ns, 5.0, 1e-6);
    ASSERT_NEAR(ds.slowest()->median_ns, 25.0, 1e-6);
    ASSERT_NEAR(ds.mean_median_ns(), 15.0, 1e-6);
}

void test_dataset_filter() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,1,0.1,10.0), make_dp(TestStrategy::Beta,4,0.5,0.1)};
    auto noisy = make_dp(TestStrategy::Gamma,8,0.1,1.0); noisy.mad_ns = 5.0;
    pts.push_back(noisy);
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    ASSERT_EQ(ds.filter_by_quality(0.10).size(), 2u);
}

void test_dataset_invariant() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,1,0.1,10.0)};
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    ASSERT_TRUE(ds.invariant());
}

void test_dataset_callable_type() {
    using ds_t = ctdp::calibrator::calibration_dataset<test_space,test_callable,null_snap>;
    static_assert(std::same_as<ds_t::callable_type, test_callable>);
    ASSERT_TRUE(true);
}

// ═══════════ 3.5a calibration_profile ═══════════

void test_lookup_model_predict() {
    ctdp::calibrator::lookup_model<test_point> lm;
    test_point p1{TestStrategy::Alpha,4,0.1}, p2{TestStrategy::Beta,8,0.5};
    lm.entries.emplace_back(p1, 10.0); lm.entries.emplace_back(p2, 20.0);
    ASSERT_NEAR(lm.predict(p1), 10.0, 1e-6);
    ASSERT_NEAR(lm.predict({TestStrategy::Gamma,1,0.5}), -1.0, 1e-6);
}

void test_linear_model_predict() {
    ctdp::calibrator::linear_model<2> lm;
    lm.intercept = 2.0;
    lm.coefficients = {3.0, -1.0};
    std::array<float, 2> f1{1.f, 1.f};
    std::array<float, 2> f2{0.f, 0.f};
    ASSERT_NEAR(lm.predict(f1), 4.0, 1e-6);
    ASSERT_NEAR(lm.predict(f2), 2.0, 1e-6);
}

void test_fit_lookup_profile() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,1,0.1,5.0),
        make_dp(TestStrategy::Beta,4,0.5,15.0), make_dp(TestStrategy::Gamma,8,0.1,30.0)};
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    auto prof = ctdp::calibrator::fit_lookup_profile(ds);
    using mt = ctdp::calibrator::calibration_profile<test_space,test_callable>::model_type;
    ASSERT_TRUE(prof.active_model == mt::lookup);
    ASSERT_EQ(prof.training_points, 3u);
    ASSERT_NEAR(prof.predict_lookup({TestStrategy::Beta,4,0.5}), 15.0, 1e-6);
}

void test_fit_linear_profile() {
    std::vector<dp_t> pts;
    for (int sz : {1,2,4,8,16}) pts.push_back(make_dp(TestStrategy::Alpha, sz, 0.1, 10.0*sz+1.0));
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    auto prof = ctdp::calibrator::fit_linear_profile(ds, size_encoder{});
    ASSERT_TRUE(prof.linear_r_squared > 0.99);
    ASSERT_NEAR(prof.linear_coefficients[0], 10.0, 0.1);
    ASSERT_NEAR(prof.linear_intercept, 1.0, 0.1);
}

void test_profile_callable_type() {
    using p = ctdp::calibrator::calibration_profile<test_space,test_callable>;
    static_assert(std::same_as<p::callable_type, test_callable>);
    ASSERT_TRUE(true);
}

// ═══════════ 3.5b ct_dp_emit ═══════════

void test_emit_constexpr_lookup() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,1,0.1,5.0), make_dp(TestStrategy::Beta,4,0.5,15.0)};
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    auto prof = ctdp::calibrator::fit_lookup_profile(ds);
    std::ostringstream oss;
    ctdp::calibrator::emit_constexpr_profile(oss, prof, "test_cost_table", "test_space", "test_callable");
    auto c = oss.str();
    ASSERT_TRUE(c.find("#ifndef CTDP_GENERATED_TEST_COST_TABLE_H") != std::string::npos);
    ASSERT_TRUE(c.find("5.00") != std::string::npos);
    ASSERT_TRUE(c.find("15.00") != std::string::npos);
}

void test_emit_constexpr_linear() {
    std::vector<dp_t> pts;
    for (int sz : {1,2,4,8}) pts.push_back(make_dp(TestStrategy::Alpha, sz, 0.1, 10.0*sz));
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    auto prof = ctdp::calibrator::fit_linear_profile(ds, size_encoder{});
    std::ostringstream oss;
    ctdp::calibrator::emit_constexpr_profile(oss, prof, "linear_cost", "test_space", "test_callable");
    auto c = oss.str();
    ASSERT_TRUE(c.find("intercept") != std::string::npos);
    ASSERT_TRUE(c.find("num_features = 1") != std::string::npos);
}

void test_emit_none_model() {
    ctdp::calibrator::calibration_profile<test_space,test_callable> prof;
    std::ostringstream oss;
    ctdp::calibrator::emit_constexpr_profile(oss, prof, "empty");
    ASSERT_TRUE(oss.str().find("ERROR") != std::string::npos);
}

// ═══════════ 3.6 plan_validate ═══════════

void test_point_validation_within_tolerance() {
    ctdp::calibrator::point_validation pv{.predicted_ns=10.0,.measured_ns=10.5,
        .absolute_error=0.5,.relative_error=0.05};
    ASSERT_TRUE(pv.within_tolerance(0.10)); ASSERT_TRUE(!pv.within_tolerance(0.04));
}

void test_validation_result_summary() {
    ctdp::calibrator::validation_result vr;
    for (int i=0; i<9; ++i)
        vr.points.push_back({10.0, 10.0+i*0.1, 0.0, std::abs(i*0.1), std::abs(i*0.1)/10.0});
    vr.points.push_back({10.0, 11.5, 0.0, 1.5, 0.15});
    vr.total_points = 10; vr.points_within_tol = 9;
    vr.max_relative_error = 0.15; vr.mean_relative_error = 0.05;
    ASSERT_TRUE(!vr.passed(0.10)); ASSERT_TRUE(vr.passed(0.20));
    ASSERT_TRUE(vr.summary(0.10).find("validation") != std::string::npos);
}

void test_validation_result_pass_rate() {
    ctdp::calibrator::validation_result vr; vr.total_points=4; vr.points_within_tol=3;
    ASSERT_NEAR(vr.pass_rate(), 0.75, 1e-6);
}

void test_validation_config_defaults() {
    ctdp::calibrator::validation_config cfg;
    ASSERT_EQ(cfg.reps, 20u); ASSERT_NEAR(cfg.tolerance, 0.10, 1e-6);
}

void test_validate_profile_live() {
    test_scenario scen;
    auto subset = ctdp::calibrator::stride_sampler{.max_points=3}.sample(scen);
    std::vector<dp_t> pts;
    ctdp::bench::null_metric nm;
    for (auto const& pt : subset) {
        scen.prepare(pt);
        auto fn = [&]() -> ctdp::bench::result_token { return scen.execute(pt); };
        auto r = ctdp::bench::measure_repeated(fn, []{}, nm, 5, 10, 1);
        pts.push_back(dp_t{.space_point=pt, .median_ns=r.median_ns, .mad_ns=r.mad_ns,
            .raw_timings=std::move(r.all_ns), .raw_snapshots=std::move(r.all_snapshots), .env={}});
    }
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    auto prof = ctdp::calibrator::fit_lookup_profile(ds);
#ifndef CTDP_SKIP_HW_TIMING_TESTS
    // Hardware path: tight tolerance — validates prediction accuracy.
    // Enable with:  cmake -DCTDP_SKIP_HW_TIMING_TESTS=OFF ..
    ctdp::calibrator::validation_config vcfg; vcfg.reps=5; vcfg.tolerance=0.50; vcfg.flush_cache=false;
    auto vr = ctdp::calibrator::validate_profile<test_space,test_callable>(prof, scen, vcfg);
    ASSERT_EQ(vr.total_points, 3u);
    ASSERT_TRUE(vr.pass_rate() > 0.5);
#else
    // CI path: tolerance=0.99 — exercises the code path, no timing assertion.
    // Virtualised MSVC runners have too much scheduling jitter for tight checks.
    ctdp::calibrator::validation_config vcfg; vcfg.reps=5; vcfg.tolerance=0.99; vcfg.flush_cache=false;
    auto vr = ctdp::calibrator::validate_profile<test_space,test_callable>(prof, scen, vcfg);
    ASSERT_EQ(vr.total_points, 3u);
    (void)vr.pass_rate();  // structural check only
#endif
}

// ═══════════ Integration ═══════════

void test_end_to_end_scenario_to_emit() {
    test_scenario scen;
    auto subset = ctdp::calibrator::stride_sampler{.max_points=4}.sample(scen);
    std::vector<dp_t> pts;
    ctdp::bench::null_metric nm;
    for (auto const& pt : subset) {
        scen.prepare(pt);
        auto fn = [&]() -> ctdp::bench::result_token { return scen.execute(pt); };
        auto r = ctdp::bench::measure_repeated(fn, []{}, nm, 3, 5, 1);
        pts.push_back(dp_t{.space_point=pt, .median_ns=r.median_ns, .mad_ns=r.mad_ns,
            .raw_timings=std::move(r.all_ns), .raw_snapshots=std::move(r.all_snapshots),
            .env=ctdp::bench::capture_environment()});
    }
    auto env = ctdp::bench::capture_environment();
    auto prov = ctdp::calibrator::capture_provenance(scen, env, 3);
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts), prov);
    ASSERT_TRUE(ds.size() > 0u && ds.invariant());
    auto prof = ctdp::calibrator::fit_lookup_profile(ds);
    std::ostringstream oss;
    ctdp::calibrator::emit_constexpr_profile(oss, prof, "e2e_profile", "test_space", "test_callable");
    ASSERT_TRUE(oss.str().find("constexpr") != std::string::npos);
}

void test_callable_type_flows_through() {
    using ds_t = ctdp::calibrator::calibration_dataset<test_space, test_callable, null_snap>;
    using pr_t = ctdp::calibrator::calibration_profile<test_space, test_callable>;
    static_assert(std::same_as<test_scenario::callable_type, test_callable>);
    static_assert(std::same_as<ds_t::callable_type, test_callable>);
    static_assert(std::same_as<pr_t::callable_type, test_callable>);
    ASSERT_TRUE(true);
}

void test_dataset_to_csv_with_encoder() {
    std::vector<dp_t> pts = {make_dp(TestStrategy::Alpha,4,0.1,10.0), make_dp(TestStrategy::Beta,8,0.5,20.0)};
    auto ds = ctdp::calibrator::make_dataset<test_space,test_callable,null_snap>(std::move(pts));
    ctdp::calibrator::feature_point_formatter<test_encoder,test_point> fmt{test_encoder{}};
    std::ostringstream oss;
    oss << fmt.csv_header() << ",median_ns,mad_ns\n";
    for (auto const& dp : ds.points)
        oss << fmt.to_csv(dp.space_point) << "," << dp.median_ns << "," << dp.mad_ns << "\n";
    auto csv = oss.str();
    ASSERT_TRUE(csv.find("strategy_alpha") != std::string::npos);
    int lines = 0; for (auto c : csv) if(c=='\n') ++lines;
    ASSERT_EQ(lines, 3);
}

int main() {
    std::printf("ctdp-calibrator Phase 3 test suite\n");
    std::printf("═════════════════════════════════════════\n\n");

    std::printf("feature_encoder tests:\n");
    TEST(encode_scalar_int); TEST(encode_scalar_double); TEST(encode_one_hot);
    TEST(encode_one_hot_out_of_range); TEST(copy_features);
    TEST(encoder_full_point); TEST(feature_point_formatter);

    std::printf("\nsampler tests:\n");
    TEST(full_sampler); TEST(stride_sampler_all); TEST(stride_sampler_subset);
    TEST(random_sampler_deterministic); TEST(random_sampler_different_seed); TEST(sampler_concept);

    std::printf("\nprovenance tests:\n");
    TEST(schema_version_compatible); TEST(schema_version_to_string);
    TEST(capture_provenance); TEST(write_provenance_header);

    std::printf("\ncalibration_dataset tests:\n");
    TEST(make_dataset); TEST(dataset_queries); TEST(dataset_filter);
    TEST(dataset_invariant); TEST(dataset_callable_type);

    std::printf("\ncalibration_profile tests:\n");
    TEST(lookup_model_predict); TEST(linear_model_predict);
    TEST(fit_lookup_profile); TEST(fit_linear_profile); TEST(profile_callable_type);

    std::printf("\nct_dp_emit tests:\n");
    TEST(emit_constexpr_lookup); TEST(emit_constexpr_linear); TEST(emit_none_model);

    std::printf("\nplan_validate tests:\n");
    TEST(point_validation_within_tolerance); TEST(validation_result_summary);
    TEST(validation_result_pass_rate); TEST(validation_config_defaults); TEST(validate_profile_live);

    std::printf("\nIntegration tests:\n");
    TEST(end_to_end_scenario_to_emit); TEST(callable_type_flows_through); TEST(dataset_to_csv_with_encoder);

    std::printf("\n═════════════════════════════════════════\n");
    std::printf("%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
