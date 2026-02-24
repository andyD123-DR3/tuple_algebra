// ctdp v0.7.0 — Performance model test suite
// Tests in-sample fitting, cross-validation OOS metrics, and model factory.

#include "ctdp/solver/cost_models/performance_model.h"
#include "ctdp/solver/cost_models/feature_extract.h"
#include "ctdp/solver/cost_models/cross_validation.h"
#include "ctdp/solver/cost_models/linear_model.h"
#include "ctdp/solver/cost_models/mlp_model.h"
#include "ctdp/solver/cost_models/svr_model.h"
#include "ctdp/solver/cost_models/model_factory.h"
#include "ctdp/solver/cost_models/data_preprocess.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ctdp::cost_models;

// ─── Minimal test framework ─────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;
static std::string g_section;

#define SECTION(name) do { g_section = (name); std::cerr << "\n=== " << g_section << " ===\n"; } while(0)

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; std::cerr << "  PASS: " << (msg) << "\n"; } \
    else { ++g_fail; std::cerr << "  FAIL: " << (msg) << " [" << g_section << "]\n"; } \
} while(0)

#define CHECK_FINITE(val, msg) CHECK(std::isfinite(val), msg)
#define CHECK_NEAR(a, b, tol, msg) CHECK(std::abs((a) - (b)) < (tol), msg)
#define CHECK_GT(a, b, msg) CHECK((a) > (b), msg)
#define CHECK_LT(a, b, msg) CHECK((a) < (b), msg)

// ─── Test point type ────────────────────────────────────────────────────

struct tile_shape {
    int tm, tn, tk;
    auto dims() const { return std::array<int, 3>{tm, tn, tk}; }
};

// ─── Synthetic data generators ──────────────────────────────────────────

// Linear relationship: cost = 2*log2(tm) + 3*log2(tn) - 1.5*log2(tk) + noise
std::vector<observation<tile_shape>> make_linear_data(std::size_t n = 50) {
    std::vector<observation<tile_shape>> obs;
    std::mt19937 rng(12345);
    std::normal_distribution<double> noise(0.0, 0.3);
    std::vector<int> vals = {2, 4, 8, 16, 32};

    for (std::size_t i = 0; i < n; ++i) {
        int tm = vals[rng() % vals.size()];
        int tn = vals[rng() % vals.size()];
        int tk = vals[rng() % vals.size()];
        double cost = 2.0 * std::log2(tm) + 3.0 * std::log2(tn)
                    - 1.5 * std::log2(tk) + 10.0 + noise(rng);
        obs.push_back({tile_shape{tm, tn, tk}, cost});
    }
    return obs;
}

// Nonlinear relationship: cost ~ interaction terms + nonlinear
std::vector<observation<tile_shape>> make_nonlinear_data(std::size_t n = 60) {
    std::vector<observation<tile_shape>> obs;
    std::mt19937 rng(67890);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<int> vals = {2, 4, 8, 16, 32};

    for (std::size_t i = 0; i < n; ++i) {
        int tm = vals[rng() % vals.size()];
        int tn = vals[rng() % vals.size()];
        int tk = vals[rng() % vals.size()];
        double l_tm = std::log2(tm), l_tn = std::log2(tn), l_tk = std::log2(tk);
        double cost = 5.0 + 2.0 * l_tm + 1.5 * l_tn - 1.0 * l_tk
                    + 0.3 * l_tm * l_tn
                    - 0.2 * l_tm * l_tk
                    + 0.1 * std::sin(l_tn * 2.0)
                    + noise(rng);
        obs.push_back({tile_shape{tm, tn, tk}, cost});
    }
    return obs;
}

// Data with a clear optimum at tm=8, tn=16, tk=4
std::vector<observation<tile_shape>> make_optimum_data() {
    std::vector<observation<tile_shape>> obs;
    std::mt19937 rng(11111);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<int> vals = {2, 4, 8, 16, 32};

    for (int tm : vals)
        for (int tn : vals)
            for (int tk : vals) {
                // Quadratic cost in log-space, minimum at log2(8)=3, log2(16)=4, log2(4)=2
                double l_tm = std::log2(tm), l_tn = std::log2(tn), l_tk = std::log2(tk);
                double cost = (l_tm - 3.0) * (l_tm - 3.0)
                            + (l_tn - 4.0) * (l_tn - 4.0)
                            + (l_tk - 2.0) * (l_tk - 2.0)
                            + noise(rng);
                obs.push_back({tile_shape{tm, tn, tk}, cost});
            }
    return obs;
}

// ─── Section 1: Metric helpers ──────────────────────────────────────────

void test_metrics() {
    SECTION("1. Metric helpers");

    std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b = {1.1, 2.05, 2.95, 4.1, 4.9};

    double r2 = compute_r2(a, b);
    CHECK_GT(r2, 0.99, "R² > 0.99 for close predictions");

    double rmse = compute_rmse(a, b);
    CHECK_LT(rmse, 0.15, "RMSE < 0.15 for close predictions");

    double rho = compute_spearman(
        std::span<const double>(a), std::span<const double>(b));
    CHECK_NEAR(rho, 1.0, 0.01, "Spearman ρ ≈ 1.0 for monotonic");

    // Perfect negation
    std::vector<double> c = {5.0, 4.0, 3.0, 2.0, 1.0};
    double rho_neg = compute_spearman(
        std::span<const double>(a), std::span<const double>(c));
    CHECK_NEAR(rho_neg, -1.0, 0.01, "Spearman ρ ≈ -1.0 for reversed");
}

// ─── Section 2: Feature extractors ──────────────────────────────────────

void test_extractors() {
    SECTION("2. Feature extractors");

    tile_shape ts{8, 16, 4};

    raw_extractor re;
    auto rf = re(ts);
    CHECK(rf.size() == 3, "raw: 3 features");
    CHECK_NEAR(rf[0], 8.0, 1e-10, "raw[0] = 8");

    log2_extractor le;
    auto lf = le(ts);
    CHECK(lf.size() == 3, "log2: 3 features");
    CHECK_NEAR(lf[0], 3.0, 1e-10, "log2(8) = 3");
    CHECK_NEAR(lf[1], 4.0, 1e-10, "log2(16) = 4");

    log2_interactions_extractor lie;
    auto lif = lie(ts);
    // 3 base + (3+2+1) = 6 interactions = 9 total
    CHECK(lif.size() == 9, "log2_interactions: 9 features (3 base + 6 interactions)");

    reciprocal_extractor rce;
    auto rcf = rce(ts);
    CHECK(rcf.size() == 6, "reciprocal: 6 features (3 values + 3 reciprocals)");
    CHECK_NEAR(rcf[1], 1.0/8.0, 1e-10, "reciprocal of 8");
}

// ─── Section 3: Cross-validation infrastructure ─────────────────────────

void test_cross_validation() {
    SECTION("3. Cross-validation infrastructure");

    auto data = make_linear_data(50);

    // Fold assignment
    std::vector<double> targets;
    for (const auto& o : data) targets.push_back(o.cost);
    auto folds = stratified_fold_assignment(targets, 5);

    // Check all folds have roughly equal size
    std::vector<int> fold_counts(5, 0);
    for (auto f : folds) fold_counts[f]++;
    CHECK(fold_counts[0] == 10, "Fold 0 has 10 samples");
    CHECK(fold_counts[4] == 10, "Fold 4 has 10 samples");

    // Run CV with linear model
    log2_interactions_extractor ext;
    linear_trainer<tile_shape> lt(ext);
    auto cv = k_fold_cv<tile_shape>(data, 5,
        [&](const std::vector<observation<tile_shape>>& train) {
            return lt.build_raw(train);
        });

    CHECK_FINITE(cv.oos_r2, "CV OOS R² is finite");
    CHECK_FINITE(cv.oos_spearman_rho, "CV OOS Spearman ρ is finite");
    CHECK_FINITE(cv.oos_rmse, "CV OOS RMSE is finite");
    CHECK_GT(cv.oos_spearman_rho, 0.5, "CV OOS Spearman ρ > 0.5 for linear data");
    CHECK(cv.oos_predictions.size() == 50, "CV predictions for all 50 samples");
}

// ─── Section 4: Linear model with CV ────────────────────────────────────

void test_linear_model() {
    SECTION("4. Linear model with cross-validation");

    auto data = make_linear_data(50);
    log2_interactions_extractor ext;
    linear_trainer<tile_shape> lt(ext);
    auto model = lt.build(data);

    auto q = model.quality();

    // In-sample metrics
    CHECK_FINITE(q.r2, "In-sample R² is finite");
    CHECK_FINITE(q.spearman_rho, "In-sample Spearman ρ is finite");
    CHECK_GT(q.r2, 0.8, "In-sample R² > 0.8 for linear data");
    CHECK_GT(q.spearman_rho, 0.8, "In-sample Spearman ρ > 0.8");

    // OOS metrics
    CHECK_FINITE(q.oos_r2, "OOS R² is finite");
    CHECK_FINITE(q.oos_spearman_rho, "OOS Spearman ρ is finite");
    CHECK_FINITE(q.oos_rmse, "OOS RMSE is finite");
    CHECK_GT(q.oos_r2, 0.5, "OOS R² > 0.5 for linear data");
    CHECK_GT(q.oos_spearman_rho, 0.5, "OOS Spearman ρ > 0.5");

    // Sanity: OOS should be worse than in-sample
    CHECK_LT(q.oos_spearman_rho, q.spearman_rho + 0.01,
             "OOS ρ ≤ in-sample ρ (generalisation gap)");

    // LOO R² should be computed (closed-form)
    CHECK_FINITE(q.loo_r2, "LOO R² is finite (closed-form)");

    // Prediction works
    double pred = model.predict(tile_shape{8, 16, 4});
    CHECK_FINITE(pred, "Prediction is finite");

    CHECK(q.n_samples == 50, "n_samples = 50");
    CHECK(q.n_params > 0, "n_params > 0");
}

// ─── Section 5: MLP model with CV ───────────────────────────────────────

void test_mlp_model() {
    SECTION("5. MLP model with cross-validation");

    auto data = make_nonlinear_data(60);
    log2_interactions_extractor ext;
    // Smaller MLP for test speed
    mlp_trainer<tile_shape> mt(ext, 8, 4, 200, 0.01, 0.9, 42);
    auto model = mt.build(data);

    auto q = model.quality();

    CHECK_FINITE(q.r2, "In-sample R² is finite");
    CHECK_FINITE(q.spearman_rho, "In-sample Spearman ρ is finite");

    CHECK_FINITE(q.oos_r2, "OOS R² is finite");
    CHECK_FINITE(q.oos_spearman_rho, "OOS Spearman ρ is finite");
    CHECK_FINITE(q.oos_rmse, "OOS RMSE is finite");

    // MLP should have some predictive power on this data
    CHECK_GT(q.oos_spearman_rho, 0.0, "OOS Spearman ρ > 0 (some predictive power)");

    // Sanity: OOS worse than in-sample
    CHECK_LT(q.oos_r2, q.r2 + 0.01, "OOS R² ≤ in-sample R²");

    CHECK(q.model_name == "mlp", "Model name is 'mlp'");
}

// ─── Section 6: SVR model with two-stage CV tuning ──────────────────────

void test_svr_model() {
    SECTION("6. SVR model with two-stage CV tuning");

    auto data = make_nonlinear_data(40);
    log2_interactions_extractor ext;
    svr_trainer<tile_shape> st(ext);
    auto model = st.build(data);

    auto q = model.quality();

    CHECK_FINITE(q.r2, "In-sample R² is finite");
    CHECK_FINITE(q.spearman_rho, "In-sample Spearman ρ is finite");

    CHECK_FINITE(q.oos_r2, "OOS R² is finite");
    CHECK_FINITE(q.oos_spearman_rho, "OOS Spearman ρ is finite");
    CHECK_FINITE(q.oos_rmse, "OOS RMSE is finite");

    // Critical: SVR should NOT trivially memorise
    // If it were memorising, OOS R² would be very low or negative
    // but in-sample R² would be ~1.0
    // We check the gap is not extreme
    if (q.r2 > 0.99) {
        CHECK_GT(q.oos_r2, -0.5,
                 "SVR not trivially memorising (OOS R² > -0.5 when in-sample ≈ 1.0)");
    }

    // OOS should be meaningfully worse than in-sample for a kernel method
    CHECK_LT(q.oos_spearman_rho, q.spearman_rho + 0.01,
             "OOS ρ ≤ in-sample ρ (SVR generalisation gap)");

    // Hyperparameters should be tuned (not trivial values)
    CHECK_GT(model.gamma(), 0.0, "γ > 0");
    CHECK_GT(model.lambda(), 0.0, "λ > 0");

    // Prediction works
    double pred = model.predict(tile_shape{8, 16, 4});
    CHECK_FINITE(pred, "Prediction is finite");
}

// ─── Section 7: Model factory (select on OOS ρ) ────────────────────────

void test_model_factory() {
    SECTION("7. Model factory (OOS Spearman ρ selection)");

    auto data = make_linear_data(50);
    auto_model_factory<tile_shape> factory(log2_interactions_extractor{}, true);
    auto result = factory.build(data);

    CHECK(static_cast<bool>(result.model), "Factory returns a valid model");
    CHECK(result.all_qualities.size() == 3, "Factory evaluated 3 models");

    auto q = result.model.quality();
    CHECK_FINITE(q.oos_spearman_rho, "Selected model has finite OOS ρ");
    CHECK_GT(q.oos_spearman_rho, 0.3, "Selected model OOS ρ > 0.3");

    // Verify the factory selected the model with highest OOS ρ
    double best_oos_rho = -2.0;
    for (const auto& mq : result.all_qualities) {
        double rho = std::isfinite(mq.oos_spearman_rho)
            ? mq.oos_spearman_rho : mq.spearman_rho;
        if (rho > best_oos_rho) best_oos_rho = rho;
    }
    double selected_rho = std::isfinite(q.oos_spearman_rho)
        ? q.oos_spearman_rho : q.spearman_rho;
    CHECK_NEAR(selected_rho, best_oos_rho, 0.01,
               "Factory selected model with highest OOS ρ");
}

// ─── Section 8: Factory finds correct optimum ───────────────────────────

void test_factory_optimum() {
    SECTION("8. Factory finds correct optimum on synthetic data");

    auto data = make_optimum_data();
    auto_model_factory<tile_shape> factory(log2_interactions_extractor{});
    auto result = factory.build(data);

    CHECK(static_cast<bool>(result.model), "Factory returns valid model");

    // The true optimum is at tm=8, tn=16, tk=4 (cost ≈ 0)
    // Find the point with minimum predicted cost
    std::vector<int> vals = {2, 4, 8, 16, 32};
    double best_pred = 1e30;
    tile_shape best_point{0, 0, 0};

    for (int tm : vals)
        for (int tn : vals)
            for (int tk : vals) {
                tile_shape ts{tm, tn, tk};
                double pred = result.model.predict(ts);
                if (pred < best_pred) {
                    best_pred = pred;
                    best_point = ts;
                }
            }

    std::cerr << "  Best predicted point: (" << best_point.tm << ", "
              << best_point.tn << ", " << best_point.tk << ")"
              << " cost=" << best_pred << "\n";

    CHECK(best_point.tm == 8,  "Optimum tm = 8");
    CHECK(best_point.tn == 16, "Optimum tn = 16");
    CHECK(best_point.tk == 4,  "Optimum tk = 4");
}

// ─── Section 9: Data preprocessing — normalise & centre ─────────────────

void test_preprocess_normalise() {
    SECTION("9. Data preprocessing — normalise & centre");

    auto data = make_linear_data(50);

    data_preprocessor<tile_shape> pp;
    auto result = pp.fit_transform(data);

    CHECK(result.observations.size() == 50, "No samples removed (no outlier filter)");
    CHECK(result.n_original == 50, "n_original = 50");
    CHECK(!result.outliers.has_value(), "No outlier report when filter disabled");

    // Normalised targets should have mean ≈ 0, std ≈ 1
    double sum = 0.0;
    for (const auto& o : result.observations) sum += o.cost;
    double mean = sum / 50.0;
    CHECK_NEAR(mean, 0.0, 0.01, "Normalised target mean ≈ 0");

    double var = 0.0;
    for (const auto& o : result.observations) {
        double d = o.cost - mean;
        var += d * d;
    }
    double std_dev = std::sqrt(var / 50.0);
    CHECK_NEAR(std_dev, 1.0, 0.01, "Normalised target std ≈ 1");

    // Inverse transform round-trips
    double raw_val = data[0].cost;
    double norm_val = result.transform.forward(raw_val);
    double back = result.transform.inverse(norm_val);
    CHECK_NEAR(back, raw_val, 1e-10, "Inverse transform round-trips");

    // Transform params are sensible
    CHECK(std::isfinite(result.transform.mean), "Transform mean is finite");
    CHECK_GT(result.transform.std, 0.0, "Transform std > 0");
}

// ─── Section 10: Spike outlier removal ──────────────────────────────────

void test_preprocess_outlier_removal() {
    SECTION("10. Spike outlier removal");

    // Start with clean data, inject a few extreme spikes
    auto data = make_linear_data(50);
    double max_cost = 0.0;
    for (const auto& o : data) max_cost = std::max(max_cost, o.cost);

    // Inject 3 outliers well beyond the data range
    data.push_back({tile_shape{4, 4, 4}, max_cost + 100.0});  // huge spike
    data.push_back({tile_shape{8, 8, 8}, max_cost + 200.0});  // bigger spike
    data.push_back({tile_shape{2, 2, 2}, -500.0});             // negative spike

    data_preprocessor<tile_shape> pp;
    pp.set_outlier_sigma(3.0);  // 3σ threshold
    auto result = pp.fit_transform(data);

    CHECK(result.n_original == 53, "n_original = 53");
    CHECK(result.outliers.has_value(), "Outlier report present");

    auto& report = result.outliers.value();
    std::cerr << "  Outliers removed: " << report.removed_indices.size()
              << " (robust centre=" << report.centre
              << ", σ=" << report.spread << ")\n";
    std::cerr << "  Thresholds: [" << report.threshold_low
              << ", " << report.threshold_high << "]\n";

    CHECK_GT(report.removed_indices.size(), static_cast<std::size_t>(0),
             "At least some outliers removed");
    CHECK(result.observations.size() < 53,
          "Fewer observations after filtering");
    CHECK(result.observations.size() + report.removed_indices.size() == 53,
          "Removed + kept = original");

    // The 3 injected spikes should be gone
    // (they're 100+ units away from the data centre)
    CHECK(report.removed_indices.size() >= 3,
          "All 3 injected spikes detected");

    // Remaining data should be normalised
    double sum = 0.0;
    for (const auto& o : result.observations) sum += o.cost;
    double mean = sum / static_cast<double>(result.observations.size());
    CHECK_NEAR(mean, 0.0, 0.05, "Post-filter normalised mean ≈ 0");
}

// ─── Section 11: Outlier removal with tight threshold ───────────────────

void test_preprocess_tight_threshold() {
    SECTION("11. Outlier removal — tight threshold");

    // Clustered data near cost=10 with a few mild spikes
    std::vector<observation<tile_shape>> data;
    std::mt19937 rng(77777);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<int> vals = {2, 4, 8, 16, 32};

    for (int i = 0; i < 44; ++i) {
        int tm = vals[rng() % vals.size()];
        int tn = vals[rng() % vals.size()];
        int tk = vals[rng() % vals.size()];
        data.push_back({tile_shape{tm, tn, tk}, 10.0 + noise(rng)});
    }
    // Mild spikes — outside 1.5σ but inside 3σ
    for (int i = 0; i < 6; ++i)
        data.push_back({tile_shape{4, 4, 4}, 10.0 + 2.0 * (i % 2 == 0 ? 1.0 : -1.0)});

    data_preprocessor<tile_shape> pp;
    pp.set_outlier_sigma(1.5);  // tight: 1.5σ
    auto result = pp.fit_transform(data);

    std::cerr << "  1.5σ threshold: kept " << result.observations.size()
              << " of " << result.n_original << "\n";

    // With 1.5σ on tight cluster + mild spikes, some should be removed
    CHECK_GT(result.observations.size(), static_cast<std::size_t>(20),
             "Keeps >20 samples at 1.5σ");
    CHECK_LT(result.observations.size(), result.n_original,
             "Removes some samples at 1.5σ");
}

// ─── Section 12: No filtering on small datasets ─────────────────────────

void test_preprocess_small_data() {
    SECTION("12. Preprocessing small datasets");

    // 3 observations — too few for outlier detection (need >= 4)
    std::vector<observation<tile_shape>> tiny = {
        {tile_shape{2, 2, 2}, 1.0},
        {tile_shape{4, 4, 4}, 2.0},
        {tile_shape{8, 8, 8}, 3.0},
    };

    data_preprocessor<tile_shape> pp;
    pp.set_outlier_sigma(3.0);
    auto result = pp.fit_transform(tiny);

    CHECK(result.observations.size() == 3, "All 3 kept (too few for filtering)");
    CHECK(!result.outliers.has_value(), "No outlier report for tiny dataset");

    // Still normalised
    double sum = 0.0;
    for (const auto& o : result.observations) sum += o.cost;
    CHECK_NEAR(sum / 3.0, 0.0, 0.01, "Tiny dataset normalised mean ≈ 0");
}

// ─── Section 13: Transformed model wrapper ──────────────────────────────

void test_transformed_model() {
    SECTION("13. Transformed model (preprocess → train → predict)");

    auto raw_data = make_optimum_data();

    // Preprocess
    data_preprocessor<tile_shape> pp;
    pp.set_outlier_sigma(3.5);
    auto pre = pp.fit_transform(raw_data);

    std::cerr << "  Preprocessed: " << pre.observations.size()
              << " of " << pre.n_original << " kept\n";

    // Train factory on normalised data
    auto_model_factory<tile_shape> factory(log2_interactions_extractor{});
    auto build_result = factory.build(pre.observations);

    // Wrap with inverse transform
    auto model = make_transformed_model<tile_shape>(
        std::move(build_result.model), pre.transform);

    CHECK(static_cast<bool>(model), "Transformed model is valid");

    // Prediction at the known optimum should give a low cost
    // The true minimum cost is ≈ 0 (quadratic bowl centred at (8,16,4))
    double pred_opt = model.predict(tile_shape{8, 16, 4});
    std::cerr << "  Prediction at optimum (8,16,4): " << pred_opt << "\n";

    // Prediction at a far point should give a higher cost
    double pred_far = model.predict(tile_shape{32, 2, 32});
    std::cerr << "  Prediction at (32,2,32): " << pred_far << "\n";

    CHECK_LT(pred_opt, pred_far,
             "Optimum has lower predicted cost than far point");

    // The prediction should be in the ORIGINAL scale (not normalised)
    // Original costs range roughly [0, ~20] for this data
    CHECK_GT(pred_far, 1.0,
             "Far-point prediction in original scale (> 1.0)");

    // Verify factory still finds correct optimum through the full pipeline
    std::vector<int> vals = {2, 4, 8, 16, 32};
    double best_pred = 1e30;
    tile_shape best_point{0, 0, 0};
    for (int tm : vals)
        for (int tn : vals)
            for (int tk : vals) {
                tile_shape ts{tm, tn, tk};
                double p = model.predict(ts);
                if (p < best_pred) { best_pred = p; best_point = ts; }
            }

    std::cerr << "  Best via transformed model: (" << best_point.tm << ", "
              << best_point.tn << ", " << best_point.tk
              << ") cost=" << best_pred << "\n";

    CHECK(best_point.tm == 8,  "Transformed model: optimum tm = 8");
    CHECK(best_point.tn == 16, "Transformed model: optimum tn = 16");
    CHECK(best_point.tk == 4,  "Transformed model: optimum tk = 4");
}

// ─── Section 14: Robust detection (median+MAD) correctness ──────────────

void test_robust_detection() {
    SECTION("14. Robust outlier detection (median+MAD)");

    // Construct data where mean-based detection would fail:
    // 47 samples clustered near 10.0, plus 3 extreme spikes at 1000.0.
    // Mean ≈ 72, std ≈ 144, so 3σ from mean covers [−360, 504] — misses spikes!
    // Median = 10.0, MAD-based σ ≈ small, so 3σ from median catches spikes.

    std::vector<observation<tile_shape>> data;
    std::mt19937 rng(99999);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<int> vals = {2, 4, 8, 16, 32};

    for (int i = 0; i < 47; ++i) {
        int tm = vals[rng() % vals.size()];
        int tn = vals[rng() % vals.size()];
        int tk = vals[rng() % vals.size()];
        data.push_back({tile_shape{tm, tn, tk}, 10.0 + noise(rng)});
    }
    // 3 spikes
    data.push_back({tile_shape{2, 2, 2}, 1000.0});
    data.push_back({tile_shape{4, 4, 4}, 1200.0});
    data.push_back({tile_shape{8, 8, 8}, -800.0});

    data_preprocessor<tile_shape> pp;
    pp.set_outlier_sigma(3.0);
    auto result = pp.fit_transform(data);

    auto& report = result.outliers.value();
    std::cerr << "  Robust centre (median): " << report.centre
              << ", robust σ: " << report.spread << "\n";
    std::cerr << "  Removed: " << report.removed_indices.size() << "\n";

    CHECK(report.removed_indices.size() == 3,
          "Robust detection catches all 3 spikes");
    CHECK(result.observations.size() == 47,
          "47 clean observations remain");

    // Verify the robust centre is near 10.0 (not pulled by spikes)
    CHECK_NEAR(report.centre, 10.0, 1.0,
               "Robust centre ≈ 10.0 (not contaminated by spikes)");
}

// ─── Main ───────────────────────────────────────────────────────────────

int main() {
    std::cerr << "ctdp v0.7.0 — Performance model test suite\n"
              << "==========================================\n";

    test_metrics();
    test_extractors();
    test_cross_validation();
    test_linear_model();
    test_mlp_model();
    test_svr_model();
    test_model_factory();
    test_factory_optimum();
    test_preprocess_normalise();
    test_preprocess_outlier_removal();
    test_preprocess_tight_threshold();
    test_preprocess_small_data();
    test_transformed_model();
    test_robust_detection();

    std::cerr << "\n==========================================\n"
              << "Results: " << g_pass << " passed, " << g_fail << " failed\n";

    return g_fail > 0 ? 1 : 0;
}
