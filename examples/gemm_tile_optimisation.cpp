// ctdp v0.7.0 — Example: GEMM Tile Configuration Optimisation
//
// Full pipeline:
//   1. Simulate noisy hardware benchmarks for a GEMM tile kernel
//   2. Preprocess: remove OS-jitter spike outliers, normalise targets
//   3. Train 3 competing models (linear, MLP, SVR) with 5-fold CV
//   4. Factory selects winner by out-of-sample Spearman ρ
//   5. Exhaustive search over tile space → predicted optimum
//   6. Compare to ground truth
//
// This parallels the FIX parser calibration story: measure a subset of
// the configuration space, learn a surrogate cost model, then search
// the full space to find the optimal plan.

#include "ctdp/solver/cost_models/performance_model.h"
#include "ctdp/solver/cost_models/feature_extract.h"
#include "ctdp/solver/cost_models/data_preprocess.h"
#include "ctdp/solver/cost_models/model_factory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace ctdp::cost_models;

// ─── Tile shape ─────────────────────────────────────────────────────────

struct tile_shape {
    int tm, tn, tk;
    auto dims() const { return std::array<int, 3>{tm, tn, tk}; }
};

std::ostream& operator<<(std::ostream& os, const tile_shape& ts) {
    return os << "(" << ts.tm << "," << ts.tn << "," << ts.tk << ")";
}

// ─── Simulated hardware cost function ───────────────────────────────────
// Models a realistic GEMM tile kernel where performance depends on:
//   - Register pressure (too-large tiles spill to stack)
//   - L1 occupancy (tile footprint vs 32KB L1)
//   - Loop overhead (too-small tiles waste time on loop control)
//   - Vectorisation alignment (TK multiples of 8 get a bonus)
//   - Interaction effects (TM × TN determines output tile area)
//
// True optimum: TM=16, TN=32, TK=8  →  cost ≈ 10.0 ns/element

namespace ground_truth {

double true_cost(const tile_shape& ts) {
    double ltm = std::log2(ts.tm);
    double ltn = std::log2(ts.tn);
    double ltk = std::log2(ts.tk);

    // Quadratic bowl centred at (log2(16), log2(32), log2(8)) = (4, 5, 3)
    double cost = 10.0
        + 2.5 * (ltm - 4.0) * (ltm - 4.0)
        + 1.8 * (ltn - 5.0) * (ltn - 5.0)
        + 3.0 * (ltk - 3.0) * (ltk - 3.0);

    // Interaction: large output tile area stresses register file
    cost += 0.4 * (ltm - 4.0) * (ltn - 5.0);

    // Reciprocal: loop overhead dominates for tiny tiles
    cost += 8.0 / ts.tm + 6.0 / ts.tn + 4.0 / ts.tk;

    // L1 pressure: total tile footprint in doubles
    // A tile: TM×TK, B tile: TK×TN, C tile: TM×TN  (all doubles = 8 bytes)
    double bytes = 8.0 * (ts.tm * ts.tk + ts.tk * ts.tn + ts.tm * ts.tn);
    double l1 = 32768.0;  // 32KB L1
    if (bytes > l1)        cost += 0.005 * (bytes - l1);      // L1 spill penalty
    if (bytes > 4.0 * l1)  cost += 0.01  * (bytes - 4 * l1);  // severe spill

    // Vectorisation: TK divisible by 8 (AVX-512 doubles) gets a bonus
    if (ts.tk % 8 == 0) cost -= 1.5;

    return cost;
}

// Add measurement noise and occasional OS-jitter spikes
double noisy_measure(const tile_shape& ts, std::mt19937& rng) {
    double base = true_cost(ts);
    std::normal_distribution<double> noise(0.0, 0.6);
    double measured = base + noise(rng);

    // ~5% chance of a spike (context switch, cache flush, TLB miss storm)
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) < 0.05) {
        std::uniform_real_distribution<double> spike(30.0, 80.0);
        measured += spike(rng);
    }

    return std::max(measured, 0.1);
}

} // namespace ground_truth

// ─── Search space ───────────────────────────────────────────────────────
// Powers of 2 from 2..64 for each dimension → 6³ = 216 configurations

std::vector<tile_shape> build_tile_space() {
    std::vector<tile_shape> space;
    for (int tm : {2, 4, 8, 16, 32, 64})
        for (int tn : {2, 4, 8, 16, 32, 64})
            for (int tk : {2, 4, 8, 16, 32, 64})
                space.push_back({tm, tn, tk});
    return space;
}

// ─── Reporting ──────────────────────────────────────────────────────────

void rule(char c = '-', int w = 72) { std::cerr << std::string(static_cast<std::size_t>(w), c) << "\n"; }

void print_quality_row(const model_quality& q) {
    std::cerr << std::fixed << std::setprecision(4)
              << "  " << std::setw(26) << std::left << q.model_name
              << "  ρ_in=" << std::setw(7) << q.spearman_rho
              << "  ρ_oos=" << std::setw(7) << q.oos_spearman_rho
              << "  R²_oos=" << std::setw(7) << q.oos_r2
              << "  RMSE_oos=" << std::setw(7) << q.oos_rmse
              << "\n";
}

// ═══════════════════════════════════════════════════════════════════════
int main() {
    std::cerr << R"(
  ╔══════════════════════════════════════════════════════════════════╗
  ║  ctdp v0.7.0 — GEMM Tile Configuration Optimisation Example   ║
  ║                                                                 ║
  ║  Calibrate → Preprocess → Train → Select → Search → Validate   ║
  ╚══════════════════════════════════════════════════════════════════╝
)" << "\n";

    auto full_space = build_tile_space();

    // ── Ground truth ────────────────────────────────────────────────
    tile_shape true_opt{};
    double true_best = 1e30;
    for (const auto& ts : full_space) {
        double c = ground_truth::true_cost(ts);
        if (c < true_best) { true_best = c; true_opt = ts; }
    }
    std::cerr << "Search space:  " << full_space.size() << " tile configs\n";
    std::cerr << "True optimum:  " << true_opt
              << "  cost = " << std::fixed << std::setprecision(2)
              << true_best << " ns/elem\n\n";

    // ════════════════════════════════════════════════════════════════
    // STEP 1: Calibration — benchmark a random subset
    // ════════════════════════════════════════════════════════════════
    rule('=');
    std::cerr << "STEP 1: Calibration — benchmark 80 of "
              << full_space.size() << " configs (37%)\n";
    rule();

    std::mt19937 rng(42);
    std::vector<std::size_t> idx(full_space.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    constexpr std::size_t N_SAMPLES = 80;
    std::vector<observation<tile_shape>> raw_obs;
    raw_obs.reserve(N_SAMPLES);
    for (std::size_t i = 0; i < N_SAMPLES; ++i) {
        auto& ts = full_space[idx[i]];
        raw_obs.push_back({ts, ground_truth::noisy_measure(ts, rng)});
    }

    // Show a few
    std::cerr << "\n  Sample measurements:\n";
    for (std::size_t i = 0; i < 6; ++i) {
        double truth = ground_truth::true_cost(raw_obs[i].point);
        std::cerr << "    " << raw_obs[i].point
                  << "  measured=" << std::setw(7) << std::setprecision(2)
                  << raw_obs[i].cost
                  << "  true=" << std::setw(7) << truth
                  << "  error=" << std::setw(6) << std::setprecision(1)
                  << (raw_obs[i].cost - truth) << "\n";
    }

    // Count spikes in raw data
    double med = 0.0; {
        std::vector<double> costs;
        for (auto& o : raw_obs) costs.push_back(o.cost);
        std::sort(costs.begin(), costs.end());
        med = costs[costs.size() / 2];
    }
    int n_suspicious = 0;
    for (auto& o : raw_obs)
        if (o.cost > med + 30.0) ++n_suspicious;
    std::cerr << "\n  Suspicious spikes (> median+30): " << n_suspicious
              << " of " << N_SAMPLES << "\n\n";

    // ════════════════════════════════════════════════════════════════
    // STEP 2: Preprocessing — outlier removal + normalisation
    // ════════════════════════════════════════════════════════════════
    rule('=');
    std::cerr << "STEP 2: Preprocessing — 3σ outlier removal + z-score normalise\n";
    rule();

    data_preprocessor<tile_shape> pp;
    pp.set_outlier_sigma(3.0);
    auto pre = pp.fit_transform(raw_obs);

    std::cerr << "\n  Input:              " << pre.n_original << " observations\n";
    if (pre.outliers) {
        auto& ol = *pre.outliers;
        std::cerr << "  Outliers removed:   " << ol.removed_indices.size()
                  << " (robust centre=" << std::setprecision(1) << ol.centre
                  << ", σ=" << ol.spread << ")\n";
        std::cerr << "  Accept window:      [" << std::setprecision(1)
                  << ol.threshold_low << ", " << ol.threshold_high << "]\n";
        if (!ol.removed_observations.empty()) {
            std::cerr << "  Removed values:     ";
            for (std::size_t i = 0; i < ol.removed_observations.size() && i < 5; ++i)
                std::cerr << std::setprecision(1) << ol.removed_observations[i].cost << " ";
            std::cerr << "\n";
        }
    }
    std::cerr << "  Clean observations: " << pre.observations.size() << "\n";
    std::cerr << "  Target transform:   mean=" << std::setprecision(2)
              << pre.transform.mean << "  std=" << pre.transform.std << "\n\n";

    // ════════════════════════════════════════════════════════════════
    // STEP 3: Model training — factory trains all 3 with 5-fold CV
    // ════════════════════════════════════════════════════════════════
    rule('=');
    std::cerr << "STEP 3: Model training — linear / MLP / SVR (5-fold CV)\n";
    rule();
    std::cerr << "\n";

    auto_model_factory<tile_shape> factory(log2_interactions_extractor{}, true);
    auto build = factory.build(pre.observations);

    std::cerr << "\n  Model comparison (sorted by OOS ρ):\n";
    auto qualities = build.all_qualities;
    std::sort(qualities.begin(), qualities.end(),
              [](auto& a, auto& b) {
                  double ra = std::isfinite(a.oos_spearman_rho) ? a.oos_spearman_rho : -9;
                  double rb = std::isfinite(b.oos_spearman_rho) ? b.oos_spearman_rho : -9;
                  return ra > rb;
              });
    for (auto& q : qualities) print_quality_row(q);

    auto winner_q = build.model.quality();
    std::cerr << "\n  ★ Winner: " << build.model.name()
              << " (OOS ρ = " << std::setprecision(4)
              << winner_q.oos_spearman_rho << ")\n\n";

    // ════════════════════════════════════════════════════════════════
    // STEP 4: Wrap model with inverse transform
    // ════════════════════════════════════════════════════════════════
    auto model = make_transformed_model<tile_shape>(
        std::move(build.model), pre.transform);

    // ════════════════════════════════════════════════════════════════
    // STEP 5: Exhaustive search — predict every config, find minimum
    // ════════════════════════════════════════════════════════════════
    rule('=');
    std::cerr << "STEP 5: Exhaustive search — predict all "
              << full_space.size() << " configurations\n";
    rule();

    struct ranked {
        tile_shape ts;
        double predicted;
        double true_cost;
    };
    std::vector<ranked> results;
    results.reserve(full_space.size());
    for (const auto& ts : full_space)
        results.push_back({ts, model.predict(ts), ground_truth::true_cost(ts)});

    std::sort(results.begin(), results.end(),
              [](auto& a, auto& b) { return a.predicted < b.predicted; });

    std::cerr << "\n  Top 10 by predicted cost:\n";
    std::cerr << "  " << std::setw(5) << "Rank"
              << std::setw(16) << "Config"
              << std::setw(12) << "Predicted"
              << std::setw(12) << "True"
              << std::setw(10) << "Error" << "\n";

    for (std::size_t i = 0; i < 10; ++i) {
        auto& r = results[i];
        std::cerr << "  " << std::setw(5) << (i + 1)
                  << "  " << std::setw(14) << std::left << r.ts << std::right
                  << std::setw(12) << std::setprecision(2) << r.predicted
                  << std::setw(12) << r.true_cost
                  << std::setw(10) << std::setprecision(1)
                  << (r.predicted - r.true_cost) << "\n";
    }

    // Worst 3 for contrast
    std::cerr << "\n  Bottom 3 by predicted cost:\n";
    for (std::size_t i = results.size() - 3; i < results.size(); ++i) {
        auto& r = results[i];
        std::cerr << "  " << std::setw(5) << (i + 1)
                  << "  " << std::setw(14) << std::left << r.ts << std::right
                  << std::setw(12) << std::setprecision(2) << r.predicted
                  << std::setw(12) << r.true_cost << "\n";
    }

    // ════════════════════════════════════════════════════════════════
    // STEP 6: Validation — compare predicted vs true optimum
    // ════════════════════════════════════════════════════════════════
    rule('=');
    std::cerr << "STEP 6: Validation\n";
    rule();

    auto& pred_best = results[0];
    std::cerr << "\n  Predicted optimum:  " << pred_best.ts
              << "  predicted=" << std::setprecision(2) << pred_best.predicted
              << "  true=" << pred_best.true_cost << "\n";
    std::cerr << "  True optimum:       " << true_opt
              << "  true=" << true_best << "\n";

    bool found_exact = (pred_best.ts.tm == true_opt.tm &&
                        pred_best.ts.tn == true_opt.tn &&
                        pred_best.ts.tk == true_opt.tk);

    double regret = pred_best.true_cost - true_best;

    std::cerr << "\n  Exact match:   " << (found_exact ? "YES ✓" : "NO") << "\n";
    std::cerr << "  Regret:        " << std::setprecision(2) << regret
              << " ns/elem (" << std::setprecision(1)
              << (100.0 * regret / true_best) << "% of optimum)\n";

    // Where does the true optimum rank in predictions?
    std::size_t true_rank = 0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i].ts.tm == true_opt.tm &&
            results[i].ts.tn == true_opt.tn &&
            results[i].ts.tk == true_opt.tk) {
            true_rank = i + 1;
            break;
        }
    }
    std::cerr << "  True optimum predicted rank: " << true_rank
              << " / " << results.size() << "\n";

    // Spearman ρ between predicted and true across whole space
    std::vector<double> pred_vals, true_vals;
    for (auto& r : results) {
        pred_vals.push_back(r.predicted);
        true_vals.push_back(r.true_cost);
    }
    double global_rho = compute_spearman(
        std::span<const double>(true_vals),
        std::span<const double>(pred_vals));
    std::cerr << "  Ranking correlation (full space): ρ = "
              << std::setprecision(4) << global_rho << "\n";

    // ════════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════════
    rule('=');
    std::cerr << "SUMMARY\n";
    rule();
    std::cerr << "\n  Measured:          " << N_SAMPLES << " of "
              << full_space.size() << " configs ("
              << (100 * N_SAMPLES / full_space.size()) << "%)\n";
    std::cerr << "  Outliers removed:  "
              << (pre.outliers ? pre.outliers->removed_indices.size() : 0) << "\n";
    std::cerr << "  Best model:        " << model.name() << "\n";
    std::cerr << "  OOS Spearman ρ:    " << std::setprecision(4)
              << winner_q.oos_spearman_rho << "\n";
    std::cerr << "  Predicted optimum: " << pred_best.ts << "\n";
    std::cerr << "  True optimum:      " << true_opt << "\n";
    std::cerr << "  Regret:            " << std::setprecision(2) << regret
              << " ns/elem\n";
    std::cerr << "  Ranking ρ:         " << std::setprecision(4) << global_rho
              << "\n\n";

    if (found_exact)
        std::cerr << "  → Model found the exact optimal tile configuration.\n";
    else if (regret < 1.0)
        std::cerr << "  → Model found a near-optimal config (< 1 ns/elem regret).\n";
    else
        std::cerr << "  → Model missed the optimum by " << std::setprecision(1)
                  << regret << " ns/elem.\n";

    std::cerr << "\n";
    return found_exact ? 0 : (regret < 2.0 ? 0 : 1);
}
