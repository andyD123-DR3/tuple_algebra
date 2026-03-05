// examples/fix_p99_beam_search.cpp -- SVR + Beam Search for FIX parser p99
//
// Extends fix_p99_svr_search.cpp with a beam search phase that uses
// SVR predictions to guide exploration through the 4^12 strategy space.
//
// Five phases:
//   1. Measure training set (200 random configs, batched rdtsc)
//   2. Train SVR cost model with 5-fold CV hyperparameter tuning
//   3. Beam search over 12 fields using SVR as scoring function
//   4. Verify beam search winners with real measurements
//   5. Compare: direct best vs SVR-predicted best vs beam search best
//
// The beam search works field-by-field:
//   - Start with W seed configs (best from training data)
//   - For each field position 0..11:
//       - For each beam member, try all 4 strategies at that field
//       - Score all candidates with SVR (cheap, no compilation needed)
//       - Keep top W by SVR-predicted p99
//   - Final beam: W configs, all scored by SVR
//   - Measure the top candidates to verify SVR's ranking
//
// This demonstrates the key CT-DP workflow: cheap model steers expensive
// measurement, beam search navigates exponential space efficiently.
//
// Build: cmake --build build --target fix_p99_beam_search
// Run:   ./build/examples/fix_p99_beam_search [--quick] [--csv]

#include <ctdp/calibrator/fix_et_parser.h>
#include <ctdp/solver/cost_models/linear_model.h>
#include <ctdp/solver/cost_models/svr_model.h>
#include <ctdp/solver/cost_models/cross_validation.h>
#include <ctdp/solver/cost_models/performance_model.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

namespace fix = ctdp::calibrator::fix;
namespace cm  = ctdp::cost_models;

// Phase 10g optimal from prior investigation
inline constexpr fix::fix_config phase10g_optimal = {
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
    fix::Strategy::SWAR,     fix::Strategy::Loop,
    fix::Strategy::SWAR,     fix::Strategy::Unrolled,
    fix::Strategy::Unrolled, fix::Strategy::SWAR,
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
}; // UUSLSUUSUUUU


// =====================================================================
// 1. Point type adapter
// =====================================================================

struct fix_point {
    fix::fix_config config;

    std::array<double, fix::num_fields> dims() const {
        std::array<double, fix::num_fields> d{};
        for (int i = 0; i < fix::num_fields; ++i)
            d[i] = static_cast<double>(config[static_cast<std::size_t>(i)]);
        return d;
    }
};

// =====================================================================
// 2. One-hot feature extractor (36 features, no collinearity)
// =====================================================================

struct fix_onehot_extractor {
    std::vector<double> operator()(const fix_point& p) const {
        std::vector<double> f(fix::num_fields * 3, 0.0);
        for (int i = 0; i < fix::num_fields; ++i) {
            int s = static_cast<int>(
                p.config[static_cast<std::size_t>(i)]);
            if (s < 3) {
                f[static_cast<std::size_t>(i * 3 + s)] = 1.0;
            }
        }
        return f;
    }

    static constexpr const char* feature_name() {
        return "onehot_per_field";
    }
};


// =====================================================================
// 3. Dispatch tables (200 train + 100 verify + 5 baselines = 305)
// =====================================================================

constexpr auto train_pool  = fix::generate_random_configs<200>(12345);
constexpr auto verify_pool = fix::generate_random_configs<100>(67890);

struct config_entry {
    fix::fix_config config;
    const char* group;
    ctdp::bench::percentile_result (*measure)(
        std::vector<std::string> const&, std::size_t);
};

template<fix::fix_config Cfg>
ctdp::bench::percentile_result measure_wrapper(
        std::vector<std::string> const& pool,
        std::size_t samples) {
    return fix::measure_config<Cfg>(pool, samples);
}

template<auto const& Configs, std::size_t... Is>
constexpr auto make_entries_impl(
        const char* group, std::index_sequence<Is...>)
{
    return std::array<config_entry, sizeof...(Is)>{{
        config_entry{
            Configs[Is], group,
            &measure_wrapper<Configs[Is]>
        }...
    }};
}

template<auto const& Configs, std::size_t N>
constexpr auto make_entries(const char* group)
{
    return make_entries_impl<Configs>(group,
        std::make_index_sequence<N>{});
}

constexpr auto train_entries  = make_entries<train_pool, 200>("train");
constexpr auto verify_entries = make_entries<verify_pool, 100>("verify");

constexpr auto baseline_entries = std::array<config_entry, 5>{{
    {fix::all_unrolled, "baseline", &measure_wrapper<fix::all_unrolled>},
    {fix::all_swar,     "baseline", &measure_wrapper<fix::all_swar>},
    {fix::all_loop,     "baseline", &measure_wrapper<fix::all_loop>},
    {fix::all_generic,  "baseline", &measure_wrapper<fix::all_generic>},
    {phase10g_optimal,  "phase10g", &measure_wrapper<phase10g_optimal>},
}};


// =====================================================================
// 4. Beam search using SVR as scoring function
// =====================================================================

struct beam_entry {
    fix::fix_config config;
    double svr_score;   // SVR-predicted p99

    bool operator<(const beam_entry& o) const {
        return svr_score < o.svr_score;
    }
};

// Config to string key for deduplication
inline std::string config_key(const fix::fix_config& c) {
    return fix::config_to_string(c);
}

template<typename Model>
std::vector<beam_entry> beam_search(
        const Model& svr,
        const std::vector<fix::fix_config>& seeds,
        std::size_t beam_width,
        bool verbose = false)
{
    // Initialize beam with seeds
    std::vector<beam_entry> beam;
    beam.reserve(seeds.size());
    for (auto const& cfg : seeds) {
        fix_point pt{cfg};
        double score = svr.predict(pt);
        beam.push_back(beam_entry{cfg, score});
    }
    std::sort(beam.begin(), beam.end());
    if (beam.size() > beam_width)
        beam.resize(beam_width);

    if (verbose) {
        std::printf("  Beam search: width=%zu, 12 fields, 4 strategies\n",
            beam_width);
        std::printf("  Initial beam best: %s  predicted=%.1f ns\n",
            fix::config_to_string(beam[0].config).c_str(),
            beam[0].svr_score);
    }

    // Sweep each field position
    for (int field = 0; field < fix::num_fields; ++field) {
        std::set<std::string> seen;
        std::vector<beam_entry> candidates;
        candidates.reserve(beam.size() * 4);

        for (auto const& entry : beam) {
            // Try all 4 strategies at this field
            for (int s = 0; s < 4; ++s) {
                fix::fix_config trial = entry.config;
                trial[static_cast<std::size_t>(field)] =
                    static_cast<fix::Strategy>(s);

                std::string key = config_key(trial);
                if (seen.count(key)) continue;
                seen.insert(key);

                fix_point pt{trial};
                double score = svr.predict(pt);
                candidates.push_back(beam_entry{trial, score});
            }
        }

        // Keep top beam_width
        std::sort(candidates.begin(), candidates.end());
        if (candidates.size() > beam_width)
            candidates.resize(beam_width);
        beam = std::move(candidates);

        if (verbose) {
            std::printf("  Field %2d (%d digits): beam best=%s  predicted=%.1f ns"
                        "  candidates=%zu\n",
                field, fix::field_digits[static_cast<std::size_t>(field)],
                fix::config_to_string(beam[0].config).c_str(),
                beam[0].svr_score,
                seen.size());
        }
    }

    return beam;
}


// =====================================================================
// 5. Main
// =====================================================================

struct options {
    bool quick          = false;
    bool csv_mode       = false;
    std::size_t train_count   = 200;
    std::size_t predict_count = 10000;
    std::size_t verify_count  = 50;
    std::size_t beam_width    = 20;
    std::size_t samples       = 50000;
    std::size_t pool_size     = 5000;
};

int main(int argc, char** argv) {
    options opts;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            opts.quick = true;
            opts.train_count   = 100;
            opts.predict_count = 2000;
            opts.verify_count  = 20;
            opts.beam_width    = 10;
            opts.samples       = 20000;
            opts.pool_size     = 2000;
        } else if (std::strcmp(argv[i], "--csv") == 0) {
            opts.csv_mode = true;
        }
    }

    if (opts.train_count > 200) opts.train_count = 200;
    if (opts.verify_count > 100) opts.verify_count = 100;

    if (!opts.csv_mode) {
        std::printf(
            "+-------------------------------------------------------+\n"
            "|  FIX Parser p99: SVR + Beam Search                    |\n"
            "+-------------------------------------------------------+\n\n");

        std::printf("  Training configs:    %zu (measured)\n", opts.train_count);
        std::printf("  Prediction pool:     %zu (SVR-evaluated)\n", opts.predict_count);
        std::printf("  Beam width:          %zu\n", opts.beam_width);
        std::printf("  Verification:        %zu (measured)\n", opts.verify_count);
        std::printf("  Samples/config:      %zu\n", opts.samples);
        std::printf("  Batch size:          64 parses/sample\n");
        std::printf("  Feature encoding:    one-hot per field (36 features)\n");
        std::printf("  Model:               SVR (RBF kernel, 2-stage CV tuning)\n");
        std::printf("  Fields:              %d\n", fix::num_fields);
        std::printf("  Strategy space:      4^12 = 16777216 configurations\n\n");
    }

    // -- Calibrate TSC ------------------------------------------------
    if (!opts.csv_mode) {
        std::printf("Calibrating TSC... ");
        std::fflush(stdout);
    }
    double cpns = fix::calibrate_tsc();
    if (!opts.csv_mode) {
        std::printf("%.3f GHz (%.2f cycles/ns)\n", cpns, cpns);
    }

    // -- Generate message pool ----------------------------------------
    if (!opts.csv_mode) {
        std::printf("Generating %zu-message pool... ", opts.pool_size);
        std::fflush(stdout);
    }
    auto messages = fix::generate_message_pool(opts.pool_size);
    if (!opts.csv_mode) std::printf("done.\n\n");

    // =================================================================
    // Phase 1: Measure training set
    // =================================================================
    if (!opts.csv_mode) {
        std::printf("=== Phase 1: Measuring %zu training configs ===\n\n",
                    opts.train_count);
    }

    using obs_t = cm::observation<fix_point>;
    std::vector<obs_t> training_obs;
    training_obs.reserve(opts.train_count);

    auto t_start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < opts.train_count; ++i) {
        auto const& entry = train_entries[i];
        auto pctl = entry.measure(messages, opts.samples);

        fix_point pt{entry.config};
        training_obs.push_back(obs_t{pt, pctl.p99});

        if (!opts.csv_mode) {
            std::printf("  [%3zu/%3zu] %s   mean=%6.1f  p50=%6.1f  p99=%6.1f\n",
                i + 1, opts.train_count,
                fix::config_to_string(entry.config).c_str(),
                pctl.mean, pctl.p50, pctl.p99);
        }
    }

    auto t_train_end = std::chrono::steady_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(
        t_train_end - t_start).count();

    if (!opts.csv_mode) {
        std::printf("\n  Training data: %zu configs in %.1f ms (%.0f configs/sec)\n",
            opts.train_count, train_ms,
            static_cast<double>(opts.train_count) * 1000.0 / train_ms);

        double min_p99 = 1e9, max_p99 = 0, sum_p99 = 0;
        for (auto const& o : training_obs) {
            min_p99 = std::min(min_p99, o.cost);
            max_p99 = std::max(max_p99, o.cost);
            sum_p99 += o.cost;
        }
        std::printf("  p99 range: %.1f - %.1f ns  (mean=%.1f)\n\n",
            min_p99, max_p99, sum_p99 / static_cast<double>(opts.train_count));
    }

    // =================================================================
    // Phase 2: Train SVR model
    // =================================================================
    if (!opts.csv_mode) {
        std::printf("=== Phase 2: Training SVR (RBF kernel, 2-stage CV) ===\n\n");
    }

    auto t_svr_start = std::chrono::steady_clock::now();

    cm::svr_trainer<fix_point, fix_onehot_extractor> trainer{
        fix_onehot_extractor{}};
    auto svr = trainer.build(training_obs);

    auto t_svr_end = std::chrono::steady_clock::now();
    double svr_ms = std::chrono::duration<double, std::milli>(
        t_svr_end - t_svr_start).count();

    auto q = svr.quality();
    if (!opts.csv_mode) {
        std::printf("  SVR trained in %.1f ms\n", svr_ms);
        std::printf("  Hyperparameters: lambda=%.4f  gamma=%.4f\n",
            svr.lambda(), svr.gamma());
        std::printf("  In-sample:    R^2=%.4f  Spearman=%.4f\n",
            q.r2, q.spearman_rho);
        std::printf("  Out-of-sample: R^2=%.4f  Spearman=%.4f  RMSE=%.2f ns\n",
            q.oos_r2, q.oos_spearman_rho, q.oos_rmse);
        std::printf("  Training points: %zu  Features: 36 (one-hot per field)\n\n",
            q.n_samples);
    }

    // =================================================================
    // Phase 3: Beam search using SVR predictions
    // =================================================================
    if (!opts.csv_mode) {
        std::printf("=== Phase 3: Beam search (width=%zu, SVR-guided) ===\n\n",
                    opts.beam_width);
    }

    auto t_beam_start = std::chrono::steady_clock::now();

    // Seed beam with top training configs (by measured p99)
    std::vector<std::size_t> train_ranked(opts.train_count);
    std::iota(train_ranked.begin(), train_ranked.end(), 0);
    std::sort(train_ranked.begin(), train_ranked.end(),
        [&](auto a, auto b) {
            return training_obs[a].cost < training_obs[b].cost;
        });

    // Take top 2*beam_width as seeds (measured-good configs)
    std::size_t n_seeds = std::min(opts.beam_width * 2, opts.train_count);
    std::vector<fix::fix_config> seeds;
    seeds.reserve(n_seeds);
    for (std::size_t i = 0; i < n_seeds; ++i)
        seeds.push_back(training_obs[train_ranked[i]].point.config);

    auto beam_result = beam_search(svr, seeds, opts.beam_width,
                                   !opts.csv_mode);

    auto t_beam_end = std::chrono::steady_clock::now();
    double beam_ms = std::chrono::duration<double, std::milli>(
        t_beam_end - t_beam_start).count();

    if (!opts.csv_mode) {
        std::printf("\n  Beam search completed in %.1f ms\n", beam_ms);
        std::printf("  Final beam (%zu configs):\n", beam_result.size());
        for (std::size_t i = 0; i < beam_result.size() && i < 10; ++i) {
            std::printf("    #%-2zu  %s  predicted=%.1f ns\n",
                i + 1,
                fix::config_to_string(beam_result[i].config).c_str(),
                beam_result[i].svr_score);
        }
        std::printf("\n");
    }

    // =================================================================
    // Phase 4: Verify with real measurements
    // =================================================================
    if (!opts.csv_mode) {
        std::printf("=== Phase 4: Verification ===\n\n");
    }

    // 4a. Measure verify pool (independent configs SVR has never seen)
    struct verified_result {
        fix::fix_config config;
        const char* source;
        double predicted_p99;
        double actual_p99;
        double actual_mean;
    };

    std::vector<verified_result> verified;

    if (!opts.csv_mode) {
        std::printf("  Measuring %zu independent verification configs...\n",
                    opts.verify_count);
    }

    for (std::size_t i = 0; i < opts.verify_count; ++i) {
        auto const& entry = verify_entries[i];
        auto pctl = entry.measure(messages, opts.samples);

        fix_point pt{entry.config};
        double pred = svr.predict(pt);

        verified.push_back(verified_result{
            entry.config, "verify", pred, pctl.p99, pctl.mean});

        if (!opts.csv_mode) {
            std::printf("  [%3zu/%3zu] %s   predicted=%6.1f  actual=%6.1f"
                        "  err=%+.1f\n",
                i + 1, opts.verify_count,
                fix::config_to_string(entry.config).c_str(),
                pred, pctl.p99, pred - pctl.p99);
        }
    }

    // 4b. OOS metrics on verification set
    std::vector<double> v_actual, v_predicted;
    for (auto const& vr : verified) {
        v_actual.push_back(vr.actual_p99);
        v_predicted.push_back(vr.predicted_p99);
    }

    double verify_r2 = cm::compute_r2(v_actual, v_predicted);
    double verify_spearman = cm::compute_spearman(v_actual, v_predicted);
    double verify_rmse = cm::compute_rmse(v_actual, v_predicted);

    auto t_verify_end = std::chrono::steady_clock::now();
    double verify_ms = std::chrono::duration<double, std::milli>(
        t_verify_end - t_beam_end).count();

    if (!opts.csv_mode) {
        std::printf("\n  Verification: %zu configs in %.1f ms\n",
            opts.verify_count, verify_ms);
        std::printf("  Hold-out R^2=%.4f  Spearman=%.4f  RMSE=%.2f ns\n\n",
            verify_r2, verify_spearman, verify_rmse);
    }

    // =================================================================
    // Phase 5: Comparison
    // =================================================================

    // Measure baselines
    struct baseline_result {
        const char* name;
        fix::fix_config config;
        double p99;
        double mean;
    };
    std::vector<baseline_result> baselines;

    if (!opts.csv_mode) {
        std::printf("=== Phase 5: Comparison ===\n\n");
        std::printf("  Measuring baselines...\n");
    }

    for (auto const& entry : baseline_entries) {
        auto pctl = entry.measure(messages, opts.samples);
        baselines.push_back(baseline_result{
            entry.group, entry.config, pctl.p99, pctl.mean});
    }

    // Find bests from each source
    auto best_train = std::min_element(training_obs.begin(), training_obs.end(),
        [](auto const& a, auto const& b) { return a.cost < b.cost; });

    auto best_verify = std::min_element(verified.begin(), verified.end(),
        [](auto const& a, auto const& b) {
            return a.actual_p99 < b.actual_p99;
        });

    // Best SVR-predicted from training data
    double best_svr_pred = 1e9;
    std::size_t best_svr_idx = 0;
    for (std::size_t i = 0; i < training_obs.size(); ++i) {
        double pred = svr.predict(training_obs[i].point);
        if (pred < best_svr_pred) {
            best_svr_pred = pred;
            best_svr_idx = i;
        }
    }

    // Beam search winner (SVR-predicted, not measured)
    auto const& beam_winner = beam_result[0];

    // Check if beam winner was in training set (so we have a measurement)
    std::string beam_key = config_key(beam_winner.config);
    double beam_actual = -1.0;
    for (std::size_t i = 0; i < training_obs.size(); ++i) {
        if (config_key(training_obs[i].point.config) == beam_key) {
            beam_actual = training_obs[i].cost;
            break;
        }
    }

    if (!opts.csv_mode) {
        std::printf("\n  --- Results Summary ---\n\n");

        std::printf("  %-16s  %-14s  %8s  %8s  %8s\n",
            "Source", "Config", "p99(ns)", "SVR pred", "method");
        std::printf("  %-16s  %-14s  %8s  %8s  %8s\n",
            "----------------", "--------------",
            "--------", "--------", "--------");

        // Baselines
        const char* baseline_names[] = {
            "All-Unrolled", "All-SWAR", "All-Loop", "All-Generic", "Phase10g"};
        for (std::size_t i = 0; i < baselines.size(); ++i) {
            fix_point pt{baselines[i].config};
            double pred = svr.predict(pt);
            std::printf("  %-16s  %-14s  %8.1f  %8.1f  %8s\n",
                baseline_names[i],
                fix::config_to_string(baselines[i].config).c_str(),
                baselines[i].p99, pred, "measured");
        }

        // Best from training (direct measurement)
        {
            fix_point pt{best_train->point.config};
            double pred = svr.predict(pt);
            std::printf("  %-16s  %-14s  %8.1f  %8.1f  %8s\n",
                "Best(train)",
                fix::config_to_string(best_train->point.config).c_str(),
                best_train->cost, pred, "measured");
        }

        // Best SVR-recommended from training
        {
            std::printf("  %-16s  %-14s  %8.1f  %8.1f  %8s\n",
                "SVR-best(train)",
                fix::config_to_string(
                    training_obs[best_svr_idx].point.config).c_str(),
                training_obs[best_svr_idx].cost, best_svr_pred, "measured");
        }

        // Beam search winner
        {
            if (beam_actual > 0) {
                std::printf("  %-16s  %-14s  %8.1f  %8.1f  %8s\n",
                    "Beam winner",
                    fix::config_to_string(beam_winner.config).c_str(),
                    beam_actual, beam_winner.svr_score, "measured");
            } else {
                std::printf("  %-16s  %-14s  %8s  %8.1f  %8s\n",
                    "Beam winner",
                    fix::config_to_string(beam_winner.config).c_str(),
                    "--", beam_winner.svr_score, "SVR only");
            }
        }

        // Best from verification
        {
            std::printf("  %-16s  %-14s  %8.1f  %8.1f  %8s\n",
                "Best(verify)",
                fix::config_to_string(best_verify->config).c_str(),
                best_verify->actual_p99, best_verify->predicted_p99,
                "measured");
        }

        // --- Beam search analysis ---
        std::printf("\n  --- Beam Search Analysis ---\n\n");

        // Strategy frequency in beam
        std::array<int, 4> strategy_counts{};
        for (auto const& be : beam_result) {
            for (int f = 0; f < fix::num_fields; ++f) {
                int s = static_cast<int>(
                    be.config[static_cast<std::size_t>(f)]);
                strategy_counts[static_cast<std::size_t>(s)]++;
            }
        }
        double total_slots = static_cast<double>(
            beam_result.size() * static_cast<std::size_t>(fix::num_fields));

        std::printf("  Strategy distribution in final beam (%zu configs):\n",
            beam_result.size());
        std::printf("    Unrolled: %5.1f%%\n",
            100.0 * static_cast<double>(strategy_counts[0]) / total_slots);
        std::printf("    SWAR:     %5.1f%%\n",
            100.0 * static_cast<double>(strategy_counts[1]) / total_slots);
        std::printf("    Loop:     %5.1f%%\n",
            100.0 * static_cast<double>(strategy_counts[2]) / total_slots);
        std::printf("    Generic:  %5.1f%%\n",
            100.0 * static_cast<double>(strategy_counts[3]) / total_slots);

        // Per-field consensus in beam
        std::printf("\n  Per-field strategy consensus (beam top %zu):\n",
            beam_result.size());
        std::printf("    Field  Digits  U    S    L    G    Consensus\n");
        std::printf("    -----  ------  ---  ---  ---  ---  ---------\n");

        for (int f = 0; f < fix::num_fields; ++f) {
            std::array<int, 4> fc{};
            for (auto const& be : beam_result) {
                int s = static_cast<int>(
                    be.config[static_cast<std::size_t>(f)]);
                fc[static_cast<std::size_t>(s)]++;
            }
            int max_count = *std::max_element(fc.begin(), fc.end());
            int max_strategy = 0;
            for (int s = 0; s < 4; ++s) {
                if (fc[static_cast<std::size_t>(s)] == max_count)
                    max_strategy = s;
            }
            const char* snames[] = {"U", "S", "L", "G"};
            double pct = 100.0 * static_cast<double>(max_count) /
                         static_cast<double>(beam_result.size());

            std::printf("    %5d  %6d  %3d  %3d  %3d  %3d  %s (%.0f%%)\n",
                f,
                fix::field_digits[static_cast<std::size_t>(f)],
                fc[0], fc[1], fc[2], fc[3],
                snames[max_strategy], pct);
        }

        // --- SVR model quality ---
        std::printf("\n  --- SVR Model Quality ---\n\n");
        std::printf("  Training:     %zu observations, 36 one-hot features\n",
            q.n_samples);
        std::printf("  In-sample:    R^2=%.4f  Spearman=%.4f\n",
            q.r2, q.spearman_rho);
        std::printf("  5-fold CV:    R^2=%.4f  Spearman=%.4f  RMSE=%.2f ns\n",
            q.oos_r2, q.oos_spearman_rho, q.oos_rmse);
        std::printf("  Hold-out:     R^2=%.4f  Spearman=%.4f  RMSE=%.2f ns\n",
            verify_r2, verify_spearman, verify_rmse);

        // --- Timing ---
        std::printf("\n  --- Timing ---\n\n");
        double total_ms = std::chrono::duration<double, std::milli>(
            t_verify_end - t_start).count();
        std::printf("  Phase 1 (measure %zu train):     %8.1f ms\n",
            opts.train_count, train_ms);
        std::printf("  Phase 2 (train SVR):            %8.1f ms\n", svr_ms);
        std::printf("  Phase 3 (beam search):          %8.1f ms\n", beam_ms);
        std::printf("  Phase 4 (verify %zu):             %8.1f ms\n",
            opts.verify_count, verify_ms);
        std::printf("  Total:                          %8.1f ms\n", total_ms);

        std::printf("\n  Key insight: beam search explores 12*4*W = %zu\n"
                    "  SVR evaluations to navigate 16.7M configs.\n"
                    "  Cost: %.0f measurements + %.0f SVR predictions.\n\n",
            static_cast<std::size_t>(12 * 4) * opts.beam_width,
            static_cast<double>(opts.train_count + opts.verify_count + 5),
            static_cast<double>(12 * 4) * static_cast<double>(opts.beam_width));
    }

    // CSV output
    if (opts.csv_mode) {
        std::printf("phase,config,group,actual_p99,svr_predicted_p99\n");

        for (std::size_t i = 0; i < training_obs.size(); ++i) {
            double pred = svr.predict(training_obs[i].point);
            std::printf("train,%s,train,%.2f,%.2f\n",
                fix::config_to_string(
                    training_obs[i].point.config).c_str(),
                training_obs[i].cost, pred);
        }

        for (auto const& be : beam_result) {
            std::printf("beam,%s,beam,--,%.2f\n",
                fix::config_to_string(be.config).c_str(),
                be.svr_score);
        }

        for (auto const& vr : verified) {
            std::printf("verify,%s,verify,%.2f,%.2f\n",
                fix::config_to_string(vr.config).c_str(),
                vr.actual_p99, vr.predicted_p99);
        }

        for (std::size_t i = 0; i < baselines.size(); ++i) {
            fix_point pt{baselines[i].config};
            double pred = svr.predict(pt);
            std::printf("baseline,%s,%s,%.2f,%.2f\n",
                fix::config_to_string(baselines[i].config).c_str(),
                baselines[i].name, baselines[i].p99, pred);
        }
    }

    return 0;
}
