// examples/fix_p99_beam_search.cpp -- SVR + Beam Search with hardware counters
//
// Six phases:
//   1. Measure training set with hardware counters (200 configs)
//      -> p99 timing + IPC + L1D/L1I/DTLB miss rates per config
//   2. Train SVR models: one per metric (p99, IPC, L1D, L1I, DTLB)
//   3. Beam search using SVR-predicted p99 as scoring function
//   4. Verify beam search winners with real measurements + counters
//   5. Compare: direct best vs SVR-predicted best vs beam search best
//   6. Per-field consensus analysis across all metrics
//
// Hardware counters (Linux only, Tier 1 + Tier 2):
//   Tier 1: instructions, branches, branch misses, LL cache refs/misses
//   Tier 2: L1D read access/miss, L1I read access/miss, DTLB read access/miss
//   On Windows/non-Linux: counters read as zero, timing still works.
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
#include <memory>
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
        for (std::size_t i = 0; i < static_cast<std::size_t>(fix::num_fields); ++i)
            d[i] = static_cast<double>(config[i]);
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
// 3. Dispatch tables using config_metrics (timing + counters)
// =====================================================================

constexpr auto train_pool  = fix::generate_random_configs<200>(12345);
constexpr auto verify_pool = fix::generate_random_configs<100>(67890);

struct config_entry {
    fix::fix_config config;
    const char* group;
    fix::config_metrics (*measure_full)(
        std::vector<std::string> const&, fix::measurement_config const&);
};

template<fix::fix_config Cfg>
fix::config_metrics measure_full_wrapper(
        std::vector<std::string> const& pool,
        fix::measurement_config const& cfg) {
    return fix::measure_config_with_counters<Cfg>(pool, cfg);
}

template<auto const& Configs, std::size_t... Is>
constexpr auto make_entries_impl(
        const char* group, std::index_sequence<Is...>)
{
    return std::array<config_entry, sizeof...(Is)>{{
        config_entry{
            Configs[Is], group,
            &measure_full_wrapper<Configs[Is]>
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
    {fix::all_unrolled, "baseline", &measure_full_wrapper<fix::all_unrolled>},
    {fix::all_swar,     "baseline", &measure_full_wrapper<fix::all_swar>},
    {fix::all_loop,     "baseline", &measure_full_wrapper<fix::all_loop>},
    {fix::all_generic,  "baseline", &measure_full_wrapper<fix::all_generic>},
    {phase10g_optimal,  "phase10g", &measure_full_wrapper<phase10g_optimal>},
}};


// =====================================================================
// 4. Training data row
// =====================================================================

struct training_row {
    fix_point point;
    fix::config_metrics metrics;
};


// =====================================================================
// 5. Beam search using SVR as scoring function
// =====================================================================

struct beam_entry {
    fix::fix_config config;
    double svr_score;

    bool operator<(const beam_entry& o) const {
        return svr_score < o.svr_score;
    }
};

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

    for (int field = 0; field < fix::num_fields; ++field) {
        std::set<std::string> seen;
        std::vector<beam_entry> candidates;
        candidates.reserve(beam.size() * 4);

        for (auto const& entry : beam) {
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
// 6. Main
// =====================================================================

struct options {
    bool quick          = false;
    bool csv_mode       = false;
    std::size_t train_count   = 200;
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

    fix::measurement_config mcfg;
    mcfg.samples       = opts.samples;
    mcfg.batch_size    = 64;
    mcfg.warmup_parses = 4000;

    if (!opts.csv_mode) {
        std::printf(
            "+-------------------------------------------------------+\n"
            "|  FIX Parser p99: SVR + Beam Search + HW Counters      |\n"
            "+-------------------------------------------------------+\n\n");

        std::printf("  Training configs:    %zu (measured + counters)\n", opts.train_count);
        std::printf("  Beam width:          %zu\n", opts.beam_width);
        std::printf("  Verification:        %zu (measured + counters)\n", opts.verify_count);
        std::printf("  Samples/config:      %zu\n", opts.samples);
        std::printf("  Batch size:          64 parses/sample\n");
        std::printf("  Feature encoding:    one-hot per field (36 features)\n");
        std::printf("  Model:               SVR (RBF kernel, 2-stage CV tuning)\n");
        std::printf("  Metrics:             p99, IPC, L1D, L1I(icache), DTLB\n");
        std::printf("  Fields:              %d\n", fix::num_fields);
        std::printf("  Strategy space:      4^12 = 16777216 configurations\n\n");
    }

    // -- Calibrate TSC ------------------------------------------------
    if (!opts.csv_mode) {
        std::printf("Calibrating TSC... ");
        std::fflush(stdout);
    }
    double cpns = fix::calibrate_tsc();
    if (!opts.csv_mode)
        std::printf("%.3f GHz (%.2f cycles/ns)\n", cpns, cpns);
    mcfg.cycles_per_ns = cpns;

    // -- Generate message pool ----------------------------------------
    if (!opts.csv_mode) {
        std::printf("Generating %zu-message pool... ", opts.pool_size);
        std::fflush(stdout);
    }
    auto messages = fix::generate_message_pool(opts.pool_size);
    if (!opts.csv_mode) std::printf("done.\n\n");

    // =================================================================
    // Phase 1: Measure training set with hardware counters
    // =================================================================
    if (!opts.csv_mode)
        std::printf("=== Phase 1: Measuring %zu training configs (timing + counters) ===\n\n",
                    opts.train_count);

    std::vector<training_row> train_data;
    train_data.reserve(opts.train_count);

    auto t_start = std::chrono::steady_clock::now();
    bool counters_t1 = false, counters_t2 = false;

    for (std::size_t i = 0; i < opts.train_count; ++i) {
        auto const& entry = train_entries[i];
        auto m = entry.measure_full(messages, mcfg);
        fix_point pt{entry.config};
        train_data.push_back(training_row{pt, m});

        if (i == 0) {
            counters_t1 = m.tier1_available;
            counters_t2 = m.tier2_available;
        }

        if (!opts.csv_mode) {
            if (counters_t1 || counters_t2) {
                std::printf("  [%3zu/%3zu] %s   p99=%5.1f  IPC=%4.2f"
                            "  L1D=%5.2f%%  L1I=%5.2f%%  DTLB=%5.3f%%\n",
                    i + 1, opts.train_count,
                    fix::config_to_string(entry.config).c_str(),
                    m.timing.p99, m.ipc,
                    m.l1d_miss_rate * 100.0,
                    m.l1i_miss_rate * 100.0,
                    m.dtlb_miss_rate * 100.0);
            } else {
                std::printf("  [%3zu/%3zu] %s   p99=%5.1f  (no HW counters)\n",
                    i + 1, opts.train_count,
                    fix::config_to_string(entry.config).c_str(),
                    m.timing.p99);
            }
        }
    }

    auto t_train_end = std::chrono::steady_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(
        t_train_end - t_start).count();

    if (!opts.csv_mode) {
        std::printf("\n  Training data: %zu configs in %.1f ms (%.0f configs/sec)\n",
            opts.train_count, train_ms,
            static_cast<double>(opts.train_count) * 1000.0 / train_ms);
        std::printf("  Counter availability: Tier1=%s  Tier2=%s\n",
            counters_t1 ? "YES" : "no", counters_t2 ? "YES" : "no");
        if (counters_t1) std::printf("    Tier1: instructions, branches, LL cache\n");
        if (counters_t2) std::printf("    Tier2: L1D, L1I (icache), DTLB\n");

        // Summary table
        double min_p99 = 1e9, max_p99 = 0, sum_p99 = 0;
        double min_ipc = 1e9, max_ipc = 0, sum_ipc = 0;
        double min_l1d = 1e9, max_l1d = 0, sum_l1d = 0;
        double min_l1i = 1e9, max_l1i = 0, sum_l1i = 0;
        double min_dtlb = 1e9, max_dtlb = 0, sum_dtlb = 0;
        for (auto const& row : train_data) {
            auto const& m = row.metrics;
            min_p99 = std::min(min_p99, m.timing.p99);
            max_p99 = std::max(max_p99, m.timing.p99);
            sum_p99 += m.timing.p99;
            min_ipc = std::min(min_ipc, m.ipc); max_ipc = std::max(max_ipc, m.ipc); sum_ipc += m.ipc;
            min_l1d = std::min(min_l1d, m.l1d_miss_rate); max_l1d = std::max(max_l1d, m.l1d_miss_rate); sum_l1d += m.l1d_miss_rate;
            min_l1i = std::min(min_l1i, m.l1i_miss_rate); max_l1i = std::max(max_l1i, m.l1i_miss_rate); sum_l1i += m.l1i_miss_rate;
            min_dtlb = std::min(min_dtlb, m.dtlb_miss_rate); max_dtlb = std::max(max_dtlb, m.dtlb_miss_rate); sum_dtlb += m.dtlb_miss_rate;
        }
        double n = static_cast<double>(opts.train_count);

        std::printf("\n  --- Training Data Summary ---\n\n");
        std::printf("  %-12s  %10s  %10s  %10s\n", "Metric", "Min", "Mean", "Max");
        std::printf("  %-12s  %10s  %10s  %10s\n", "------------", "----------", "----------", "----------");
        std::printf("  %-12s  %8.1f ns  %8.1f ns  %8.1f ns\n", "p99", min_p99, sum_p99/n, max_p99);
        if (counters_t1)
            std::printf("  %-12s  %10.2f  %10.2f  %10.2f\n", "IPC", min_ipc, sum_ipc/n, max_ipc);
        if (counters_t2) {
            std::printf("  %-12s  %9.2f%%  %9.2f%%  %9.2f%%\n", "L1D miss", min_l1d*100, (sum_l1d/n)*100, max_l1d*100);
            std::printf("  %-12s  %9.2f%%  %9.2f%%  %9.2f%%\n", "L1I miss", min_l1i*100, (sum_l1i/n)*100, max_l1i*100);
            std::printf("  %-12s  %9.3f%%  %9.3f%%  %9.3f%%\n", "DTLB miss", min_dtlb*100, (sum_dtlb/n)*100, max_dtlb*100);
        }
        std::printf("\n");
    }

    // =================================================================
    // Phase 2: Train SVR models (one per metric)
    // =================================================================
    if (!opts.csv_mode)
        std::printf("=== Phase 2: Training SVR models ===\n\n");

    auto t_svr_start = std::chrono::steady_clock::now();

    using obs_t = cm::observation<fix_point>;
    using svr_type = cm::svr_model<fix_point, fix_onehot_extractor>;
    cm::svr_trainer<fix_point, fix_onehot_extractor> trainer{fix_onehot_extractor{}};

    std::vector<obs_t> obs_p99, obs_ipc, obs_l1d, obs_l1i, obs_dtlb;
    for (auto const& row : train_data) {
        obs_p99.push_back(obs_t{row.point, row.metrics.timing.p99});
        if (counters_t1)
            obs_ipc.push_back(obs_t{row.point, row.metrics.ipc});
        if (counters_t2) {
            obs_l1d.push_back(obs_t{row.point, row.metrics.l1d_miss_rate});
            obs_l1i.push_back(obs_t{row.point, row.metrics.l1i_miss_rate});
            obs_dtlb.push_back(obs_t{row.point, row.metrics.dtlb_miss_rate});
        }
    }

    auto svr_p99 = trainer.build(obs_p99);
    std::unique_ptr<svr_type> svr_ipc, svr_l1d, svr_l1i, svr_dtlb;
    if (counters_t1 && !obs_ipc.empty())
        svr_ipc = std::make_unique<svr_type>(trainer.build(obs_ipc));
    if (counters_t2 && !obs_l1d.empty()) {
        svr_l1d = std::make_unique<svr_type>(trainer.build(obs_l1d));
        svr_l1i = std::make_unique<svr_type>(trainer.build(obs_l1i));
        svr_dtlb = std::make_unique<svr_type>(trainer.build(obs_dtlb));
    }

    auto t_svr_end = std::chrono::steady_clock::now();
    double svr_ms = std::chrono::duration<double, std::milli>(t_svr_end - t_svr_start).count();

    if (!opts.csv_mode) {
        int n_models = 1 + (svr_ipc ? 1 : 0) + (svr_l1d ? 3 : 0);
        std::printf("  %d SVR models trained in %.1f ms\n\n", n_models, svr_ms);
        std::printf("  %-10s  %8s  %10s  %8s  %10s  %8s\n",
            "Metric", "R^2(in)", "Spear(in)", "R^2(CV)", "Spear(CV)", "RMSE(CV)");
        std::printf("  %-10s  %8s  %10s  %8s  %10s  %8s\n",
            "----------", "--------", "----------", "--------", "----------", "--------");

        auto pq = [](const char* name, const cm::model_quality& q) {
            std::printf("  %-10s  %8.4f  %10.4f  %8.4f  %10.4f  %8.4f\n",
                name, q.r2, q.spearman_rho, q.oos_r2, q.oos_spearman_rho, q.oos_rmse);
        };
        pq("p99", svr_p99.quality());
        if (svr_ipc)  pq("IPC", svr_ipc->quality());
        if (svr_l1d)  pq("L1D miss", svr_l1d->quality());
        if (svr_l1i)  pq("L1I miss", svr_l1i->quality());
        if (svr_dtlb) pq("DTLB miss", svr_dtlb->quality());
        std::printf("\n");
    }

    // =================================================================
    // Phase 3: Beam search using SVR-predicted p99
    // =================================================================
    if (!opts.csv_mode)
        std::printf("=== Phase 3: Beam search (width=%zu, SVR-guided) ===\n\n", opts.beam_width);

    auto t_beam_start = std::chrono::steady_clock::now();

    std::vector<std::size_t> train_ranked(opts.train_count);
    std::iota(train_ranked.begin(), train_ranked.end(), 0);
    std::sort(train_ranked.begin(), train_ranked.end(),
        [&](auto a, auto b) {
            return train_data[a].metrics.timing.p99 < train_data[b].metrics.timing.p99;
        });

    std::size_t n_seeds = std::min(opts.beam_width * 2, opts.train_count);
    std::vector<fix::fix_config> seeds;
    seeds.reserve(n_seeds);
    for (std::size_t i = 0; i < n_seeds; ++i)
        seeds.push_back(train_data[train_ranked[i]].point.config);

    auto beam_result = beam_search(svr_p99, seeds, opts.beam_width, !opts.csv_mode);

    auto t_beam_end = std::chrono::steady_clock::now();
    double beam_ms = std::chrono::duration<double, std::milli>(t_beam_end - t_beam_start).count();

    if (!opts.csv_mode) {
        std::printf("\n  Beam search completed in %.1f ms\n", beam_ms);
        std::printf("  Final beam with multi-metric SVR predictions:\n\n");

        bool has_counters = (svr_ipc || svr_l1d);
        if (has_counters) {
            std::printf("  %3s  %-14s  %8s  %6s  %7s  %7s  %8s\n",
                "#", "Config", "p99(SVR)", "IPC", "L1D%%", "L1I%%", "DTLB%%");
            std::printf("  %3s  %-14s  %8s  %6s  %7s  %7s  %8s\n",
                "---", "--------------", "--------", "------", "-------", "-------", "--------");
        } else {
            std::printf("  %3s  %-14s  %8s\n", "#", "Config", "p99(SVR)");
            std::printf("  %3s  %-14s  %8s\n", "---", "--------------", "--------");
        }

        for (std::size_t i = 0; i < beam_result.size() && i < 10; ++i) {
            fix_point pt{beam_result[i].config};
            if (has_counters) {
                std::printf("  %3zu  %-14s  %8.1f  %6.2f  %6.2f%%  %6.2f%%  %7.3f%%\n",
                    i + 1,
                    fix::config_to_string(beam_result[i].config).c_str(),
                    beam_result[i].svr_score,
                    svr_ipc ? svr_ipc->predict(pt) : 0.0,
                    svr_l1d ? svr_l1d->predict(pt) * 100.0 : 0.0,
                    svr_l1i ? svr_l1i->predict(pt) * 100.0 : 0.0,
                    svr_dtlb ? svr_dtlb->predict(pt) * 100.0 : 0.0);
            } else {
                std::printf("  %3zu  %-14s  %8.1f\n",
                    i + 1,
                    fix::config_to_string(beam_result[i].config).c_str(),
                    beam_result[i].svr_score);
            }
        }
        std::printf("\n");
    }

    // =================================================================
    // Phase 4: Verify with real measurements + counters
    // =================================================================
    if (!opts.csv_mode)
        std::printf("=== Phase 4: Verification (timing + counters) ===\n\n");

    struct verified_row {
        fix::fix_config config;
        fix::config_metrics metrics;
        double pred_p99, pred_ipc, pred_l1d, pred_l1i, pred_dtlb;
    };
    std::vector<verified_row> verified;

    for (std::size_t i = 0; i < opts.verify_count; ++i) {
        auto const& entry = verify_entries[i];
        auto m = entry.measure_full(messages, mcfg);
        fix_point pt{entry.config};

        verified_row vr;
        vr.config = entry.config;
        vr.metrics = m;
        vr.pred_p99  = svr_p99.predict(pt);
        vr.pred_ipc  = svr_ipc ? svr_ipc->predict(pt) : 0.0;
        vr.pred_l1d  = svr_l1d ? svr_l1d->predict(pt) : 0.0;
        vr.pred_l1i  = svr_l1i ? svr_l1i->predict(pt) : 0.0;
        vr.pred_dtlb = svr_dtlb ? svr_dtlb->predict(pt) : 0.0;
        verified.push_back(vr);

        if (!opts.csv_mode) {
            if (counters_t1 || counters_t2) {
                std::printf("  [%3zu/%3zu] %s"
                    "  p99: %5.1f/%5.1f"
                    "  IPC: %4.2f/%4.2f"
                    "  L1D: %5.2f%%/%5.2f%%\n",
                    i + 1, opts.verify_count,
                    fix::config_to_string(entry.config).c_str(),
                    vr.pred_p99, m.timing.p99,
                    vr.pred_ipc, m.ipc,
                    vr.pred_l1d * 100.0, m.l1d_miss_rate * 100.0);
            } else {
                std::printf("  [%3zu/%3zu] %s  pred=%5.1f  actual=%5.1f  err=%+.1f\n",
                    i + 1, opts.verify_count,
                    fix::config_to_string(entry.config).c_str(),
                    vr.pred_p99, m.timing.p99, vr.pred_p99 - m.timing.p99);
            }
        }
    }

    // OOS metrics
    std::vector<double> va_p99, vp_p99, va_ipc, vp_ipc;
    std::vector<double> va_l1d, vp_l1d, va_l1i, vp_l1i, va_dtlb, vp_dtlb;
    for (auto const& v : verified) {
        va_p99.push_back(v.metrics.timing.p99); vp_p99.push_back(v.pred_p99);
        if (counters_t1) { va_ipc.push_back(v.metrics.ipc); vp_ipc.push_back(v.pred_ipc); }
        if (counters_t2) {
            va_l1d.push_back(v.metrics.l1d_miss_rate); vp_l1d.push_back(v.pred_l1d);
            va_l1i.push_back(v.metrics.l1i_miss_rate); vp_l1i.push_back(v.pred_l1i);
            va_dtlb.push_back(v.metrics.dtlb_miss_rate); vp_dtlb.push_back(v.pred_dtlb);
        }
    }

    auto t_verify_end = std::chrono::steady_clock::now();
    double verify_ms = std::chrono::duration<double, std::milli>(t_verify_end - t_beam_end).count();

    if (!opts.csv_mode) {
        std::printf("\n  --- Hold-out Verification (%zu configs) ---\n\n", opts.verify_count);
        std::printf("  %-10s  %8s  %10s  %10s\n", "Metric", "R^2", "Spearman", "RMSE");
        std::printf("  %-10s  %8s  %10s  %10s\n", "----------", "--------", "----------", "----------");

        auto pr = [](const char* name, auto const& a, auto const& p) {
            std::printf("  %-10s  %8.4f  %10.4f  %10.4f\n", name,
                cm::compute_r2(a, p), cm::compute_spearman(a, p), cm::compute_rmse(a, p));
        };
        pr("p99", va_p99, vp_p99);
        if (counters_t1 && !va_ipc.empty())   pr("IPC", va_ipc, vp_ipc);
        if (counters_t2 && !va_l1d.empty()) {
            pr("L1D miss", va_l1d, vp_l1d);
            pr("L1I miss", va_l1i, vp_l1i);
            pr("DTLB miss", va_dtlb, vp_dtlb);
        }
        std::printf("\n");
    }

    // =================================================================
    // Phase 5: Comparison table
    // =================================================================
    if (!opts.csv_mode) {
        std::printf("=== Phase 5: Comparison ===\n\n  Measuring baselines...\n");
    }

    struct baseline_row {
        const char* name;
        fix::fix_config config;
        fix::config_metrics metrics;
    };
    std::vector<baseline_row> baselines;
    for (auto const& entry : baseline_entries) {
        auto m = entry.measure_full(messages, mcfg);
        baselines.push_back(baseline_row{entry.group, entry.config, m});
    }

    auto best_train_it = std::min_element(train_data.begin(), train_data.end(),
        [](auto const& a, auto const& b) { return a.metrics.timing.p99 < b.metrics.timing.p99; });
    auto best_verify_it = std::min_element(verified.begin(), verified.end(),
        [](auto const& a, auto const& b) { return a.metrics.timing.p99 < b.metrics.timing.p99; });

    if (!opts.csv_mode) {
        std::printf("\n  --- Full Metrics Table ---\n\n");

        bool hc = (counters_t1 || counters_t2);
        if (hc) {
            std::printf("  %-16s  %-14s  %6s  %6s  %5s  %6s  %6s  %7s\n",
                "Source", "Config", "p99", "mean", "IPC", "L1D%%", "L1I%%", "DTLB%%");
            std::printf("  %-16s  %-14s  %6s  %6s  %5s  %6s  %6s  %7s\n",
                "----------------", "--------------",
                "------", "------", "-----", "------", "------", "-------");
        } else {
            std::printf("  %-16s  %-14s  %8s  %8s\n", "Source", "Config", "p99(ns)", "mean(ns)");
            std::printf("  %-16s  %-14s  %8s  %8s\n", "----------------", "--------------", "--------", "--------");
        }

        auto print_row = [&](const char* src, const fix::fix_config& cfg,
                             const fix::config_metrics& m) {
            if (hc) {
                std::printf("  %-16s  %-14s  %6.1f  %6.1f  %5.2f  %5.2f%%  %5.2f%%  %6.3f%%\n",
                    src, fix::config_to_string(cfg).c_str(),
                    m.timing.p99, m.timing.mean, m.ipc,
                    m.l1d_miss_rate * 100.0, m.l1i_miss_rate * 100.0,
                    m.dtlb_miss_rate * 100.0);
            } else {
                std::printf("  %-16s  %-14s  %8.1f  %8.1f\n",
                    src, fix::config_to_string(cfg).c_str(),
                    m.timing.p99, m.timing.mean);
            }
        };

        const char* bnames[] = {"All-Unrolled", "All-SWAR", "All-Loop", "All-Generic", "Phase10g"};
        for (std::size_t i = 0; i < baselines.size(); ++i)
            print_row(bnames[i], baselines[i].config, baselines[i].metrics);
        print_row("Best(train)", best_train_it->point.config, best_train_it->metrics);
        print_row("Best(verify)", best_verify_it->config, best_verify_it->metrics);

        // Beam winner
        fix_point bpt{beam_result[0].config};
        std::printf("\n  Beam winner:      %s  (SVR: p99=%.1f",
            fix::config_to_string(beam_result[0].config).c_str(),
            beam_result[0].svr_score);
        if (svr_ipc)  std::printf("  IPC=%.2f", svr_ipc->predict(bpt));
        if (svr_l1d)  std::printf("  L1D=%.2f%%", svr_l1d->predict(bpt) * 100.0);
        if (svr_l1i)  std::printf("  L1I=%.2f%%", svr_l1i->predict(bpt) * 100.0);
        std::printf(")\n");

        // Per-field consensus
        std::printf("\n  --- Per-field Strategy Consensus (beam top %zu) ---\n\n",
            beam_result.size());
        std::printf("    Field  Digits  U    S    L    G    Consensus\n");
        std::printf("    -----  ------  ---  ---  ---  ---  ---------\n");

        for (int f = 0; f < fix::num_fields; ++f) {
            std::array<int, 4> fc{};
            for (auto const& be : beam_result) {
                int s = static_cast<int>(be.config[static_cast<std::size_t>(f)]);
                fc[static_cast<std::size_t>(s)]++;
            }
            int max_count = *std::max_element(fc.begin(), fc.end());
            int max_s = 0;
            for (int s = 0; s < 4; ++s)
                if (fc[static_cast<std::size_t>(s)] == max_count) max_s = s;
            const char* sn[] = {"U", "S", "L", "G"};
            std::printf("    %5d  %6d  %3d  %3d  %3d  %3d  %s (%.0f%%)\n",
                f, fix::field_digits[static_cast<std::size_t>(f)],
                fc[0], fc[1], fc[2], fc[3], sn[max_s],
                100.0 * static_cast<double>(max_count) / static_cast<double>(beam_result.size()));
        }

        // Timing
        double total_ms = std::chrono::duration<double, std::milli>(t_verify_end - t_start).count();
        int n_models = 1 + (svr_ipc ? 1 : 0) + (svr_l1d ? 3 : 0);
        std::printf("\n  --- Timing ---\n\n");
        std::printf("  Phase 1 (measure + counters):   %8.1f ms\n", train_ms);
        std::printf("  Phase 2 (train SVR x %d):       %8.1f ms\n", n_models, svr_ms);
        std::printf("  Phase 3 (beam search):          %8.1f ms\n", beam_ms);
        std::printf("  Phase 4 (verify + counters):    %8.1f ms\n", verify_ms);
        std::printf("  Total:                          %8.1f ms\n\n", total_ms);
    }

    // CSV output
    if (opts.csv_mode) {
        std::printf("phase,config,p99,mean,ipc,l1d_miss,l1i_miss,dtlb_miss,"
                    "branch_miss,instructions,cycles,svr_p99,svr_ipc,svr_l1d\n");
        for (auto const& row : train_data) {
            fix_point pt{row.point.config};
            auto const& m = row.metrics;
            std::printf("train,%s,%.2f,%.2f,%.4f,%.6f,%.6f,%.6f,%.6f,%.1f,%.1f,%.2f,%.4f,%.6f\n",
                fix::config_to_string(row.point.config).c_str(),
                m.timing.p99, m.timing.mean, m.ipc,
                m.l1d_miss_rate, m.l1i_miss_rate, m.dtlb_miss_rate, m.branch_miss_rate,
                m.instructions, m.cycles,
                svr_p99.predict(pt),
                svr_ipc ? svr_ipc->predict(pt) : 0.0,
                svr_l1d ? svr_l1d->predict(pt) : 0.0);
        }
        for (auto const& v : verified) {
            auto const& m = v.metrics;
            std::printf("verify,%s,%.2f,%.2f,%.4f,%.6f,%.6f,%.6f,%.6f,%.1f,%.1f,%.2f,%.4f,%.6f\n",
                fix::config_to_string(v.config).c_str(),
                m.timing.p99, m.timing.mean, m.ipc,
                m.l1d_miss_rate, m.l1i_miss_rate, m.dtlb_miss_rate, m.branch_miss_rate,
                m.instructions, m.cycles,
                v.pred_p99, v.pred_ipc, v.pred_l1d);
        }
        for (auto const& be : beam_result) {
            fix_point pt{be.config};
            std::printf("beam,%s,,,,,,,,,,,%.2f,%.4f,%.6f\n",
                fix::config_to_string(be.config).c_str(),
                be.svr_score,
                svr_ipc ? svr_ipc->predict(pt) : 0.0,
                svr_l1d ? svr_l1d->predict(pt) : 0.0);
        }
    }

    return 0;
}
