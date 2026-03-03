// examples/fix_p99_calibration.cpp — FIX parser p99 calibration via CT-DP
//
// Demonstrates the complete p99 calibration pipeline:
//
//   1. Generate 200 uniformly random ET parser configurations
//   2. Add baselines (all-U, all-S, all-L, all-G)
//   3. Add simplex neighborhood of claimed optimal (UUSLSUUSUUUU)
//   4. Template-specialise all configs as expression-template parsers
//   5. Measure each with 100K samples for stable p99
//   6. Report full latency distributions
//   7. Identify p99-optimal configuration
//   8. Compare against baselines
//
// Key findings from Phase 10:
//   - Runtime dispatch overestimates latency by 1.2–1.9× vs ET
//   - p99 needs 100K+ samples for stability (not 10K)
//   - Uniform random sampling is critical (biased breaks everything)
//   - Direct measurement beats ML models (NN R²=0.54 was insufficient)
//   - Performance plateaus: many configs achieve similar p99
//
// Build:
//   g++ -std=c++20 -O2 -march=native -Wall -Wextra -Wpedantic
//       -I include examples/fix_p99_calibration.cpp -o fix_p99_cal
//
// Run:
//   ./fix_p99_cal                   # Full run (200 random + baselines + simplex)
//   ./fix_p99_cal --csv > data.csv  # CSV output for analysis
//   ./fix_p99_cal --quick           # Quick: 50 configs, 50K samples
//
// Part of the compile-time DP framework (ctdp v0.7.0)

#include <ctdp/calibrator/fix_et_parser.h>
#include <ctdp/bench/compiler_barrier.h>
#include <ctdp/bench/percentile.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fix  = ctdp::calibrator::fix;
namespace bench = ctdp::bench;

// ═══════════════════════════════════════════════════════════════════
// Compile-time configuration generation
// ═══════════════════════════════════════════════════════════════════

// 200 uniformly random configurations (seed=42 for reproducibility)
inline constexpr auto random_configs =
    fix::generate_random_configs<200>(42);

// Phase 10g claimed optimal
inline constexpr fix::fix_config phase10g_optimal = {
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
    fix::Strategy::SWAR,     fix::Strategy::Loop,
    fix::Strategy::SWAR,     fix::Strategy::Unrolled,
    fix::Strategy::Unrolled, fix::Strategy::SWAR,
    fix::Strategy::Unrolled, fix::Strategy::Unrolled,
    fix::Strategy::Unrolled, fix::Strategy::Unrolled
};

// ─── Simplex neighbors of phase10g_optimal ──────────────────────
// For each of 12 fields, try the other 3 strategies → 36 neighbors.

template<fix::fix_config Base, int Field, fix::Strategy NewS>
constexpr fix::fix_config make_neighbor() {
    fix::fix_config cfg = Base;
    cfg[static_cast<std::size_t>(Field)] = NewS;
    return cfg;
}

// Generate all 36 neighbors at compile time.
// Using a flat array rather than nested iteration for simplicity.
inline constexpr auto simplex_neighbors = [] {
    std::array<fix::fix_config, 36> nbrs{};
    int idx = 0;
    constexpr fix::Strategy all_strats[] = {
        fix::Strategy::Unrolled, fix::Strategy::SWAR,
        fix::Strategy::Loop,     fix::Strategy::Generic
    };
    for (int f = 0; f < fix::num_fields; ++f) {
        for (auto s : all_strats) {
            if (s != phase10g_optimal[static_cast<std::size_t>(f)]) {
                nbrs[static_cast<std::size_t>(idx)] = phase10g_optimal;
                nbrs[static_cast<std::size_t>(idx)][static_cast<std::size_t>(f)] = s;
                ++idx;
            }
        }
    }
    return nbrs;
}();


// ═══════════════════════════════════════════════════════════════════
// Dispatch table: map runtime index → template-specialised measure
// ═══════════════════════════════════════════════════════════════════

// Each entry: { config, name, measure_function_ptr }
struct config_entry {
    fix::fix_config config;
    const char*     group;      // "random", "baseline", "simplex"
    // Function pointer to template-specialised measurement
    bench::percentile_result (*measure)(
        std::vector<std::string> const&, std::size_t);
};

// Helper: create measure function for a specific compile-time config
template<fix::fix_config Cfg>
bench::percentile_result measure_wrapper(
    std::vector<std::string> const& msgs, std::size_t samples)
{
    return fix::measure_config<Cfg>(msgs, samples);
}

// ─── Build the dispatch table at compile time ───────────────────

// Random configs (200)
template<std::size_t... Is>
constexpr auto make_random_entries(std::index_sequence<Is...>) {
    return std::array<config_entry, sizeof...(Is)>{{
        config_entry{
            random_configs[Is],
            "random",
            &measure_wrapper<random_configs[Is]>
        }...
    }};
}

// Simplex neighbors (36)
template<std::size_t... Is>
constexpr auto make_simplex_entries(std::index_sequence<Is...>) {
    return std::array<config_entry, sizeof...(Is)>{{
        config_entry{
            simplex_neighbors[Is],
            "simplex",
            &measure_wrapper<simplex_neighbors[Is]>
        }...
    }};
}

// Baselines (5: all-U, all-S, all-L, all-G, phase10g)
inline auto baseline_entries = std::array<config_entry, 5>{{
    { fix::all_unrolled,   "baseline", &measure_wrapper<fix::all_unrolled>   },
    { fix::all_swar,       "baseline", &measure_wrapper<fix::all_swar>       },
    { fix::all_loop,       "baseline", &measure_wrapper<fix::all_loop>       },
    { fix::all_generic,    "baseline", &measure_wrapper<fix::all_generic>    },
    { phase10g_optimal,    "phase10g", &measure_wrapper<phase10g_optimal>    },
}};


// ═══════════════════════════════════════════════════════════════════
// Result storage
// ═══════════════════════════════════════════════════════════════════

struct calibration_result {
    std::string          config_str;
    const char*          group;
    bench::percentile_result pctl;
};


// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

struct cli_options {
    bool        csv_mode   = false;
    bool        quick_mode = false;
    std::size_t samples    = 100'000;
    std::size_t pool_size  = 10'000;
};

cli_options parse_cli(int argc, char** argv) {
    cli_options opts;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};
        if (arg == "--csv")   opts.csv_mode = true;
        if (arg == "--quick") {
            opts.quick_mode = true;
            opts.samples   = 50'000;
            opts.pool_size = 5'000;
        }
        if (arg == "--samples" && i + 1 < argc) {
            opts.samples = static_cast<std::size_t>(std::atol(argv[++i]));
        }
        if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--csv] [--quick] [--samples N]\n", argv[0]);
            std::printf("  --csv       Output CSV to stdout\n");
            std::printf("  --quick     Quick mode: 50 configs, 50K samples\n");
            std::printf("  --samples N Samples per config (default: 100000)\n");
            std::exit(0);
        }
    }
    return opts;
}


// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    auto opts = parse_cli(argc, argv);

    // Build the complete dispatch table
    auto random_table  = make_random_entries(std::make_index_sequence<200>{});
    auto simplex_table = make_simplex_entries(std::make_index_sequence<36>{});

    // Combine into a single vector for iteration
    std::vector<config_entry const*> all_entries;

    // Baselines first
    for (auto const& e : baseline_entries) all_entries.push_back(&e);

    // Simplex neighbors
    for (auto const& e : simplex_table)   all_entries.push_back(&e);

    // Random configs (all 200 unless quick mode)
    std::size_t random_count = opts.quick_mode ? 50 : 200;
    for (std::size_t i = 0; i < random_count; ++i) {
        all_entries.push_back(&random_table[i]);
    }

    // ─── Header ─────────────────────────────────────────────────
    if (!opts.csv_mode) {
        std::printf(
            "╔═══════════════════════════════════════════════════════╗\n"
            "║  FIX Parser p99 Calibration (Expression Templates)   ║\n"
            "╚═══════════════════════════════════════════════════════╝\n\n");

        std::printf("  Configurations:  %zu\n", all_entries.size());
        std::printf("    Baselines:     %zu\n", baseline_entries.size());
        std::printf("    Simplex:       %zu\n", simplex_table.size());
        std::printf("    Random:        %zu\n", random_count);
        std::printf("  Samples/config:  %zu\n", opts.samples);
        std::printf("  Fields:          %d\n", fix::num_fields);
        std::printf("  Total digits:    %d\n", fix::total_digits);
        std::printf("  Strategy space:  4^%d = %d configurations\n",
                    fix::num_fields, 1 << (2 * fix::num_fields));
        std::printf("  Compiler:        %s\n\n", __VERSION__);
    }

    // ─── Generate message pool ──────────────────────────────────
    if (!opts.csv_mode) {
        std::printf("Generating %zu-message pool... ", opts.pool_size);
        std::fflush(stdout);
    }

    auto messages = fix::generate_message_pool(opts.pool_size, 12345);

    if (!opts.csv_mode) {
        std::printf("done.\n\n");
    }

    // ─── Measure all configurations ─────────────────────────────
    std::vector<calibration_result> results;
    results.reserve(all_entries.size());

    auto t_start = std::chrono::steady_clock::now();

    if (opts.csv_mode) {
        std::printf("config,group,mean,p50,p90,p95,p99,p999,max,samples,tail_ratio\n");
    }

    for (std::size_t i = 0; i < all_entries.size(); ++i) {
        auto const& entry = *all_entries[i];
        auto cfg_str = fix::config_to_string(entry.config);

        // Measure
        auto pctl = entry.measure(messages, opts.samples);

        if (opts.csv_mode) {
            std::printf("%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%zu,%.2f\n",
                cfg_str.c_str(), entry.group,
                pctl.mean, pctl.p50, pctl.p90, pctl.p95,
                pctl.p99, pctl.p999, pctl.max,
                pctl.samples, pctl.tail_ratio());
        } else {
            std::printf("  [%3zu/%3zu] %-14s %-8s  mean=%6.1f  p50=%6.1f"
                        "  p99=%6.1f  p999=%6.1f  tail=%.1fx\n",
                i + 1, all_entries.size(),
                cfg_str.c_str(), entry.group,
                pctl.mean, pctl.p50, pctl.p99, pctl.p999,
                pctl.tail_ratio());
        }

        results.push_back({std::move(cfg_str), entry.group, pctl});
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    if (opts.csv_mode) return 0;

    // ─── Analysis ───────────────────────────────────────────────
    std::printf("\n");
    std::printf("Measurement complete: %.1f ms (%.0f configs/sec)\n\n",
        elapsed_ms,
        static_cast<double>(all_entries.size()) / (elapsed_ms / 1000.0));

    // Sort by p99
    std::sort(results.begin(), results.end(),
        [](auto const& a, auto const& b) { return a.pctl.p99 < b.pctl.p99; });

    // ─── Top 10 by p99 ──────────────────────────────────────────
    std::printf("═══ Top 10 by p99 Latency ═══\n\n");
    std::printf("  %-4s  %-14s %-8s  %8s  %8s  %8s  %8s  %6s\n",
                "Rank", "Config", "Group", "Mean", "p50", "p99", "p999", "Tail");
    std::printf("  %-4s  %-14s %-8s  %8s  %8s  %8s  %8s  %6s\n",
                "────", "──────────────", "────────",
                "────────", "────────", "────────", "────────", "──────");

    std::size_t top_n = std::min<std::size_t>(10, results.size());
    for (std::size_t i = 0; i < top_n; ++i) {
        auto const& r = results[i];
        std::printf("  #%-3zu  %-14s %-8s  %7.1f  %7.1f  %7.1f  %7.1f  %5.1fx\n",
            i + 1, r.config_str.c_str(), r.group,
            r.pctl.mean, r.pctl.p50, r.pctl.p99, r.pctl.p999,
            r.pctl.tail_ratio());
    }

    // ─── Baseline comparison ────────────────────────────────────
    std::printf("\n═══ Baseline Comparison ═══\n\n");

    // Find baselines in results
    auto find_result = [&](std::string_view cfg_str) -> calibration_result const* {
        for (auto const& r : results) {
            if (r.config_str == cfg_str) return &r;
        }
        return nullptr;
    };

    struct named_baseline {
        const char* name;
        const char* config;
    };
    named_baseline baselines[] = {
        {"All-Unrolled", "UUUUUUUUUUUU"},
        {"All-SWAR",     "SSSSSSSSSSSS"},
        {"All-Loop",     "LLLLLLLLLLLL"},
        {"All-Generic",  "GGGGGGGGGGGG"},
        {"Phase10g",     "UUSLSUUSUUUU"},
    };

    auto const& best = results.front();

    std::printf("  %-14s  %8s  %8s  %8s  %10s\n",
                "Config", "Mean", "p99", "p999", "vs Best p99");
    std::printf("  %-14s  %8s  %8s  %8s  %10s\n",
                "──────────────", "────────", "────────", "────────", "──────────");

    for (auto const& bl : baselines) {
        auto const* r = find_result(bl.config);
        if (r) {
            double pct = (r->pctl.p99 / best.pctl.p99 - 1.0) * 100.0;
            std::printf("  %-14s  %7.1f  %7.1f  %7.1f  %+8.1f%%\n",
                bl.name, r->pctl.mean, r->pctl.p99, r->pctl.p999, pct);
        }
    }

    // ─── Best overall ───────────────────────────────────────────
    std::printf("\n═══ Optimal Configuration ═══\n\n");
    std::printf("  Config:  %s  (%s)\n", best.config_str.c_str(), best.group);
    std::printf("  Mean:    %.1f ns\n", best.pctl.mean);
    std::printf("  p50:     %.1f ns\n", best.pctl.p50);
    std::printf("  p90:     %.1f ns\n", best.pctl.p90);
    std::printf("  p95:     %.1f ns\n", best.pctl.p95);
    std::printf("  p99:     %.1f ns\n", best.pctl.p99);
    std::printf("  p99.9:   %.1f ns\n", best.pctl.p999);
    std::printf("  Max:     %.1f ns\n", best.pctl.max);
    std::printf("  Tail:    %.1fx (p99/p50)\n", best.pctl.tail_ratio());

    // Strategy breakdown
    int counts[4] = {};
    auto best_cfg = fix::config_from_string(best.config_str);
    for (auto s : best_cfg) counts[static_cast<int>(s)]++;
    std::printf("\n  Strategy mix: %dU %dS %dL %dG\n",
                counts[0], counts[1], counts[2], counts[3]);

    // ─── Summary ────────────────────────────────────────────────
    auto const* phase10g = find_result("UUSLSUUSUUUU");

    std::printf("\n═══ Summary ═══\n\n");
    std::printf("  Configs measured:    %zu\n", results.size());
    std::printf("  Samples per config:  %zu\n", opts.samples);
    std::printf("  Measurement time:    %.1f ms\n", elapsed_ms);
    std::printf("  Throughput:          %.0f configs/sec\n",
        static_cast<double>(results.size()) / (elapsed_ms / 1000.0));

    if (phase10g) {
        double improvement = (1.0 - best.pctl.p99 / phase10g->pctl.p99) * 100.0;
        std::printf("  p99 vs Phase10g:     %+.1f%%\n", -improvement);
    }

    std::printf("\n  Lesson: \"Success is not skill in optimising —\n"
                "           it is skill in defining and measuring.\"\n\n");

    return 0;
}
