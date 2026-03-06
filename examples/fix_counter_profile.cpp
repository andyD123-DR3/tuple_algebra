// fix_counter_profile.cpp  —  CT-DP FIX Parser Hardware Counter Profile
//
// For a representative set of N=4 trivial-schema plans, runs
// measure_config_with_counters() and prints a side-by-side table of:
//
//   p99 (ns) | instructions/parse | cycles/parse | IPC
//   L1D miss-rate | branch miss-rate | LL cache miss-rate
//
// This is the "why is plan X faster?" diagnostic companion to
// fix_calibrate_trivial (which finds the optimal plan).
// Hardware counters explain the mechanism; p99 gives the verdict.
//
// Tier 0 (RDTSC) is always available.
// Tier 1 (perf_event_open) requires:
//   Linux: sudo sysctl -w kernel.perf_event_paranoid=1
//   or:    CAP_PERFMON / CAP_SYS_ADMIN
// Counter columns show 0.000 and [NO HW COUNTERS] if Tier 1 unavailable.
//
// Build:
//   cmake --build . --target fix_counter_profile
//
// Run (pin to a quiet core for stable readings):
//   taskset -c 2 ./fix_counter_profile
//   taskset -c 2 ./fix_counter_profile --all      # all 64 trivial plans
//   taskset -c 2 ./fix_counter_profile --csv      # CSV output

#include <ctdp/calibrator/fix_et_parser.h>
#include <ctdp/calibrator/fix/fix_strategy_ids.h>
#include <ctdp/calibrator/fix/fix_schema.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace fix   = ctdp::calibrator::fix;
namespace bench = ctdp::bench;

// ─────────────────────────────────────────────────────────────────────────────
//  Probe set — plans chosen to show contrast across the counter dimensions.
//
//  The trivial schema has N=4 fields:
//    field 0: BeginString  (short, alpha — SWAR shines)
//    field 1: BodyLength   (4 digits  — Unrolled shines)
//    field 2: MsgType      (1 char    — Loop has no overhead)
//    field 3: CheckSum     (3 digits  — Generic is safe baseline)
//
//  We measure:
//    ALL_U  — all Unrolled: tight inner loop, lowest IPC, fewest instructions
//    ALL_S  — all SWAR:     widest load, highest IPC on numeric fields
//    ALL_L  — all Loop:     branch-heavy, highest branch-miss-rate
//    ALL_G  — all Generic:  function-call overhead, most cache pressure
//    USLG   — one of each: balanced mix, useful reference point
//    SULS   — SWAR-Unrolled-Loop-SWAR: plausible near-optimal for this schema
//    UUUG   — 3xUnrolled + Generic: tests isolated Generic penalty
//    LLLL   — all Loop: shows branch predictor saturation on trivial schema
// ─────────────────────────────────────────────────────────────────────────────

struct ProbeEntry {
    const char*    name;
    fix::fix_config cfg;  // N=4 entries
};

// Build a 4-field fix_config from four Strategy values.
// The trivial schema only uses the first 4 slots of a fix_config[12].
static constexpr fix::fix_config make4(
    fix::Strategy s0, fix::Strategy s1,
    fix::Strategy s2, fix::Strategy s3) noexcept
{
    fix::fix_config c{};
    c[0] = s0; c[1] = s1; c[2] = s2; c[3] = s3;
    // Remaining 8 slots are zero-initialised (Strategy::Unrolled == 0)
    return c;
}

using S = fix::Strategy;
static constexpr std::array<ProbeEntry, 8> probe_set {{
    { "ALL_U  (UUUU)", make4(S::Unrolled, S::Unrolled, S::Unrolled, S::Unrolled) },
    { "ALL_S  (SSSS)", make4(S::SWAR,     S::SWAR,     S::SWAR,     S::SWAR    ) },
    { "ALL_L  (LLLL)", make4(S::Loop,     S::Loop,     S::Loop,     S::Loop    ) },
    { "ALL_G  (GGGG)", make4(S::Generic,  S::Generic,  S::Generic,  S::Generic ) },
    { "USLG        ", make4(S::Unrolled, S::SWAR,     S::Loop,     S::Generic ) },
    { "SULS        ", make4(S::SWAR,     S::Unrolled, S::Loop,     S::SWAR    ) },
    { "UUUG        ", make4(S::Unrolled, S::Unrolled, S::Unrolled, S::Generic ) },
    { "USUU        ", make4(S::Unrolled, S::SWAR,     S::Unrolled, S::Unrolled) },
}};

// ─────────────────────────────────────────────────────────────────────────────
//  Measurement — dispatch via trivial_index (avoids 64-entry dispatch table)
// ─────────────────────────────────────────────────────────────────────────────

// We want measure_config_with_counters<cfg> but cfg is runtime here.
// Use the 64-entry trivial dispatch table from fix_bench.h detail namespace.
// For the probe set (8 fixed plans) we can instantiate directly.

template<fix::fix_config Cfg>
static fix::config_metrics measure_probe(
    const std::vector<std::string>& msgs,
    const fix::measurement_config&  mcfg)
{
    return fix::measure_config_with_counters<Cfg>(msgs, mcfg);
}

// Function pointer type
using MeasureFn = fix::config_metrics(*)(
    const std::vector<std::string>&,
    const fix::measurement_config&);

// Compile-time dispatch table for the 8 probes
static constexpr std::array<MeasureFn, 8> probe_fns {{
    &measure_probe<probe_set[0].cfg>,
    &measure_probe<probe_set[1].cfg>,
    &measure_probe<probe_set[2].cfg>,
    &measure_probe<probe_set[3].cfg>,
    &measure_probe<probe_set[4].cfg>,
    &measure_probe<probe_set[5].cfg>,
    &measure_probe<probe_set[6].cfg>,
    &measure_probe<probe_set[7].cfg>,
}};

// ─────────────────────────────────────────────────────────────────────────────
//  Result storage
// ─────────────────────────────────────────────────────────────────────────────

struct ProfileRow {
    const char*           name;
    fix::config_metrics   m;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Printing
// ─────────────────────────────────────────────────────────────────────────────

static void print_header_table() {
    std::printf("\n");
    std::printf("  %-14s  %8s  %8s  %12s  %6s  %9s  %9s  %9s\n",
        "Plan", "p99(ns)", "fit-p99", "instr/parse",
        "IPC", "L1D-miss%", "br-miss%", "LL-miss%");
    std::printf("  %-14s  %8s  %8s  %12s  %6s  %9s  %9s  %9s\n",
        "--------------", "--------", "--------", "------------",
        "------", "---------", "--------", "--------");
}

static void print_row(const ProfileRow& r) {
    const auto& m = r.m;
    std::printf("  %-14s  %8.2f  %8.2f  %12.1f  %6.3f  %8.3f%%  %7.3f%%  %7.3f%%\n",
        r.name,
        m.timing.p99,
        m.fitted_p99,
        m.instructions,
        m.ipc,
        m.l1d_miss_rate   * 100.0,
        m.branch_miss_rate * 100.0,
        m.cache_miss_rate  * 100.0);
}

static void print_csv_header() {
    std::printf("plan,p99_ns,fitted_p99_ns,instructions_per_parse,cycles_per_parse,"
                "ipc,l1d_miss_rate,l1i_miss_rate,branch_miss_rate,ll_cache_miss_rate,"
                "tier1_available\n");
}

static void print_csv_row(const ProfileRow& r) {
    const auto& m = r.m;
    std::printf("%s,%.3f,%.3f,%.1f,%.1f,%.4f,%.6f,%.6f,%.6f,%.6f,%d\n",
        r.name,
        m.timing.p99, m.fitted_p99,
        m.instructions, m.cycles,
        m.ipc,
        m.l1d_miss_rate, m.l1i_miss_rate,
        m.branch_miss_rate, m.cache_miss_rate,
        static_cast<int>(m.tier1_available));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Analysis helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_analysis(const std::vector<ProfileRow>& rows) {
    // Sort by fitted_p99 ascending
    auto sorted = rows;
    std::sort(sorted.begin(), sorted.end(),
        [](const ProfileRow& a, const ProfileRow& b) {
            return a.m.fitted_p99 < b.m.fitted_p99;
        });

    const auto& best  = sorted.front();
    const auto& worst = sorted.back();

    std::printf("\n=== Analysis ===\n\n");
    std::printf("  Best  fitted p99:  %-14s  %.2f ns\n",
        best.name, best.m.fitted_p99);
    std::printf("  Worst fitted p99:  %-14s  %.2f ns\n",
        worst.name, worst.m.fitted_p99);
    if (best.m.fitted_p99 > 0.0) {
        std::printf("  Ratio worst/best:  %.2fx\n",
            worst.m.fitted_p99 / best.m.fitted_p99);
    }

    if (best.m.tier1_available) {
        std::printf("\n  Counter insights (best vs worst):\n");
        std::printf("    Instructions/parse:  %.1f  vs  %.1f  (%.1fx)\n",
            best.m.instructions, worst.m.instructions,
            worst.m.instructions > 0 ? best.m.instructions / worst.m.instructions : 0.0);
        std::printf("    IPC:                 %.3f  vs  %.3f\n",
            best.m.ipc, worst.m.ipc);
        std::printf("    L1D miss rate:       %.3f%%  vs  %.3f%%\n",
            best.m.l1d_miss_rate * 100.0, worst.m.l1d_miss_rate * 100.0);
        std::printf("    Branch miss rate:    %.3f%%  vs  %.3f%%\n",
            best.m.branch_miss_rate * 100.0, worst.m.branch_miss_rate * 100.0);
    } else {
        std::printf("\n  [NO HW COUNTERS] — counter columns are zero.\n");
        std::printf("  Enable with:  sudo sysctl -w kernel.perf_event_paranoid=1\n");
    }

    std::printf("\n  Ranked by fitted p99:\n");
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        double delta_pct = (sorted[i].m.fitted_p99 / best.m.fitted_p99 - 1.0) * 100.0;
        std::printf("    #%zu  %-14s  %.2f ns  (+%.1f%%)\n",
            i + 1, sorted[i].name, sorted[i].m.fitted_p99, delta_pct);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  CLI
// ─────────────────────────────────────────────────────────────────────────────

struct Options {
    bool   csv      = false;
    bool   verbose  = true;
    std::size_t samples   = 50'000;
    std::size_t pool_size = 1024;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a == "--csv")     { o.csv = true; o.verbose = false; }
        if (a == "--quiet")   { o.verbose = false; }
        if (a == "--samples" && i + 1 < argc)
            o.samples = static_cast<std::size_t>(std::atol(argv[++i]));
        if (a == "--help" || a == "-h") {
            std::printf("Usage: fix_counter_profile [--csv] [--quiet] [--samples N]\n");
            std::printf("  --csv         CSV output (suitable for plotting)\n");
            std::printf("  --quiet       Suppress progress messages\n");
            std::printf("  --samples N   Timing samples per plan (default 50000)\n");
            std::printf("\nHardware counters require: sudo sysctl -w kernel.perf_event_paranoid=1\n");
            std::exit(0);
        }
    }
    return o;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    auto opts = parse_args(argc, argv);

    if (opts.verbose) {
        std::printf("+------------------------------------------------------+\n");
        std::printf("|  FIX Parser Hardware Counter Profile (Trivial N=4)  |\n");
        std::printf("+------------------------------------------------------+\n\n");
        std::printf("  Plans:            %zu (selected probes)\n", probe_set.size());
        std::printf("  Samples/plan:     %zu\n", opts.samples);
        std::printf("  Message pool:     %zu\n", opts.pool_size);
        std::printf("\n  Calibrating TSC... ");
        std::fflush(stdout);
    }

    fix::measurement_config mcfg;
    mcfg.cycles_per_ns = fix::calibrate_tsc();
    mcfg.samples       = opts.samples;
    mcfg.batch_size    = 64;
    mcfg.warmup_parses = 4096;

    if (opts.verbose)
        std::printf("%.3f GHz\n", mcfg.cycles_per_ns);

    if (opts.verbose) {
        std::printf("  Generating message pool... ");
        std::fflush(stdout);
    }

    auto messages = fix::generate_message_pool(opts.pool_size);

    if (opts.verbose)
        std::printf("done.\n\n  Measuring %zu plans (timing + hw counters)...\n",
                    probe_set.size());

    // ── Measure ─────────────────────────────────────────────────────────────

    std::vector<ProfileRow> rows;
    rows.reserve(probe_set.size());

    if (opts.csv) print_csv_header();
    if (!opts.csv && opts.verbose) print_header_table();

    for (std::size_t i = 0; i < probe_set.size(); ++i) {
        if (opts.verbose && !opts.csv)
            std::printf("  [%zu/%zu] %s\r", i+1, probe_set.size(), probe_set[i].name);

        auto m = probe_fns[i](messages, mcfg);
        ProfileRow row{ probe_set[i].name, m };

        if (opts.csv)        print_csv_row(row);
        else if (opts.verbose) { /* printed after all done */ }

        rows.push_back(row);
    }

    if (opts.csv) return 0;

    // ── Print table ──────────────────────────────────────────────────────────

    std::printf("\n");
    print_header_table();
    for (const auto& r : rows) print_row(r);

    // ── Analysis ─────────────────────────────────────────────────────────────

    print_analysis(rows);

    std::printf("\n  Tip: pipe to CSV with --csv for plotting.\n");
    std::printf("  Tip: run with taskset -c 2 for stable counter readings.\n\n");

    return 0;
}
