// ctdp-calibrator CLI main
// Runs example scenarios and produces CSV output.
//
// Usage:
//   ./calibrator_main [scenario] [options]
//
// Scenarios:
//   memory   — Memory regime characterisation (default)
//   parser   — Parser strategy comparison
//
// Options:
//   --reps N         Number of repetitions per point (default: 10)
//   --warmup N       Warmup iterations (default: 200)
//   --no-pin         Don't pin to CPU
//   --no-flush       Don't flush cache between measurements
//   --no-boost       Don't elevate priority
//   --quiet          Suppress progress output
//   --summary        Write summary CSV (one row per point)
//   --output FILE    Write CSV to file (default: stdout)
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
//        -I../../ctdp-bench/include -I../include
//        calibrator_main.cpp -o calibrator_main

#include "calibration_scenarios.h"

#include <ctdp/calibrator/calibration_harness.h>
#include <ctdp/calibrator/csv_writer.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

// ─── CLI parsing ────────────────────────────────────────────────────

struct cli_args {
    std::string scenario   = "memory";
    std::string output     = "";  // empty = stdout
    std::size_t reps       = 10;
    std::size_t warmup     = 200;
    bool        pin_cpu    = true;
    bool        flush      = true;
    bool        boost      = true;
    bool        verbose    = true;
    bool        summary    = false;
};

cli_args parse_args(int argc, char** argv) {
    cli_args args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};
        if (arg == "memory" || arg == "parser") {
            args.scenario = std::string(arg);
        } else if (arg == "--reps" && i + 1 < argc) {
            args.reps = static_cast<std::size_t>(std::atol(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = static_cast<std::size_t>(std::atol(argv[++i]));
        } else if (arg == "--no-pin") {
            args.pin_cpu = false;
        } else if (arg == "--no-flush") {
            args.flush = false;
        } else if (arg == "--no-boost") {
            args.boost = false;
        } else if (arg == "--quiet") {
            args.verbose = false;
        } else if (arg == "--summary") {
            args.summary = true;
        } else if (arg == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [memory|parser] [options]\n"
                      << "  --reps N      Repetitions per point\n"
                      << "  --warmup N    Warmup iterations\n"
                      << "  --no-pin      Don't pin to CPU\n"
                      << "  --no-flush    Don't flush cache\n"
                      << "  --no-boost    Don't elevate priority\n"
                      << "  --quiet       Suppress progress\n"
                      << "  --summary     Summary CSV (one row per point)\n"
                      << "  --output FILE Write CSV to file\n";
            std::exit(0);
        }
    }
    return args;
}

ctdp::calibrator::harness_config make_config(cli_args const& args) {
    ctdp::calibrator::harness_config cfg;
    cfg.reps           = args.reps;
    cfg.warmup_iters   = args.warmup;
    cfg.measure_iters  = 1;
    cfg.pin_cpu        = args.pin_cpu;
    cfg.boost_priority = args.boost;
    cfg.flush_cache    = args.flush;
    cfg.verbose        = args.verbose;
    return cfg;
}

// ─── Scenario runners ───────────────────────────────────────────────

void run_memory_scenario(cli_args const& args, std::ostream& out) {
    using namespace ctdp::calibrator;
    using namespace ctdp::calibrator::examples;

    // 16 KiB → 64 MiB, 20 logarithmically-spaced points
    memory_regime_scenario scenario(16, 64 * 1024, 20);

    auto harness = calibration_harness<memory_regime_scenario>{
        std::move(scenario), make_config(args)};

    auto results = harness.run();

    if (args.summary) {
        write_csv_summary(out,
            std::span<const decltype(results)::value_type>{results},
            memory_point_formatter{},
            counter_snapshot_formatter{});
    } else {
        write_csv(out,
            std::span<const decltype(results)::value_type>{results},
            memory_point_formatter{},
            counter_snapshot_formatter{});
    }
}

void run_parser_scenario(cli_args const& args, std::ostream& out) {
    using namespace ctdp::calibrator;
    using namespace ctdp::calibrator::examples;

    // Full cross-product: digits 1–12 × 4 strategies = 48 points
    parser_strategy_scenario scenario(1, 12);

    auto harness = calibration_harness<parser_strategy_scenario>{
        std::move(scenario), make_config(args)};

    auto results = harness.run();

    if (args.summary) {
        write_csv_summary(out,
            std::span<const decltype(results)::value_type>{results},
            parser_point_formatter{},
            counter_snapshot_formatter{});
    } else {
        write_csv(out,
            std::span<const decltype(results)::value_type>{results},
            parser_point_formatter{},
            counter_snapshot_formatter{});
    }
}

// ─── main ───────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    // Set up output
    std::ofstream file_out;
    std::ostream* out = &std::cout;
    if (!args.output.empty()) {
        file_out.open(args.output);
        if (!file_out.is_open()) {
            std::cerr << "Error: cannot open " << args.output << "\n";
            return 1;
        }
        out = &file_out;
    }

    // Run selected scenario
    if (args.scenario == "memory") {
        run_memory_scenario(args, *out);
    } else if (args.scenario == "parser") {
        run_parser_scenario(args, *out);
    } else {
        std::cerr << "Unknown scenario: " << args.scenario << "\n";
        return 1;
    }

    return 0;
}
