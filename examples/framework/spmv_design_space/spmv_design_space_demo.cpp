#include "spmv_design_space/search.hpp"
#include "spmv_design_space/reporting.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

struct demo_options {
    std::size_t width = 512;
    std::size_t height = 512;
    ctdp::spmv_dsl::search_options search{};
    bool sweep = false;
    bool hybrid = false;
    std::vector<std::size_t> sweep_sizes{32, 64, 128, 256, 512, 1024};
    std::size_t remainder_period = 17;
    std::string summary_prefix;
    std::string platform_label = ctdp::spmv_dsl::default_platform_label();
};

std::size_t parse_size(const char* s) {
    return static_cast<std::size_t>(std::strtoull(s, nullptr, 10));
}

std::vector<std::size_t> parse_size_list(const char* text) {
    std::vector<std::size_t> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        out.push_back(static_cast<std::size_t>(std::strtoull(item.c_str(), nullptr, 10)));
    }
    if (out.empty()) {
        throw std::runtime_error("--sizes requires at least one positive size");
    }
    return out;
}

void append_unique_size(std::vector<std::size_t>& sizes, std::size_t value) {
    for (const auto existing : sizes) {
        if (existing == value) {
            return;
        }
    }
    sizes.push_back(value);
}

void print_usage(const char* exe) {
    std::cout
        << "usage: " << exe << " [width height]\n"
        << "       " << exe << " [--size N | --width W --height H] [--iterations N] [--warmup N]\n"
        << "              [--threads N] [--grain N] [--relaxed] [--sweep] [--sizes LIST]\n"
        << "              [--timing-observation full|solver-state]\n"
        << "              [--summary-prefix PATH] [--platform LABEL]\n"
        << "              [--hybrid] [--include-2048] [--remainder-period N] [--max-simd-lanes N]\n"
        << "              [--single-thread-target] [--no-task-runtime]\n\n"
        << "defaults are tuned to show threaded candidates on a non-trivial matrix:\n"
        << "  --size 512 --iterations 31 --warmup 3 --threads auto --grain 2048\n\n"
        << "examples:\n"
        << "  " << exe << " --size 512 --iterations 41 --threads 4 --grain 4096\n"
        << "  " << exe << " --size 512 --iterations 17 --threads 4 --relaxed\n"
        << "  " << exe << " --size 512 --iterations 17 --threads 4 --timing-observation solver-state\n"
        << "  " << exe << " --sweep --iterations 17 --threads 4\n"
        << "  " << exe << " --sweep --timing-observation solver-state --summary-prefix spmv_solver_state\n"
        << "  " << exe << " --sweep --sizes 32,64,128,256,512,1024,2048 --summary-prefix spmv_large\n"
        << "  " << exe << " --hybrid --size 256 --iterations 17 --threads 4\n"
        << "  " << exe << " --size 256 --max-simd-lanes 4 --single-thread-target\n";
}

demo_options parse_args(int argc, char** argv) {
    demo_options options;
    options.search.task_grain = 2048;

    if (argc == 3 && argv[1][0] != '-' && argv[2][0] != '-') {
        options.width = parse_size(argv[1]);
        options.height = parse_size(argv[2]);
        return options;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << name << '\n';
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--size") {
            options.width = options.height = parse_size(need_value("--size"));
        } else if (arg == "--width") {
            options.width = parse_size(need_value("--width"));
        } else if (arg == "--height") {
            options.height = parse_size(need_value("--height"));
        } else if (arg == "--iterations") {
            options.search.iterations = parse_size(need_value("--iterations"));
        } else if (arg == "--warmup") {
            options.search.warmup = parse_size(need_value("--warmup"));
        } else if (arg == "--threads") {
            options.search.threads = parse_size(need_value("--threads"));
        } else if (arg == "--grain") {
            options.search.task_grain = parse_size(need_value("--grain"));
        } else if (arg == "--timing-observation") {
            const std::string mode = need_value("--timing-observation");
            if (mode == "full" || mode == "full-trace") {
                options.search.timing_observation = ctdp::spmv_dsl::timing_observation_mode::full_trace;
            } else if (mode == "solver-state" || mode == "solver-state-only") {
                options.search.timing_observation = ctdp::spmv_dsl::timing_observation_mode::solver_state_only;
            } else {
                std::cerr << "unknown timing observation mode: " << mode << "\n";
                print_usage(argv[0]);
                std::exit(2);
            }
        } else if (arg == "--max-simd-lanes") {
            options.search.hardware.max_simd_lanes = parse_size(need_value("--max-simd-lanes"));
        } else if (arg == "--single-thread-target") {
            options.search.hardware.allow_parallel_candidates = false;
            options.search.hardware.max_worker_threads = 1;
        } else if (arg == "--no-task-runtime") {
            options.search.hardware.task_runtime_available = false;
        } else if (arg == "--summary-prefix") {
            options.summary_prefix = need_value("--summary-prefix");
        } else if (arg == "--platform") {
            options.platform_label = need_value("--platform");
        } else if (arg == "--hybrid") {
            options.hybrid = true;
        } else if (arg == "--remainder-period") {
            options.remainder_period = parse_size(need_value("--remainder-period"));
        } else if (arg == "--relaxed") {
            options.search.scope = ctdp::spmv_dsl::candidate_scope::strict_and_relaxed_executable;
        } else if (arg == "--strict-only") {
            options.search.scope = ctdp::spmv_dsl::candidate_scope::strict_conforming_only;
        } else if (arg == "--sweep") {
            options.sweep = true;
        } else if (arg == "--sizes") {
            options.sweep_sizes = parse_size_list(need_value("--sizes"));
        } else if (arg == "--include-2048") {
            append_unique_size(options.sweep_sizes, 2048);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return options;
}

void write_summary_files(const std::string& prefix,
                         const std::vector<ctdp::spmv_dsl::selected_plan_summary>& selected_rows,
                         const std::vector<ctdp::spmv_dsl::candidate_plan_summary>& candidate_rows,
                         const std::vector<ctdp::spmv_dsl::search_run_count_summary>& count_rows) {
    using namespace ctdp::spmv_dsl;

    if (prefix.empty()) {
        return;
    }

    const auto selected_csv = prefix + "_selected.csv";
    const auto selected_md = prefix + "_selected.md";
    const auto selected_families_md = prefix + "_families.md";
    const auto candidates_csv = prefix + "_candidates.csv";
    const auto candidate_families_md = prefix + "_candidate_families.md";
    const auto counts_csv = prefix + "_candidate_counts.csv";
    const auto counts_md = prefix + "_candidate_counts.md";

    {
        std::ofstream out(selected_csv);
        if (!out) {
            throw std::runtime_error("failed to open summary CSV: " + selected_csv);
        }
        write_selected_plan_summary_csv(out, selected_rows);
    }
    {
        std::ofstream out(selected_md);
        if (!out) {
            throw std::runtime_error("failed to open selected-plan Markdown summary: " + selected_md);
        }
        write_selected_plan_summary_markdown(out, selected_rows);
    }
    {
        std::ofstream out(selected_families_md);
        if (!out) {
            throw std::runtime_error("failed to open selected family Markdown summary: " + selected_families_md);
        }
        write_family_summary_markdown(out, selected_rows);
    }
    {
        std::ofstream out(candidates_csv);
        if (!out) {
            throw std::runtime_error("failed to open candidate CSV: " + candidates_csv);
        }
        write_candidate_plan_summary_csv(out, candidate_rows);
    }
    {
        std::ofstream out(candidate_families_md);
        if (!out) {
            throw std::runtime_error("failed to open candidate family Markdown summary: " + candidate_families_md);
        }
        write_candidate_family_summary_markdown(out, candidate_rows);
    }
    {
        std::ofstream out(counts_csv);
        if (!out) {
            throw std::runtime_error("failed to open candidate-count CSV: " + counts_csv);
        }
        write_search_run_count_summary_csv(out, count_rows);
    }
    {
        std::ofstream out(counts_md);
        if (!out) {
            throw std::runtime_error("failed to open candidate-count Markdown summary: " + counts_md);
        }
        write_search_run_count_summary_markdown(out, count_rows);
    }

    std::cout << "\nWrote summary reports:\n"
              << "  " << selected_csv << "\n"
              << "  " << selected_md << "\n"
              << "  " << selected_families_md << "\n"
              << "  " << candidates_csv << "\n"
              << "  " << candidate_families_md << "\n"
              << "  " << counts_csv << "\n"
              << "  " << counts_md << "\n";
}

int run_single(const demo_options& options) {
    using namespace ctdp::spmv_dsl;

    const auto problem = options.hybrid
        ? make_stencil_with_remainder_problem(options.width, options.height, options.remainder_period)
        : make_stencil_problem(options.width, options.height);
    const auto facts = analyse_problem(problem);
    const expression_contract contract{};
    const auto results = run_design_space_search(problem, contract, 0.125, options.search);
    print_report(std::cout, problem, facts, results, options.search);

    const auto* best = select_best_legal(results);
    if (!options.summary_prefix.empty()) {
        const std::vector<selected_plan_summary> selected_rows{
            make_selected_plan_summary(problem, facts, options.search, best, options.platform_label)
        };
        const auto candidate_rows = make_candidate_plan_summaries(
            problem, facts, options.search, results, best, options.platform_label);
        const std::vector<search_run_count_summary> count_rows{
            make_search_run_count_summary(problem, facts, options.search, results, options.platform_label)
        };
        write_summary_files(options.summary_prefix, selected_rows, candidate_rows, count_rows);
    }
    return best == nullptr ? 1 : 0;
}

int run_sweep(demo_options options) {
    using namespace ctdp::spmv_dsl;

    const auto sizes = options.sweep_sizes;
    const expression_contract contract{};
    bool saw_threaded_selection = false;
    std::vector<selected_plan_summary> selected_rows;
    std::vector<candidate_plan_summary> candidate_rows;
    std::vector<search_run_count_summary> count_rows;

    std::cout << "Sparse Expression Decomposition DSL Size Sweep\n";
    std::cout << "==============================================\n\n";
    std::cout << "iterations=" << options.search.iterations
              << " warmup=" << options.search.warmup
              << " threads=" << effective_worker_count(options.search)
              << " grain=" << options.search.task_grain
              << " scope=" << to_string(options.search.scope)
              << " timing_observation=" << to_string(options.search.timing_observation)
              << " hybrid=" << (options.hybrid ? "yes" : "no")
              << " platform=" << options.platform_label
              << " hardware={" << describe_hardware_profile(options.search.hardware) << "}" << "\n";
    std::cout << "sweep_sizes=";
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << sizes[i];
    }
    std::cout << "\n\n";

    for (const auto n : sizes) {
        const auto problem = options.hybrid
            ? make_stencil_with_remainder_problem(n, n, options.remainder_period)
            : make_stencil_problem(n, n);
        const auto facts = analyse_problem(problem);
        const auto results = run_design_space_search(problem, contract, 0.125, options.search);
        const auto* best = select_best_legal(results);
        const auto case_candidate_rows = make_candidate_plan_summaries(
            problem, facts, options.search, results, best, options.platform_label);
        candidate_rows.insert(candidate_rows.end(), case_candidate_rows.begin(), case_candidate_rows.end());
        count_rows.push_back(make_search_run_count_summary(
            problem, facts, options.search, results, options.platform_label));
        if (best == nullptr) {
            selected_rows.push_back(make_selected_plan_summary(problem, facts, options.search, nullptr, options.platform_label));
            std::cout << n << "x" << n << ": no legal conforming candidate\n";
            continue;
        }
        selected_rows.push_back(make_selected_plan_summary(problem, facts, options.search, best, options.platform_label));
        saw_threaded_selection = saw_threaded_selection || uses_parallel_threading(best->plan.threading);
        std::cout << n << "x" << n
                  << " rows=" << problem.size()
                  << " selected='" << best->plan.name << "'"
                  << " median_ns=" << best->median_ns
                  << " mean_ns=" << best->mean_ns
                  << " rho_bits=" << hex64(best->rho_bits)
                  << " sigma_bits=" << hex64(best->sigma_bits)
                  << " alpha_bits=" << hex64(best->alpha_bits)
                  << " x_next_hash=" << hex64(best->x_next_hash)
                  << " observation_hash=" << hex64(best->observation_hash)
                  << " uses_threads=" << (uses_parallel_threading(best->plan.threading) ? "yes" : "no")
                  << "\n";
    }

    write_summary_files(options.summary_prefix, selected_rows, candidate_rows, count_rows);

    std::cout << "\nInterpretation:\n";
    std::cout << "  Threaded candidates are expected to lose on small matrices because startup,\n";
    std::cout << "  phase, and task scheduling costs dominate. The useful question is where the\n";
    std::cout << "  measured selector crosses over to a threaded plan on this machine.\n";

    return saw_threaded_selection ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const auto options = parse_args(argc, argv);
    if (options.sweep) {
        return run_sweep(options);
    }
    return run_single(options);
}
