#include "spmv_design_space/search.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct demo_options {
    std::size_t width = 512;
    std::size_t height = 512;
    ctdp::spmv_dsl::search_options search{};
    bool sweep = false;
};

std::size_t parse_size(const char* s) {
    return static_cast<std::size_t>(std::strtoull(s, nullptr, 10));
}

void print_usage(const char* exe) {
    std::cout
        << "usage: " << exe << " [width height]\n"
        << "       " << exe << " [--size N | --width W --height H] [--iterations N] [--warmup N]\n"
        << "              [--threads N] [--grain N] [--sweep]\n\n"
        << "defaults are tuned to show threaded candidates on a non-trivial matrix:\n"
        << "  --size 512 --iterations 31 --warmup 3 --threads auto --grain 2048\n\n"
        << "examples:\n"
        << "  " << exe << " --size 512 --iterations 41 --threads 4 --grain 4096\n"
        << "  " << exe << " --sweep --iterations 17 --threads 4\n";
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
        } else if (arg == "--sweep") {
            options.sweep = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return options;
}

int run_single(const demo_options& options) {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(options.width, options.height);
    const auto facts = analyse_problem(problem);
    const expression_contract contract{};
    const auto results = run_design_space_search(problem, contract, 0.125, options.search);
    print_report(std::cout, problem, facts, results);
    return select_best_legal(results) == nullptr ? 1 : 0;
}

int run_sweep(demo_options options) {
    using namespace ctdp::spmv_dsl;

    const std::vector<std::size_t> sizes{32, 64, 128, 256, 512};
    const expression_contract contract{};
    bool saw_threaded_selection = false;

    std::cout << "Sparse Expression Decomposition DSL Size Sweep\n";
    std::cout << "==============================================\n\n";
    std::cout << "iterations=" << options.search.iterations
              << " warmup=" << options.search.warmup
              << " threads=" << effective_worker_count(options.search)
              << " grain=" << options.search.task_grain << "\n\n";

    for (const auto n : sizes) {
        const auto problem = make_stencil_problem(n, n);
        const auto results = run_design_space_search(problem, contract, 0.125, options.search);
        const auto* best = select_best_legal(results);
        if (best == nullptr) {
            std::cout << n << "x" << n << ": no legal conforming candidate\n";
            continue;
        }
        saw_threaded_selection = saw_threaded_selection || uses_parallel_threading(best->plan.threading);
        std::cout << n << "x" << n
                  << " rows=" << problem.size()
                  << " selected='" << best->plan.name << "'"
                  << " p99_ns=" << best->p99_ns
                  << " median_ns=" << best->median_ns
                  << " uses_threads=" << (uses_parallel_threading(best->plan.threading) ? "yes" : "no")
                  << "\n";
    }

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
