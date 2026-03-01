// examples/matrix_chain_et.cpp
// CT-DP: Plan Instantiation via Expression Templates
//
// Demonstrates the framework's instantiation concept:
//   1. Constexpr interval DP computes optimal parenthesisation
//   2. Split table drives template recursion → plan becomes a TYPE
//   3. The type locks association order — compiler cannot reassociate
//   4. At runtime: just matrix multiplies in the right order, framework gone
//
// This is the companion to matrix_chain_demo.cpp.  That example proves
// the solver finds the optimal cost.  This example proves the plan can
// be instantiated as zero-overhead executable code via the type system.
//
// The DP here is self-contained for Godbolt portability.  In the full
// framework, interval_dp.h + interval_split_space.h produce the same
// split table, and the BuildOptimal<I,J> template is the reusable
// bridge from plan data to plan type.
//
// Key properties:
//   - The plan type Mul<Mul<A,B>, Mul<C,D>> is STRUCTURAL — the compiler
//     cannot reassociate it, even under -ffast-math
//   - Different association orders produce different FP rounding, which
//     proves the type preserved exactly the DP-chosen order
//   - The Leaf/Mul node vocabulary generalises to any binary expression:
//     FMA chains (Horner), reduction trees, parser dispatch
//
// Build:   g++ -std=c++20 -O2 matrix_chain_et.cpp -o mc_et
// Godbolt: g++ trunk, -std=c++20 -O2
//
// Copyright (c) 2025-2026 Andrew Drakeford. All rights reserved.

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ═══════════════════════════════════════════════════════════════
// Runtime matrix — heap-allocated, dimensions known at runtime
// ═══════════════════════════════════════════════════════════════

struct Matrix {
    std::vector<double> data;
    std::size_t rows = 0, cols = 0;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : data(r * c, 0.0), rows(r), cols(c) {}

    double& operator()(std::size_t i, std::size_t j)       { return data[i * cols + j]; }
    double  operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }

    void fill_random(std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& x : data) x = dist(rng);
    }

    double checksum() const {
        return std::accumulate(data.begin(), data.end(), 0.0);
    }
};

// Standard triple-loop multiply.  Counts scalar multiplications.
Matrix mat_multiply(const Matrix& a, const Matrix& b, std::size_t& ops) {
    Matrix c(a.rows, b.cols);
    for (std::size_t i = 0; i < a.rows; ++i)
        for (std::size_t k = 0; k < a.cols; ++k)
            for (std::size_t j = 0; j < b.cols; ++j) {
                c(i, j) += a(i, k) * b(k, j);
                ++ops;
            }
    return c;
}

// ═══════════════════════════════════════════════════════════════
// Compile-time interval DP — produces the split table
//
// In the framework this is interval_dp.h + interval_split_space.h.
// Self-contained here for Godbolt portability.
// ═══════════════════════════════════════════════════════════════

template<std::size_t N>
struct DPTable {
    std::size_t cost[N][N]{};
    std::size_t split[N][N]{};
};

template<std::size_t N>
constexpr DPTable<N> matrix_chain_dp(const std::array<std::size_t, N + 1>& d) {
    DPTable<N> t{};
    constexpr std::size_t INF = static_cast<std::size_t>(-1);
    for (std::size_t len = 2; len <= N; ++len)
        for (std::size_t i = 0; i + len <= N; ++i) {
            std::size_t j = i + len - 1;
            t.cost[i][j] = INF;
            for (std::size_t k = i; k < j; ++k) {
                auto c = t.cost[i][k] + t.cost[k + 1][j]
                       + d[i] * d[k + 1] * d[j + 1];
                if (c < t.cost[i][j]) {
                    t.cost[i][j] = c;
                    t.split[i][j] = k;
                }
            }
        }
    return t;
}

// ─── Problem instance ────────────────────────────────────────
// CLRS textbook dimensions × 10  (for measurable runtime)
//   Original CLRS: {30, 35, 15, 5, 10, 20, 25}  →  optimal 15,125
//   Scaled ×10:    {300,350,150,50,100,200,250}   →  optimal 15,125,000
constexpr std::array<std::size_t, 7> dims = {300, 350, 150, 50, 100, 200, 250};
constexpr std::size_t N = dims.size() - 1;   // 6 matrices

// Solve at compile time
constexpr auto dp = matrix_chain_dp<N>(dims);

// Naive (left-to-right) cost at compile time.
// After i left-folds, result is dims[0] × dims[i+1].
// Next multiply costs dims[0] × dims[i+1] × dims[i+2].
constexpr std::size_t compute_naive_cost() {
    std::size_t total = 0;
    for (std::size_t i = 0; i < N - 1; ++i)
        total += dims[0] * dims[i + 1] * dims[i + 2];
    return total;
}

constexpr std::size_t optimal_cost = dp.cost[0][N - 1];
constexpr std::size_t naive_cost   = compute_naive_cost();

// Verify at compile time
static_assert(optimal_cost == 15'125'000, "CLRS optimal (scaled)");
static_assert(naive_cost   == 40'500'000, "Naive left-to-right cost");
static_assert(optimal_cost <  naive_cost, "DP beats naive");

// ═══════════════════════════════════════════════════════════════
// Expression template nodes
//
// Leaf<I>     — references matrix I by compile-time index
// Mul<L, R>   — binary multiply node
//
// The TYPE is the association order.  The compiler cannot
// reassociate Mul<Mul<A,B>, Mul<C,D>> — the nesting is
// structural, not algebraic.
// ═══════════════════════════════════════════════════════════════

template<std::size_t I>
struct Leaf {
    static Matrix eval(const std::vector<Matrix>& ms, std::size_t& /*ops*/) {
        return ms[I];
    }
    static void print() { std::printf("M%zu", I); }
};

template<typename L, typename R>
struct Mul {
    static Matrix eval(const std::vector<Matrix>& ms, std::size_t& ops) {
        auto left  = L::eval(ms, ops);
        auto right = R::eval(ms, ops);
        return mat_multiply(left, right, ops);
    }
    static void print() {
        std::printf("(");
        L::print();
        std::printf(" x ");
        R::print();
        std::printf(")");
    }
};

// ═══════════════════════════════════════════════════════════════
// Plan instantiation: compile-time data → type
//
// BuildOptimal walks the constexpr split table, emitting a
// nested Mul<L,R> type at each level.  The template recursion
// mirrors the recursive reconstruction of optimal parenthesisation.
//
// This is the reusable bridge between any DP solver's output
// (compile-time data) and executable code (a type the compiler
// can see through).  The pattern generalises:
//   Matrix chain → Mul<L,R>
//   Horner chain → FMA<coeff, x, rest>
//   Reduction    → Reduce<op, L, R>
//   Parser       → Parse<field, strategy>
// ═══════════════════════════════════════════════════════════════

// Optimal: DP split table → nested Mul<> tree
template<std::size_t I, std::size_t J>
struct BuildOptimal {
    static constexpr std::size_t K = dp.split[I][J];
    using type = Mul<typename BuildOptimal<I, K>::type,
                     typename BuildOptimal<K + 1, J>::type>;
};

template<std::size_t I>
struct BuildOptimal<I, I> {
    using type = Leaf<I>;
};

// Naive: left-to-right fold → Mul<Mul<..., M_{n-2}>, M_{n-1}>
template<std::size_t I, std::size_t J>
struct BuildNaive {
    using type = Mul<typename BuildNaive<I, J - 1>::type, Leaf<J>>;
};

template<std::size_t I>
struct BuildNaive<I, I> {
    using type = Leaf<I>;
};

// ─── The two plan types ──────────────────────────────────────
// These are TYPES, not values.  The association order is frozen
// in the type system.  The compiler sees nested struct::eval()
// calls — fully inlineable, zero indirection.
using OptimalPlan = typename BuildOptimal<0, N - 1>::type;
using NaivePlan   = typename BuildNaive<0, N - 1>::type;

// ═══════════════════════════════════════════════════════════════
// Benchmark harness
// ═══════════════════════════════════════════════════════════════

using Clock = std::chrono::high_resolution_clock;

template<typename Plan>
double run_benchmark(const std::vector<Matrix>& ms, int trials,
                     std::size_t& total_ops, double& checksum)
{
    total_ops = 0;
    checksum  = 0.0;

    // Warmup
    for (int i = 0; i < 3; ++i) {
        std::size_t ops = 0;
        auto r = Plan::eval(ms, ops);
        checksum = r.checksum();
    }

    auto t0 = Clock::now();
    for (int i = 0; i < trials; ++i) {
        std::size_t ops = 0;
        auto r = Plan::eval(ms, ops);
        total_ops += ops;
        checksum  += r.checksum();  // prevent DCE
    }
    auto t1 = Clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    std::printf("CT-DP Matrix Chain: Expression Template Instantiation\n");
    std::printf("=====================================================\n\n");

    // ── Create matrices ─────────────────────────────────────
    std::mt19937 rng(42);
    std::vector<Matrix> matrices;
    matrices.reserve(N);

    std::printf("Matrices: %zu  (CLRS textbook dimensions x10)\n", N);
    for (std::size_t i = 0; i < N; ++i) {
        matrices.emplace_back(dims[i], dims[i + 1]);
        matrices.back().fill_random(rng);
        std::printf("  M%zu: %4zu x %zu\n", i, dims[i], dims[i + 1]);
    }

    // ── Compile-time results ────────────────────────────────
    std::printf("\nCompile-time DP results:\n");
    std::printf("  Optimal cost:  %zu scalar multiplications\n", optimal_cost);
    std::printf("  Naive cost:    %zu scalar multiplications\n", naive_cost);
    std::printf("  Theoretical:   %.2fx fewer operations\n\n",
                static_cast<double>(naive_cost) / optimal_cost);

    // ── Print plan types ────────────────────────────────────
    std::printf("Plan types (association order locked by type system):\n\n");
    std::printf("  Naive:   ");
    NaivePlan::print();
    std::printf("\n\n");
    std::printf("  Optimal: ");
    OptimalPlan::print();
    std::printf("\n\n");

    // ── Benchmark ───────────────────────────────────────────
    constexpr int trials = 20;
    std::printf("Runtime benchmark (%d trials):\n", trials);

    std::size_t naive_ops = 0, opt_ops = 0;
    double naive_cs = 0, opt_cs = 0;

    double naive_ms = run_benchmark<NaivePlan>(matrices, trials, naive_ops, naive_cs);
    double opt_ms   = run_benchmark<OptimalPlan>(matrices, trials, opt_ops, opt_cs);

    std::printf("\n");
    std::printf("  %-10s %8s  %16s  %8s\n", "", "Time(ms)", "Ops/trial", "ms/trial");
    std::printf("  %-10s %8s  %16s  %8s\n", "", "--------", "---------", "--------");
    std::printf("  %-10s %8.1f  %16zu  %8.2f\n",
                "Naive",   naive_ms, naive_ops / trials, naive_ms / trials);
    std::printf("  %-10s %8.1f  %16zu  %8.2f\n",
                "Optimal", opt_ms,   opt_ops   / trials, opt_ms   / trials);
    std::printf("\n");
    std::printf("  Measured speedup: %.2fx\n", naive_ms / opt_ms);
    std::printf("  Op count ratio:   %.2fx\n\n",
                static_cast<double>(naive_ops) / static_cast<double>(opt_ops));

    // ── Verify correctness ──────────────────────────────────
    std::size_t v = 0;
    auto r_naive = NaivePlan::eval(matrices, v);
    auto r_opt   = OptimalPlan::eval(matrices, v);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < r_naive.rows; ++i)
        for (std::size_t j = 0; j < r_naive.cols; ++j)
            max_diff = std::fmax(max_diff,
                                 std::fabs(r_naive(i, j) - r_opt(i, j)));

    std::printf("Correctness:\n");
    std::printf("  Result dimensions: %zu x %zu\n", r_opt.rows, r_opt.cols);
    std::printf("  Max element diff:  %.2e\n", max_diff);
    std::printf("  (Nonzero diff expected: different association =\n");
    std::printf("   different FP rounding.  The type PREVENTS the\n");
    std::printf("   compiler from reassociating to a different order.)\n");

    return 0;
}
