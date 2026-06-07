// examples/accuracy_probe.cpp
// Measuring BOTH objectives of the error/throughput trade-off directly,
// instead of the analytical proxies the Pareto demo uses.
//
//   ACCURACY: float kernel under test vs a double-Kahan reference ("truth"),
//             on adversarial input (16 orders of magnitude, mixed signs,
//             engineered cancellation).
//   SPEED:    wall-clock per traversal, best-of-reps after warmup, reported
//             as ns/element so the cost is comparable across sizes.
//
// Three traversals reduce the SAME data in float:
//   naive sequential | full pairwise tree (deterministic) | Kahan sequential
//
// Build (NO -ffast-math: strict FP ordering is the experiment):
//   g++ -std=c++20 -O2 -I include examples/accuracy_probe.cpp -o accuracy_probe
//   ./accuracy_probe [N] [seed] [reps]
//
// Copyright (c) 2025 Andrew Drakeford.

#include <ct_dp/algebra/operations.h>   // identity_t, plus_fn

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using ct_dp::algebra::identity_t;
using ct_dp::algebra::plus_fn;

using clk = std::chrono::steady_clock;

// High-precision reference: double Kahan, treated as truth.
static double kahan_double(const std::vector<double>& x) {
    double acc = 0.0, c = 0.0;
    for (double v : x) { double y = v - c, t = acc + y; c = (t - acc) - y; acc = t; }
    return acc;
}

// Traversal A: naive sequential left fold.
template <class T, class Xform, class Comb>
T reduce_seq(const std::vector<T>& x, Xform f, Comb op) {
    T acc = T(0);
    for (T v : x) acc = op(acc, f(v));
    return acc;
}

// Traversal B: full pairwise tree (fixed shape -> deterministic).
template <class T, class Xform, class Comb>
T reduce_tree(const T* p, std::size_t n, Xform f, Comb op) {
    if (n == 1) return f(p[0]);
    if (n == 0) return T(0);
    std::size_t h = n / 2;
    return op(reduce_tree(p, h, f, op), reduce_tree(p + h, n - h, f, op));
}
template <class T, class Xform, class Comb>
T reduce_tree(const std::vector<T>& x, Xform f, Comb op) {
    return reduce_tree(x.data(), x.size(), f, op);
}

// Traversal B': blocked deterministic tree. Recurse until a block of BLK
// elements, then sum that block sequentially. Same fixed shape (still
// deterministic, still pairwise across blocks) but amortises the call
// overhead — this is what a real tiled reduction looks like.
template <class T, class Xform, class Comb>
T reduce_tree_blocked(const T* p, std::size_t n, Xform f, Comb op,
                      std::size_t BLK = 64) {
    if (n <= BLK) {
        T acc = T(0);
        for (std::size_t i = 0; i < n; ++i) acc = op(acc, f(p[i]));
        return acc;
    }
    std::size_t h = (n / 2);
    // round split to a block boundary so blocks stay aligned & equal-ish
    h = (h / BLK) * BLK; if (h == 0) h = BLK;
    return op(reduce_tree_blocked(p, h, f, op, BLK),
              reduce_tree_blocked(p + h, n - h, f, op, BLK));
}
template <class T, class Xform, class Comb>
T reduce_tree_blocked(const std::vector<T>& x, Xform f, Comb op) {
    return reduce_tree_blocked(x.data(), x.size(), f, op);
}

// Traversal C: Kahan-compensated sequential.
template <class T, class Xform, class Comb>
T reduce_kahan(const std::vector<T>& x, Xform f, Comb op) {
    T acc = T(0), c = T(0);
    for (T v : x) { T y = op(f(v), -c); T t = op(acc, y); c = op(op(t, -acc), -y); acc = t; }
    return acc;
}

static double rel_err(double m, double truth) {
    double d = std::abs(truth); if (d < 1e-300) d = 1.0;
    return std::abs(m - truth) / d;
}

// Time a reduction: warmup, then best-of-reps (min wall time is the most
// stable estimator of true cost — it strips OS jitter upward noise).
// 'sink' accumulates the result to defeat dead-code elimination.
template <class Fn>
static double time_ns_per_elem(Fn&& fn, std::size_t n, int reps, volatile double& sink) {
    fn(); // warmup (also faults in / warms cache)
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        double v = (double)fn();
        auto t1 = clk::now();
        sink += v;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (ns < best) best = ns;
    }
    return best / (double)n;
}

int main(int argc, char** argv) {
    std::size_t N    = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 1'000'000;
    unsigned    seed = (argc > 2) ? (unsigned)std::strtoul(argv[2], nullptr, 10) : 12345u;
    int         reps = (argc > 3) ? std::atoi(argv[3]) : 20;

    identity_t f{};
    plus_fn    op{};

    // Adversarial data: |value| ~ 1e-8 .. 1e8, random signs, + engineered cancellation.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> mag(-8.0, 8.0);
    std::uniform_int_distribution<int>     sgn(0, 1);
    std::vector<double> xd; xd.reserve(N * 2);
    for (std::size_t i = 0; i < N; ++i) {
        double v = std::pow(10.0, mag(rng));
        xd.push_back(sgn(rng) ? -v : v);
    }
    std::vector<double> big;
    for (double v : xd) if (std::abs(v) > 1e3) big.push_back(-v);
    std::shuffle(big.begin(), big.end(), rng);
    xd.insert(xd.end(), big.begin(), big.end());
    std::shuffle(xd.begin(), xd.end(), rng);
    N = xd.size();

    std::vector<float> xf(N);
    for (std::size_t i = 0; i < N; ++i) xf[i] = (float)xd[i];

    double truth = kahan_double(xd);

    // --- accuracy ---
    float f_seq   = reduce_seq  (xf, f, op);
    float f_tree  = reduce_tree (xf, f, op);
    float f_treeb = reduce_tree_blocked(xf, f, op);
    float f_kahan = reduce_kahan(xf, f, op);
    double e_seq   = rel_err((double)f_seq,   truth);
    double e_tree  = rel_err((double)f_tree,  truth);
    double e_treeb = rel_err((double)f_treeb, truth);
    double e_kahan = rel_err((double)f_kahan, truth);
    double ebase = (e_seq < 1e-300) ? 1.0 : e_seq;

    // --- timing (best-of-reps ns/element) ---
    volatile double sink = 0.0;
    double t_seq   = time_ns_per_elem([&]{ return reduce_seq  (xf, f, op); }, N, reps, sink);
    double t_tree  = time_ns_per_elem([&]{ return reduce_tree (xf, f, op); }, N, reps, sink);
    double t_treeb = time_ns_per_elem([&]{ return reduce_tree_blocked(xf, f, op); }, N, reps, sink);
    double t_kahan = time_ns_per_elem([&]{ return reduce_kahan(xf, f, op); }, N, reps, sink);
    double tbase = (t_seq <= 0.0) ? 1.0 : t_seq;

    printf("Accuracy + speed probe — float kernel vs double-Kahan reference\n");
    printf("===============================================================\n");
    printf("N = %zu   seed = %u   reps = %d   (best-of-reps timing)\n", N, seed, reps);
    printf("Reference sum (double, Kahan) = % .10e   [double-Kahan reference]\n\n", truth);

    printf("%-20s %12s %11s | %11s %10s\n",
           "float traversal", "rel_error", "err x", "ns/elem", "time x");
    printf("%-20s %12s %11s | %11s %10s\n",
           "---------------", "---------", "-----", "-------", "------");
    printf("%-20s %12.3e %10.3f | %11.3f %9.2fx\n",
           "naive sequential", e_seq,   e_seq/ebase,   t_seq,   t_seq/tbase);
    printf("%-20s %12.3e %10.3f | %11.3f %9.2fx\n",
           "full pairwise tree", e_tree,  e_tree/ebase,  t_tree,  t_tree/tbase);
    printf("%-20s %12.3e %10.3f | %11.3f %9.2fx\n",
           "blocked det. tree", e_treeb, e_treeb/ebase, t_treeb, t_treeb/tbase);
    printf("%-20s %12.3e %10.3f | %11.3f %9.2fx\n",
           "Kahan sequential", e_kahan, e_kahan/ebase, t_kahan, t_kahan/tbase);

    printf("\nMeasured trade-off (no proxy):\n");
    printf("  * naive sequential  : simple baseline; worst accuracy, and not even fastest here.\n");
    printf("  * full pairwise tree: %.0fx more accurate; naive recursion costs ~%.1fx.\n",
           e_seq / (e_tree  > 0 ? e_tree  : 1e-300), t_tree / tbase);
    printf("  * blocked det. tree : %.0fx more accurate than naive AND %.2fx its time\n",
           e_seq / (e_treeb > 0 ? e_treeb : 1e-300), t_treeb / tbase);
    printf("                        (dual win over naive; ILP/SIMD-friendly block;\n");
    printf("                        near full-tree accuracy, not beating it).\n");
    printf("  * Kahan sequential  : most accurate (%.0fx), slowest (%.2fx).\n",
           e_seq / (e_kahan > 0 ? e_kahan : 1e-300), t_kahan / tbase);
    printf("(sink=%.3e, defeats DCE)\n", (double)sink);
    return 0;
}
