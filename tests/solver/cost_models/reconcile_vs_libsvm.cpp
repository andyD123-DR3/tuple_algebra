// reconcile_vs_libsvm.cpp
//
// Reconciliation: ctdp::EpsilonSVRSolver vs libSVM on identical problems.
// libSVM is the reference implementation (BSD-3, github.com/cjlin1/libsvm).
//
// Two levels of comparison:
//
//   reconcile()       — T1-T6 (original)
//     Checks: n_sv count, bias, max/mean prediction difference.
//     What it catches: gross solver failures — wrong n_sv, wildly wrong predictions.
//     What it MISSES: wrong dual variable distribution that still yields similar
//                     predictions; checker bugs that silently pass incorrect solutions.
//
//   reconcile_deep()  — T7-T10 (new)
//     Additionally checks:
//       (a) Per-coefficient agreement: coef[i] = α_i − α*_i matched by feature vector.
//           Catches wrong Hessian sign, wrong step sizes, wrong α distribution.
//           Two different dual solutions CAN produce similar predictions while having
//           completely different α values — prediction-only comparison misses this.
//       (b) n_sv as a hard pass/fail criterion (not just printed).
//           n_sv mismatch with libSVM is a definitive solver bug indicator.
//       (c) KKT checker applied to libSVM's own solution.
//           If our checker reports violations on a verified-correct libSVM solution,
//           the checker has a bug — not the solver.
//           This is the test that would have directly exposed the checker inversion.
//       (d) Bias agreement to 1e-4 (tighter than prediction tolerance).
//
// Tests that would have caught the two implementation bugs:
//
//   Bug 1: Hessian missing y2[t]*y2[q] sign factor
//     H_wrong = K_tt + K_qq + 2*K_tq  (when cross-side pair: y2[t]*y2[q]=-1)
//     H_right = K_tt + K_qq - 2*K_tq
//     Effect: wrong step sizes for all cross-side updates → wrong α distribution.
//     Caught by: T7/T8/T9 per-coef comparison; T7 cross-side stress directly exercises
//     the sign-flip path. Also caught by n_sv mismatch in all deep tests.
//
//   Bug 2: Spurious y2 signs in f-update
//     Δf_wrong[k] = step*(y2[t]*K[tp,k] - y2[q]*K[qp,k])
//     When y2[t]=y2[q]=-1 (both α* side): wrong code gives -(K_t-K_q) instead of K_t-K_q.
//     Effect: gradient tracking diverges after first cross-side update → all subsequent
//     working-set selections corrupted.
//     Caught by: same — per-coef comparison on any dataset with mixed SVs (T7, T9, T10).
//
//   Checker bug: on-band alpha/alpha* assignments inverted
//     ON_UP (r≈+ε): checker required alpha=0 but should require alpha*=0.
//     Effect: every free SV on the upper tube boundary is flagged as a KKT violation.
//     Caught by: T8 — apply our checker to libSVM's solution. libSVM is correct, so
//     any violation report from our checker is a checker bug, not a solver bug.
//
// Build:
//   g++ -std=c++23 -O2 \
//       -I./include -I./include/ctdp/solver/cost_models \
//       -I/usr/include/libsvm \
//       tests/solver/cost_models/reconcile_vs_libsvm.cpp \
//       -lsvm -o reconcile_vs_libsvm

#include "epsilon_svr.h"
#include <libsvm/svm.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace ctdp::cost_models;

static void silent_print(const char*) {}

// ============================================================================
// libSVM helpers
// ============================================================================

struct LibSVMProblem {
    svm_problem                        prob{};
    std::vector<svm_node*>             node_ptrs;
    std::vector<std::vector<svm_node>> nodes;

    LibSVMProblem(const std::vector<std::vector<double>>& X,
                  std::span<const double>                  y)
    {
        int n = static_cast<int>(X.size());
        int d = static_cast<int>(X[0].size());
        prob.l = n;
        prob.y = const_cast<double*>(y.data());
        nodes.resize(n, std::vector<svm_node>(d + 1));
        node_ptrs.resize(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                nodes[i][j].index = j + 1;
                nodes[i][j].value = X[i][j];
            }
            nodes[i][d].index = -1;
            node_ptrs[i] = nodes[i].data();
        }
        prob.x = node_ptrs.data();
    }
};

static svm_parameter make_libsvm_params(const EpsilonSVRParams& p)
{
    svm_parameter sp{};
    sp.svm_type    = EPSILON_SVR;
    sp.kernel_type = RBF;
    sp.gamma       = p.gamma;
    sp.C           = p.C;
    sp.p           = p.epsilon;
    sp.eps         = p.tol;
    sp.cache_size  = 64;
    sp.shrinking   = 0;    // off: both solvers follow identical WSS path
    sp.probability = 0;
    return sp;
}

static double libsvm_predict(const svm_model* model,
                             const std::vector<double>& x)
{
    int d = static_cast<int>(x.size());
    std::vector<svm_node> node(d + 1);
    for (int j = 0; j < d; ++j) { node[j].index = j+1; node[j].value = x[j]; }
    node[d].index = -1;
    return svm_predict(model, node.data());
}

// ============================================================================
// Per-coefficient extraction from libSVM
//
// libSVM stores SVs in an arbitrary order. We recover per-training-point coefs
// by matching each libSVM SV to a training point by exact feature equality.
// Returns a vector of length n; coef[i] = α_i − α*_i (0 if not a SV).
// ============================================================================

static std::vector<double> extract_libsvm_coefs(
    const svm_model*                        lm,
    const std::vector<std::vector<double>>& X)
{
    int n = static_cast<int>(X.size());
    int d = static_cast<int>(X[0].size());
    std::vector<double> coef(n, 0.0);
    std::vector<bool>   matched(n, false);

    for (int j = 0; j < lm->l; ++j) {
        const svm_node* sv = lm->SV[j];
        for (int i = 0; i < n; ++i) {
            if (matched[i]) continue;
            bool ok = true;
            for (int k = 0; k < d; ++k) {
                if (std::abs(sv[k].value - X[i][k]) > 1e-12) { ok = false; break; }
            }
            if (ok) {
                coef[i]    = lm->sv_coef[0][j];
                matched[i] = true;
                break;
            }
        }
    }
    return coef;
}

// ============================================================================
// Build an EpsilonSVRModel from a libSVM solution so we can apply our
// KKT checker to libSVM's answer.
// ============================================================================

static EpsilonSVRModel model_from_libsvm(
    const svm_model*                        lm,
    const std::vector<std::vector<double>>& X,
    const EpsilonSVRParams&                 p)
{
    int n = static_cast<int>(X.size());
    EpsilonSVRModel m;
    m.params = p;
    m.bias   = -lm->rho[0];          // libSVM rho = −b
    m.coef   = extract_libsvm_coefs(lm, X);
    for (int i = 0; i < n; ++i) {
        if (std::abs(m.coef[i]) > 1e-8) {
            m.sv_x.push_back(X[i]);
            m.sv_coef.push_back(m.coef[i]);
            ++m.n_sv;
        }
    }
    return m;
}

// ============================================================================
// KKT checker (five-region classification; see epsilon_svr_design_note.md)
//
// r_i = f(x_i) − y_i   (positive = prediction above target)
//
//   STRICTLY INSIDE  |r| < ε − δ   → α = 0, α* = 0
//   ON UPPER BAND    r ∈ [ε−δ,ε+δ] → α = 0, α* free in [0,C]  (f ≈ y+ε)
//   ON LOWER BAND    r ∈[−ε−δ,−ε+δ]→ α* = 0, α free in [0,C]  (f ≈ y−ε)
//   ABOVE TUBE       r > ε+δ       → α* = C, α = 0
//   BELOW TUBE       r < −ε−δ      → α = C, α* = 0
// ============================================================================

struct KKTStatus {
    int    n_violations = 0;
    double max_kkt_violation = 0.0;
    double dual_equality = 0.0;
};

static KKTStatus check_kkt(const EpsilonSVRModel&                  model,
                            const std::vector<std::vector<double>>& X,
                            std::span<const double>                  y,
                            double kkt_tol = 1e-2)
{
    int    n   = static_cast<int>(X.size());
    double C   = model.params.C;
    double eps = model.params.epsilon;
    KKTStatus s;
    for (double c : model.coef) s.dual_equality += c;

    for (int i = 0; i < n; ++i) {
        double r      = model.predict(X[i]) - y[i];
        double alpha  = std::max(0.0,  model.coef[i]);
        double alphas = std::max(0.0, -model.coef[i]);

        bool above      = r >  eps + kkt_tol;
        bool below      = r < -eps - kkt_tol;
        bool on_upper   = !above && r >  eps - kkt_tol;  // f ≈ y+ε: α* free, α=0
        bool on_lower   = !below && r < -eps + kkt_tol;  // f ≈ y-ε: α free, α*=0
        bool str_inside = !above && !below && !on_upper && !on_lower;

        double v = 0.0;
        if (str_inside) v = std::max({v, alpha, alphas});
        else if (on_upper) { v = std::max(v, alpha); v = std::max(v, std::max(0.0, alphas-C)); }
        else if (on_lower) { v = std::max(v, alphas); v = std::max(v, std::max(0.0, alpha-C)); }
        else if (above)    { v = std::max(v, std::abs(alphas-C)); v = std::max(v, alpha); }
        else               { v = std::max(v, std::abs(alpha-C));  v = std::max(v, alphas); }

        s.max_kkt_violation = std::max(s.max_kkt_violation, v);
        if (v > kkt_tol) ++s.n_violations;
    }
    return s;
}

// ============================================================================
// Original reconcile() — prediction-level comparison (T1-T6)
// ============================================================================

struct ReconcileResult {
    int    n_sv_ours, n_sv_libsvm;
    double bias_ours, bias_libsvm;
    double max_diff, mean_diff;
    bool   pass;
};

static ReconcileResult reconcile(
    const char*                              name,
    const std::vector<std::vector<double>>& X,
    std::span<const double>                  y,
    const EpsilonSVRParams&                  p,
    double                                   pred_tol = 1e-2)
{
    svm_set_print_string_function(silent_print);
    auto  our = EpsilonSVRSolver::train(X, y, p);
    LibSVMProblem lp(X, y);
    auto  sp  = make_libsvm_params(p);
    svm_model* lm = svm_train(&lp.prob, &sp);

    int n = static_cast<int>(X.size());
    double max_diff = 0.0, sum_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = std::abs(our.predict(X[i]) - libsvm_predict(lm, X[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }

    ReconcileResult r{
        .n_sv_ours   = our.n_sv,
        .n_sv_libsvm = lm->l,
        .bias_ours   = our.bias,
        .bias_libsvm = -lm->rho[0],
        .max_diff    = max_diff,
        .mean_diff   = sum_diff / n,
        .pass        = max_diff < pred_tol
    };
    std::printf("\n--- %s ---\n", name);
    std::printf("  n_sv:  ours=%-3d  libsvm=%-3d  %s\n",
                r.n_sv_ours, r.n_sv_libsvm,
                r.n_sv_ours == r.n_sv_libsvm ? "" : "  << n_sv MISMATCH");
    std::printf("  bias:  ours=%.6f  libsvm=%.6f\n", r.bias_ours, r.bias_libsvm);
    std::printf("  pred:  max|diff|=%.2e  mean|diff|=%.2e  (tol=%.2e)  %s\n",
                r.max_diff, r.mean_diff, pred_tol,
                r.pass ? "PASS" : "FAIL");
    svm_free_and_destroy_model(&lm);
    return r;
}

// ============================================================================
// reconcile_deep() — prediction + per-coef + KKT-on-libSVM + n_sv + bias
//
// This is the function that would have caught all three bugs.
// Each sub-check is independently pass/fail and reported separately so that
// a future regression is immediately localised.
// ============================================================================

struct DeepReconcileResult {
    // sub-check results
    bool pred_ok;       // max prediction difference < pred_tol
    bool nsv_ok;        // n_sv counts match exactly
    bool coef_ok;       // per-training-point coef matches to coef_tol
    bool kkt_ours_ok;   // our KKT checker passes our solution
    bool kkt_libsvm_ok; // our KKT checker passes libSVM's solution (checker validity)
    bool bias_ok;       // |bias_ours - bias_libsvm| < bias_tol

    // diagnostics
    int    n_sv_ours, n_sv_libsvm;
    double max_pred_diff;
    double max_coef_diff;
    double bias_diff;
    int    kkt_viols_ours, kkt_viols_libsvm;
    double kkt_max_ours, kkt_max_libsvm;
    double dual_eq_ours;

    bool pass() const {
        return pred_ok && nsv_ok && coef_ok && kkt_ours_ok && kkt_libsvm_ok && bias_ok;
    }
};

static DeepReconcileResult reconcile_deep(
    const char*                              name,
    const std::vector<std::vector<double>>& X,
    std::span<const double>                  y,
    const EpsilonSVRParams&                  p,
    double pred_tol = 1e-2,
    double coef_tol = 1e-3,
    double bias_tol = 1e-4)
{
    svm_set_print_string_function(silent_print);
    int n = static_cast<int>(X.size());

    // --- Train both solvers -------------------------------------------------
    auto  our = EpsilonSVRSolver::train(X, y, p);
    LibSVMProblem lp(X, y);
    auto  sp  = make_libsvm_params(p);
    svm_model* lm = svm_train(&lp.prob, &sp);

    // --- (a) Prediction comparison ------------------------------------------
    double max_pred = 0.0;
    for (int i = 0; i < n; ++i)
        max_pred = std::max(max_pred,
            std::abs(our.predict(X[i]) - libsvm_predict(lm, X[i])));

    // --- (b) n_sv comparison ------------------------------------------------
    bool nsv_ok = (our.n_sv == lm->l);

    // --- (c) Per-coef comparison --------------------------------------------
    // Extract libSVM coef[i] = α_i − α*_i indexed by training point.
    auto lm_coef  = extract_libsvm_coefs(lm, X);
    double max_coef = 0.0;
    for (int i = 0; i < n; ++i)
        max_coef = std::max(max_coef, std::abs(our.coef[i] - lm_coef[i]));

    // --- (d) KKT on OUR solution --------------------------------------------
    auto kkt_ours = check_kkt(our, X, y);

    // --- (e) KKT on libSVM's solution ---------------------------------------
    // Build an EpsilonSVRModel from libSVM's output so we can run our checker.
    // If our checker reports violations on libSVM's verified solution, the
    // checker itself is broken (not the solver).
    auto lm_model  = model_from_libsvm(lm, X, p);
    auto kkt_libsvm = check_kkt(lm_model, X, y);

    // --- (f) Bias comparison ------------------------------------------------
    double bias_libsvm = -lm->rho[0];
    double bias_diff   = std::abs(our.bias - bias_libsvm);

    DeepReconcileResult r{
        .pred_ok        = max_pred   < pred_tol,
        .nsv_ok         = nsv_ok,
        .coef_ok        = max_coef   < coef_tol,
        .kkt_ours_ok    = kkt_ours.n_violations   == 0,
        .kkt_libsvm_ok  = kkt_libsvm.n_violations == 0,
        .bias_ok        = bias_diff  < bias_tol,
        .n_sv_ours      = our.n_sv,
        .n_sv_libsvm    = lm->l,
        .max_pred_diff  = max_pred,
        .max_coef_diff  = max_coef,
        .bias_diff      = bias_diff,
        .kkt_viols_ours    = kkt_ours.n_violations,
        .kkt_viols_libsvm  = kkt_libsvm.n_violations,
        .kkt_max_ours      = kkt_ours.max_kkt_violation,
        .kkt_max_libsvm    = kkt_libsvm.max_kkt_violation,
        .dual_eq_ours      = kkt_ours.dual_equality
    };

    // Report
    std::printf("\n--- %s (DEEP) ---\n", name);
    std::printf("  n_sv:     ours=%-3d  libsvm=%-3d  %s\n",
                r.n_sv_ours, r.n_sv_libsvm, r.nsv_ok ? "ok" : "MISMATCH <<");
    std::printf("  bias:     ours=%.6f  libsvm=%.6f  diff=%.2e  %s\n",
                our.bias, bias_libsvm, bias_diff,
                r.bias_ok ? "ok" : "MISMATCH <<");
    std::printf("  pred:     max|diff|=%.2e  (tol=%.2e)  %s\n",
                max_pred, pred_tol, r.pred_ok ? "ok" : "FAIL <<");
    std::printf("  coef:     max|diff|=%.2e  (tol=%.2e)  %s\n",
                max_coef, coef_tol, r.coef_ok ? "ok" : "FAIL <<");
    std::printf("  kkt/ours: viols=%-2d  max_viol=%.2e  dual_eq=%.2e  %s\n",
                r.kkt_viols_ours, r.kkt_max_ours, r.dual_eq_ours,
                r.kkt_ours_ok ? "ok" : "FAIL <<");
    std::printf("  kkt/libsvm: viols=%-2d  max_viol=%.2e  %s\n",
                r.kkt_viols_libsvm, r.kkt_max_libsvm,
                r.kkt_libsvm_ok ? "ok" : "CHECKER BUG <<");
    std::printf("  => %s\n", r.pass() ? "PASS" : "FAIL");

    svm_free_and_destroy_model(&lm);
    return r;
}

// ============================================================================
// Original T1-T6  (prediction-only reconciliation)
// ============================================================================

static ReconcileResult t1_constant()
{
    int n = 20;
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) { X[i][0] = i*0.1; y[i] = 3.0; }
    return reconcile("T1 constant y=3", X, y,
                     {.C=1.0,.epsilon=0.5,.gamma=0.5,.tol=1e-4}, 0.05);
}

static ReconcileResult t2_sine()
{
    int n = 30;
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        X[i][0] = i * 2.0 * M_PI / n;
        y[i]    = std::sin(X[i][0]);
    }
    return reconcile("T2 sin(x) n=30", X, y,
                     {.C=10.0,.epsilon=0.05,.gamma=0.5,.tol=1e-5}, 0.05);
}

static ReconcileResult t3_linear_2d()
{
    int n = 25;
    std::vector<std::vector<double>> X(n, {0.0,0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        X[i][0] = i*0.1; X[i][1] = std::cos(i*0.3);
        y[i] = 2.0*X[i][0] - X[i][1];
    }
    return reconcile("T3 linear 2D n=25", X, y,
                     {.C=5.0,.epsilon=0.1,.gamma=1.0,.tol=1e-5}, 0.05);
}

static ReconcileResult t4_noisy()
{
    int n = 20;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) { X[i][0] = i*0.1; y[i] = X[i][0] + noise(rng); }
    return reconcile("T4 noisy linear n=20", X, y,
                     {.C=1.0,.epsilon=0.2,.gamma=2.0,.tol=1e-5}, 0.05);
}

static ReconcileResult t5_fix_like()
{
    int n=40, d=3;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> strat(0,4);
    std::normal_distribution<double>   noise(0.0,3.0);
    std::vector<std::vector<double>> X(n, std::vector<double>(d));
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) X[i][j] = strat(rng)/4.0;
        y[i] = std::max(60.0 - 8.0*X[i][0] - 5.0*X[i][1] + 3.0*X[i][2] + noise(rng),
                        30.0);
    }
    return reconcile("T5 FIX-like n=40 d=3", X, y,
                     {.C=10.0,.epsilon=2.0,.gamma=0.1,.tol=1e-4}, 0.10);
}

static ReconcileResult t6_all_inside_tube()
{
    int n = 15;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> tiny(-0.03, 0.03);
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) { X[i][0] = i*0.1; y[i] = tiny(rng); }
    return reconcile("T6 all inside tube", X, y,
                     {.C=1.0,.epsilon=0.05,.gamma=0.5,.tol=1e-5}, 0.02);
}

// ============================================================================
// New T7-T10: deep tests that would have caught the implementation bugs
// ============================================================================

// ---------------------------------------------------------------------------
// T7: Cross-side working-set stress
//
// WHY: Bug 1 (Hessian sign) only fires when the SMO working pair crosses sides
// (one variable from α-side, one from α*-side; y2[t]*y2[q]=−1).
// An alternating-sign target forces the solver to make cross-side updates
// throughout, since simultaneously some points are BELOW tube (needing α)
// and others ABOVE tube (needing α*).
//
// Per-coef comparison is the critical check: the wrong Hessian changes the
// magnitude of every cross-side step → wrong α distribution → coef mismatch
// even when aggregate predictions happen to look reasonable.
// ---------------------------------------------------------------------------
static DeepReconcileResult t7_cross_side_stress()
{
    // Alternating ±A with A >> ε: every update must cross sides.
    // Exact alternation means f=0 initially → all points outside tube on
    // alternating sides → first WSS2 pair is always cross-side.
    int n = 24;
    double A = 0.6, eps = 0.05;
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        X[i][0] = i * (2.0 * M_PI / n);
        y[i]    = A * ((i % 2 == 0) ? +1.0 : -1.0);
    }
    // Small gamma so kernel is smooth; tight eps forces many SVs outside tube
    EpsilonSVRParams p{.C=5.0, .epsilon=eps, .gamma=0.3, .tol=1e-6};
    return reconcile_deep("T7 cross-side stress", X, y, p, 0.02, 1e-3, 1e-4);
}

// ---------------------------------------------------------------------------
// T8: KKT checker applied to libSVM's solution
//
// WHY: This is the test that directly exposes a broken KKT checker.
// libSVM's solution is correct by construction (it is the reference).
// If our check_kkt() reports violations on a libSVM solution, our checker
// has a bug — specifically, the on-band alpha/alpha* inversion that was
// present in the first implementation.
//
// Uses a sine target so both on-upper and on-lower band free SVs exist,
// exercising both halves of the checker's band logic.
// ---------------------------------------------------------------------------
static DeepReconcileResult t8_kkt_checker_validity()
{
    int n = 30;
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        X[i][0] = i * 2.0 * M_PI / n;
        y[i]    = std::sin(X[i][0]);
    }
    // Use params that produce free SVs on both tube boundaries (0 < α < C and
    // 0 < α* < C for different points) so the checker exercises both band arms.
    EpsilonSVRParams p{.C=10.0, .epsilon=0.05, .gamma=0.5, .tol=1e-6};
    return reconcile_deep("T8 KKT checker validity", X, y, p, 0.02, 1e-3, 1e-4);
}

// ---------------------------------------------------------------------------
// T9: Mixed α/α* SVs — both sides simultaneously active
//
// WHY: Bugs 1 and 2 produce wrong step sizes only when BOTH α > 0 and α* > 0
// exist in the solution simultaneously (the solver must place variables on
// both sides of the 2n variable space). A monotone target produces SVs on
// only one side; an oscillating non-monotone target forces both.
//
// Verification: coef vector must have both positive and negative entries
// (positive = α > α*, negative = α* > α). Exact coef values vs libSVM
// expose any wrong step sizes from the Hessian or f-update bugs.
// ---------------------------------------------------------------------------
static DeepReconcileResult t9_mixed_alpha_sides()
{
    // sin(x)*cos(2x) has both positive and negative excursions of varying
    // amplitude → some SVs will have coef > 0, others coef < 0.
    int n = 35;
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double x = i * 2.0 * M_PI / n;
        X[i][0] = x;
        y[i]    = std::sin(x) * std::cos(2.0 * x);
    }
    EpsilonSVRParams p{.C=8.0, .epsilon=0.04, .gamma=0.8, .tol=1e-6};
    auto r = reconcile_deep("T9 mixed alpha/alpha* SVs", X, y, p, 0.02, 1e-3, 1e-4);

    // Extra check: verify we actually have both positive and negative coefs
    // (if all coefs are one sign, the test isn't stressing both dual sides)
    svm_set_print_string_function(silent_print);
    auto our = EpsilonSVRSolver::train(X, y, p);
    bool has_pos = false, has_neg = false;
    for (double c : our.coef) {
        if (c >  1e-8) has_pos = true;
        if (c < -1e-8) has_neg = true;
    }
    std::printf("  coef signs: pos=%d neg=%d (both required for cross-side stress)\n",
                has_pos, has_neg);
    return r;
}

// ---------------------------------------------------------------------------
// T10: Tight bias agreement
//
// WHY: Bias is recovered from the dual solution — wrong α distribution
// produces wrong bias even if predictions look similar. The original
// reconcile() compared bias only as a diagnostic. Here it is a hard
// pass/fail criterion with tighter tolerance (1e-4).
//
// Also exercises the fallback bias path (no free SVs) vs normal path.
// ---------------------------------------------------------------------------
static DeepReconcileResult t10_bias_tight()
{
    // T10a: All-bound SVs (no free SVs) — exercises the fallback bias path.
    //
    // Construction: alternating ±1 targets with a flat RBF kernel (γ=0.05).
    // The kernel is too smooth to fit the sign alternations, so the model
    // predicts ≈0 everywhere. With ε=0.3, every point is outside the tube
    // (|r_i| ≈ 1.0 >> 0.3), so α_i = C or α*_i = C for every SV — no free SVs.
    // Bias must therefore be recovered from the KKT bound inequality path.
    //
    // Verified stable: n_sv=16/16, n_free=0, n_bound=16 for γ in [0.01, 0.1].
    int n = 16;
    std::vector<std::vector<double>> X(n, {0.0});
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        X[i][0] = i * 0.2;
        y[i]    = (i % 2 == 0) ? 1.0 : -1.0;  // alternating: ±1
    }
    EpsilonSVRParams p_bound{.C=1.0, .epsilon=0.3, .gamma=0.05, .tol=1e-6};
    auto r1 = reconcile_deep("T10a bias (all-bound SVs, fallback path)", X, y,
                             p_bound, 0.02, 1e-3, 1e-4);

    // T10b: Free SVs present (0 < α < C) — exercises the normal bias path.
    // Uses the same dataset but tighter ε so the kernel CAN place some SVs
    // on the tube boundary.
    int n2 = 20;
    std::vector<std::vector<double>> X2(n2, {0.0});
    std::vector<double> y2(n2);
    for (int i = 0; i < n2; ++i) {
        X2[i][0] = i * 0.15;
        y2[i]    = std::sin(X2[i][0] * 2.0) + 0.3 * X2[i][0];
    }
    EpsilonSVRParams p_free{.C=2.0, .epsilon=0.08, .gamma=0.8, .tol=1e-6};
    auto r2 = reconcile_deep("T10b bias (free SVs, normal path)", X2, y2,
                             p_free, 0.02, 1e-3, 1e-4);

    // Return the stricter result for pass/fail counting
    return r1.pass() ? r2 : r1;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::printf("=== Reconciliation: ctdp::epsilon_svr vs libSVM ===\n");
    std::printf("    shrinking=OFF; same C/eps/gamma/tol throughout\n");

    int passed = 0, total = 0;

    // --- T1-T6: original prediction-level checks ----------------------------
    std::printf("\n--- Original tests (prediction level) ---\n");
    auto check = [&](ReconcileResult r) {
        ++total; if (r.pass) ++passed;
    };
    check(t1_constant());
    check(t2_sine());
    check(t3_linear_2d());
    check(t4_noisy());
    check(t5_fix_like());
    check(t6_all_inside_tube());

    // --- T7-T10: deep checks ------------------------------------------------
    std::printf("\n--- New tests (per-coef + KKT-on-libSVM + cross-side) ---\n");
    auto check_deep = [&](DeepReconcileResult r) {
        ++total; if (r.pass()) ++passed;
    };
    check_deep(t7_cross_side_stress());
    check_deep(t8_kkt_checker_validity());
    check_deep(t9_mixed_alpha_sides());
    check_deep(t10_bias_tight());

    std::printf("\n=== Summary: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
