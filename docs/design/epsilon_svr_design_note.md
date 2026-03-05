# `epsilon_svr.h` — Design Note

**Location:** `include/ctdp/solver/cost_models/epsilon_svr.h`
**Namespace:** `ctdp::cost_models`
**Date:** March 2026
**Status:** Implemented, reconciled against libSVM, 41/41 tests passing

---

## Purpose

Replaces the KRR (`svr_model.h`) surrogate with a proper epsilon-insensitive Support Vector Regression model. The primary motivation is the FIX parser calibration problem, where:

- Empirical p99 measurements have ~6 ns run-to-run variance
- A fixed epsilon tube absorbs this noise without penalising or fitting to it
- Outlier measurements are bounded in influence by the capacity parameter C
- The resulting sparse model (30–80 SVs from ~200 samples) generalises better than a dense KRR solution

---

## Mathematical Formulation

### Primal (Smola & Schölkopf, 1998)

Minimise:

```
0.5 ||w||² + C Σ (ξ_i + ξ*_i)

subject to:
  y_i - f(x_i) ≤ ε + ξ_i
  f(x_i) - y_i ≤ ε + ξ*_i
  ξ_i, ξ*_i ≥ 0
```

where `f(x) = w^T φ(x) + b` and `ε` is the half-width of the insensitive tube.

### Dual

Maximise:

```
-0.5 (α - α*)^T Q (α - α*) - ε Σ(α_i + α*_i) + Σ y_i(α_i - α*_i)

subject to:
  Σ (α_i - α*_i) = 0
  0 ≤ α_i, α*_i ≤ C
```

where `Q_ij = K(x_i, x_j)` is the kernel matrix. The decision function is:

```
f(x) = Σ_i (α_i - α*_i) K(x_i, x) + b
```

Points with `α_i - α*_i ≠ 0` are **support vectors** and are the only ones contributing to prediction.

### Kernel

RBF (Radial Basis Function):

```
K(x, z) = exp(-γ ||x - z||²)
```

---

## Implementation: 2n-Variable Encoding

The implementation uses the standard libSVM 2n-variable encoding to avoid handling the equality constraint `Σ(α_i - α*_i) = 0` explicitly.

Define 2n variables:

| Index | Variable | Sign `y2` |
|-------|----------|-----------|
| `i ∈ [0, n)` | `u[i] = α_i` | `+1` |
| `i ∈ [n, 2n)` | `u[i] = α*_{i-n}` | `−1` |

Linear terms:

```
p[i]   = ε − y[i]      for i < n
p[i+n] = ε + y[i]      for i < n
```

The gradient becomes:

```
G[i] = y2[i] · f[i%n] + p[i]
```

where `f[k] = Σ_j y2[j] · u[j] · K(x_{j%n}, x_k)` is the kernel-sum component of the prediction (no bias).

---

## SMO Solver

**Algorithm:** Sequential Minimal Optimisation with WSS2 second-order working-set selection (Fan, Chen, Lin, JMLR 2005).

### Working Sets

```
I_up = { i : y2[i]=+1, u[i] < C } ∪ { i : y2[i]=−1, u[i] > 0 }
I_lo = { i : y2[i]=+1, u[i] > 0 } ∪ { i : y2[i]=−1, u[i] < C }
```

At KKT optimality: `max_{I_up}(−y2·G) = min_{I_lo}(−y2·G)`.

### Working-Set Selection

1. **Select `t`:** `argmax_{I_up}(−y2[t]·G[t])` — the most-violated upper constraint.
2. **Select `q`:** `argmin_{I_lo} H / (gmax + y2[q]·G[q])²` where `H = K_{tt} + K_{qq} − 2·y2[t]·y2[q]·K_{tq}` — second-order improvement (WSS2).

> **Key bug caught during implementation:** The Hessian element `H` must include the `y2[t]·y2[q]` sign factor — `H = K_tt + K_qq − 2·y2[t]·y2[q]·K_tq`. Omitting this produced wrong step sizes (up to 2× error) causing the solver to converge to a wrong solution with 100% sparsity failure on the sine benchmark.

### Update Step

```
u[t] += y2[t] · step,   u[q] −= y2[q] · step
step = (gmax + y2[q]·G[q]) / H,  clipped to box constraints
```

The `f` update cancels all `y2` factors (since `y2² = 1`):

```
Δf[k] = step · (K[t%n, k] − K[q%n, k])
```

> **Second bug caught:** The initial `f`-update included `y2[t]` and `y2[q]` factors that should cancel. When both are `-1`, the net sign was wrong, causing divergent gradient tracking.

### Convergence

The solver terminates when the KKT gap `gmax − min_{I_lo}(−y2·G) ≤ tol`. Default `tol = 1e-3`.

### Bias Recovery

For free SVs (`0 < u[i] < C`):

```
b = −G[i]   (if i < n, α side)
b = +G[i]   (if i ≥ n, α* side)
```

Bias is the mean over all free SVs. If no free SVs exist (fully regularised solution), `b` is computed from the KKT bound-variable interval.

---

## KKT Conditions Reference

The five regions defined by residual `r_i = f(x_i) − y_i`:

| Region | Condition | α_i | α*_i |
|--------|-----------|-----|------|
| Strictly inside | `\|r\| < ε − δ` | `= 0` | `= 0` |
| On upper band | `r ∈ [ε−δ, ε+δ]` | `= 0` | `∈ [0, C]` (free) |
| On lower band | `r ∈ [−ε−δ, −ε+δ]` | `∈ [0, C]` (free) | `= 0` |
| Above tube | `r > ε + δ` | `= 0` | `= C` |
| Below tube | `r < −ε − δ` | `= C` | `= 0` |

where `δ` is the checker tolerance. Note: `coef[i] = α_i − α*_i`.

**Sign mnemonic:** `r = f − y`. If `f` is *above* `y` by more than `ε`, the upper-slack constraint is saturated → `α*_i = C` → `coef = −C`. If `f` is *below* `y` by more than `ε`, the lower-slack constraint is saturated → `α_i = C` → `coef = +C`.

> **Checker bug corrected during testing:** The initial KKT checker had the on-band alpha assignments inverted (testing `alpha* = 0` on the upper band instead of `alpha = 0`). This produced spurious violations for every free SV. The five-region classifier with strict `ε ± δ` band edges is required to separate free SVs (boundary band) from interior points (alpha must be zero).

---

## Reconciliation Against libSVM

All tests use `shrinking=0` in libSVM for a deterministic comparison path.

| Test | n_sv match | max pred diff |
|------|-----------|---------------|
| Constant y=3, n=20 | ✓ (0/0) | 0.000000 |
| sin(x), n=30 | ✓ (7/7) | 1.4×10⁻⁵ |
| Linear 2D, n=25 | ✓ (10/10) | 2.2×10⁻⁵ |
| Noisy linear, n=20 | ✓ (11/11) | 2×10⁻⁶ |
| FIX-like, n=40 d=3 | ✓ (24/24) | 7.6×10⁻⁵ |
| All in tube, n=15 | ✓ (0/0) | 0.000000 |

Residual prediction differences are floating-point order-of-operations noise from the bias averaging, not algorithm divergence.

---

## Test Suite

**File:** `test_epsilon_svr_full.cpp`
**Build:** `g++ -std=c++23 -O2 -I. -I/usr/include/libsvm test_epsilon_svr_full.cpp -lsvm -o test_epsilon_svr_full`
**Result:** 41/41 pass

### Group A: Mathematical invariants

| ID | Test | What it checks |
|----|------|----------------|
| A1 | KKT complementary slackness — sine | No point violates its region's alpha constraint |
| A2 | KKT with gross outliers | Outlier alphas saturate at C; correct sign (coef = ±C) |
| A3 | Dual equality Σ(α − α*) = 0 | Maintained by every SMO step; verified post-convergence |
| A4 | Dual objective matches libSVM | Same optimal value, confirmed via prediction agreement |
| A5 | KKT valid after grid search | Best model from LOO grid search is still a valid dual solution |

### Group B: Behavioural properties

| ID | Test | What it checks |
|----|------|----------------|
| B1 | C-monotonicity | Training RMSE non-increasing as C grows |
| B2 | ε-monotonicity | n_sv non-increasing as ε grows; zero SVs at ε >> range(y) |
| B3 | Shift invariance: y → y + δ | Predictions shift by exactly δ; n_sv unchanged |
| B4 | Negation invariance: y → −y | Predictions negate; n_sv unchanged |
| B5 | Bias at large ε | n_sv = 0; bias ≈ mean(y) |
| B6 | Rank order preserved | Spearman ρ ≈ 1 on monotone target |

**Note on C-monotonicity:** n_sv is **not** monotone in C and no such assertion is made. At very small C the model is heavily regularised but still needs many small-α SVs for points outside the tube; as C grows the model fits better and those points move inside the tube, reducing n_sv before eventually re-increasing at large C. Only RMSE is guaranteed monotone.

### Group C: Degenerate and edge cases

| ID | Test | What it checks |
|----|------|----------------|
| C1 | n=1 | Single-point problem: `\|f(x₀) − y₀\| ≤ ε` |
| C2 | Duplicate feature vectors | No crash, no NaN when kernel matrix has repeated rows |
| C3 | Very small C (1e-4) | All `\|coef[i]\| ≤ C`; box constraint respected |
| C4 | Large C, small ε | Near-interpolation: training RMSE < 0.05 |
| C5 | All targets identical | n_sv = 0; bias equals the constant value |
| C6 | n=2, both in tube | n_sv = 0 |
| C7 | Near-collinear features | Dual equality holds despite near-singular kernel |

### Group D: Regression correctness

| ID | Test | What it checks |
|----|------|----------------|
| D1 | sin(x), n=30 | RMSE < 0.1; 0 < n_sv < n (genuine sparsity) |
| D2 | Linear 2D, n=25 | RMSE < 0.15 |
| D3 | FIX-like, n=40 d=3 | RMSE < 15 ns; KKT valid; predictions in [20, 90] ns |
| D4 | predict_batch | Batch predictions identical to individual predictions |
| D5 | Spearman ρ helper | Perfect ±1 on known monotone/anti-monotone pairs |
| D6 | Matches libSVM (regression guard) | n_sv identical; max pred diff < 0.01 ns |

---

## API Reference

```cpp
// Hyper-parameters
struct EpsilonSVRParams {
    double C        = 1.0;       // Penalty for points outside tube
    double epsilon  = 0.1;       // Tube half-width (same units as y)
    double gamma    = 0.1;       // RBF kernel width
    double tol      = 1e-3;      // KKT gap convergence tolerance
    int    max_iter = 100'000;   // Iteration cap
};

// Train
EpsilonSVRModel model = EpsilonSVRSolver::train(X, y, params);
// or:
EpsilonSVRModel model = fit_epsilon_svr(X, y, params);

// Predict
double y_hat = model.predict(x);                    // single sample
auto   preds = model.predict_batch(X);              // batch

// Inspect
int    n_sv    = model.n_sv;
double bias    = model.bias;
double sparse  = model.sparsity_ratio();            // n_sv / n_train

// Grid search (LOO cross-validation)
SVRGridResult r = svr_grid_search_loo(X, y,
    C_grid, epsilon_grid, gamma_grid, tol);
// r.best_params, r.best_cv_rmse, r.best_cv_spearman, r.best_n_sv

// Rank correlation helper
double rho = spearman_rho(predictions, targets);
```

---

## Hyper-parameter Guidance for FIX Parser Calibration

Given the observed problem characteristics (n ≈ 40–200 configs, d = 3–5 features, p99 in [30, 80] ns, 6 ns run-to-run noise):

| Parameter | Recommended starting range | Rationale |
|-----------|---------------------------|-----------|
| C | {1, 10, 100} | Low C gives bias-dominated model; high C overfits noise |
| ε | {1, 2, 5} ns | Must cover the ~6 ns run-to-run variance floor |
| γ | {1e-4, 1e-3, 0.01} | KRR was selecting ~1e-4 (near-flat kernel); try wider range |

The epsilon tube is the key advantage over KRR for this use case. Setting ε ≈ 2–3× the measurement noise standard deviation gives the tube the right coverage without absorbing genuine performance differences between strategies.

---

## References

- Smola, A. & Schölkopf, B. "A tutorial on support vector regression." *Statistics and Computing* 14, 2004.
- Fan, R., Chen, P. & Lin, C. "Working set selection using second order information for training support vector machines." *JMLR* 6:1889–1918, 2005.
- libSVM source: https://github.com/cjlin1/libsvm (BSD-3 licence)
