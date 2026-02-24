# Performance Model Cross-Validation — ctdp v0.7.0

## Overview

The performance model subsystem now measures **genuine out-of-sample (OOS)** generalisation performance via 5-fold cross-validation. Previously, all quality metrics (R², Spearman ρ) were computed on the training data — meaning a model that memorised the data perfectly would appear to have perfect quality. This is fixed.

## Architecture

```
observations → feature_extract → trainer.build(obs)
                                    ├─ fit on full data (in-sample metrics)
                                    └─ 5-fold CV       (OOS metrics)
                                         → model_quality
                                              → model_factory selects on oos_spearman_rho
```

### Files

| File | Role |
|------|------|
| `performance_model.h` | Concept, `observation<Point>`, `model_quality`, metric helpers, `any_model<Point>` |
| `feature_extract.h` | `raw`, `log2`, `log2_interactions`, `reciprocal` extractors |
| `cross_validation.h` | **NEW** — generic `k_fold_cv` with stratified fold assignment |
| `linear_model.h` | OLS via normal equations, z-score normalisation, closed-form LOO, 5-fold CV |
| `mlp_model.h` | 3-layer MLP (NF→H1→H2→1), SGD+momentum, Xavier init, 5-fold CV |
| `svr_model.h` | Kernel ridge regression, RBF kernel, **two-stage CV grid search** |
| `model_factory.h` | Trains all 3, selects winner by **`oos_spearman_rho`** |

## model_quality struct

```cpp
struct model_quality {
    double r2;                // in-sample
    double spearman_rho;      // in-sample
    double oos_r2;            // 5-fold CV
    double oos_spearman_rho;  // 5-fold CV ← factory selection criterion
    double oos_rmse;          // 5-fold CV
    double loo_r2;            // leave-one-out (linear only, closed-form)
    std::size_t n_samples;
    std::size_t n_params;
    std::string model_name;
};
```

## Cross-validation design

### Stratified K-fold splitting
Observations are sorted by target value rank and dealt round-robin into K folds. This ensures each fold contains a representative spread of easy/hard cases, preventing pathological splits where (e.g.) all the fast tile shapes end up in one fold.

### Generic interface
```cpp
auto cv = k_fold_cv<Point>(observations, 5,
    [&](const auto& train_obs) { return trainer.build_raw(train_obs); });
// cv.oos_r2, cv.oos_spearman_rho, cv.oos_rmse
```
Works with any trainer whose `.build_raw(data)` returns a model with `.predict(point)`.

## SVR two-stage hyperparameter tuning

Previously the SVR selected (λ, γ) by in-sample Spearman ρ — with tiny λ this led to interpolation/memorisation (R² ≈ 1.0 in-sample, poor generalisation).

### Stage 1: Coarse grid
5 × 5 grid over log₁₀(λ) ∈ {−3, −2, −1, 0, 1} and log₁₀(γ) ∈ {−3, −2, −1, 0, 1}. Each combination evaluated by 5-fold CV Spearman ρ.

### Stage 2: Fine-tune
5 × 5 grid of ±0.5 log-step offsets around the coarse winner. Best combination selected.

### Final model
Retrained on full data with tuned (λ, γ). OOS metrics reported from CV at the final hyperparameters.

## Test results

```
Section 1: Metric helpers                    — 4/4 pass
Section 2: Feature extractors                — 8/8 pass
Section 3: Cross-validation infrastructure   — 7/7 pass
Section 4: Linear model with CV              — 14/14 pass
Section 5: MLP model with CV                 — 8/8 pass
Section 6: SVR with two-stage CV tuning      — 10/10 pass
Section 7: Model factory (OOS ρ selection)   — 5/5 pass
Section 8: Factory finds correct optimum     — 4/4 pass
Total: 60 passed, 0 failed
```

Key validations:
- OOS metrics are finite and populated for all three models
- OOS ρ ≤ in-sample ρ (generalisation gap exists, as expected)
- SVR no longer trivially memorises (OOS R² meaningful when in-sample R² ≈ 1.0)
- Factory selects the model with the highest OOS Spearman ρ
- Factory-selected model correctly identifies the optimum tile shape on synthetic data

## Build

```bash
cd build
cmake .. -DCTDP_BUILD_TESTS=ON -DCTDP_BUILD_EXAMPLES=OFF
make test_performance_models -j$(nproc)
./tests/test_performance_models
```
