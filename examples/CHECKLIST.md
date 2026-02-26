# Example PR Checklist

Every example must pass all seven checks before commit.
Each check traces to a bug that occurred at least once during development.

## The Checks

- [ ] **1. No dead knobs** — every search dimension changes generated code. If the solver can select AVX512, the executor must implement AVX512. If `aligned` is a dimension, aligned loads must actually be used when `aligned == true`.

- [ ] **2. No lying metadata** — `cardinality()` and `rank` are truthful. A filtered space must not report the unfiltered count. If you can't implement `cardinality()` honestly, don't expose it.

- [ ] **3. Single solve function** — compile-time and runtime paths share one `make_problem()` or equivalent. There must not be separate space construction for `static constexpr` vs runtime evaluation.

- [ ] **4. Search matches executor** — every value the solver can select has a corresponding `if constexpr` branch in the executor, or the executor `static_assert(false)` on unhandled values.

- [ ] **5. Constraints structural** — validity is in the space (via `filter_valid` or equivalent), not `cost = +infinity` inside the cost function.

- [ ] **6. Bridge from same space** — feature encoding uses the same (filtered) space that the solver searched, not the unfiltered base.

- [ ] **7. Non-trivial correctness** — non-uniform inputs (not all-ones), verified against a scalar reference implementation. All-ones inputs hide transposition and indexing bugs.

## Why This Exists

We proved that knowing better doesn't prevent regression.
A list does.
