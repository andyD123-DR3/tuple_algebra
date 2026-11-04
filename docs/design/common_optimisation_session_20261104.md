# CT-DP common optimisation session — consolidation tranche A

## Boundary and scope

`include/ctdp/solver/optimisation_session.h` composes existing CT-DP
solvers, arbitrary domain-native search spaces, domain-specific physical
evaluation and a reusable result/evidence lifecycle.

The shared `run(request, solve, proposals, stable_id, assess)` invocation
does **not** replace the rich CT-DP space algebra, interval/subset/beam/local/
Pareto algorithms, CT-DP calibration or existing specialised numerical
lowerers. `solve()` returns its original, unmodified native result.
`proposals(native)` exposes domain-native candidate objects from that result;
the session retains the complete native report and all exploration statistics.

The protocol is deliberately C++20 statically typed rather than type-erased;
compile-time solvers and NTTP specialisation paths need not be wrapped in a
runtime optimisation engine.

## Evidence contract

`model_only`: a planning/search result, never eligible for selection.
`measured_unqualified`: physically measured, but no numerical claim.
`numerically_qualified`: caller asserts actual physical execution, independent
reference, complete declared observation scope, numerical admission and finite
measured costs. Every qualified record must match expression, observation,
environment, input scope, target and compiler/build identities.

Only qualified candidates may be selected, by measured total workload
`preparation + calibration + invocations * runtime`, with stable ID ties.
The domain is responsible for numeric certification; a bounded-ULP result may
properly have a candidate output digest different from the reference digest.
The framework cannot turn a boolean into a proof. Invalid purported
qualification, foreign evidence, duplicate IDs and invalid costs fail closed.

## Executable regression

The standalone experiment exercises:
1. a genuine CT-DP `compositional_pareto` frontier with model-only proposals;
2. a real CT-DP descriptor/exhaustive search, preserving its native statistics;
3. qualified measured selection with preparation amortisation and stable ties;
4. rejection of missing scope, numerical failure, missing physical work,
   non-finite timing, duplicated identity and exhausted candidate budgets.

CI runs GCC, Clang, MSVC and native Linux ARM64. Domain-specific MPFR or
floating-point conformance remains the responsibility of DN clients.

## Integration sequence

DN matrix, FFT/NTT and gamma adapters should call this one session boundary
while leaving their existing physical-plan selection, independent numerical
oracles, and model-only search algorithms in place. CT-DP owns the generic
session; DN owns typed admission and execution semantics. Do not duplicate
DN numerical or backend contracts into CT-DP.

Deferred: general mathematical rewrite discovery, parallel expression theory,
domain-specific search-space reduction, Pólya++ integration and production
kernel dispatch.
