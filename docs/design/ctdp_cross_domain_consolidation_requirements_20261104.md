# CT-DP cross-domain consolidation: requirements and design review

Date: 2026-11-04
Status: REQUIREMENTS REVIEW, NOT AN APPROVED IMPLEMENTATION.
Source: discussion about consolidating CT-DP, Deterministic Numerics, FFT/NTT, matrix/SpMV, special functions, parallel algorithms, and ML training.

## Decision and boundary

We need a common approach and infrastructure across existing optimisation domains, not another optimiser, another mathematical intermediate representation, or a generic post-hoc winner selector. Preserve CT-DP's existing rich grammar, optimiser portfolio, typed compile-time route, cost modelling and calibrator. Consolidate their integration points.

Deferred: new algorithmic expression theory, general parallel rewrite/lifting discovery, Polya++ discovery, and domain-specific search-space reduction research. Existing legality/conditional filtering and domain-specific prefilters continue to work; deferring research does not mean removing those capabilities.

This document is the prerequisite for any adoption of the pre-existing branch feature/common-optimisation-session-20261104. That prototype was not written against these full requirements and is NOT accepted on architectural grounds merely because it compiles.

## Baseline: technologies to reuse, not replace

- CT-DP space descriptors, typed scalar/product/conditional/heterogeneous/permutation/partition/tree/interval/graph-oriented spaces; search-space capability tiers; NTTP/constexpr paths.
- CT-DP solver families: exhaustive, beam, local, per-element, interval and memoised compositional Pareto, plus domain-specific specialisations. Solvers impose their own space concepts.
- CT-DP existing core plan and solve_stats, calibrator::plan, provenance, feature bridge/schema v2 and wisdom serialization. They have different roles: do not quietly collapse them.
- CT-DP shared physical_recursive, index_geometry and compositional_pareto; these are already shared by multiple DN domains.
- DN planning/contract.hpp, ctdp_selector.hpp, domain-native numerical proof, measurements, actual physical plans and pinned target/build identity. Keep DN as authority on numerical contracts and backend qualification.
- Domain clients: qualified experimental matrix-DAG/prepared BLAS planning; exact modular NTT radix planning; SpMV/replay; opt-in gamma formulation/segmentation, MPFR smoke qualification and separate generated tol8 kernels; deterministic training/AD and checkpoint planning.

## Requirements (MUST unless stated SHOULD)

### R01 — Domain-native, typed problem and search space
A client supplies a problem instance and its own legal space. Preserve structure, types, constraints, dependent subproblems, optional dimensions, and capability concepts. No mandatory vector-of-configurations, universal Cartesian projection, common algorithm-expression IR or runtime type erasure. Existing constexpr/NTTP routes remain zero-runtime-overhead.

### R02 — Use existing solver portfolio, not just one solver
A consistent invocation and outcome protocol must work with exhaustive/beam/local/interval/Pareto and specialised search without rewriting implementations. Solver applicability remains concept/property constrained; no assertion that every optimiser can solve every space. Retain native solver result and native candidate type, including multiobjective frontiers and approximation/incompleteness flags.

### R03 — Separate problem, search, candidate construction, qualification and physical selection
A domain may explore model-only abstract candidates without any executable lowering, may qualify a subset later, and may select a physical candidate after independent evidence arrives. The common interface must support these phases separately and their asynchronous/offline provenance; do not force assessment or timing of every model candidate inside one run() call. A model-only result is not a qualified implementation.

### R04 — Explicit legality and numerical semantics
Keep structural admissibility (shape, geometry, identity), mathematical legality, physical lowering availability, numerical accuracy/reproducibility and device qualification distinct. Numeric contracts can be exact-bit, specified deterministic expression, exact accumulation, tolerance/ULP, or domain-specific. A generic 'numerical_pass' boolean or equal hashes does not constitute a universal proof. A structurally valid NTT, mathematically equivalent gamma transformation and strictly reproducible FP32 GEMM have distinct verification requirements.

### R05 — Numerical backend and evidence ownership
CT-DP stays domain-neutral. DN owns MPFR oracles, correct/faithful/tol8 evidence, FP environments, mixed-precision semantics, actual CPU/CUDA/HIP/MLX kernels and validation of backend-specific numerical execution. Common infrastructure carries typed/generic evidence descriptors and references without duplicating DN's full numerical contract type.

### R06 — Multiobjective and solver-specific search statistics
Retain work, span, memory, numerical-pressure heuristics, setup costs and other objective vectors; no mandatory scalarization or premature pruning. Distinguish heuristic model costs, measured physical costs, qualified feasibility and Pareto bounds. Preserve native search statistics, budgets and 'approximate/truncated' states. Never claim global optimality from bounded beam/Pareto runs.

### R07 — Actual cost economics and resource admission
Support runtime latency/throughput, cold preparation, build/compilation, calibration and certification costs where relevant, invocation count and amortisation. Enforce workspace and known peak-memory limits (unknown full-process peak must fail under a hard peak cap). Model confidence, measurement noise and deployment reuse are recorded distinctly from verified runtime. Do not hard-code 'prepare + calibrate + K*runtime' as the only universal objective.

### R08 — Backend and precision extensibility without backend replacement
CPU, CUDA, HIP and MLX/Metal are physical realisations supplied by their existing libraries. Availability and qualification are properties of actual implementations, not of enum names. Precision choices, deterministic reduction geometry, and future AD/backprop optimisers can be domain metadata and candidate dimensions without requiring CT-DP to own a unified device-execution API.

### R09 — Stable and reproducible evidence, results and cache identity
A candidate, problem, source expression, observation and FP environment, input/workload scope, compiler/build, device, feature schema, cost model and evidence version must be attributable. Reuse existing CT-DP provenance/wisdom and DN identities where possible; do not fork competing serialization schemas. Failed/stale/partial/mismatched-scope qualification must fail closed. Record rejected alternatives and reason codes.

### R10 — Deterministic planning itself
Reproducible candidate ordering and tie-breaking, explicit search budgets, consistent results independent of enumeration order where promised. Model-only rankings must not be silently treated as physical selection evidence. Compile-time and offline/runtime planning must be permitted without changing numeric execution semantics.

### R11 — Minimal migration and compatibility
Add adapters at existing boundaries instead of rewriting core solvers or domain DSLs. Keep the existing matrix/NTT/function/SpMV numerical/test witnesses and default production dispatch unaffected. One pinned, qualified CT-DP revision used by DN integration CI. Separate CT-DP core from DN, with no reverse library dependency.

### R12 — Real cross-domain proof of consolidation
At minimum, matrix-DAG native qualified search, NTT recursive Pareto planning, and gamma formulation/segmentation must share the new protocol and preserve their original semantic admission statuses. SpMV is a fourth irregular-domain regression. The gamma smoke must remain smoke until complete tol8 reconstruction/coverage is qualified. No claim of full cross-platform ML training optimisation merely from a shared request structure.

### R13 — Documentation and incremental delivery
Document concepts, domain ownership, example invocations and admitted limitations. Stage reusable infrastructure in CT-DP first, then opt-in DN adapters, tests and CI. Merge reviewed, fully qualified infrastructure incrementally; do not require shipping new algorithms to finish consolidation.

## Existing optimisation-session prototype: preliminary gap review

Candidate source: feature/common-optimisation-session-20261104, include/ctdp/solver/optimisation_session.h.

Useful ideas to retain selectively:
- Native report is retained; domain plan types are not erased; no forced Cartesian enumeration.
- Separate model-only, measured-unqualified and numerically-qualified labels.
- Workload/scope/target/build identity, bounded candidate count and stable winner tie-breaking.
- Rejects overtly invalid claims of completed qualification.

Mismatches/gaps versus the requirements:
1. Its solve() / proposals() / stable_id() / assess() lifecycle is a *post-solver assessment and qualified winner selector*, not a common optimisation interface. No actual solver-family adaptation, structured-space protocol, session decomposition or reusable search orchestration (R01–R03).
2. One eager assessment pass over every returned proposal couples potentially costly physical compilation/oracle work to model search; no staged shortlisting, deferred evidence import or explicit search economics (R03, R07).
3. It reduces final selection to one scalar cost formula and winner; Pareto/native result is carried opaquely but there is no general multiobjective evaluation/result integration (R06).
4. It invents identity/evidence/cost structures alongside existing DN contract and CT-DP calibration/wisdom. There is no interoperation or schema migration rule, memory limits, numerical tolerance policy, independent proof type or measurement confidence (R04, R05, R07, R09).
5. It treats a domain-supplied numerically_qualified flag plus booleans as admission. That may be acceptable as a domain assertion, but must not be mistaken for actual ULP/bitwise/full-domain verification. Model-only, materially incomplete and mixed precision cases need distinct status and scope (R04).
6. A dynamic vector of copyable domain plans and post-hoc candidate cap is not itself an adequate compile-time optimisation protocol; it also cannot bound preceding recursive searches (R01, R02, R10).
7. No adapter migration or end-to-end matrix/NTT/gamma tests demonstrate common use (R11, R12).

Conclusion: DO NOT adopt the prototype as the consolidated framework. The phase-separated qualified selection idea may become an adapter, after reconciling existing contracts.

## Proposed minimal architecture to review before coding

- **Common problem/view concepts**: domain-owned instance, space, constraints, objectives, optional calibration context and identity; zero replacement of native search spaces.
- **Solver adapters**: constrained, typed façade into established CT-DP solver overloads and domain-specialist solvers; preserve native plan(s), cost vector, stats, bounds and exact/approximate indicators.
- **Exploration report**: model candidates and search provenance, no numerical qualification implied.
- **Physical/evidence adapter**: domain callback to construct, run, verify and time actual candidate(s); optional offline phase. Reuse existing DN admission, CT-DP measurement/provenance/wisdom rather than define another numerical oracle.
- **Qualified plan selection**: explicit domain contract and resource budget, ranking on verified measured costs, preserving alternate plans and rejection reasons.
- **Persistence and replay**: only once compatibility/migration with existing CT-DP calibrator and DN evidence is established. No new wisdom format merely to have one.

Each layer should be separable; do not require a monolithic session orchestration object.

## Proof and acceptance gates

1. Compile-time: existing constexpr search and NTTP plan tests unchanged. New adapters do not require allocations to run constexpr searches.
2. Search-family: at least one finite descriptor exhaustive, interval/partition structured, and compositional Pareto case through common typed invocation, retaining native reports. Existing first-party solver results are independent references.
3. Model-only safety: FFT/gamma model candidates cannot pass as qualified by providing a plausible cost, identifier or unverified fingerprint.
4. Numeric safety: matrix strict bitwise, NTT exact modular and gamma explicitly scoped tol8/smoke evidence stay distinct and fail under foreign scope/build/device/environment.
5. Cost/resource: amortisation can change winner; incomplete timing, missing lowerings and uncertified peak-memory budgets are rejected; search cost not silently zero.
6. Stability: deterministic ties, candidate identity, truncated-frontier warnings and rejected reasons are visible.
7. Cross-domain: identical current numerical witnesses for matrix, FFT/NTT, gamma and SpMV; GCC/Clang/MSVC/ARM64 gates where current domain gates exist.
8. Integration: default DN paths unchanged; one pinned CT-DP revision; no duplicate implementation of optimisers or generated numerical functions.

## Order of work AFTER requirements review

A. Confirm requirements against CT-DP maintainers/design and baseline test inventory; identify overlaps with pending permutation/interval branches.
B. Make a minimal shared *typed invocation + report* adapter using existing solver concepts. Prove compile-time and multiobjective preservation.
C. Connect evaluation/admission to existing CT-DP calibrator and DN planning contract; do not duplicate their data structures.
D. Migrate matrix, NTT and gamma clients one at a time with cross-domain regression; SpMV fourth.
E. Only then consider small, opt-in ML training and reverse-AD clients. Defer new research grammar and new backend APIs.

The implementation is NOT approved merely by the existence of this requirements note.
