# Red-team reference: exact chain fusion and the first broken boundary

Status: bounded C++20 research experiment. No numerical kernel or production
code generation. The results are exact **only for the declared integer-cost
models**. They are NOT hardware throughput measurements or proofs that
arbitrary floating-point expressions may be fused or duplicated.

## Hypothesis and falsification criteria

Hypothesis H1: for a pure linear single-consumer chain and an independently
specified nonnegative finite total cost for every admissible contiguous
fusion segment, the optimal fusion partition is found by the prefix recurrence

    D(0) = 0
    D(j) = min over 0 <= i < j of D(i) + C(i,j).

Every segment cost includes its entire execution and boundary overhead.
There are no shared producers, externally observed intermediates, resource
interactions across segments, layout choices or unmodelled numerical effects.
The DP is O(n^2) in its bounded cost table and records a canonical cut mask.
The test independently reconstructs every 2^(n-1) possible partition.

**Gate A** is exact equality of feasibility, minimum cost and canonical cut
mask against an independently implemented exhaustive oracle for:
- Every 3^10 assignment of {zero, positive, forbidden} to the ten contiguous
  segments of a four-operation chain (59,049 tables).
- 1,200 deterministic random larger chains, with missing segments,
  zero costs, ties and infeasible instances.
- Integer-overflow and malformed-interval negative cases.
- Compile-time C++20 static_assert cases, independent of test build mode.

No associative rearrangement, floating-point equivalence or memory use is
established by this model.

## Boundary-state counterexample, before branching

A two-operation chain can already defeat a prefix-greedy algorithm if the
intermediate layout is part of the continuation cost. The generator searches
positive producer/consumer costs in increasing aggregate weight and returns:

    producer: layout A = 1, layout B = 2
    consumer: layout A = 3, layout B = 1
    conversions: not available

Keeping only the producer's cheapest layout A gives 4. Preserving both
boundary states gives layout B with global cost 3. The minimum witness has
weight 7 within the enumerated four-cost grammar.

The corrected state-aware recurrence carries the output layout:

    D(j,layout) = C(j,layout) +
                  min_previous (D(j-1,previous) +
                                conversion(previous,layout)).

**Gate B** compares this DP against independent enumeration of all 2^n
layout sequences for 1,500 deterministic irregular chains, including
unavailable transitions and costs. It tests that preserving boundary state
fixes the known failure.

## First branch: producer with two consumers

The smallest directed fork has three nodes: a producer P and consumers L, R.
Assume pure, freely duplicable computation for the independent-chain lift.

Two explicitly costed physical strategies:

    duplicate producer: 2*P + L + R - fusion_gain_L - fusion_gain_R
    materialize once:   P + L + R + materialization_overhead

The naive auxiliary lift solves P->L and P->R as independent optimal chains,
then executes both chains. It must pay twice for P; the independent solutions
cannot share a producer implicitly. Materialization instead evaluates P
once and pays explicit intermediate storage/reading overhead.

The generator searches all admissible positive integer parameters within
P <= 8, L/R <= 4, overhead <= 3, gains <= 3. It canonically removes consumer
symmetries, prioritises total model weight, then lexicographic parameter
order, and returns the first eight examples where independent-chain
lifting is valid but **suboptimal**.

The minimal witness in this grammar is:

    P=4, L=1, R=1, memory=1, gain_L=1, gain_R=1
    independently fused duplicate chains: cost 8
    materialize once:                  cost 7
    regret: 1, model weight: 9

Both fusion gains and materialization overhead must be strictly positive,
so this is not a vacuous free-materialization/no-fusion case.
The corpus includes cases of rising weight and a second, unrelated
layout-boundary witness. The test independently evaluates explicit action
counts for pure and impure producers, checks every smaller-cost model for
a missed minimal counterexample, and verifies that there are cases where
duplication *does* win.

An effectful producer cannot be duplicated by this auxiliary lift; it is
rejected, rather than having its effects silently performed twice.

**Gate C** is no misclassified fork solution under the enumerated small
model, plus a deterministic machine-readable witness corpus.

## Artifacts and reproducibility

- Chain/layout reference: include/ctdp/solver/algorithms/fusion_chain_probe.h
- Branch and layout generators: include/ctdp/solver/algorithms/fusion_boundary_witness.h
- Exhaustive independent oracles: tests/solver/test_fusion_boundary_probe.cpp
- JSONL witness generator: experiments/fusion_boundary/report.cpp
- Offline CMake: experiments/fusion_boundary/CMakeLists.txt
- Cross-compiler workflow: .github/workflows/fusion-boundary-redteam.yml

Each line of the emitted JSONL corpus contains model version, explicit
assumptions, source parameters, lifted cost, globally best cost and regret.
CI compares the complete corpus bytes across GCC 13, Clang 18 and MSVC,
rather than treating successful compilation as sufficient.

This experiment is also registered as a standard CTest in the root CT-DP
test suite. A standalone CMake entry allows qualification without fetching
GoogleTest or building unrelated examples.

## What this deliberately does not prove

1. An additive integer segment cost table is not a reliable performance
   model of fused CPU/GPU kernels. It excludes register spills, occupancy,
   cache effects, code generation and overall search/calibration overhead.
2. A cost-model "materialize" winner need not be fastest on real hardware.
   The model supplies no instructions and no executable numerical lowering.
3. Equal cost under the model is not numerical equivalence. Real DN
   execution still needs full observation semantics, especially for
   floating-point associativity, NaNs, signed zero, exceptions and effects.
4. The minimality claim is scoped to the stated small graph grammar and
   bounded integer cost alphabet. It is not a theorem about every graph.
5. The generator finds obstructions to a *particular auxiliary lift*. It
   does not yet construct a general certificate and discover a novel lifting
   transformation automatically.

## Next gated work after this experiment

D: Connect a real pure pointwise chain and one fork implementation to the
DN numerical-evidence admission boundary; measure complete execution
cost, allocation, optimization overhead and break-even invocation count.

E: Extend the boundary-state model to live intermediates, peak resources,
layout and schedule under a sufficient DP state; test every proposed
dominance rule against exhaustive small-graph enumeration.

F: Ask a system with a typed intervention vocabulary to generate and
validate the missing shared-producer/materialize/duplicate alternatives.
A useful intervention must be discovered using shared machinery and
produce an independently checked lowering, not a graph-specific heuristic.

Promotion is conditional on Gates A, B and C passing on GCC/Clang/MSVC and
on an honest review of the counterexamples. Do not infer runtime speedups
or claim general algorithm synthesis from these artificial cost models.
