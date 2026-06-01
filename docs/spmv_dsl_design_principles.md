# Sparse Expression Decomposition DSL Design Principles

## Working Theme

```text
Search the decomposition, not the kernel.
```

The DSL is not primarily a sparse-kernel selector. It is a framework for representing a sparse numerical expression, deriving facts about it, searching legal decomposition plans, and lowering a selected plan into an implementation while preserving a declared numerical contract.

## Core Principle

```text
Local optimisation is allowed.
Expression identity is not negotiable.
```

A plan may choose different storage, ordering, decomposition, vector layout, threading, backend, and materialisation strategies for different targets. It must not silently change the floating-point expression, the environment, the observed reduction tree, or the set of observable values.

## Artefact Separation

The design should separate four artefacts.

### Expression Artefact

The expression artefact says what computation is specified:

```text
r    = b - A*x
z    = M^{-1}r
rho  = canonical_dot(r,z)
x'   = x + alpha*z
```

It records nodes, edges, values, operator references, observable temporaries, numeric policies, reduction structures, and environment requirements.

### Environment Artefact

The environment artefact says how the expression is interpreted:

```text
rounding mode
FMA contraction policy
FTZ/DAZ policy
exception policy
primitive function contract
conversion rules
signed-zero and NaN policy
reproducibility scope
```

For reproducible numerics, the environment is part of the contract, not an implementation detail.

### Plan Artefact

The plan artefact says how the expression is realised on a target:

```text
matrix decomposition
graph view
ordering
colouring
blocking
storage transform
vector layout
fusion and materialisation
threading schedule
ISA mapping
precision mapping
reduction implementation
lowering target
```

The plan is printable, serialisable, hashable, comparable, cacheable, and lowerable.

### Realisation Artefact

The realisation artefact is the selected implementation:

```text
specialised C++ template instantiation
generated C++
CUDA kernel
backend library call
cached binary
runtime-dispatched executor
```

The realisation is not the plan. It is one lowering of a selected plan.

## Reproducibility Rule

For strict reproducible modes, the contract is:

```text
same expression
+ same floating-point environment
+ same observation contract
= reproducible numerics
```

The DSL should distinguish mathematical equivalence from floating-point expression identity. These are different expressions:

```text
((a + b) + c) + d
(a + b) + (c + d)
canonical_pairwise_reduce(a,b,c,d)
block_dyadic_reduce(a,b,c,d)
```

A plan may only change the reduction tree if the contract permits it.

## Contracts Flow Down, Capabilities Flow Up

The plan object should be hierarchical and compositional.

```text
contracts flow downward
capabilities flow upward
facts flow forward
costs accumulate upward
lowering composes downward
```

A parent contract imposes requirements on child components. Each child reports its capabilities. The parent accepts or rejects the composed plan.

For example:

```text
strict-expression contract requires preserved observed rho
    -> canonical pairwise reduction is admissible
    -> unordered thread-local merge is rejected
```

This is better than a central `is_legal(plan)` switch because new components can be added independently.

## Dependent Search

The search space should be grammar-like and fact-dependent, not a flat Cartesian product.

Example:

```text
if the operator is stencil-like
    matrix-free stencil realisation is available
    red-black colouring is available
    recursive grid bisection is available

if row-length variance is high
    row-length buckets may be useful
    SELL-C-sigma may be available

if a reduction is externally observed
    the named reduction tree must be preserved

if mixed precision is permitted for M^{-1}
    fp32 preconditioner candidates may be generated
```

The search protocol is:

```text
start with expression contract and target model
compute sparse and graph facts
generate legal next operations
propagate facts
reject failed preconditions early
accumulate capabilities
reject plans that fail the contract
estimate cost
measure selected legal plans
report the selected plan and rejected alternatives
```

## Contract Levels

The DSL should support several contract levels.

```cpp
enum class contract_level {
    strict_expression,
    expression_family,
    solver_family,
    backend_defined,
    unchecked
};
```

`strict_expression` fixes the expression, environment, operator bindings, observed values, and reduction observations.

`expression_family` allows approved variants within a named family.

`solver_family` allows solver-level choices such as preconditioner family, smoother, or refinement policy, subject to convergence and validation rules.

`backend_defined` delegates numerical details to a backend library and reports that fact.

`unchecked` is for raw experiments and should be visibly marked as unaudited.

## Preconditioners Are Contract-Bearing

A preconditioner is not just a performance option. Under strict-expression mode, the binding of `M` is part of the expression. The plan may change how `M` is realised, but it may not silently replace `M` with another preconditioner.

Therefore:

```text
strict expression:
    M is fixed
    allowed choices are realisations of the same M

solver family:
    M may vary within an approved preconditioner family
    the report must say that the contract is solver-family, not strict-expression
```

This distinction is essential for credibility.

## Observation Contracts Control Fusion and Materialisation

The observation contract says which values are externally visible.

Examples:

```text
observe only rho
observe x' and rho
observe r, x', and rho
observe every scan prefix
observe full debug trace
```

If `r` is unobserved and has one consumer, it may be streamed. If `r` is observed or reused, it must be materialised or reproduced under the observation contract.

## Executors Are Lowering Targets

C++ executors, senders, thread pools, CUDA streams, CUDA graphs, and backend libraries are execution mechanisms. They are not the plan object.

The relationship is:

```text
DSL plan:
    selects, validates, explains, and audits the numerical realisation

Execution system:
    schedules and runs the selected realisation on an execution resource
```

A sender graph can be an excellent lowering for a selected plan, but it does not by itself express the numerical legality of the plan.

## C++26 Code Generation and Reflection

C++26 static reflection is relevant to this design because the plan can become a compile-time object that drives generation of specialised runtime code. The reflection proposal P2996 introduces `std::meta::info` and static reflection facilities. Expansion statements from P1306 provide compile-time iteration. P3491 adds `define_static_{string,object,array}` facilities for promoting compile-time values to static storage. These features make it more realistic to generate plan-specific dispatch tables, reports, wrappers, and lowered executor skeletons in standard C++26.

The immediate recommendation is conservative:

```text
Use C++20 plan tags and templates for the current demonstrator.
Design the plan IR so that it can later be reflected.
Use C++26 reflection/generation as an optional future backend, not as a hard dependency.
```

The plan object should therefore be kept simple, structural, and named. That makes it a good candidate for future reflection-driven report generation, executor binding, serialisation, and compile-time plan registry construction.

## Reporting Is Part of the Product

The report should not be an afterthought. It is the visible proof that the DSL is doing structured search rather than random benchmarking.

A useful report includes:

```text
expression
contract level
environment policy
sparse facts
graph facts
candidate plans
legality results
rejection reasons
conformance results
measurement results
selected plan
```

If a performance engineer cannot understand why the selected plan is legal and why it won, the abstraction has failed.

## Summary

The design converts sparse optimisation from kernel selection into expression realisation search.

```text
The compiler sees one expression.
The performance engineer sees a space of possible decompositions.
The DSL makes that space explicit, searchable, measurable, and lowerable.
```
