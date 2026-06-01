# Sparse Expression Decomposition DSL Demonstrator

## Purpose

This demonstrator turns the current SpMV examples into a small subproject that shows the central DSL idea:

> Search the decomposition, not the kernel.
>
> Local optimisation is allowed; expression identity is not negotiable.

The demonstrator is intentionally compact. It is not a full sparse matrix library. It is a proof that a sparse numerical expression can be represented as a contract-bearing artefact, searched as a dependent design space, lowered through plan/executor bindings, and checked against a canonical reference result.

## What the Demonstrator Shows

The demonstrator builds a 2D five-point stencil problem and evaluates a residual/preconditioner/dot/update expression:

```text
r    = b - A*x
z    = M^{-1}r
rho  = canonical_dot(r, z)
x'   = x + alpha*z
```

The search space includes storage, decomposition, ordering, colouring, preconditioner, threading, SIMD descriptor, reduction descriptor, and executor descriptor. Some choices are executable and measured. Some are descriptor-only but still useful because the planner can reason about legality and contracts. Some are deliberately illegal, so the tests and report can show rejection.

The point is not to claim that the generated executor is the fastest possible SpMV kernel. The point is to demonstrate that the design space is structured and contract-aware.

## Demonstration 1: Expression Contract Versus Realisation

The expression contract is the portable numerical meaning. It fixes the expression, the floating-point environment, and the observed values. The realisation is allowed to vary through storage, decomposition, scheduling, and executor choices.

The strict-expression candidates preserve:

```text
same pointwise residual expression
same preconditioner binding
same canonical observed dot-product tree
same update expression
same floating-point environment policy
```

The demonstrator compares legal realisations such as CSR execution and matrix-free stencil execution while checking both against the same canonical reference executor.

## Demonstration 2: Dependent Search Rather Than Flat Enum Search

The candidate generator uses facts about the problem to produce legal next moves. For example:

```text
stencil-like operator
    -> matrix-free stencil realisation becomes available
    -> red-black colouring becomes available
    -> recursive grid bisection becomes available

fixed diagonal preconditioner
    -> Jacobi apply is available in strict-expression mode

strict observed rho
    -> unordered thread-local merge is rejected
```

This is the intended shape of the DSL: facts constrain the next search choices. The planner does not blindly form a Cartesian product of every descriptor.

## Demonstration 3: Contract Gates and Negative Witnesses

The subproject includes illegal candidates on purpose. For example:

```text
thread-local unordered rho merge
```

is rejected under strict-expression mode because it does not preserve the observed canonical reduction tree.

Similarly, a coloured smoother without a colouring fact is rejected because its structural precondition is not satisfied.

This is important for the public story. The DSL is not just selecting fast kernels. It is proving that only contract-preserving plans are admitted to strict modes.

## Demonstration 4: Plans, Executors, and Reports

A plan is not an executor. A plan describes what is legal, what is preserved, and what can be lowered. An executor runs one realisation of a selected plan.

The demonstrator models this separation explicitly:

```text
expression contract
    -> plan descriptor
        -> legality analysis
            -> executor binding
                -> measured result and conformance check
```

The report prints each candidate with:

```text
plan name
contract level
storage descriptor
decomposition descriptor
colouring descriptor
preconditioner descriptor
reduction descriptor
executor descriptor
legality result
conformance result
median and p99 timing
rejection reason, where relevant
```

## Demonstration 5: A Small Solver-Shaped Example

Although the executable kernel is small, the expression is solver-shaped. It is closer to a Krylov or PDE update step than to a single raw SpMV call.

The demonstrator therefore prepares the ground for later examples:

```text
residual computation
preconditioner application
canonical dot product
update step
conformance check
measurement report
```

This makes it a better bridge toward PDE residual/update/norm pipelines, sparse chains, mixed-precision preconditioners, and heterogeneous CPU/GPU plans.

## Build and Run

The patch adds a self-contained CMake subproject:

```text
examples/framework/spmv_design_space
```

The demonstrator executable is:

```text
spmv_design_space_demo
```

The tests are plain C++ assertion tests registered with CTest:

```text
spmv_design_space_contract_tests
spmv_design_space_fact_tests
spmv_design_space_plan_tests
spmv_design_space_executor_tests
```

Typical build:

```bash
cmake -S . -B build -DCTDP_BUILD_TESTS=OFF -DCTDP_BUILD_EXAMPLES=ON
cmake --build build --target spmv_design_space_demo
./build/examples/framework/spmv_design_space/spmv_design_space_demo
```

To run the subproject tests:

```bash
cmake -S . -B build -DCTDP_BUILD_TESTS=OFF -DCTDP_BUILD_EXAMPLES=ON
cmake --build build --target \
  spmv_design_space_contract_tests \
  spmv_design_space_fact_tests \
  spmv_design_space_plan_tests \
  spmv_design_space_executor_tests
ctest --test-dir build -R spmv_design_space
```

The root project's existing `CTDP_BUILD_TESTS` option may still pull in external test dependencies. These demonstrator tests are deliberately attached to the examples subproject so they can be built without adding a new external test framework.

## What It Does Not Yet Claim

The first merged demonstrator does not claim:

```text
full CSR/SELL/DIA autotuning
full executor/sender integration
actual SIMD specialisation for every SIMD descriptor
actual thread-pool implementation for every threading descriptor
CUDA lowering
complete solver-family correctness proof
```

Those are next stages. The first demonstrator establishes the artefact model, the contract gate, the executable reference, the measurement path, and the report.

## Intended Follow-Up Extensions

The next useful extensions are:

```text
add SELL-C-sigma and DIA storage descriptors
add actual row-block and recursive executors
add multi-RHS vector layout descriptors
add mixed-precision preconditioner descriptors
add executor/sender lowering prototype
add CUDA launch-sequence lowering prototype
add plan serialisation and hashing
add a remote benchmark hook
```

## Summary

The demonstrator shows the intended architecture in miniature:

```text
The expression is the portable numerical contract.
The sparse facts expose legal decomposition choices.
The plan is a contract-preserving realisation.
The executor runs the selected realisation.
The report explains why the result is legal and what it measured.
```

That is the useful public story. It is no longer just an SpMV benchmark. It is a small sparse-expression decomposition DSL demonstrator.
