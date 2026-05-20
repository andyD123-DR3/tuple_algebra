# Phase B1 Concrete C++ API Shape — `interval_solver` Rooted Output

**Status:** Implemented slice reference note  
**Date:** May 20, 2026  
**Scope:** Concrete API shape for Stage 2 Phase B1 choice tracking  
**Related:** `docs/design/phase_b1_interval_choice_tracking_scope.md`, `docs/design/interval_stage2_execution_plan.md`, `include/ctdp/solver/algorithms/interval_solver.h`, `include/ctdp/solver/interval_rooted_candidate.h`, `include/ctdp/solver/algorithms/interval_dp.h`

## 1. Purpose

This note refines the Phase B1 scope into a concrete API target.

The corresponding implementation has now landed as an additive Stage 2 Phase B1 slice in `include/ctdp/solver/algorithms/interval_solver.h`; this note remains useful as the API-shape reference.

Its job is to answer a narrower question than the scope note:

> If we add direct rooted-output support to `interval_solver`, what exact public API shape should we prefer first?

## 2. Design constraints carried forward

The B1 API shape must preserve the following already-landed constraints.

### 2.1 Stage 1 compatibility is mandatory

The current public Stage 1 surface must remain valid:

- `interval_recurrence`
- `interval_solver::solve(...)`
- `interval_solver::solve_with_stats(...)`

No existing value-only recurrence should need to change.

### 2.2 Phase A public output is already defined

The new solve path should target the existing public rooted output vocabulary:

- `ctdp::solver::interval_rooted_candidate<MaxN>`
- `ctdp::solver::interval_rooted_plan<MaxN>`

B1 should not invent a parallel tree/result type.

### 2.3 B1 should remain narrower than B2/B3/B4

This API shape should not pull in:

- sparse memo decisions
- infeasible-branch protocols
- new split policies
- `select_and_run` routing changes

## 3. Recommended public API direction

The preferred first B1 shape is an **additive member API** on `interval_solver`.

Recommended direction:

```cpp
namespace ctdp::solver::algorithms {

template<class Recurrence,
         class SplitPolicy = ctdp::solver::policies::all_binary_splits,
         class Compare = std::less<>>
class interval_solver {
public:
    using value_type = typename Recurrence::value_type;

    template<std::size_t MaxN, class Memo>
    [[nodiscard]] auto solve_rooted(ctdp::solver::interval_context ctx,
                                    Memo& memo) const
        -> ctdp::solver::interval_rooted_plan<MaxN>;

    template<std::size_t MaxN, class Memo>
    [[nodiscard]] auto solve_rooted_with_stats(ctdp::solver::interval_context ctx,
                                               Memo& memo) const
        -> ctdp::solver::interval_rooted_plan<MaxN>;
};

} // namespace ctdp::solver::algorithms
```

## 4. Why a member API first

A member API is preferred over a free helper for the first B1 slice because it:

- keeps the rooted-output solve path visibly attached to the current solver object
- reuses the already-configured `Recurrence`, `SplitPolicy`, and `Compare`
- avoids a second layer of user-facing wrapper configuration
- makes Stage 1 and B1 read as sibling solve paths on the same abstraction

This is the simplest story for users already holding an `interval_solver` instance.

## 5. Why `interval_rooted_plan<MaxN>` should be the return type

The return type should be:

- `ctdp::solver::interval_rooted_plan<MaxN>`

not a new bespoke B1 result wrapper.

Reasons:

- `ctdp::plan<Candidate>` is already the framework vocabulary
- `predicted_cost` belongs naturally in the returned plan
- `solve_stats` already belongs naturally in the returned plan
- Phase A has already established `interval_rooted_plan<MaxN>` as the readable alias

This means B1 should preserve stats directly in the plan rather than nesting them in another result carrier.

## 6. Relationship to `solve_rooted` vs `solve_rooted_with_stats`

For B1, the pair should mirror the existing Stage 1 naming pattern.

### 6.1 `solve_rooted(...)`

Recommended meaning:

- produce rooted output
- return `interval_rooted_plan<MaxN>`
- populate `predicted_cost`
- populate `stats`

Even the non-`with_stats` rooted path should still return a full `plan`, because `plan` is the established result vocabulary.

### 6.2 `solve_rooted_with_stats(...)`

Recommended near-term meaning:

- same return type: `interval_rooted_plan<MaxN>`
- name retained only if the implementation internally mirrors Stage 1 flow and wants an explicit pair

However, unlike Stage 1, there is less benefit in having two materially different rooted return shapes.

That suggests the implementation should seriously consider one of these two options:

1. keep both names for symmetry, but make both return `interval_rooted_plan<MaxN>`
2. expose only `solve_rooted(...)` publicly and let it always fill plan stats

### Recommendation

For the first implementation ticket, prefer:

- public `solve_rooted<MaxN>(...) -> interval_rooted_plan<MaxN>`
- internal helpers may still thread `solve_stats&` explicitly

In other words, `solve_rooted_with_stats(...)` is optional in the concrete implementation.

### Implementation status

The current landed B1 slice now provides both rooted entry points:

- `solve_rooted<MaxN>(...)`
- `solve_rooted_with_stats<MaxN>(...)`

Both return `ctdp::solver::interval_rooted_plan<MaxN>`.

The current direct public demo path is:

- `examples/matrix_chain_demo.cpp`

That example now shows both:

- `interval_dp` + `reconstruct_interval_rooted_plan(...)`
- direct rooted solve via `interval_solver::solve_rooted_with_stats(...)`

## 7. Recommended internal helper types

B1 likely needs one internal choice-tracking structure.

### 7.1 Dedicated split-choice table

Recommended internal helper direction:

```cpp
namespace ctdp::solver::algorithms::detail {

template<std::size_t MaxN>
struct interval_choice_table {
    std::size_t n{};
    std::array<std::size_t, MaxN * MaxN> best_split{};

    [[nodiscard]] constexpr bool contains(std::size_t i, std::size_t j) const noexcept;
    constexpr void store(std::size_t i, std::size_t j, std::size_t k) noexcept;
    [[nodiscard]] constexpr std::size_t split(std::size_t i, std::size_t j) const noexcept;
};

} // namespace ctdp::solver::algorithms::detail
```

This note does **not** require this exact type name.

What matters is the shape of the responsibility:

- solver-owned
- fixed-capacity
- interval-indexed
- stores only winning split choice

### 7.2 Why not reuse `triangular_memo`

`triangular_memo<Value>` is for value memoization.

The B1 choice structure is different in role:

- it stores reconstruction choice, not solved objective value
- it only needs to record winning internal splits
- it does not participate in the same lookup/store semantics as value memoization

So B1 should keep value memo and choice tracking conceptually separate.

## 8. Recommended internal solve flow

A plausible internal flow for `solve_rooted<MaxN>(...)` is:

1. validate `ctx` and `MaxN` compatibility
2. allocate `solve_stats`
3. allocate value memo (supplied by caller) and a local choice table
4. run recursive solve as today
5. whenever a better candidate is selected for an internal interval, store its winning split in the choice table
6. reconstruct `interval_rooted_candidate<MaxN>` from the choice table
7. return `interval_rooted_plan<MaxN>{candidate, cost_as_double, stats}`

The key design point is that reconstruction should happen **after** solving, using Phase A machinery.

## 9. Reconstruction bridge direction

B1 will likely need a narrow bridge from the internal choice table to Phase A reconstruction.

Recommended direction:

- keep that bridge near `interval_rooted_candidate` reconstruction helpers, or
- keep it as an internal helper inside `interval_solver.h`

But either way, the semantic contract should stay:

- half-open intervals `[i, j)`
- absolute split `k`
- canonical rooted candidate as final public output

## 10. Capacity model

Because `interval_rooted_plan<MaxN>` is fixed-capacity, B1 should make the reconstruction capacity explicit at the API boundary.

That is why the preferred rooted solve shape is:

```cpp
template<std::size_t MaxN, class Memo>
auto solve_rooted(interval_context ctx, Memo& memo) const
    -> interval_rooted_plan<MaxN>;
```

This keeps B1 aligned with the framework’s existing fixed-capacity design style.

## 11. Relation to `interval_dp`

`interval_dp` remains the reference example of solver-owned split-table reconstruction.

B1 should imitate only the narrow part of that story that is relevant:

- keep best split for each internal interval
- reconstruct after the value solve

B1 should **not** imitate the parts that are specific to classic closed-interval DP storage or specialized `interval_split_candidate` output.

## 12. Tests the eventual implementation should unlock

A B1 implementation based on this API shape should be able to support tests like:

- rooted solve matches value-only solve on the CLRS matrix-chain case
- rooted solve preserves Stage 1 stats semantics
- rooted solve returns canonical `interval_rooted_candidate<MaxN>`
- rooted solve respects custom split policies
- rooted solve respects compare direction (`std::less<>` / `std::greater<>`)
- rooted solve coexists with the current `interval_dp` reconstruction adapter story

## 13. Summary

The preferred first B1 API shape is:

- additive member solve path on `interval_solver`
- explicit fixed-capacity return via `interval_rooted_plan<MaxN>`
- solver-owned internal split-choice table
- post-solve reconstruction through the landed Phase A rooted candidate machinery

That is concrete enough to guide implementation without prematurely solving B2/B3/B4 or redesigning the Stage 1 recurrence contract.



