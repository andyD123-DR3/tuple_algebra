# CT-DP Framework Architecture

**Version 0.7.2 — February 2026**

## 1. What This Framework Does

CT-DP finds optimal configurations at compile time and instantiates them as zero-cost specialised code. The input is a search space of configuration choices. The output is a fully-specialised template instantiation — a concrete type with no runtime branching on the configuration.

The canonical pipeline:

```
define space → filter by validity → search for optimum → dispatch to executor
```

A concrete example: given a 3D loop nest, the framework searches over loop orderings and SIMD strategies, finds that IKJ with innermost SIMD on AVX2 with FMA is optimal for a 512×512 matmul, and produces `matmul_executor<IKJ, INNERMOST_SIMD, AVX2, true, true>` — one concrete type, fully specialised, 47× faster than naive.

The framework's value is not in the search algorithm (a nested `for` loop suffices for small spaces). The value is in the contracts between components: space defines what exists, validity defines what's legal, cost ranks what's legal, bridge encodes for learning, dispatch maps the winner to code. These boundaries are named, typed, and enforced — so when you change the space you can't silently break the executor, and when you swap the cost model the search still works.


## 2. The Pipeline in Detail

### 2.1 Space — What to Search

A space is a set of points. Each point is a candidate configuration. The space knows how many candidates exist, how to enumerate them, and what type they are.

The simplest space is a Cartesian product of enumerated dimensions:

```cpp
descriptor_space loop_opt("loop_opt",
    make_enum_vals("order", loop_orders),     // 6 values
    make_enum_vals("simd",  simd_strategies), // 4 values
    make_enum_vals("isa",   isa_levels),      // 2 values
    bool_flag("fma"),                         // 2 values
    bool_flag("aligned")                      // 2 values
);
// 6 × 4 × 2 × 2 × 2 = 192 points
// point_type = std::tuple<loop_order, simd_strategy, isa_level, bool, bool>
```

Each dimension is a **descriptor** — a named, typed, enumerable set of values. Five descriptor kinds exist: `positive_int` (1..N), `power_2` (1,2,4,...,2^k), `int_set` (arbitrary integers), `bool_flag` (true/false), `enum_vals` (user-defined enum). New descriptor kinds can be added by satisfying the descriptor concept.

The point type is `std::tuple<V0, V1, ..., Vn>` where each `Vi` is the descriptor's value type. For `enum_vals` that's the user's enum. For `bool_flag` it's `bool`. For `int_set` it's `int`. The tuple is a structural type in C++20 — valid as a non-type template parameter — which is what makes the dispatch step possible.


### 2.2 Validity — What's Legal

Not every point in the Cartesian product is a legal configuration. INNERMOST_SIMD requires the innermost loop dimension to be vectorisable. AVX512 requires the hardware to support it. FMA with SSE requires checking instruction availability.

Validity is expressed as a predicate over complete points:

```cpp
auto valid_space = filter_valid(base_space, [&](const auto& pt) {
    auto [ord, sim, isa, fma] = pt;
    int innermost = order_to_innermost(ord);
    if (sim == INNERMOST_SIMD && !inst.vectorisable[innermost]) return false;
    if (fma && isa == isa_level::SSE) return false;
    return true;
});
```

`filter_valid` returns a `valid_view` — a wrapper that enumerates only points passing the predicate. The cost function never sees invalid points. This separation matters: cost ranks what exists; validity defines what exists. Mixing them (returning `+infinity` for invalid points) made every cost function defensive and hid the feasible count.

Constraints are evaluated on complete candidates only. A constraint like "INNERMOST_SIMD requires vectorisable innermost" cannot be checked until both the `order` and `simd` dimensions have values. Per-dimension pruning (constraint propagation) is a future capability for large spaces; for current problems with ≤300 points, post-hoc filtering is sufficient and simpler.


### 2.3 Search — Finding the Optimum

The solver takes a (possibly filtered) space and a cost function, and returns the point with minimum cost:

```cpp
constexpr auto [best, cost] = exhaustive_search_with_cost(valid_space,
    [&](const auto& pt) { return evaluate_cost(pt, instance); });
```

For spaces under ~10,000 feasible points, exhaustive search is correct. The framework also supports beam search for larger spaces (using the factored structure of `descriptor_space` to extend partial assignments dimension by dimension) and custom solvers for problems with exploitable structure (subset DP for cache-line packing, branch and bound for tiling).

The solver is generic over the space concept. It doesn't know whether the space is a Cartesian product, a filtered view, or a hand-written enumeration. It calls `enumerate(fn)` and tracks the minimum. This genericity is the point — the solver doesn't change when the space changes.


### 2.4 Bridge — Encoding for Learning

The bridge converts a space point into a numeric feature vector suitable for a cost model:

```cpp
// 5 dimensions → 15 features:
// order (6 values)    → one-hot [6 floats]
// simd  (4 values)    → one-hot [4 floats]
// isa   (2 values)    → one-hot [2 floats]
// fma   (bool)        → binary  [1 float]
// aligned (bool)      → binary  [1 float]

auto bridge = default_bridge(space);
double features[15];
bridge.write_features(point, std::span{features});
```

The encoding is determined by the descriptor type: enums and int_sets get one-hot encoding, bools get binary, power_2 gets log₂, positive_int gets normalised. `write_features` writes into a caller-provided span — no allocation.

The bridge earns its keep when the cost function is learned rather than hand-written. Given a set of measured timings for sampled configurations, the features become the input to a regression model (linear, SVR, MLP). The model predicts cost for unmeasured configurations. The bridge doesn't change between hand-written and learned cost — it encodes the same way regardless. Space defines what to search, bridge defines how to encode, model is swappable.


### 2.5 Dispatch — Zero-Cost Instantiation

The search produces a constexpr result. The dispatch step turns that result into template arguments:

```cpp
// The solve result is constexpr
static constexpr auto ct_result = solve(make_problem(matmul_3d), make_cost(matmul_3d));

// Convert tuple to a config struct (structural type)
static constexpr auto cfg = config_point{
    std::get<0>(ct_result.best),   // loop_order
    std::get<1>(ct_result.best),   // simd_strategy
    std::get<2>(ct_result.best),   // isa_level
    std::get<3>(ct_result.best),   // use_fma
    std::get<4>(ct_result.best)    // aligned
};

// The config struct IS the template argument
using optimal = matmul_executor<cfg>;
optimal::execute(A, B, C, N);
```

This works because C++20 allows structural types (all-public members, all scalar or structural) as non-type template parameters (NTTPs). A struct with enum, int, and bool fields qualifies. The framework provides a dispatch alias:

```cpp
template<template<auto> class Executor, auto Point>
using dispatch = Executor<Point>;
```

The compiler sees one concrete type. No virtual dispatch, no function pointer, no switch statement, no runtime branch on the configuration choice. The loop body shape, SIMD width, instruction selection — all resolved at compile time. The optimizer works on a fully-specialised function.

This is the critical property that distinguishes CT-DP from runtime configuration systems. A runtime system selects a code path via a dispatch table or switch. CT-DP eliminates the selection entirely — there is only one path, chosen at compile time. The cost is that a different configuration requires recompilation. The benefit is that the compiler optimises through the configuration boundary.


### 2.6 Executor — The Generated Code

The executor is a class template parameterised on a configuration:

```cpp
template<config_point Cfg>
struct matmul_executor {
    static void execute(const float* A, const float* B, float* C, size_t N) {
        // Cfg.order determines loop nesting
        // Cfg.simd determines vectorisation strategy
        // Cfg.isa determines instruction width
        // All resolved at compile time via if constexpr
    }
};
```

The executor is user-written and problem-specific. The framework does not generate it. The framework's contract with the executor is: every value that the solver can select must have a corresponding `if constexpr` branch (or the executor must `static_assert(false)` on unsupported values). This is checklist item 4: "search matches executor."

The executor is also the correctness boundary. Each specialisation is tested against a known-correct reference implementation with non-uniform inputs. This catches indexing errors that uniform inputs (all-ones) would miss, and verifies that the compile-time configuration produces a functionally correct code path, not just a fast one.


## 3. The Space Architecture

### 3.1 Dimensions Are Decisions

A dimension of the search space is a decision the solver makes. It is not the internal structure of a decision.

Example: loop ordering. A 3D loop nest has 6 possible orderings (IJK, IKJ, JIK, JKI, KIJ, KJI). This is one dimension with 6 values — not three dimensions (slot₀, slot₁, slot₂) each with 3 values and a no-repeat constraint. The latter decomposition is an encoding choice that inflates the space from 6 to 27 points and introduces a constraint that couples all three dimensions. The former keeps the space minimal and the constraint implicit in the value set.

The principle: dimensions should correspond to independent decisions. When choices are structurally coupled (like positions in a permutation), treat the coupled group as a single dimension with a structured value type. When choices are genuinely independent (loop order vs SIMD strategy), they are separate dimensions whose interaction appears only in the cost function and validity predicate.


### 3.2 Instances Are Observations

The space defines what the solver can choose. The **instance** defines what it's choosing for.

For a loop nest optimiser, the space is {order × simd × isa × fma × aligned} — the same regardless of whether you're optimising a 512×512 matmul or a 7-point stencil. What changes between problems is the instance: loop sizes, strides, vectorisability per dimension, cache sizes, alignment guarantees. The instance shapes the cost landscape, not the search geometry.

```cpp
struct loop_instance {
    std::array<size_t, N> size;       // iteration count per dim
    std::array<int, N>    stride;     // memory stride per dim
    std::array<bool, N>   vectorisable;
    std::array<size_t, N> alignment;
    size_t L1_size;
    size_t simd_width;
};
```

The cost function is where decision meets observation: `(point, instance) → score`. The validity predicate similarly: `(point, instance) → bool`. The instance is captured in a closure at the point where the space is constructed:

```cpp
constexpr auto make_problem(const loop_instance& inst) {
    auto base = make_loop_opt_space();
    return filter_valid(base, [&inst](const auto& pt) {
        return is_valid_config(pt, inst);
    });
}
```

`make_problem` is the single construction point. Both compile-time and runtime paths call it. If the compile-time path uses a different space construction than the runtime path, the divergence is a bug — and the architecture makes it structurally impossible by having one function that both paths share.

Instance data falls into layers:

- **Target context** (fixed per machine): cache sizes, pipeline depth, issue width, SIMD capabilities. Measured once, shared across all problems on the same hardware.
- **Problem context** (fixed per problem instance): strides, associativity, data types, access patterns. Arrives fresh with each optimisation request.
- **Derived context** (computed): vectorisability = f(stride, simd_width). Derived from the other two.

Currently the instance is a plain struct — no framework class, no projection algebra, no reactive dependency tracking. That's deliberate. A struct with known field names is a coordination point that multiple problems can read from. The value of naming fields consistently (always `L1_size`, never sometimes `l1_cache_bytes`) is that cost models trained on one problem can transfer features to another. Shared vocabulary makes feature spaces commensurable. This becomes important when the ML model generalises across problems, but it requires only a naming convention, not a framework abstraction.


### 3.3 Access Tiers

Not all spaces support the same operations. A Cartesian product has closed-form cardinality and random access. A filtered space can only enumerate forward. The framework makes this explicit with four concept tiers:

```
search_space      enumerate(fn)                                  [tier 1]
countable_space   + cardinality() → size_t                       [tier 2]
indexable_space   + point_at(i) → point_type                     [tier 3]
factored_space    + num_dims(), dim_cardinality(d), dim_value(j)  [orthogonal]
```

`descriptor_space` satisfies all four. It has closed-form cardinality (product of per-dimension cardinalities), closed-form point_at (div/mod decomposition of a flat index), and per-dimension enumeration (each descriptor is independently enumerable).

`valid_view` satisfies only `search_space`. Filtering destroys cardinality (you can't count feasible points without enumerating), indexing (you can't compute the 47th feasible point without knowing which points are feasible), and factored structure (the filter couples dimensions — valid SIMD choices depend on which order was chosen).

The tier concepts enforce honesty. A wrapper that can't truthfully implement `cardinality()` doesn't expose it. A solver that needs random access declares `indexable_space` in its concept constraint. If you pass a `valid_view` to such a solver, the compiler rejects it at the call site with a concept failure pointing at exactly the missing capability.

Wrappers compose with honest tier propagation:

```
filter(tier3_space, pred)     → tier 1  (filtering destroys indexing)
materialise(tier1_space)      → tier 3  (enumerates once, caches all points)
product(tier3_a, tier3_b)     → tier 3  (both indexable → product indexable)
product(tier3_a, tier1_b)     → tier 1  (lowest tier wins)
section(tier3_space, dim, v)  → tier 3  (inherits from base)
```

`materialise` is not yet implemented — it will be added when calibration (which needs random sampling from filtered spaces) demands it. The concept exists now so the type system documents the capability, even though no wrapper provides it yet.


### 3.4 Factored Structure vs Indexing

These are independent properties, and confusing them caused design errors early on.

**Indexable** means you can say "give me point number 47." Random access on the flat product. Useful for uniform random sampling (calibration).

**Factored** means you can say "for dimension 2, what are the choices?" Per-dimension enumeration. Useful for beam search, which builds solutions incrementally — extending partial assignments one dimension at a time, pruning after each extension.

`descriptor_space` has both. `valid_view` has neither — the filter couples dimensions (can't enumerate dimension 2's choices without knowing dimension 0's value) and destroys indexing.

Beam search on a constrained space works by extending candidates on the unfiltered factored space and evaluating constraints on complete candidates. For small per-dimension cardinalities (6 loop orders, 4 SIMD strategies), trying all choices and discarding invalids post-hoc is cheap. For large per-dimension ranges (tile sizes 1–1024), per-dimension constraint propagation would narrow the range before extension — but that's constraint programming, deferred until a problem demands it.


### 3.5 Spaces the Framework Doesn't Own

Some problems have spaces that aren't Cartesian products. Cache-line layout is a partition of N fields into bins. The state space is the power set (2^N), not a product of N independent dimensions. The subset DP exploits this structure — it evaluates 3^N states instead of K^N for K bins, because it recognises that line labels are symmetric (a quotient) and that packing a subset is independent of how remaining fields are handled (optimal substructure).

The framework does not replace the DP. It wraps it. The wrapping pattern is: the framework searches over meta-decisions (which cost model weights to use), and the DP handles the structured subproblem (which fields go on which line, given those weights):

```cpp
auto policy_space = descriptor_space("policy",
    make_int_set("hot_line_w",  {1000, 5000, 10000, 50000}),
    make_int_set("mix_penalty", {10000, 100000, 1000000}),
    make_int_set("waste_w",     {1, 5, 10}),
    make_int_set("conflict_w",  {100, 500, 2000})
);
// 4 × 3 × 3 × 3 = 108 policies

auto cost = [&](const auto& pt) -> double {
    auto policy = to_policy(pt);
    auto plan = subset_dp_solve(fields, policy);  // DP runs per policy
    return plan.total_cost;
};
```

This is the general pattern for structured subproblems: the framework owns the outer search over a Cartesian space of configuration parameters; a problem-specific solver owns the inner search that exploits structure the framework can't see. The framework's value is in the outer loop — calibrating the inner solver's parameters — not in replacing it.

Hand-written spaces satisfy the same `search_space` concept as `descriptor_space`. They provide `enumerate(fn)` and `point_type`. The solver doesn't know or care about the internal representation. This is the adapter pattern: any space that can enumerate its points works with any solver that accepts `search_space`.


## 4. The Zero-Cost Instantiation Path

This is the central technical contribution. Every design decision in the framework serves this path.

### 4.1 The Problem

A runtime configuration system works like this:

```cpp
double* dispatch_table[] = { &kernel_v1, &kernel_v2, &kernel_v3 };
dispatch_table[config]();   // indirect call, opaque to optimizer
```

The compiler can't inline through a function pointer. It can't constant-fold the configuration. It can't eliminate dead branches inside the kernel. The dispatch boundary is a wall.

### 4.2 The Solution

CT-DP eliminates the dispatch boundary:

```cpp
// Compile-time search
static constexpr auto result = solve(space, cost);

// Result becomes a type, not a runtime value
using optimal = matmul_executor<result.best_config>;

// Direct call to a fully-specialised function
optimal::execute(A, B, C, N);
```

The compiler sees exactly one specialisation of `matmul_executor`. All `if constexpr` branches resolve. Dead code is eliminated. The optimizer works on a concrete function with no configuration uncertainty.

### 4.3 What Makes It Work

Three C++20 features combine:

1. **constexpr evaluation** — the search runs at compile time. `static constexpr auto result = solve(...)` forces the compiler to evaluate the entire search, including constraint checking, cost evaluation, and min-tracking, during compilation.

2. **Structural types as NTTPs** — a struct with all-public members of scalar type is a "structural type" that can be used as a non-type template parameter. The search result struct (containing enums, ints, bools) qualifies. So `matmul_executor<result.best_config>` is legal — the struct value IS the template argument.

3. **`if constexpr`** — inside the executor, branches on the configuration are resolved at compile time. `if constexpr (Cfg.isa == isa_level::AVX2)` is not a runtime check — the compiler evaluates it during template instantiation and discards the false branch entirely.

The pipeline is: `constexpr search` → `struct result` → `NTTP on executor template` → `if constexpr resolution` → `one concrete function`. Each step is standard C++20. The framework's contribution is making the pipeline systematic and reusable across problems.

### 4.4 What It Requires

The executor template must handle every configuration the solver can produce. If the space contains `isa_level::AVX512` but the executor only implements AVX2, the solver might select AVX512, the executor silently falls through to scalar code, and the "optimised" path is slower than expected. This happened twice during development — once with AVX512, once with the `aligned` flag.

The architectural rule: every dimension value must have a corresponding `if constexpr` branch in the executor, or the executor must `static_assert(false)` on unhandled values. This is checklist item 4 and it prevents the most common class of bugs in the system.


## 5. Cost Models: From Intuition to Measurement

### 5.1 Hand-Written Cost Functions

The simplest cost model is a weighted sum of indicator features:

```cpp
auto cost = [&](const auto& pt, const auto& inst) -> double {
    double c = 0.0;
    if (inst.stride[innermost(pt)] != 1)     c += 5000;   // cache hostility
    if (can_vectorise(pt, inst))              c -= 4000;   // SIMD opportunity
    if (inst.size[innermost(pt)] <= L1_SIZE)  c -= 2000;   // L1 fit
    return c;
};
```

This is a linear model with human-assigned coefficients. For small spaces with clear dominance relationships (unit-stride innermost is always better than non-unit-stride), it's sufficient. The optimal point is obvious to anyone who understands the hardware.

### 5.2 Cost-as-Simulation

A more principled approach: simulate the hardware behaviour for each configuration. The cache-size example does this — it runs an LRU simulation at compile time:

```cpp
constexpr auto cost = [&](size_t cache_size, const auto& pattern) {
    auto sim = lru_cache<cache_size>{};
    int misses = 0;
    for (auto access : pattern.sequence)
        if (!sim.lookup(access)) ++misses;
    return misses;
};
```

The simulation shares its LRU engine with the runtime cache — single source of truth. But simulation captures single-pass behaviour. Real workloads have warm-start effects across invocations, branch predictor training, TLB state. The cache-size example demonstrated this gap: the simulation says size 3 ties with size 6 for a phase-shifting access pattern, but at runtime size 6 is 48× faster because it stays warm across phase boundaries.

### 5.3 Learned Cost Models

The bridge + calibration pipeline replaces human intuition and incomplete simulation with measurement:

1. **Sample** — instantiate and measure a subset of feasible configurations
2. **Encode** — the bridge converts each measured point into a feature vector
3. **Fit** — train a model (linear, SVR, MLP) on (features, timing) pairs
4. **Predict** — use the model to score unmeasured configurations
5. **Search** — find the configuration with minimum predicted cost

The space, bridge, and search machinery don't change between paths 5.1, 5.2, and 5.3. The cost function is swappable. That's the three-layer separation earning its keep: space defines what to search, bridge defines how to encode, model is swappable.

For the loop nest demo with 138 feasible configurations, exhaustive measurement takes about 2 seconds. You don't need to sample — measure everything, fit, done. For Level 1 tiling spaces with thousands of feasible points, you sample 50–100, fit, predict the rest. The bridge encoding matters here — a good encoding lets the model generalise from 50 measurements to 5000 predictions because it captures the physics (stride, cache fit, vectorisability) rather than arbitrary point indices.


## 6. The Descriptor System

### 6.1 Five Descriptor Kinds

Each descriptor is a named, typed, enumerable set of values with a default encoding:

| Descriptor | Value type | Example | Encoding |
|------------|-----------|---------|----------|
| `positive_int(name, max)` | `size_t` | Tile size 1..64 | normalised |
| `power_2(name, max)` | `size_t` | Block size 1,2,4,...,64 | log₂ |
| `int_set(name, vals)` | `int` | Unroll factor {1,2,4,8} | ordinal/one-hot |
| `bool_flag(name)` | `bool` | Use FMA? | binary |
| `enum_vals(name, vals)` | user enum | SIMD strategy | one-hot |

### 6.2 descriptor_space Composition

`descriptor_space` is constructed from descriptors:

```cpp
descriptor_space s("name", desc0, desc1, ..., descN);
```

The point type is `std::tuple<V0, V1, ..., VN>`. Cardinality is the product of per-descriptor cardinalities. Enumeration visits the Cartesian product in row-major order.

Composition operations:

- **section(space, dim, val)** — fix one dimension, reduce rank by 1
- **filter_section(space, dim, pred)** — fix one dimension to values matching a predicate
- **product(space_a, space_b)** — Cartesian product of two spaces
- **descriptor_product(space_a, space_b, name)** — named product with qualified dimension names

### 6.3 Extending with New Descriptors

A new descriptor kind must satisfy the descriptor concept:

```cpp
template<typename D>
concept descriptor = requires(const D& d) {
    { d.name() } -> std::convertible_to<std::string_view>;
    typename D::value_type;
    { D::cardinality() } -> std::same_as<std::size_t>;
    { D::value_at(size_t{}) } -> std::same_as<typename D::value_type>;
    { D::contains(std::declval<typename D::value_type>()) } -> std::same_as<bool>;
};
```

This is the extensibility path for permutation descriptors, interval descriptors, or domain-specific value types. The descriptor concept is small — four operations — and `descriptor_space` is generic over any set of types satisfying it.


## 7. Wrappers and Composition

### 7.1 The Wrapper Model

A wrapper holds a reference to an inner space and modifies its behaviour:

```
valid_view<Space, Pred>     — filters enumeration by predicate
materialise<Space>          — caches points for random access    [planned]
section<Space>              — fixes one dimension
product<SpaceA, SpaceB>     — Cartesian product
```

Wrappers compose by nesting:

```
section(
  materialise(
    product(
      filter_valid(descriptor_space, pred),
      descriptor_space
    )
  ),
  dim, val
)
```

The user never spells these types. They live behind `auto`. Solvers accept concept-constrained references: `search_space auto&& s` or `factored_space auto&& s`. The wrapper chain is an implementation detail.

### 7.2 Tier Propagation Rules

Each wrapper honestly declares what capabilities it provides:

| Wrapper | search | countable | indexable | factored |
|---------|--------|-----------|-----------|----------|
| `descriptor_space` | ✓ | ✓ | ✓ | ✓ |
| `valid_view` | ✓ | ✗ | ✗ | ✗ |
| `materialise` (planned) | ✓ | ✓ | ✓ | ✗ |
| `section` | inherits | inherits | inherits | inherits−1 dim |
| `product` | ✓ | min(a,b) | min(a,b) | if both factored |

The rule: a wrapper only exposes capabilities it can truthfully implement. A `valid_view` with a `cardinality()` that returns the unfiltered count is a lie. Lying to satisfy a concept is worse than failing to match it.


## 8. Connection to C++ Standards and Future Execution Models

### 8.1 P2300 Senders/Receivers

In P2300 terms, the CT-DP result determines the **shape** of a sender chain — which bulk operations nest inside which, whether there's a split for loop fission, what the reduction combiner looks like. The execution context (thread pool, inline, SIMD unit) is another searchable dimension:

```cpp
enum class exec_context : int { INLINE, STATIC_THREAD_POOL, SIMD_BULK };
```

The mapping is:

| CT-DP dimension | P2300 concept |
|-----------------|---------------|
| loop_order | Sender chain nesting (which `bulk` wraps which) |
| simd_strategy | Execution policy on innermost `bulk` |
| isa_level | Scheduler property / hardware capability |
| fission/pipelining | Sender adaptor composition (`split`, `when_all`) |
| num_accumulators | Reduction tree width in `reduce` sender |
| exec_context | `scheduler` — a new searchable dimension |

The instantiation path currently produces a template type. With P2300, it produces a sender expression — a composed sender whose structure is determined by the CT-DP result. The space, bridge, and search are unchanged. The dispatch table maps `plan<C>` to a sender factory instead of a function pointer.

### 8.2 P4016R0 Canonical Parallel Reduction

The reduction tree shape — pairwise, compensated, blocked — is a dimension of the space. With P4016 + P2300:

```cpp
auto reduction = ctdp_reduce(
    bulk_schedule(simd_scheduler, N),
    data,
    std::plus<>{},
    reduction_tree<ct_result.tree_shape, ct_result.block_size>{}
);
```

The tree shape is the CT-DP output. The scheduler is the P2300 executor. The determinism guarantee is P4016. Three standards proposals, one pipeline.


## 9. Worked Examples — What Each Teaches

Four examples validate the architecture. Each exercises different aspects:

| # | Example | Dimensions | Space | Key lesson |
|---|---------|-----------|-------|------------|
| 1 | Loop nest | 5 (order, simd, isa, fma, aligned) | 192→92 | Cross-dimension constraints |
| 2 | Dependency breaking | 3 (strategy, unroll, accum) | 105 | Same space, different instance, different winner |
| 3 | Cache size | 1 (cache_size 1–8) | 8 | Cost model ≠ reality |
| 4 | Cache-line layout | 4 policy dims wrapping DP | 270 | Framework wraps solvers it can't replace |

### 9.1 Loop Nest Optimisation (Standalone)

The most complete exercise of the pipeline. Five enumerated dimensions. Cross-dimension constraint: INNERMOST_SIMD requires the innermost loop dimension (determined by `order`) to be vectorisable (a property of the instance). The constraint couples `order` and `simd` — you can't evaluate SIMD validity without knowing which dimension landed innermost.

Bug found during development: the original demo's hand-rolled code computed order and SIMD strategy independently. It selected IKJ (j innermost) but reported OUTER_SIMD because the strategy evaluator checked the original dimension 2 (k, non-vectorisable) rather than the reordered innermost (j, vectorisable). The execution worked by accident — the executor checked the actual innermost dimension correctly. The CTDP formulation prevents this: the constraint sees the complete point (order + strategy together) and rejects incoherent combinations.

Safety enforcement: `static_assert(d2 == 1)` in the SIMD branch — the constraint already prevents INNERMOST_SIMD when j isn't innermost, but the executor enforces it explicitly. If someone changes the constraint or instance data, the compiler catches it at instantiation time.

### 9.2 Dependency Breaking (Standalone)

Same architecture, three instances with different optimal strategies. Accumulation (associative, commutative) → REDUCTION_TREE with 8 accumulators. Recurrence with distance 3 → SOFTWARE_PIPELINING with unroll=3. Prefix sum (non-associative) → SCALAR_EXPANSION.

Teaches conditional dimension relevance: `unroll_factor` is meaningless for REDUCTION_TREE; `num_accumulators` is meaningless for LOOP_UNROLLING. The space contains all combinations (7×5×3 = 105). The cost function ignores irrelevant dimensions. This is the "don't-care" pattern — valid, just not influential.

### 9.3 Cache Size Optimisation (Standalone)

One-dimensional space. The cost function is a compile-time LRU simulation sharing its engine with the runtime cache (single source of truth). Three instances show different optimal sizes.

The teaching point: the analytical cost model gets the wrong answer for the `phase_shift` instance under realistic multi-iteration usage. Single-pass simulation says size 3 suffices. Runtime measurement shows size 6 is 48× faster because it stays warm across phase boundaries. This is the motivation for Layer 3 — learned models trained on measurements, not hand-written simulations.

### 9.4 Cache-Line Layout (Framework)

The architecturally significant example. A subset DP over field-to-cache-line assignment is too structured for `descriptor_space` — the state space is a power set (2^N), not a Cartesian product. The framework can't replace the DP. But it can calibrate it.

The framework searches over 270 cost-model policy configurations (weights for hot-line cost, mixing penalty, waste, false sharing). For each policy, the DP finds the optimal layout. The framework discovers which policy produces the best layout by comparing DP outputs. The DP is a subroutine inside the cost function.

Demonstrates: the framework's value when the inner solver is better than anything generic. The DP exploits optimal substructure and label symmetry (a quotient). Exhaustive search over the equivalent Cartesian product would be intractable for N>10. The framework doesn't need to understand the subproblem — it just evaluates the DP's output and searches over the DP's parameters.


## 10. Architectural Principles

Nine principles emerged from building and stress-testing the examples. Each traces to a bug that occurred at least once.

**1. Constraints are structure, not cost.** Validity belongs to the space (`valid_view`), not the cost function (`+infinity` hack). Cost ranks what exists. Validity defines what exists. Mixing them made every cost function defensive and hid the feasible count.

**2. Every dimension must have a wire.** If the solver can distinguish two points, the executor must distinguish them too. AVX512 searched but AVX2 executed. `aligned` searched but `loadu` always emitted. Dead dimensions are silent correctness bugs.

**3. The instance is the observation, distinct from the decision.** Cost is `(point, instance) → score`. The instance parameterises the search; the solver varies the point while holding the instance fixed. `make_problem(instance) → filtered_space` is the single construction point.

**4. Wrappers must be honest about capabilities.** A `valid_view` with `cardinality()` returning the unfiltered count is a lie. Lying to satisfy a concept is worse than failing to match it. The tier system enforces honesty.

**5. Factored structure and indexing are independent.** Beam search needs per-dimension enumeration (factored). Sampling needs random access (indexable). Filtering destroys both. The concepts are separate so the type system documents which capability is missing and why.

**6. Single source of truth for the solve path.** One `make_problem` function, callable at both compile time and runtime. Divergence between CT and RT paths is a bug. `static constexpr auto result = solve(make_problem(inst), cost)` — if the function isn't constexpr-capable, that's the bug to fix, not a reason to duplicate.

**7. Correctness tests are architecture, not afterthoughts.** Non-uniform inputs with scalar reference comparison. All-ones inputs hide transposition bugs. The correctness harness should be a reusable component, not ad-hoc code in `main` that disappears during refactoring.

**8. The framework's value is in the boundaries, not the algorithms.** Exhaustive search is a nested for loop. The cost model is problem-specific. The executor is problem-specific. What the framework provides is the protocol between them — named, typed, enforced. The algorithms are trivial. The contracts are the product.

**9. Small examples expose architectural gaps that large designs hide.** 192 points and 105 points. Tiny spaces. But they forced confrontation with: instance parameterisation, tier honesty, factored vs filtered structure, dead dimensions, constraint placement, test regression, and wrapper composition. A 10,000-point space would have hidden all of these behind "works well enough on average."


## 11. What a Usable Framework Looks Like

### 11.1 The Twenty-Line Test

A competent C++ developer, given the headers and a new problem domain (not one of the existing examples), should be able to:

1. Define a space with 3–6 dimensions
2. Attach validity constraints via `filter_valid`
3. Search for the optimum with a hand-written cost function
4. Dispatch the result to an executor template

...in approximately 20 lines, without reading the framework source. Only the concepts header and the descriptor constructors should be needed. If the user needs to understand `valid_view` internals or tuple index mechanics, that's a failure.

### 11.2 The Composition Test

The following composition chain must work without special-casing:

```
section(product(space_A, space_B), dim_from_B, value) → enumerate
```

This is where index remapping breaks if the composition algebra isn't clean. A sectioned product where the fixed dimension comes from the second constituent space requires the product to correctly offset dimension indices. This is tested in the existing 76 space algebra tests.

### 11.3 The Extensibility Test

Adding a new descriptor kind should require:

1. A struct satisfying the descriptor concept (4 functions)
2. Nothing else — `descriptor_space`, bridge encoding, search, all work automatically

This has been verified with the existing five descriptor kinds. The extensibility test for a permutation descriptor would additionally test non-scalar `value_type` (std::array<int, N> instead of int/enum/bool), which propagates through the tuple point type and bridge encoding.

### 11.4 The API Surface

The user-facing API is deliberately small:

| Component | User touches | User doesn't touch |
|-----------|-------------|-------------------|
| **Space** | `descriptor_space`, `make_enum_vals`, `make_int_set`, `bool_flag`, `filter_valid` | `valid_view` internals, tier concepts, factored accessors |
| **Search** | `exhaustive_search_with_cost` | Search loop mechanics |
| **Bridge** | `default_bridge(space)`, `write_features(pt, span)` | Encoding dispatch, feature width calculation |
| **Dispatch** | `dispatch<Executor, point>` | NTTP mechanics |
| **Composition** | `section`, `product`, `descriptor_product` | Index remapping, qualified name generation |

Everything below the user-facing line is framework infrastructure that the concepts enforce without the user needing to see it.


## 12. Roadmap

### 12.1 Phase 1 — Calibration and Beam Search (next)

| Component | What | Status |
|-----------|------|--------|
| `calibration<Space>` | Dispatch table → measure each feasible point → (features, timing) table | Design complete, not built |
| `beam_search` | Factored space + complete-candidate validation | Concept infrastructure in place |
| `materialise(tier1)` | Enumerate once, cache all points for random access | Needed by calibration sampling |
| Instance concept | Calibration harness iterates instances, pairs with measurements | Convention exists, not formalised |
| Integration test | End-to-end: make_problem → search → bridge → dispatch → execute → verify | Blocked on calibration |

### 12.2 Phase 2 — Tiling and Larger Spaces

Adding tile size dimensions to the loop nest space. Each dimension d gets a power-of-two tile choice `tile[d]`. The space becomes `order(2N) × simd × isa × tile[0] × tile[1] × ... × tile[N-1]` — much larger, requiring:

- Per-dimension constraint propagation (tile sizes bounded by cache capacity)
- Beam search as the primary solver (exhaustive infeasible)
- Calibration with sampling (can't measure all points)

### 12.3 Phase 3 — Tree-Structured Spaces

Loop fission creates a partition of statements into groups. Each group gets its own tile/order/SIMD subspace. The number of subspaces depends on the fission decision — a tree, not a flat product.

Requires: `conditional_space` or `tree_space` where downstream dimensions exist only when a parent dimension has certain values. Beam search over the tree root, exhaustive within branches. DP over the tree when groups are independent.

### 12.4 Deferred Capabilities

| Capability | Trigger | Why deferred |
|-----------|---------|-------------|
| Quotient spaces | Bin-packing with symmetric labels | Need a concrete problem beyond cache-line layout |
| `permutation_space<N>` | Struct field ordering | Next example to build |
| Domain observable vocabulary | 3+ problems sharing observables | Need more examples first |
| Per-dimension constraint propagation | Tiling spaces with 1000+ points per dimension | Phase 2 |
| Learned model selection (linear/SVR/MLP) | Calibration producing training data | Phase 1 calibration first |


## 13. Version History

| Version | Date | Key changes |
|---------|------|-------------|
| v0.4 | Feb 2026 | Two solver halves, separate libraries |
| v0.5 | Feb 18 | Merged solvers, candidate_cache fix, solve_stats reconciled. 394 tests |
| v0.6.0 | Feb 23 | GitHub upload, CI (GCC 13/Clang 17/MSVC), 213 files |
| v0.7.1 | Feb 25 | space.h + descriptor.h rewritten, two-layer API. 249 tests |
| v0.7.2 | Feb 26 | valid_view, tier concepts, factored accessors, dispatch, 4 examples. 254 tests |


## 14. Example Checklist (PR Gate)

Every example must pass before commit:

1. **No dead knobs** — every search dimension changes generated code
2. **No lying metadata** — cardinality and rank are truthful
3. **Single solve function** — CT and RT paths share one `make_problem`
4. **Search matches executor** — every searchable value has a corresponding branch
5. **Constraints structural** — validity is in the space, not `cost = +inf`
6. **Bridge from same space** — encode from the filtered space the solver uses
7. **Non-trivial correctness** — non-uniform inputs, scalar reference comparison

This checklist exists because we proved that knowing better doesn't prevent regression. A list does.
