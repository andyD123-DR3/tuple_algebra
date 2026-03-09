# CT-DP Unified Descriptor Model

## Design Document v4 — Dense Developer Draft

### Author: Andrew Drakeford

---

## 1. Purpose

This document defines the implementation-facing specification for the CT-DP framework’s descriptor system. The intention is not to provide a light overview, nor to provide a marketing statement about a future framework. The intention is to state, in enough detail to guide implementation, how domain meaning, algebraic legality, searchable optimisation structure, plan construction, and instantiation are supposed to fit together.

The system has grown from several lines of work: domain-specific search spaces in the FIX and SpMV work, the `descriptor_space` and feature-bridge machinery in the `ctdp::space` tier, and the algebraic descriptor machinery in the tuple algebra work. Those lines of work solved real problems, but they did so in relative isolation. This document is about turning them into a single design that developers can implement against and use.

The working claim of this revision is deliberately scoped. The claim is not that CT-DP now has a universal compiler IR, a complete tensor-layout algebra, or a general-purpose optimisation theory. The claim is that a small set of scalar descriptor primitives, together with a small structured core, is enough to describe the discrete structured optimisation spaces that are currently in scope for CT-DP. In concrete terms, those spaces include the spaces needed for the talk demonstrators, the FIX parser family, the current SpMV format/strategy searches, reduction-topology generation from reduction descriptors, and the immediate descriptor-driven design work around tiling, fusion, ordering, and layout family selection.

This document therefore has two jobs. The first job is to define the layers and the primitive basis precisely enough that an implementer can write code against them without discovering, only at compile time, that the semantics were underspecified. The second job is to act as a developer guide: it must explain not only what the primitives are, but how they interoperate, how they are supposed to be encoded for calibration, how they relate to reduction descriptors, and what code written with them is supposed to look like.

---

## 2. Scope and non-goals

The scope of this document is the discrete, typed, enumerable optimisation problem that CT-DP is trying to solve. The system described here is meant to describe finite spaces of legal program choices. It is designed for descriptor-driven search, legality filtering, calibration, Pareto ranking, and plan construction. That includes ordinary Cartesian products of choices, constrained products, heterogeneous per-element spaces, ordering spaces, grouping spaces, property-driven conditional dimensions, dependent subproblems, regular tessellation of regular domains, and plan-level runtime choice between already-solved alternatives.

The document does not claim to solve every representation problem that appears in performance engineering. It does not define a full CuTe-style layout algebra with arbitrary function composition on coordinate maps. It does not define a general compiler IR for CFG or SSA structure. It does not attempt to represent arbitrary graph rewrite systems. It does not yet attempt continuous real-valued hyperparameter optimisation. It also does not pretend that every ambitious future primitive is at the same maturity level as `int_set` or `enum_vals`. Where a primitive is mature and already reflected in existing code, that is stated. Where a primitive is designed but not yet at implementation maturity, that is also stated.

The design of the document is driven by five principles. First, properties generate structure: algebraic and instance properties decide which optimisation axes exist and which points are legal. Second, identity rides along: domain meaning must remain available to cost functions and instantiation logic even when structure is derived automatically. Third, structure must be enumerable: the search-space layer is for finite typed spaces. Fourth, search and instantiation must remain separate: the search-space layer describes alternatives, while the plan and instantiation layers describe what is chosen and how it becomes code. Fifth, feature encoding must remain stable: spaces with conditional structure must still be encodable into a calibration/ML interface in a way that does not make the learning interface incoherent.

---

## 3. The three descriptor generations

The descriptor story in CT-DP did not start from a single clean abstraction. It emerged in three separate forms.

### 3.1 Generation 1 — domain descriptors

Generation 1 descriptors are opaque, domain-specific types. They were the first serious expression of the principle that “the descriptor list is the program.” In that model, the framework did not interpret the internal fields of the descriptor. Instead, it carried them through the system so that domain logic could interpret them when computing costs, checking legality, or performing instantiation.

A FIX field descriptor is the canonical example.

```cpp
struct fix_field {
    std::string_view name;
    int tag;
    int digits;
    field_type type;
    tier_t tier;
    strategy_mask allowed;
};
```

This descriptor carries information that is not merely decorative. A 2-digit field and a 10-digit field do not have the same cost profile. A checksum field and a price field are not interchangeable. A field’s tier, type, and legal parsing strategies affect both legality and cost. The framework, however, need not understand the meaning of `digits` or `tier`; it need only preserve the descriptor so that the domain code can use it.

That is the central strength of Generation 1. It preserves domain meaning honestly. Its weakness is that it does not itself provide a uniform structure for enumeration or feature encoding.

### 3.2 Generation 2 — space descriptors

Generation 2 descriptors are the framework-defined structural primitives in `ctdp::space`. These are not opaque. They are deliberately inspectable. They define legal values, cardinality, deterministic enumeration, and feature encoding.

The implemented scalar primitives are:

```cpp
using namespace ctdp::space;

constexpr auto tile_m   = power_2("tile_m", 8, 256);
constexpr auto tile_n   = power_2("tile_n", 8, 256);
constexpr auto unroll   = int_set("unroll", {1, 2, 4, 8});
constexpr auto use_fma  = bool_flag("use_fma");
constexpr auto simd     = enum_vals("simd", {simd_kind::scalar, simd_kind::sse2, simd_kind::avx2});
```

These descriptors are good at one thing: describing search axes. They are finite, typed, and enumerable. They fit naturally into feature encoding and exhaustive search. Their weakness is that, taken alone, they know nothing about domain identity and nothing about algebraic legality. If a user writes `int_set("vec", {1, 4, 8})`, the descriptor knows the legal values and their encoding, but it does not know whether vector width is legal for the current reduction state or whether a particular field may use SWAR.

### 3.3 Generation 3 — reduction descriptors

Generation 3 descriptors arise from the tuple algebra and reduction work. They describe semantic structure and algebraic legality. They answer questions such as whether an operation is associative, whether it has an identity, whether it is commutative, and how a tuple-valued reduction is structured.

A representative example is a statistics reduction:

```cpp
using namespace ct_dp::algebra;

constexpr auto stats = make_reduction(
    reduction_lane{constant_t<1>{}, plus_fn{}, 0},
    reduction_lane{identity_t{},    plus_fn{}, 0.0},
    reduction_lane{power_t<2>{},    plus_fn{}, 0.0},
    reduction_lane{identity_t{},    min_fn{},  +std::numeric_limits<double>::infinity()},
    reduction_lane{identity_t{},    max_fn{},  -std::numeric_limits<double>::infinity()}
);
```

This object is not a search space. It is a semantic contract. It says what the reduction means, how lanes are updated and combined, and what algebraic facts hold. Those facts are exactly the facts that ought to decide whether tree reduction is legal, whether vector-width exploration should exist, and whether grouping freedoms are available.

### 3.4 Unified interpretation

The unified model treats the three generations as three different contributions to one pipeline. Generation 1 carries identity and domain meaning. Generation 2 carries enumerable structure. Generation 3 carries semantic legality facts. They meet in the space-generation layer. The framework should therefore be able to take Generation 3 facts, use them to generate Generation 2 structure, and attach Generation 1 identity so that the resulting search space is both searchable and meaningful.

---

## 4. Architectural layers

The descriptor story only becomes implementable when it is described as a stack of layers. If the document talks about “descriptors” as if they are simultaneously the semantic contract, the search space, the selected plan, and the final executor shape, then it collapses several different abstraction boundaries and becomes difficult to implement coherently.

### 4.1 Layer 1 — semantic layer

The semantic layer describes the problem. It includes domain descriptors, reduction descriptors, iteration descriptors, hardware descriptors, and intent descriptors.

A domain descriptor describes what the program is about. A reduction descriptor describes what algebraic operations are legal. An iteration descriptor describes the shape and traversal-relevant properties of the domain. A hardware descriptor contributes budgets and capabilities. An intent descriptor states required guarantees such as a minimum reproducibility level or a preference for latency versus energy.

A useful iteration descriptor looks like this:

```cpp
struct iteration_dim {
    std::string_view name;
    int extent;
    int stride;
    int alignment;
    bool vectorizable;
};
```

A useful hardware descriptor looks like this:

```cpp
struct hardware {
    int l1_bytes;
    int l2_bytes;
    int simd_width;
    int num_registers;
    bool has_fma;
};
```

An intent descriptor might be:

```cpp
enum class repro_level {
    nondet,
    deterministic,
    reproducible,
    bitwise_portable
};

struct intent {
    repro_level minimum_repro;
    bool latency_critical;
    bool energy_sensitive;
};
```

The semantic layer is not supposed to enumerate optimisation points. It is supposed to answer: what is being optimised, what guarantees are required, what algebraic facts hold, what domain facts hold, and what hardware budgets must be respected.

### 4.2 Layer 2 — search-space layer

The search-space layer turns semantic meaning into searchable structure. This is the layer that owns scalar primitives, structural combinators, structured constructors, and space transformers. It is the layer in which a user can enumerate legal alternatives, apply legality filtering, extract features, and drive search.

This layer is where the primitive basis lives. It is not enough to say that a primitive exists; the document must say what it means. The remainder of Sections 5 through 10 does exactly that.

### 4.3 Layer 3 — plan layer

The plan layer records the result of search. A plan is not a descriptor space. A plan is not an algebraic descriptor. A plan is the chosen configuration. A flat plan is a concrete aggregate of chosen values. A grouped plan contains grouping structure plus a sub-plan per group. A hierarchical plan contains one chosen sub-plan per level of a nested search. A runtime `choose<Pred, A, B>` belongs in this layer because it describes a runtime choice between already-solved alternatives.

A simple flat plan might be:

```cpp
struct loop_plan {
    int tile_x;
    int tile_y;
    loop_order order;
    fusion_pattern fusion;
    int vec;
    tree_shape tree;
};
```

A grouped plan for a tuple reduction might be:

```cpp
struct grouped_plan {
    std::array<int, 5> lane_groups;
    std::vector<loop_plan> group_plans;
};
```

The plan layer is about the output of search, not the structure of the search itself.

### 4.4 Layer 4 — instantiation layer

The instantiation layer consumes plans and semantic information to produce executable artefacts. These artefacts may be executor specialisations, runtime dispatch tables, layout-aware containers, proxy views, or generated kernels.

An executor specialisation pattern is the most obvious example:

```cpp
template<loop_plan P>
struct executor;
```

A layout-aware container pathway for the CERN-style layout work might look like:

```cpp
auto data = make_layout_container<chosen_layout_plan>(...);
auto view = make_proxy_view(data);
```

The instantiation layer is where plans become code, storage, and runtime behaviour.

---

## 5. Scalar descriptor primitives

The scalar primitives form the atomic basis of ordinary search axes. They are small, but each one needs a precise contract. The contract matters because these primitives are not just used for enumeration. They are used for cost functions, legality filters, feature extraction, calibration, and plan construction.

Every scalar primitive must satisfy the `dimension_descriptor` contract. In operational terms, that means it must provide a name, a kind, a finite cardinality, a deterministic enumeration order, a `value_type`, a total mapping from valid indices to values, a membership check for values, and a stable mapping from valid values back to indices.

### 5.1 `positive_int(name, lo, hi)`

`positive_int` represents a dense closed integer interval from `lo` to `hi`, where `lo` and `hi` are positive and `lo <= hi`.

```cpp
constexpr auto threads = positive_int("threads", 1, 16);
```

The legal domain here is the ordered set `{1, 2, 3, ..., 16}`. Enumeration is ascending. `cardinality()` is `16`. `value_at(0)` is `1`. `value_at(15)` is `16`. `contains(4)` is true. `contains(0)` and `contains(17)` are false.

`positive_int` is appropriate when the search genuinely wants a dense integer interval. It is appropriate for things such as the number of accumulators, thread count, chunk sizes, or other quantities for which intermediate values are meaningful and not artificially excluded. It is not the correct primitive for powers of two, nor for explicit sparse sets of values. If the legal values are `{1, 2, 4, 8}`, then `int_set` or `power_2` is a better semantic fit.

The default encoding is raw integer value. That is appropriate because the descriptor does not claim any special structure beyond density.

A minimal test sketch is:

```cpp
static_assert(threads.cardinality() == 16);
static_assert(threads.contains(1));
static_assert(threads.contains(16));
static_assert(!threads.contains(0));
static_assert(!threads.contains(17));
```

### 5.2 `power_2(name, lo, hi)`

`power_2` represents the subset of powers of two in the closed interval `[lo, hi]`.

```cpp
constexpr auto tile_m = power_2("tile_m", 8, 256);
```

The legal domain is `{8, 16, 32, 64, 128, 256}`. Enumeration is ascending. `cardinality()` is `6`. `contains(32)` is true. `contains(48)` is false.

This primitive exists because powers of two are not just an incidental subset of integers in CT-DP. They appear repeatedly in tiling, AoSoA widths, cache-sensitive blocking, and SIMD-friendly chunking. Using `positive_int` for such axes obscures the real structure of the space. Using `power_2` makes the intended search structure explicit and gives a more natural feature encoding.

The default encoding is `log2(value)`. That means `8` encodes to `3`, `16` to `4`, `32` to `5`, and so on. This matters when features are later passed to calibration code.

A concrete example of use is a GEMM tile space:

```cpp
constexpr auto gemm_tiles = descriptor_space(
    "gemm_tiles",
    power_2("tile_m", 8, 128),
    power_2("tile_n", 8, 128),
    power_2("tile_k", 8, 256)
);
```

### 5.3 `int_set(name, {v...})`

`int_set` represents an explicit finite set of integer values.

```cpp
constexpr auto unroll = int_set("unroll", {1, 2, 4, 8});
```

Unlike `positive_int`, `int_set` does not imply density. Unlike `power_2`, it does not imply logarithmic structure. It simply says that the legal values are exactly the values given.

This is the correct primitive for unroll factors, manually chosen candidate counts, vector widths that are not semantically “all powers of two in a range”, or any other integer axis where the legal set is sparse and explicit.

The default encoding is raw integer value. Enumeration order should be the order induced by the descriptor’s canonical value sequence. The implementation should guarantee a stable order, normally ascending or user-given canonical order.

A representative use in a mixed execution space is:

```cpp
constexpr auto exec_space = descriptor_space(
    "exec",
    int_set("unroll", {1, 2, 4, 8}),
    int_set("vec", {1, 4, 8})
);
```

### 5.4 `bool_flag(name)`

`bool_flag` represents a binary decision.

```cpp
constexpr auto use_fma = bool_flag("use_fma");
```

Its domain is `{false, true}`. Enumeration order must be stable; the usual convention is `false` then `true`. The default encoding is binary.

This primitive is appropriate for choices like whether to enable FMA, whether to force aligned accesses, whether an optimisation pass is on or off, or whether a certain optimisation mode is enabled. It is not the right primitive when the choice set will grow beyond two meaningful values, or when the values have an ordering that should be preserved.

A common use is:

```cpp
constexpr auto hw_space = descriptor_space(
    "hw_space",
    bool_flag("use_fma"),
    bool_flag("aligned")
);
```

### 5.5 `enum_vals(name, {v...})`

`enum_vals` represents a finite unordered categorical axis.

```cpp
enum class format { CSR, ELL, DIA, BCSR };
constexpr auto fmt = enum_vals("format", {
    format::CSR, format::ELL, format::DIA, format::BCSR
});
```

This is the correct primitive for categorical choices where the alternatives are distinct but not naturally ordered. Sparse format family, preconditioner kind, fusion pattern names in a talk demo, tail strategy, layout family, and algorithm family are all good examples.

The default encoding is one-hot. If the current point uses `format::ELL`, then the format contribution to the feature vector is conceptually `[0, 1, 0, 0]`.

This primitive is strong because it is simple and honest. Its weakness is that it does not carry order semantics. When the values genuinely form a ladder or a monotone guarantee hierarchy, `ordinal` is the right tool instead.

### 5.6 `ordinal(name, {A < B < C})`

`ordinal` is the missing scalar primitive needed for ordered categorical choices.

```cpp
enum class repro_level {
    nondet,
    deterministic,
    reproducible,
    bitwise_portable
};

constexpr auto repro = ordinal(
    "repro",
    {repro_level::nondet,
     repro_level::deterministic,
     repro_level::reproducible,
     repro_level::bitwise_portable}
);
```

`ordinal` is not merely a convenience wrapper around `enum_vals`. Its purpose is to preserve order semantics in the descriptor itself. Reproducibility levels, precision ladders, and placement ladders all have meaningful order. If they are represented as plain enums, the ordering has to be recreated in legality predicates, dominance logic, and user explanations. That is brittle and obscures meaning.

The intended semantics are that the domain is a finite ordered chain. Enumeration order follows that chain. The default encoding should preserve that order, and for ML use it is often helpful to include both an ordinal scalar position and, where appropriate, a cumulative or activity-mask interpretation. The exact encoding choice should be specified once and applied consistently.

A placement example is:

```cpp
enum class compute_level { root, outer, inner, inline_ };
constexpr auto level = ordinal(
    "compute_level",
    {compute_level::root,
     compute_level::outer,
     compute_level::inner,
     compute_level::inline_}
);
```

This is appropriate only for regular ladder-like placement. It does not replace a general topological placement-site descriptor for arbitrary Halide-like loop nests.

---

## 6. Feature encoding

The descriptor system does not stop at enumeration. It also has to support stable feature extraction for calibration and ML. The feature vector must describe the selected point in a way that is deterministic, reproducible, and stable under refactoring of cost models.

A simple example shows the intended interpretation.

```cpp
enum class format { CSR, ELL, DIA };

constexpr auto space = descriptor_space(
    "demo",
    power_2("tile_x", 8, 64),
    bool_flag("use_fma"),
    enum_vals("format", {format::CSR, format::ELL, format::DIA})
);

auto bridge = default_bridge(space);
auto pt = std::tuple{16, true, format::ELL};
std::vector<double> features(bridge.width());
bridge.write_features(pt, std::span<double>(features));
```

The conceptual interpretation of the features is as follows. `tile_x = 16` contributes `log2(16) = 4` because `power_2` encodes logarithmically. `use_fma = true` contributes `1`. `format = ELL` contributes the one-hot block `[0, 1, 0]`. The resulting conceptual feature vector is therefore `[4, 1, 0, 1, 0]`.

That example matters because it makes feature meaning visible. The features are not arbitrary numbers; they are a typed encoding of descriptor choices.

For conditionally present dimensions, the ML interface must remain stable. That means the eventual `conditional_dim` and `tree_space` design has to specify how inactive or absent child structure is encoded. The current design principle is that feature width must remain stable. The implementation may therefore need activity bits or null sentinels in the encoding layer even when the logical space rank changes. The space semantics and the encoding semantics are related but not identical, and the document must say so explicitly.

---

## 7. Structural space combinators

The scalar primitives are not enough on their own. Real problems need a way to build larger spaces, constrain them, and slice them.

### 7.1 `descriptor_product(A, B)`

`descriptor_product` combines independent subspaces into a Cartesian product.

```cpp
constexpr auto tile_space = descriptor_space(
    "tile",
    power_2("tile_x", 8, 256),
    power_2("tile_y", 8, 256)
);

constexpr auto exec_space = descriptor_space(
    "exec",
    int_set("vec", {1, 4, 8}),
    bool_flag("use_fma")
);

constexpr auto full = descriptor_product(tile_space, exec_space, "full");
```

The point of `full` is a tuple concatenation of the points of its operands. If `tile_space` yields `(tile_x, tile_y)` and `exec_space` yields `(vec, use_fma)`, then `full` yields `(tile_x, tile_y, vec, use_fma)`.

The important semantic condition is independence. `descriptor_product` is correct when the spaces are structurally independent. It is not the right abstraction for permutations, partitions, or child spaces that depend on parent values.

### 7.2 `valid_view(space, pred)`

`valid_view` is a legality filter over a base space.

```cpp
constexpr hardware hw{32 * 1024, 8, 16, true};

constexpr auto feasible = valid_view(full, [&](auto const& pt) {
    auto const& [tx, ty, vec, use_fma] = pt;
    auto bytes = tx * ty * 3 * int(sizeof(double));
    return bytes <= hw.l1_bytes && tx % vec == 0;
});
```

This is the standard way to express instance-constrained Cartesian search. The base descriptors define the candidate axes. The predicate removes illegal points.

`valid_view` is strong for cache-fit constraints, vector-width divisibility constraints, register-budget constraints, and other legality conditions that do not change the logical meaning of the axes. Its weakness is that, if used alone, it produces spaces with dead dimensions when an entire axis should not exist. That is why `conditional_dim` is needed.

### 7.3 `section<I>(space, value)`

`section` fixes one dimension to one chosen value and returns the resulting subspace.

```cpp
auto vec8_only = section<2>(full, 8);
```

This is useful for debugging, partial evaluation, compare-and-contrast studies, or nested search in which one outer choice is fixed before searching the inner subspace.

`section` should preserve the meaning of the remaining dimensions and behave like a logical slice, not a new ad-hoc space.

### 7.4 `heterogeneous_per_element_space(...)`

This constructor is essential for spaces where the legal options vary by position.

The FIX parser is the reference example. Each field has its own allowed parsing strategies. The correct representation is not a flat Cartesian product with the same choices everywhere, but a per-position choice space.

```cpp
auto fix_space = heterogeneous_per_element_space(
    fix_fields,
    [](fix_field const& f) {
        return allowed_strategies_for(f);
    }
);
```

This is one of CT-DP’s real strengths. It allows the framework to express “field i has different legal strategies from field j” directly. Many optimisation systems do not have an honest abstraction for this and instead resort to hand-built domain code.

---

## 8. Higher-order structured spaces

The next group of primitives is where the space stops being flat.

### 8.1 `make_permutation(name, N)`

`make_permutation` represents ordering transformations directly. For loop reordering, field ordering, or tiled loop-band reordering, a permutation is the honest object. Treating a permutation as N independent enums plus an all-different legality filter is both semantically weak and operationally clumsy.

```cpp
auto order_space = make_permutation("loop_order", 3);
```

The intended `point_type` is a fixed-size array of indices such as `std::array<int, 3>{0, 2, 1}`, meaning “visit dimension 0 first, then 2, then 1”. Enumeration should range over all `N!` permutations directly, not over `N^N` candidates with filtering.

For the small 2D talk demo, a plain categorical axis is enough:

```cpp
enum class loop_order { ij, ji };
constexpr auto order = enum_vals("loop_order", {loop_order::ij, loop_order::ji});
```

But the general framework still needs a true permutation space.

### 8.2 `partition(N)`

`partition(N)` represents grouping of N items. This is the correct abstraction for fusion groups, fission groups, lane grouping in tuple reductions, and similar grouping problems.

The intended point representation is `std::array<int, N>` in canonical first-occurrence form. That means the first group encountered is labelled 0, the next new group encountered is labelled 1, and so on.

```cpp
auto groups = partition(5);

std::array<int, 5>{0, 0, 1, 1, 2}; // valid canonical representative
std::array<int, 5>{1, 1, 0, 0, 2}; // same grouping, but non-canonical
```

The first array means items 0 and 1 are together, items 2 and 3 are together, and item 4 is alone. The second array denotes the same grouping but with relabelled groups. The canonical representation rule ensures that only one representative is enumerated.

Enumeration order should be lexicographic over canonical representatives. Cardinality is the Bell number `B(N)`.

This primitive is the honest solution to grouping. It is not equivalent to any scalar descriptor.

### 8.3 `conditional_dim(pred, dim)`

`conditional_dim` is for property-driven conditionality known before search begins.

```cpp
auto maybe_tree = conditional_dim(
    reduction_properties<Reduction>::all_associative,
    enum_vals("tree_shape", {flat, binary, canonical})
);
```

If the predicate is true, the wrapped dimension participates normally. If the predicate is false, the dimension contributes no searched degree of freedom.

This is the right tool when the condition is a global fact. Associativity of a reduction, existence of an identity element, or the fact that a struct contains no jagged field are all examples of conditions known before the search enumerates points.

It is not the correct tool for point-dependent conditionality. If `layout` itself is being searched, and `AoSoA width` only exists when `layout == AoSoA`, that condition depends on a searched point. In that case the correct abstraction is a dependent space, which in the current model means `tree_space`.

### 8.4 `tree_space(root, children_fn)`

`tree_space` represents a hierarchical search in which child spaces depend on the chosen root point.

```cpp
auto fusion_then_sched = tree_space(
    partition(4),
    [&](auto const& grouping) {
        return per_group_schedule_space(grouping);
    }
);
```

This is the right abstraction when grouping choices generate group-specific subproblems, or when a searched layout choice enables a child subspace. In the current MVP design, `tree_space` is implemented by nested solve calls. The outer solver enumerates the root points. For each root point it constructs the child space, solves it, composes the resulting plan, and ranks the resulting composite plan. That is the honest current solver protocol.

The child-schema restriction for the MVP should be stated clearly: although the child spaces are dependent on the root value, they should share one implementation schema so that the executor and calibration machinery do not have to cope with unrestricted structural heterogeneity at every node.

---

## 9. Tessellation and layout-transforming spaces

### 9.1 `tile_space(base, tiler)`

`tile_space` is the first-class representation of regular tessellation of a regular domain.

```cpp
struct surface_2d {
    int nx;
    int ny;
};

constexpr surface_2d dom{1024, 1024};

auto tiled = tile_space(
    dom,
    tiler{
        power_2("tile_x", 8, 256),
        power_2("tile_y", 8, 256)
    }
);
```

In the MVP design, the searched point of `tile_space` is the tuple of tile sizes only. For the example above, the point is effectively `(tile_x, tile_y)`. Tile counts, remainder structure, interior/boundary decomposition, and any derived iteration structure are computed from the domain extents and the chosen tile sizes. They are not themselves searched.

That design makes `tile_space` useful immediately without forcing a full layout algebra into the MVP. It lets the framework say “regular tessellation is a first-class concept” while keeping the searched structure simple and concrete.

`tile_space` is more than just two `power_2` descriptors. The descriptors parameterise the tessellation, but `tile_space` records that the result is meant to be interpreted as a tiled domain rather than merely as two unrelated integers.

### 9.2 Future `layout_compose`

A full CuTe-style layout algebra is beyond the current scope, but the document should acknowledge the future extension point. `layout_compose(A, B)`, `complement`, or richer logical divide constructs would allow the framework to describe transformations on coordinate maps themselves rather than just search over format family choices. That is future work, but the current basis should leave space for it.

---

## 10. Plan layer details

The output of search must be describable with the same precision as the input space. A multi-level search produces a multi-level plan.

A flat search over tiling, fusion, and reduction topology might yield:

```cpp
struct demo_plan {
    int tile_x;
    int tile_y;
    loop_order order;
    fusion_pattern fusion;
    int vec;
    tree_shape tree;
};
```

A grouped plan for a tuple reduction might yield:

```cpp
struct reduction_group_plan {
    std::array<int, 5> grouping;
    std::vector<demo_plan> group_plans;
};
```

A hierarchical plan for tiled reduction can be described as a composite:

```cpp
struct tiled_reduction_plan {
    tile_plan tiling;
    std::array<int, 5> fusion_groups;
    tree_shape reduction_tree;
};
```

The executor layer must consume these plans directly. That means the document must not leave plan composition as an unspoken assumption. If Level 1 produces a tiling plan, Level 2 a grouping plan, and Level 3 a tree-shape plan, then the composite plan is a typed aggregate containing one sub-plan per level.

`choose<Pred, A, B>` belongs here. It is a plan-level combinator for runtime selection between already-solved alternatives. It is not a search-space primitive because it does not define one search over a combined domain. It defines runtime dispatch between two plans obtained by separate searches.

---

## 11. How reduction descriptors fit the primitive basis

The correct relationship is simple and strict. A reduction descriptor is not itself the optimisation space. A reduction descriptor is the semantic source from which optimisation structure is generated.

That gives the following pattern.

```cpp
auto reduction = make_reduction(...);
auto props     = summarise(reduction);
auto space     = make_reduction_opt_space(reduction, iteration, hw, intent);
```

The descriptor is the semantic contract. The summary is the derived legality and footprint facts. The generated descriptor space is the searchable optimisation structure.

### 11.1 Generated axes from algebraic facts

Associativity should generate a tree-shape axis.

```cpp
conditional_dim(all_associative,
    enum_vals("tree_shape", {flat, binary, pairwise, canonical}))
```

Identity availability should generate vector-width exploration.

```cpp
conditional_dim(all_have_identity,
    int_set("vec", {1, 4, 8, 16}))
```

Tuple lane count should enable grouping structure.

```cpp
tree_space(
    partition(num_lanes),
    [&](auto const& grouping) {
        return per_group_subspaces(grouping, reduction, hw, iter);
    }
)
```

Reproducibility requirements should appear as an ordered descriptor.

```cpp
ordinal("repro", {
    repro_level::nondet,
    repro_level::deterministic,
    repro_level::reproducible,
    repro_level::bitwise_portable
})
```

State footprint does not usually create a new axis directly. Instead, it contributes to legality and cost. Tile size legality, register-budget legality, and vector-width legality should all be filtered using the reduction summary.

### 11.2 Worked reduction-space generator

A simple entry-level reduction-space generator can already be written in today’s style.

```cpp
enum class tree_shape { flat, binary, pairwise, canonical };

template<class Reduction>
struct reduction_properties {
    static constexpr bool all_associative = true;   // schematic
    static constexpr bool all_have_identity = true; // schematic
    static constexpr int  num_lanes = 5;            // schematic
};

template<class Reduction>
constexpr auto make_reduction_opt_space() {
    using P = reduction_properties<Reduction>;

    auto base = descriptor_space(
        "reduction_base",
        power_2("tile", 64, 4096)
    );

    if constexpr (P::all_associative && P::all_have_identity) {
        auto extra = descriptor_space(
            "reduction_extra",
            enum_vals("tree", {tree_shape::flat,
                                tree_shape::binary,
                                tree_shape::pairwise,
                                tree_shape::canonical}),
            int_set("vec", {1, 4, 8, 16})
        );
        return descriptor_product(base, extra, "reduction_space");
    } else if constexpr (P::all_associative) {
        auto extra = descriptor_space(
            "reduction_extra",
            enum_vals("tree", {tree_shape::flat,
                                tree_shape::binary,
                                tree_shape::pairwise,
                                tree_shape::canonical})
        );
        return descriptor_product(base, extra, "reduction_space");
    } else {
        return base;
    }
}
```

That example is intentionally simple, but it shows the core rule clearly: properties generate structure.

---

## 12. Worked examples by domain

### 12.1 FIX parser — per-element space

The FIX parser is a per-element search problem. Each field has its own allowed strategies, and the cost model must still know which field is which.

```cpp
auto fix_space = heterogeneous_per_element_space(
    fix_fields,
    [](fix_field const& f) {
        return allowed_strategies_for(f);
    }
);
```

The domain descriptors remain attached. The space generator extracts the legal strategies per field. The resulting space is a direct fit for the FIX problem; it is not a disguised Cartesian product with a huge legality filter.

### 12.2 GEMM tiling — constrained Cartesian

```cpp
constexpr auto gemm_space = descriptor_space(
    "gemm_tile",
    power_2("TM", 8, 128),
    power_2("TN", 8, 128),
    power_2("TK", 8, 256),
    int_set("unroll", {1, 2, 4, 8})
);

constexpr auto feasible_gemm = valid_view(gemm_space, [=](auto const& pt) {
    auto const& [tm, tn, tk, unroll] = pt;
    auto footprint = (tm * tk + tk * tn + tm * tn) * int(sizeof(double));
    return footprint <= hw.l1_bytes && tn % hw.simd_width == 0;
});
```

This is the canonical instance-constrained Cartesian case.

### 12.3 SpMV — format family plus instance-derived legality

```cpp
enum class format { CSR, ELL, DIA, BCSR };
enum class reorder { natural, rcm };
enum class simd_kind { scalar, sse2, avx2 };

constexpr auto spmv_space = descriptor_space(
    "spmv",
    enum_vals("format", {format::CSR, format::ELL, format::DIA, format::BCSR}),
    enum_vals("reorder", {reorder::natural, reorder::rcm}),
    enum_vals("simd", {simd_kind::scalar, simd_kind::sse2, simd_kind::avx2})
);

auto feasible_spmv = valid_view(spmv_space, [&](auto const& pt) {
    auto const& [fmt, reord, simd] = pt;
    if (fmt == format::DIA && !matrix.has_banded_structure) return false;
    if (fmt == format::ELL && matrix.fill_ratio < 0.5) return false;
    return true;
});
```

This shows how instance properties eliminate effectively illegal points without pretending that the instance itself is the searched structure.

### 12.4 Tiling + fusion + reduction — talk demonstration

The strongest small demonstration is a 2D three-stage pipeline with an optional reduction.

```cpp
enum class loop_order { ij, ji };
enum class fusion_pattern { none, fuse_01, fuse_12, fuse_all };
enum class tree_shape { flat, binary, canonical };

constexpr auto tile = descriptor_space(
    "tile",
    power_2("tile_x", 8, 256),
    power_2("tile_y", 8, 256)
);

constexpr auto exec = descriptor_space(
    "exec",
    enum_vals("loop_order", {loop_order::ij, loop_order::ji}),
    enum_vals("fusion", {fusion_pattern::none,
                          fusion_pattern::fuse_01,
                          fusion_pattern::fuse_12,
                          fusion_pattern::fuse_all}),
    int_set("vec", {1, 4, 8})
);

constexpr auto base = descriptor_product(tile, exec, "demo");

constexpr auto full = descriptor_product(
    base,
    descriptor_space(
        "red",
        enum_vals("tree_shape", {tree_shape::flat,
                                  tree_shape::binary,
                                  tree_shape::canonical})
    ),
    "demo_with_tree"
);

auto feasible = valid_view(full, [&](auto const& pt) {
    auto const& [tx, ty, order, fusion, vec, tree] = pt;
    return fits_l1(tx, ty, hw) &&
           vectorization_legal(tx, ty, vec, dom) &&
           fusion_legal(fusion, dag);
});
```

This example is deliberately small, but it makes the thesis visible. Tiling is represented with geometric descriptor primitives. Fusion is represented as a structural categorical choice. Reduction topology is represented as a semantic choice generated from reduction legality. All three are described using the same search-space machinery.

---

## 13. Strengths and weaknesses of the primitive basis

The current primitive basis is genuinely strong for flat and mildly structured spaces. It is uniform, typed, finite, naturally encodable, and domain-independent. It already matches a large fraction of the spaces seen in FIX, SpMV, loop-nest tuning, and descriptor-driven demonstrators. The presence of `heterogeneous_per_element_space` is an especially strong point because it gives the framework a first-class way to express per-position heterogeneity.

The weaknesses are equally important to state honestly. The original five-scalar basis is biased toward flat Cartesian structure. It lacks honest order semantics for ladders such as reproducibility levels and placement ladders. It lacks honest grouping semantics for fusion and fission. It externalises conditionality into legality filters unless `conditional_dim` is added. It can parameterise tessellation, but without `tile_space` it does not describe tessellation structurally. It can search over layout families, but without richer layout-transforming constructs it does not yet offer a true layout algebra. These weaknesses are exactly why `ordinal`, `partition`, `conditional_dim`, `tree_space`, and `tile_space` are the right minimal extensions.

The expert design verdict remains simple. The framework does not need dozens of new primitives. It already has a good scalar basis. It needs a few structural primitives so that hierarchy, grouping, conditionality, and tessellation are described honestly rather than being simulated indirectly.

---

## 14. Final form of the basis

The smallest credible complete basis, in the sense required by the current CT-DP work, is this.

The scalar layer consists of `positive_int`, `power_2`, `int_set`, `bool_flag`, `enum_vals`, and `ordinal`. The structural layer consists of `descriptor_product`, `valid_view`, `section`, `heterogeneous_per_element_space`, `make_permutation`, `partition`, `conditional_dim`, `tree_space`, and `tile_space`. That is the core I would back for implementation.

The most concise correct summary of the model is therefore this: use scalar descriptors for choices and parameters, structural combinators for products and slicing, ordered descriptors for ladders, partitions for grouping, conditional dimensions for property-driven existence, and tree or tessellation spaces for hierarchical decomposition. Reduction descriptors then fit naturally as the semantic source from which optimisation structure is derived.

---

## 15. Immediate implementation consequences

The document is meant to be implementation-facing, so it must end with practical consequences.

The first practical consequence is that the search-space layer needs a common concept vocabulary. Descriptor-generated spaces are only useful if the solver layer can actually consume them. That means the project needs a clear migration or unification step between the existing solver-space concepts and the space-tier concepts.

The second practical consequence is that the next implementation steps should focus on the primitives that unblock the descriptor demonstrations and the descriptor-to-reduction bridge. `ordinal`, `conditional_dim`, `make_permutation`, pack-level reduction property queries, and a first reduction-space generator are the immediate high-value items. `partition`, `tree_space`, and `tile_space` should follow, with explicit MVP restrictions. The document has specified enough semantics here that those implementations can now be written against it rather than guessed from earlier notes.

The third practical consequence is that every primitive should be accompanied by code and tests at the same density as the examples in this document. When a developer adds `ordinal`, they should not merely make it compile. They should be able to point to the semantics here, implement them, and then write the corresponding cardinality, membership, enumeration, encoding, and composition tests.

That is the standard this document is supposed to set.
