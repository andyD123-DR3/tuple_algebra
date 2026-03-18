# Constructor Mapping: Existing Spaces → Five-Constructor Algebra

**Date:** March 2026
**Ref:** Core Semantics (Doc A), §5 constructors: `leaf`, `seq`, `nest`, `choose`, `split`

## Mapping Table

| Existing space type | Constructor tree | Shape | Notes |
|---|---|---|---|
| `descriptor_space<D0, D1, ..., Dn>` | `nest(choose(D0), choose(D1), ..., choose(Dn))` | Independent Cartesian product | Each dimension is an independent `choose`. `nest` composes them. This is the canonical space representation. |
| `per_element_space<N, Space>` | `nest(choose, choose, ..., choose)` (N copies) | Homogeneous nest | All N dimensions share the same `choose` alternatives. Sugar for `descriptor_space` with N identical descriptors. |
| `heterogeneous_per_element_space<Spaces...>` | `nest(choose(S0), choose(S1), ..., choose(Sn))` | Heterogeneous nest | Each position has its own `choose` set. Isomorphic to `descriptor_space` with per-position descriptors. FIX parser uses this. |
| `permutation_space<N>` | Dependent `seq(choose(N), choose(N-1), ..., choose(1))` | Dependent sequence | Each `choose` is constrained by prior choices (no repeats). This is `seq` not `nest` because dimensions are coupled: the available values at position k depend on selections at positions 0..k-1. |
| `interval_split_space` | `split(choose(split_point), recurse, recurse)` | Binary recursive split | Root chooses a split point; left and right subproblems recurse. This is the canonical `split` constructor — the only one that creates recursive substructure. Matrix chain, optimal BST, interval DP. |
| `cartesian_space<A, B>` | `nest(A, B)` | Binary nest | Product of two spaces. `descriptor_product` is the descriptor-level equivalent. |
| `valid_view<Space, Pred>` | Constraint wrapper (not a constructor) | Filtered space | Applies a legality predicate. Does not change the constructor tree — it constrains which points in the existing tree are legal. Maps to invariant I7 (constraints at three levels). |
| `section_space<I, Space>` / `fix<I>` | Partial assignment (not a constructor) | Fixed subproblem | Fixes one dimension to a value, reducing rank by one. The constructor tree is unchanged; one `choose` node is resolved. Maps to the resolution process (a plan is a fully-resolved tree). |
| `filter_section<Space, Acc, Val>` | Constraint wrapper (not a constructor) | Value-filtered space | Like `valid_view` but restricted to equality on one accessor. No rank reduction. |
| `conditional_dim<Descriptor>` | Compile-time conditional `choose` | Property-gated dimension | When active: behaves as `choose(Descriptor)`. When inactive: resolved to default (leaf). The condition is evaluated before search, not during. |

## Constructor Coverage

| Constructor | Used by | Status |
|---|---|---|
| `leaf` | Every resolved dimension (a plan point) | Implicit — leaf is what a `choose` becomes after resolution |
| `seq` | `permutation_space` (dependent sequence) | ✅ Implemented |
| `nest` | `descriptor_space`, `per_element_space`, `cartesian_space` | ✅ Implemented |
| `choose` | Every dimension descriptor (`positive_int`, `power_2`, `int_set`, `bool_flag`, `enum_vals`, `ordinal`) | ✅ Implemented |
| `split` | `interval_split_space` (recursive binary partition) | ✅ Implemented |

All five constructors are exercised by existing space types.

## Constructors Not Yet Needed

| Constructor combination | Would be needed for | Status |
|---|---|---|
| `choose` with dependent children (tree_space) | Fusion grouping: root = `partition(N)`, children = per-group subspaces | Sprint 4 |
| `split` with non-binary fanout | Multi-way recursive decomposition | Not planned — binary split covers all current examples |
| `nest` with non-uniform depth | Ragged per-element spaces | Covered by `heterogeneous_per_element_space` |

## Worked Examples → Constructor Trees

### Example 1: FIX Parser (per-field strategy)
```
nest(
    choose(strategy_field_0),   // e.g., {copy, skip, validate, convert}
    choose(strategy_field_1),
    ...
    choose(strategy_field_11)
)
```
→ `heterogeneous_per_element_space` with 12 fields.

### Example 2: GEMM Tiling (tile sizes + cache constraint)
```
valid_view(
    nest(
        choose(TM: power_2),    // tile M
        choose(TN: power_2),    // tile N
        choose(TK: power_2)     // tile K
    ),
    L1_fits_predicate
)
```
→ `descriptor_space<power_2, power_2, power_2>` with `valid_view`.

### Example 3: SpMV Format Selection
```
nest(
    choose(format: enum_vals),          // CSR, COO, ELL, ...
    choose(block_size: int_set),        // format-specific
    choose(thread_count: positive_int)
)
```
→ `descriptor_space` with `valid_view` for format-specific parameter legality.

### Example 4: Tiling + Fusion + Reduction
```
nest(
    choose(tile_size: power_2),
    conditional choose(tree_shape: enum_vals),     // gated by all_associative
    conditional choose(vec_width: int_set),        // gated by all_have_identity
    choose(repro_level: ordinal)                   // constrains tree_shape further
)
```
→ `descriptor_space` with `conditional_dim` and `ordinal`. Future: `tree_space` root = `partition(N)` for lane fusion grouping.
