# Code Commentary — fix_parser_demo.cpp

## What is the experiment?

The experiment answers a specific question: given a FIX protocol message with 12 numeric fields, each of which could be parsed by any of four different integer conversion algorithms, what is the performance impact of choosing the right algorithm for each field, and how much does the mechanism by which you compose those choices matter?

The demo compares four parser configurations on the same message type. The first three use expression templates — meaning the full parse chain is a single compiled function — and differ only in which algorithm is assigned to which field. The fourth uses runtime dispatch — meaning each field's parser is called through a separate noinline function — to isolate the cost of the dispatch mechanism itself.

The key finding is that the composition mechanism (expression templates vs runtime dispatch) dominates the algorithm choice (which strategy per field). On the reference system for this demo, eliminating dispatch saved 25–30 ns per message — comparable to the entire parsing workload. The absolute numbers are hardware-specific, but the structural dominance of dispatch cost over per-field strategy choice is the consistent finding.

This demo is a companion to the article "From Struct Layout to Tail Latency" and provides reproducible evidence for the decomposition described there. Everything compiles as a single translation unit with no dependencies beyond the C++20 standard library.

---

## Part 1: Representing the FIX message schema (lines 29–76)

### What FIX looks like on the wire

A FIX message is a flat sequence of tag=value pairs delimited by SOH (ASCII 0x01). A real wire message looks like:

```
8=FIX.4.4\x019=1234\x0135=X\x0134=5678901\x01...
```

Every field — including prices, quantities, timestamps, and sequence numbers — is encoded as ASCII text. The tags, `=` separator, and SOH delimiter provide framing, but there is no separate binary length or type header per field value. Each value is a variable-length ASCII string that the receiver must parse character by character.

### The FieldDesc schema descriptor

```cpp
struct FieldDesc {
    uint16_t    tag;        // FIX tag number (e.g. 34 = MsgSeqNum)
    const char* name;       // human-readable name
    uint8_t     digits;     // expected digit count for numeric fields, 0 for strings
    bool        numeric;    // whether this field requires integer conversion
};
```

This is the compile-time knowledge that the FIX data dictionary gives us. For a given message type (here, MarketDataIncrementalRefresh / MsgType=X), we know exactly which fields appear, in what order, and for numeric fields, how many digits they will contain. The `digits` field is the crucial piece — it lets us parameterise the parsers as templates on a known constant.

### The field diversity

The 22-field schema deliberately includes a wide range of numeric field sizes:

| Digits | Fields | Character |
|--------|--------|-----------|
| 2 | NoMDEntries | Group counter |
| 3 | CheckSum, MDEntryPosNo | Small integers |
| 4 | BodyLength, NumberOfOrders | Medium integers |
| 6 | MDEntrySize | Order size |
| 7 | MsgSeqNum | Sequence counter |
| 8 | SecurityID, MDEntryDate | Identifiers, dates |
| 10 | MDEntryPx | Fixed-point price |
| 17 | SendingTime, MDEntryTime | Timestamps (YYYYMMDDHHMMSS.mmm) |

This range — from 2 to 17 digits — is what makes the strategy selection problem non-trivial. If every field were 17 digits, the SWAR strategy would win everywhere and there would be nothing to optimise. But the presence of 2-digit and 3-digit fields creates crossover points where simpler strategies beat SWAR's setup overhead.

The 10 non-numeric fields (BeginString, MsgType, Symbol, etc.) are present in the schema and in the wire message but are assigned the Skip strategy — the parsers simply ignore them. They contribute to the realism of the wire format and the field-location cost, but not to the parse timing.

### NUM_IDX: the numeric field index map

```cpp
static constexpr std::array<int, NUM_NUMERIC> NUM_IDX = {
    1, 3, 4, 6, 8, 9, 12, 13, 14, 15, 16, 17
};
```

This maps the 12 numeric fields back to their positions in the 22-element FIELDS array. It exists because the Plan type assigns a strategy to every field position (including non-numeric ones, which get Skip), but we only care about the numeric assignments when printing or analysing configurations.

---

## Part 2: Wire message generation and field location (lines 78–134)

### WireMessage: the raw buffer

```cpp
struct WireMessage {
    char data[512];          // raw ASCII wire data
    int  len;                // total byte length
    struct FieldLoc {
        const char* value_ptr;  // pointer into data[]
        uint8_t value_len;      // byte length of value
    };
    FieldLoc locs[NF];       // pre-parsed field pointers
};
```

This is a deliberately simple representation. The `data` buffer holds the raw wire bytes exactly as they would arrive from a network socket. The `locs` array holds pre-parsed pointers into that buffer — for each of the 22 field positions, we store a pointer to the start of its value and its length.

The `locs` array is what the parsers actually operate on. By the time any conversion strategy runs, the field location work is already done. This is important for the benchmark: field location is shared cost, identical across all configurations, and is excluded from the timing.

### generate(): realistic wire content

The generator writes fields in schema order using two lambdas: `wf` for string fields (writes a fixed literal) and `wn` for numeric fields (writes random digits of the correct count). The first digit is always non-zero (no leading zeros), and subsequent digits are uniformly random. This prevents constant folding — the compiler cannot predict the values and optimise away the parse work.

The fixed random seed (`std::mt19937 rng(42)`) ensures reproducibility. Every run of the demo generates identical wire content.

### locate(): field pointer extraction

The locate function walks the wire buffer once, parsing tag numbers and recording value pointers. This is a linear scan — O(message length) — and is conceptually similar to what a real FIX engine does as a first pass before numeric conversion. (In practice, many engines intertwine field location and conversion in a single pass, but the conceptual separation is how people reason about it, and it lets us benchmark conversion cost in isolation.)

The tag-to-index lookup (`ti` lambda) is a linear search over 22 entries. In production you would use a perfect hash or a direct lookup table, but for 22 fields the cost is negligible and the clarity is worth it.

---

## Part 3: The four conversion strategies (lines 136–254)

This is the core design. All four strategies share the same interface — `static int64_t parse(const char* p)` — and are class templates parameterised on the digit count D. This common interface is what makes them interchangeable within the expression template framework.

### Naming the parsers

Before we dive into the four parsers, it's worth mapping them onto names you may already know. **LoopParser** is a straight Horner loop (`r = r*10 + digit`) over a fixed number of digits. **UnrolledParser** is the "small-D" case: fully unrolled kernels for very short, fixed-width integers where any loop overhead dominates. **SWARParser** is a SWAR-inspired, 4-digit block parser in the spirit of the "parse eight digits at once" tricks popularised in fast integer parsers. **GenericParser** is the validated baseline: a branchy, bounds-checked parser you might deploy on cold or untrusted paths, not on the hot FIX feed.

A note on the SWAR name: strictly speaking, classic SWAR means packed bit-parallel operations on sub-word fields within a register. Our SWARParser doesn't do that — it processes digits in arithmetic groups of four using ordinary scalar multiply-accumulate. The "four digits at a time" throughput advantage is real, but it comes from reducing multiply-accumulate chain steps, not from sub-word parallelism. We keep the name because it's established in the codebase and in the broader fast-parsing literature where "SWAR-inspired" block techniques sit on the same family tree as the true bit-parallel versions. True SWAR digit parsing — loading 8 bytes, subtracting '0' in parallel, combining with packed multiply — would be a fifth strategy the framework could accommodate.

### Strategy diversity and why it matters

The four strategies are not minor variants of each other. They produce fundamentally different machine code:

**UnrolledParser** uses `if constexpr` to select digit-count-specific expressions. For D=3, the compiler emits three loads, two multiplies, and two adds — no loop, no branch, no register spill. For larger D, it uses a constexpr power-of-ten lookup table. The table has 18 entries because `10^17` is the largest power of ten that fits in a signed 64-bit integer (10^18 = 1,000,000,000,000,000,000 exceeds `INT64_MAX` = 9,223,372,036,854,775,807).

**SWARParser** processes four digits per iteration. The block count `B = D/4` and tail size `T = D%4` are both constexpr, so the compiler unrolls the block loop completely and selects only the relevant tail branch. For SendingTime (17 digits): 4 full blocks plus 1 tail digit. The key is that `r = r * 10000 + blk` shifts the accumulator by four decimal places at once — roughly 4× fewer dependent multiply operations in the critical path compared to the Horner loop.

**LoopParser** is Horner's method: `r = r*10 + digit`. The trip count is constexpr, so the compiler can unroll and vectorise as it sees fit. At small digit counts (D=2), the compiler's own unrolling produces tighter code than SWAR's block+tail structure.

**GenericParser** adds per-character validation and is marked `NO_INLINE`. These two attributes create optimisation barriers that go beyond the validation cost itself. The noinline attribute prevents the compiler from seeing adjacent field parsers simultaneously — it can't interleave instructions across the call boundary, can't share registers across fields, and can't coalesce adjacent stores to the output struct. The `if (d > 9)` branch prevents vectorisation of the inner loop. Together, these barriers cost more than the actual validation work.

### The FORCE_INLINE / NO_INLINE contract

The inline attributes are not hints — they're the mechanism that makes the expression template architecture work. FORCE_INLINE on the trusted strategies means the compiler must inline them at every call site in the recursive expansion. NO_INLINE on GenericParser means it must remain a separate function. (The C++ standard does not guarantee inlining behaviour, but on the mainstream compilers we target — GCC, Clang, MSVC at -O3 / /O2 — `__attribute__((always_inline))` and `__forceinline` behave as intended.)

This creates two sharply different compilation models. With FORCE_INLINE, the compiler sees 12 field parsers as one flat sequence of arithmetic inside a single function body. With NO_INLINE, it sees 12 separate function calls, each opaque.

---

## Part 4: The Plan type and expression template dispatch (lines 256–325)

This is the central architectural contribution. The Plan type encodes a complete strategy assignment as a compile-time type, and the expression template machinery uses that type to generate a fully-inlined, zero-dispatch parse function.

### Plan<Strategy... Ss>

```cpp
template<Strategy... Ss>
struct Plan {
    static constexpr std::array<Strategy, sizeof...(Ss)> strategies = {Ss...};
    static constexpr Strategy get(size_t i) { return strategies[i]; }
};
```

A Plan is a variadic template. `Plan<SWAR, SWAR, Loop, ...>` is a distinct type from `Plan<SWAR, SWAR, SWAR, ...>`. This is the key property: different strategy assignments produce different types, and different types produce different compiled code.

The Plan has 22 strategy entries — one per field position, including Skip for non-numeric fields. This means the Plan fully describes the parse behaviour for the entire message. Nothing is deferred to runtime.

### FieldParser<Strategy, D>: the compile-time dispatch table

```cpp
template<Strategy S, int D> struct FieldParser;
template<int D> struct FieldParser<Strategy::Unrolled, D> : UnrolledParser<D> {};
template<int D> struct FieldParser<Strategy::SWAR, D>   : SWARParser<D> {};
template<int D> struct FieldParser<Strategy::Loop, D>     : LoopParser<D> {};
template<int D> struct FieldParser<Strategy::Generic, D>  : GenericParser<D> {};
template<int D> struct FieldParser<Strategy::Skip, D>     : SkipParser {};
```

This is a compile-time two-dimensional dispatch table. Given a strategy enum value and a digit count, it selects the correct parser struct via template specialisation. There is no switch, no function pointer, no vtable. The mapping is resolved entirely at compile time and costs nothing at runtime.

This is also where new strategies would be added. If you implemented a true SWAR bit-parallel parser, a SIMD parser using SSE/AVX intrinsics, or a lookup-table-based parser, you would add a new enum value, a new parser struct template, and a new FieldParser specialisation. The rest of the framework — Plan, ParseFieldWire, MessageParser, the benchmark harness — would require no changes.

### ParseFieldWire<PlanType, I>: the recursive expansion

```cpp
template<typename PlanType, int I = 0>
struct ParseFieldWire {
    static FORCE_INLINE void run(const WireMessage& msg, ParseResult& out) {
        constexpr Strategy S = PlanType::get(I);
        constexpr int D = FIELDS[I].digits;
        if constexpr (D > 0 && S != Strategy::Skip) {
            if (msg.locs[I].value_ptr)
                out.values[I] = FieldParser<S, D>::parse(msg.locs[I].value_ptr);
        }
        if constexpr (I + 1 < NF)
            ParseFieldWire<PlanType, I + 1>::run(msg, out);
    }
};
```

This is the expression template expansion. It recursively instantiates itself for field positions I = 0 through I = 21. At each position:

1. It reads the strategy from the Plan at compile time (`constexpr Strategy S`).
2. It reads the digit count from the schema at compile time (`constexpr int D`).
3. If the field is numeric and not Skip, it calls the appropriate parser through the FieldParser dispatch table.
4. It recurses to the next field position.

Because everything except the actual pointer dereference and arithmetic is constexpr, and because every parser is FORCE_INLINE, the compiler collapses this entire recursive chain into a single flat sequence of instructions. The recursion, the dispatch, the `if constexpr` branches — all of this vanishes. What remains is a straight-line function that loads each field's bytes, performs the appropriate arithmetic, and stores the result.

This is what we mean by "the plan is a type." `Plan<SWAR, SWAR, Loop, ...>` isn't a configuration that's interpreted at runtime. It's a type that produces a unique compiled function. Changing one field's strategy changes the type, which changes the compiled function, which changes the register allocation and instruction scheduling for all fields.

### MessageParser<PlanType>: the measurement boundary

```cpp
template<typename PlanType>
struct MessageParser {
    static NO_INLINE void parse(const WireMessage& msg, ParseResult& out) {
        ParseFieldWire<PlanType, 0>::run(msg, out);
    }
};
```

The outer MessageParser is deliberately NO_INLINE. This creates a single, well-defined function that we can benchmark without the timing call itself being optimised away or merged with adjacent code. Inside this boundary, everything is fully inlined. Outside, the benchmark harness calls it as an opaque function.

---

## Part 5: Three concrete plans (lines 327–415)

The three expression template plans are spelled out in full, with a Strategy enum value for every one of the 22 field positions. This verbosity is deliberate — it makes the assignment for each field visible and self-documenting.

**PlanGeneric** assigns Generic to every numeric field and Skip to every string field. This is the baseline: the most conservative, fully-validated parser with full optimisation barriers.

**PlanExpert** assigns SWAR to every numeric field. This represents an informed but untuned choice — an experienced developer who knows that block-wise accumulation is generally fast would reasonably apply it across the board without measuring per-field crossover points. It is analogous to deploying a specialised per-field parser (like Tonetti's hm::atoi) uniformly, rather than calibrating the choice per field. Note: this is an "expert guess," not the measured per-field optimum.

**PlanDPOptimal** assigns the strategy selected by the calibrated cost model. It differs from PlanExpert on exactly 3 of 12 numeric fields:

- NoMDEntries (2d): Loop instead of SWAR — the compiler's unrolling of a 2-iteration Horner loop beats SWAR's block+tail structure.
- CheckSum (3d): Unrolled instead of SWAR — three scalar instructions beat SWAR's zero-block + 3-digit-tail path.
- MDEntryPosNo (3d): Unrolled instead of SWAR — same reasoning as CheckSum.

The DP agrees with the expert on 9 of 12 fields and disagrees on the three smallest. This is the expected pattern: block strategies dominate at higher digit counts, but simpler strategies win where the blocking overhead isn't amortised.

---

## Part 6: The benchmark harness (lines 417–457)

### Measurement protocol

The benchmark uses a 3-level nesting: 15 trials, each consisting of 40 repetitions of the full 1000-message batch. This structure provides both statistical robustness and the ability to take the median across trials, which is resistant to outliers from OS scheduling and thermal effects.

**Warmup** (5000 iterations): brings the instruction cache and branch predictors to steady state before timing begins.

**Volatile sink** (`g_sink`): prevents the compiler from eliminating the parse work as dead code. After each message parse, we write one output value through a volatile pointer. This forces the compiler to actually compute the result.

**Cache residency** (1000 messages × ~222 bytes ≈ 222 KB): the full working set fits comfortably in L2 cache (typically 256 KB–1 MB on modern x86) though it exceeds typical L1D sizes (32–64 KB). In practice, each batch of messages streams through L1 sequentially, so the hot path is always L1-resident on a per-message basis. We are measuring parse throughput in a cache-friendly regime, not main-memory latency. This is appropriate for the inner loop of a market data engine where messages arrive sequentially and the active message is always hot.

**Median reporting**: of the 15 trials, we report the median (index 7). This is more robust than mean because it discards the effect of occasional high outliers from context switches or interrupt handling.

---

## Part 6b: Runtime-dispatched expert parser (lines 459–515)

### Why this exists

The three expression template plans (Generic, Expert, DPOptimal) all benefit from zero-dispatch composition. To measure how much of the speedup comes from the strategy choice versus the composition mechanism, we need a version that uses the same algorithm (SWAR) but dispatches through a runtime call boundary.

### rt::parse_swar(): the noinline runtime version

This is functionally identical to SWARParser<D>::parse() but takes the digit count as a runtime parameter (`int len`) rather than a template parameter. The key differences:

1. `NO_INLINE` forces a call boundary — the compiler cannot inline this into the caller.
2. `len` is a runtime value, so the compiler cannot unroll the block loop or eliminate the tail branch at compile time. It must emit code for all tail cases.
3. The caller (`parse_expert_rt`) loops over the 12 numeric fields and calls `parse_swar` for each one — 12 separate function calls per message.

This is analogous to how a specialised per-field parser is often deployed in practice: the best algorithm, but behind a function-call boundary for each field. The 25–30 ns gap between Expert(RT) and Expert(ET) on our reference system is attributable to this dispatch mechanism — all other variables (algorithm, data, field count) are held constant by construction.

---

## Part 7: The output and what to look for (lines 517–610)

The demo prints three sections:

**Plan assignments** shows the full strategy map for each configuration. Look for the SSSLSSSSSUSU signature of DPOptimal — the three deviations from S-everywhere are the DP's contribution.

**DP deviations** explicitly lists where DPOptimal differs from Expert and why. This makes the DP's reasoning visible.

**Decomposition** breaks the total speedup into three additive components:
- Specialisation alone (Generic ET → Expert RT)
- Dispatch elimination (Expert RT → Expert ET)
- DP tuning (Expert ET → DPOptimal ET)

The ratios will vary across hardware, but we expect the structural finding — that dispatch elimination dominates per-field strategy choice — to hold across modern x86 systems. Different microarchitectures may shift the relative contributions, but the noinline call-boundary cost is fundamental to how compilers handle function boundaries.

---

## Extending the demo

The demo is designed to be extended. Some directions:

**Adding a fifth strategy.** Implement a new parser struct template (e.g., `SimdParser<D>` using SSE2 intrinsics), add a Strategy enum value, add a FieldParser specialisation, and create a new Plan type that uses it for selected fields. The framework imposes no limit on the number of strategies.

**Changing the message type.** Modify the FIELDS array and the generate() function to match a different FIX message type (e.g., NewOrderSingle, ExecutionReport). The digit counts will change, which may shift the DP's optimal assignment.

**Measuring tail distributions.** Replace the median-of-trials reporting with full histogram collection. Run each configuration for many thousands of iterations, record every individual parse time, and compare p50/p99/p99.9 across configurations. This is the entry point to the Pareto frontier problem described in the article.

**Implementing the actual DP.** The demo hardcodes three plans. The full study includes a calibration loop that measures per-field costs, a DP that computes the optimal assignment, and a code generator that emits the Plan<...> type. These are separate phases that could be added as additional parts of the demo.
