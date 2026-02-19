// examples/fix_cacheline.cpp — FIX parser cache-line layout optimisation
// Part of the compile-time DP framework (ctdp v0.6.0)
//
// Demonstrates consteval subset-DP applied to a realistic problem:
// a FIX protocol session struct where the hot path (message parsing)
// touches 6 of 13 fields.  The struct evolved over time:
//
//   v1: parse state at top of struct (hot)
//   v2: session management fields added in the middle (cold, 80+ bytes)
//   v3: extended parse state appended (hot)
//
// This natural evolution creates a hot-cold-hot sandwich across cache
// lines, splitting hot fields onto lines 0 and 2 with a cold-only
// line 1 in between.  The gap defeats the adjacent-line hardware
// prefetcher — each hot line costs a full cache miss.
//
// The DP packs all 24 bytes of hot data into a single cache line,
// halving the miss count.
//
// The DP engine is O(3^N).  N=13 gives ~1.6M iterations, well within
// GCC's default constexpr-ops budget.
//
// Build (standalone, no library dependency):
//   g++ -std=c++20 -O2 -march=native fix_cacheline.cpp -o fix_cacheline
//
// Design:
//   - Field descriptors: {bytes, align, temperature, thread_mask}
//   - Subset analysis:   feasibility + cost metrics per cache-line bin
//   - Policy:            pluggable cost model (hot-line penalty, mixing)
//   - DP:                O(3^N) subset partition, fully consteval
//   - Benchmark:         pointer-chase traversal reading full 64B per
//                        hot cache line (models true hardware fetch cost),
//                        cache thrash between reps
//
// v0.1  Generic 8-field demo with synthetic separators
// v0.2  FIX session (12 fields), declaration-order baseline, typed reads
// v0.3  Hot-cold-hot sandwich (13 fields), non-adjacent hot lines,
//       full-cacheline benchmark, working set sized to exceed L3

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════
// Field descriptors
// ═══════════════════════════════════════════════════════════════════════════

constexpr int CACHELINE = 64;

enum class Temp : uint8_t { Cold, Hot };

struct Field {
    uint8_t bytes;
    uint8_t align;          // power of 2
    Temp    temp;
    uint8_t thread_mask;    // 0 => ignore
};

consteval int align_up(int x, int a) {
    return (a <= 1) ? x : ((x + (a - 1)) / a) * a;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIX session state — the struct we're optimising
// ═══════════════════════════════════════════════════════════════════════════
//
// Declaration order (as the programmer wrote it over three releases):
//
//   Line 0:  BodyPtr[8] BodyLen[4] ParsePos[4]           ← HOT (v1)
//            SenderComp[32]                               ← cold (v2, fills to 48)
//   Line 1:  TargetComp[32] OnBehalfOf[16] SessionCfg[8] ← cold (v2, all 56B)
//   Line 2:  LookupPtrs[16] SeqIn[4] SeqOut[4]          ← cold
//            MsgType[2] FieldCount[2] Checksum[4]        ← HOT (v3)
//
// Hot fields on lines 0 and 2.  Cold-only line 1 sits between them,
// defeating the adjacent-line hardware prefetcher.  Every cache miss
// costs the full DRAM latency.
//
// DP packs all 24B of hot data onto one line → 1 miss instead of 2.

enum FIX : size_t {
    BodyPtr,        // 0    8B align=8  HOT   pointer to current message body
    BodyLen,        // 1    4B align=4  HOT   remaining bytes
    ParsePos,       // 2    4B align=4  HOT   current parse offset
    SenderComp,     // 3   32B align=1  cold  SenderCompID (up to 32 chars)
    TargetComp,     // 4   32B align=1  cold  TargetCompID
    OnBehalfOf,     // 5   16B align=1  cold  OnBehalfOfCompID (routing)
    SessionCfg,     // 6    8B align=4  cold  HeartBtInt(108) + MaxMessageSize
    LookupPtrs,     // 7   16B align=8  cold  tag index + field offset ptrs
    SeqNumIn,       // 8    4B align=4  cold  inbound MsgSeqNum(34)
    SeqNumOut,      // 9    4B align=4  cold  outbound MsgSeqNum(34)
    MsgType,        // 10   2B align=2  HOT   MsgType(35) enum
    FieldCount,     // 11   2B align=2  HOT   fields parsed this message
    Checksum,       // 12   4B align=4  HOT   running 3-digit FIX checksum
    NUM_FIELDS      // 13
};

constexpr size_t N = NUM_FIELDS;

constexpr std::array<Field, N> fields{{
    { 8, 8, Temp::Hot,  0},     // BodyPtr
    { 4, 4, Temp::Hot,  0},     // BodyLen
    { 4, 4, Temp::Hot,  0},     // ParsePos
    {32, 1, Temp::Cold, 0},     // SenderComp
    {32, 1, Temp::Cold, 0},     // TargetComp
    {16, 1, Temp::Cold, 0},     // OnBehalfOf
    { 8, 4, Temp::Cold, 0},     // SessionCfg
    {16, 8, Temp::Cold, 0},     // LookupPtrs
    { 4, 4, Temp::Cold, 0},     // SeqNumIn
    { 4, 4, Temp::Cold, 0},     // SeqNumOut
    { 2, 2, Temp::Hot,  0},     // MsgType
    { 2, 2, Temp::Hot,  0},     // FieldCount
    { 4, 4, Temp::Hot,  0},     // Checksum
}};

// ═══════════════════════════════════════════════════════════════════════════
// Subset analysis — cache-line feasibility and metrics
// ═══════════════════════════════════════════════════════════════════════════

struct LineStats {
    bool     feasible           = true;
    int      used               = 0;
    int      waste              = 0;
    bool     has_hot            = false;
    bool     has_cold           = false;
    int      distinct_threads   = 0;
    bool     has_thread_conflict = false;
    uint32_t mask               = 0;
};

// Deterministic field order within a cache line: align desc, bytes desc,
// index asc.  Same ordering for feasibility check and offset assignment.
template <size_t Nf>
consteval auto subset_order(const std::array<Field, Nf>& f, size_t subset) {
    std::array<uint8_t, Nf> idx{};
    size_t k = 0;
    for (size_t i = 0; i < Nf; ++i)
        if (subset & (1u << i)) idx[k++] = (uint8_t)i;

    // selection sort (consteval-safe)
    for (size_t i = 0; i < k; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < k; ++j) {
            auto a = idx[best], b = idx[j];
            if (f[b].align > f[a].align)                                    best = j;
            else if (f[b].align == f[a].align && f[b].bytes > f[a].bytes)   best = j;
            else if (f[b].align == f[a].align && f[b].bytes == f[a].bytes
                     && b < a)                                              best = j;
        }
        auto tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
    }
    struct Order { std::array<uint8_t, Nf> v; size_t count; };
    return Order{idx, k};
}

template <size_t Nf>
consteval LineStats analyze_subset(const std::array<Field, Nf>& f, size_t subset) {
    LineStats s{};
    s.mask = (uint32_t)subset;
    auto ord = subset_order(f, subset);

    int pos = 0;
    uint8_t threads_union = 0;

    for (size_t t = 0; t < ord.count; ++t) {
        size_t i = ord.v[t];
        pos = align_up(pos, f[i].align);
        pos += f[i].bytes;
        if (pos > CACHELINE) { s.feasible = false; return s; }

        s.has_hot  |= (f[i].temp == Temp::Hot);
        s.has_cold |= (f[i].temp == Temp::Cold);
        if (f[i].thread_mask) threads_union |= f[i].thread_mask;
    }

    s.used  = pos;
    s.waste = CACHELINE - pos;

    int dt = 0;
    for (uint8_t m = threads_union; m; m &= (m - 1)) ++dt;
    s.distinct_threads    = dt;
    s.has_thread_conflict = (dt > 1);
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// Policy — pluggable cost model
// ═══════════════════════════════════════════════════════════════════════════

struct Policy {
    static consteval int hot_line_cost()     { return 10'000; }
    static consteval int any_line_overhead() { return 100; }
    static consteval int waste_per_byte()    { return 1; }
    static consteval int hot_cold_mix()      { return 1'000'000; }
    static consteval int thread_conflict()   { return 500; }

    static consteval int score(const LineStats& s) {
        if (!s.feasible) return 1'000'000'000;
        if (s.mask == 0) return 0;

        int cost = any_line_overhead();
        if (s.has_hot)                  cost += hot_line_cost();
        if (s.has_hot && s.has_cold)    cost += hot_cold_mix();
        if (s.has_thread_conflict)      cost += thread_conflict();
        cost += s.waste * waste_per_byte();
        return cost;
    }
};

template <size_t Nf>
consteval auto precompute_subset_costs(const std::array<Field, Nf>& f) {
    constexpr size_t FULL = (1u << Nf);
    std::array<int, FULL> cost{};
    for (size_t s = 0; s < FULL; ++s)
        cost[s] = Policy::score(analyze_subset(f, s));
    return cost;
}

// ═══════════════════════════════════════════════════════════════════════════
// Plan — the DP output
// ═══════════════════════════════════════════════════════════════════════════

template <size_t Nf>
struct Plan {
    int      total_cost{};
    size_t   line_count{};
    std::array<uint8_t, Nf> line_of{};
    std::array<uint8_t, Nf> offset_of{};
    std::array<size_t,  Nf> bins{};
};

// ═══════════════════════════════════════════════════════════════════════════
// DP solver — O(3^N) subset partition
// ═══════════════════════════════════════════════════════════════════════════

template <size_t Nf>
consteval Plan<Nf> optimal_partition_dp(const std::array<Field, Nf>& f) {
    static_assert(Nf <= 16);
    constexpr size_t FULL = (1u << Nf) - 1u;
    constexpr int INF = 1'000'000'000;

    auto cost = precompute_subset_costs(f);

    std::array<int,    1u << Nf> dp{};
    std::array<size_t, 1u << Nf> parent{};
    for (auto& x : dp) x = INF;
    dp[0] = 0;

    for (size_t mask = 0; mask <= FULL; ++mask) {
        if (dp[mask] == INF) continue;
        size_t rem = FULL ^ mask;

        for (size_t s = rem; s; s = (s - 1) & rem) {
            int c = cost[s];
            if (c >= INF) continue;
            int next = dp[mask] + c;
            if (next < dp[mask | s]) {
                dp[mask | s] = next;
                parent[mask | s] = s;
            }
        }
    }

    Plan<Nf> p{};
    p.total_cost = dp[FULL];
    size_t nlines = 0;
    for (size_t m = FULL; m; ) {
        size_t s = parent[m];
        p.bins[nlines++] = s;
        m ^= s;
    }
    p.line_count = nlines;

    for (size_t li = 0; li < nlines; ++li) {
        auto ord = subset_order(f, p.bins[li]);
        int pos = 0;
        for (size_t t = 0; t < ord.count; ++t) {
            size_t i = ord.v[t];
            pos = align_up(pos, f[i].align);
            p.line_of[i]   = (uint8_t)li;
            p.offset_of[i] = (uint8_t)pos;
            pos += f[i].bytes;
        }
    }
    return p;
}

// ═══════════════════════════════════════════════════════════════════════════
// Declaration-order baseline — what the naive struct gives you
// ═══════════════════════════════════════════════════════════════════════════

template <size_t Nf>
consteval Plan<Nf> declaration_order_plan(const std::array<Field, Nf>& f) {
    Plan<Nf> p{};
    int line = 0;
    int pos  = 0;

    for (size_t i = 0; i < Nf; ++i) {
        int aligned_pos = align_up(pos, f[i].align);
        if (aligned_pos + f[i].bytes > CACHELINE) {
            ++line;
            pos = 0;
            aligned_pos = align_up(0, f[i].align);
        }
        p.line_of[i]   = (uint8_t)line;
        p.offset_of[i] = (uint8_t)aligned_pos;
        pos = aligned_pos + f[i].bytes;
    }
    p.line_count = (size_t)(line + 1);

    for (size_t i = 0; i < Nf; ++i)
        p.bins[p.line_of[i]] |= (1u << i);

    int total = 0;
    for (size_t li = 0; li < p.line_count; ++li)
        total += Policy::score(analyze_subset(f, p.bins[li]));
    p.total_cost = total;
    return p;
}

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time verification
// ═══════════════════════════════════════════════════════════════════════════

constexpr auto optimal = optimal_partition_dp(fields);
constexpr auto naive   = declaration_order_plan(fields);

// Declaration order: 3 cache lines, hot on lines 0 and 2
static_assert(naive.line_count == 3);
static_assert(naive.line_of[BodyPtr]    == 0);  // hot on line 0
static_assert(naive.line_of[BodyLen]    == 0);
static_assert(naive.line_of[ParsePos]   == 0);
static_assert(naive.line_of[MsgType]    == 2);  // hot on line 2
static_assert(naive.line_of[FieldCount] == 2);
static_assert(naive.line_of[Checksum]   == 2);
// Line 1 is cold-only (TargetComp + OnBehalfOf + SessionCfg)
static_assert(naive.line_of[TargetComp] == 1);
static_assert(naive.line_of[OnBehalfOf] == 1);
static_assert(naive.line_of[SessionCfg] == 1);

// DP optimal: all 6 hot fields on one line, no hot/cold mixing
static_assert(optimal.line_of[BodyPtr]    == optimal.line_of[BodyLen]);
static_assert(optimal.line_of[BodyLen]    == optimal.line_of[ParsePos]);
static_assert(optimal.line_of[ParsePos]   == optimal.line_of[MsgType]);
static_assert(optimal.line_of[MsgType]    == optimal.line_of[FieldCount]);
static_assert(optimal.line_of[FieldCount] == optimal.line_of[Checksum]);

// No cold field shares the hot line
static_assert(optimal.line_of[BodyPtr] != optimal.line_of[SenderComp]);
static_assert(optimal.line_of[BodyPtr] != optimal.line_of[TargetComp]);
static_assert(optimal.line_of[BodyPtr] != optimal.line_of[LookupPtrs]);

// Optimal strictly beats naive
static_assert(optimal.total_cost < naive.total_cost);

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark infrastructure
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<uint32_t> make_pointer_chase(size_t n) {
    std::vector<uint32_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = (uint32_t)i;

    uint32_t x = 0x12345678u;
    for (size_t i = n; i > 1; --i) {
        x = x * 1664525u + 1013904223u;
        size_t j = (size_t)(x % (uint32_t)i);
        std::swap(perm[i - 1], perm[j]);
    }

    std::vector<uint32_t> next(n);
    for (size_t i = 0; i + 1 < n; ++i) next[perm[i]] = perm[i + 1];
    next[perm[n - 1]] = perm[0];
    return next;
}

static void thrash_cache(std::vector<uint8_t>& thrash) {
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < thrash.size(); i += 64) sink += thrash[i];
    (void)sink;
}

static inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static auto aligned_blob_64(size_t bytes) {
    auto p = std::unique_ptr<std::byte, decltype(&std::free)>(
        static_cast<std::byte*>(std::aligned_alloc(64, bytes)),
        &std::free);
    if (!p) std::abort();
    return p;
}

// Bitmask of which cache lines contain at least one hot field.
template <size_t Nf>
uint32_t hot_line_mask(const std::array<Field, Nf>& f, const Plan<Nf>& p) {
    uint32_t m = 0;
    for (size_t i = 0; i < Nf; ++i)
        if (f[i].temp == Temp::Hot) m |= (1u << p.line_of[i]);
    return m;
}

template <size_t Nf>
int count_hot_lines(const std::array<Field, Nf>& f, const Plan<Nf>& p) {
    return __builtin_popcount(hot_line_mask(f, p));
}

// ═══════════════════════════════════════════════════════════════════════════
// FIX hot-path benchmark
// ═══════════════════════════════════════════════════════════════════════════
//
// Models the per-message decode loop by reading every byte of each cache
// line that contains hot fields.  This matches hardware reality: a load
// from *any* byte in a cache line fetches the full 64B from memory.
// The cost model counts hot cache lines; the benchmark pays for them.
//
// Pointer-chase traversal defeats prefetching; cache thrash between
// reps forces cold-start misses on every iteration.

struct BenchConfig {
    size_t records;
    int    passes;
    int    reps;
    size_t thrash_bytes;
};

template <size_t Nf>
double bench_plan(const char* name,
                  const std::array<Field, Nf>& f,
                  const Plan<Nf>& p,
                  size_t stride,
                  const BenchConfig& cfg)
{
    const size_t bytes = cfg.records * stride;
    auto blob_ptr = aligned_blob_64(bytes);
    std::byte* blob = blob_ptr.get();

    // Fill deterministically
    uint32_t x = 123456789u;
    for (size_t i = 0; i < bytes; ++i) {
        x = x * 1664525u + 1013904223u;
        blob[i] = std::byte((uint8_t)(x >> 24));
    }

    auto next = make_pointer_chase(cfg.records);

    std::vector<uint8_t> thrash(cfg.thrash_bytes);
    for (auto& b : thrash) { x = x * 1664525u + 1013904223u; b = (uint8_t)(x >> 24); }

    const uint32_t hmask = hot_line_mask(f, p);

    volatile uint64_t sink = 0;

    auto run_once = [&] {
        auto t0 = std::chrono::steady_clock::now();
        size_t r = 0;
        for (int pass = 0; pass < cfg.passes; ++pass) {
            for (size_t step = 0; step < cfg.records; ++step) {
                const std::byte* rec = blob + r * stride;

                // Read every byte of each hot cache line.
                // Hardware fetches the full 64B regardless of which
                // fields you actually access — a single load anywhere
                // in a cache line pays for the entire line.
                for (uint32_t m = hmask; m; m &= (m - 1)) {
                    int li = __builtin_ctz(m & (~m + 1u));
                    const std::byte* line = rec + (size_t)li * CACHELINE;
                    for (int j = 0; j < CACHELINE; ++j)
                        sink += (uint8_t)line[j];
                }

                r = (size_t)next[r];
            }
        }
        return seconds_since(t0);
    };

    double best = 1e100;
    for (int i = 0; i < cfg.reps; ++i) {
        thrash_cache(thrash);
        best = std::min(best, run_once());
    }

    int hot = count_hot_lines(f, p);
    double ns_rec = best / (double)((size_t)cfg.passes * cfg.records) * 1e9;
    std::cout << name
              << ": hot_lines=" << hot
              << "  best=" << best << " s"
              << "  (" << ns_rec << " ns/record)"
              << "  (sink=" << sink << ")\n";
    return best;
}

// ═══════════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════════

static const char* field_names[] = {
    "BodyPtr     ", "BodyLen     ", "ParsePos    ",
    "SenderComp  ", "TargetComp  ", "OnBehalfOf  ",
    "SessionCfg  ", "LookupPtrs  ",
    "SeqNumIn    ", "SeqNumOut   ",
    "MsgType     ", "FieldCount  ", "Checksum    "
};

static void dump_plan(const char* title, const Plan<N>& p) {
    std::cout << title << " (cost=" << p.total_cost
              << "  lines=" << p.line_count << ")\n";
    for (size_t i = 0; i < N; ++i) {
        int abs_off = p.line_of[i] * CACHELINE + p.offset_of[i];
        std::cout << "  " << field_names[i]
                  << "  line " << (int)p.line_of[i]
                  << " +" << (int)p.offset_of[i]
                  << "  (abs " << abs_off << ")"
                  << (fields[i].temp == Temp::Hot ? "  HOT" : "")
                  << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== FIX Session Cache-Line Optimiser ===\n\n";

    std::cout << "FIX session struct: " << N << " fields (6 hot, 7 cold)\n";
    std::cout << "Hot payload: 8+4+4+2+2+4 = 24 bytes  (fits 1 cache line)\n";
    std::cout << "Total payload: 136 bytes  (3 cache lines in naive layout)\n\n";

    dump_plan("Declaration order (naive struct)", naive);
    dump_plan("DP-optimal partition",             optimal);

    std::cout << "Declaration order: hot on lines 0 and 2  (cold-only line 1 between)\n";
    std::cout << "  → " << count_hot_lines(fields, naive) << " hot cache lines,"
              << " adjacent-line prefetcher cannot help\n";
    std::cout << "DP-optimal:        all hot fields on 1 cache line\n";
    std::cout << "  → " << count_hot_lines(fields, optimal)
              << " hot cache line, 50% fewer cache misses\n\n";

    // In practice, FIX session state lives inside a larger connection
    // object (socket buffers, TLS context, order book refs).  Pad stride
    // to 5 cache lines to model this embedding and push the working set
    // well past L3 cache.
    constexpr size_t MIN_STRIDE_LINES = 5;
    const size_t stride_lines = []() {
        size_t m = naive.line_count;
        if (optimal.line_count > m) m = optimal.line_count;
        if (MIN_STRIDE_LINES > m)   m = MIN_STRIDE_LINES;
        return m;
    }();
    const size_t stride = stride_lines * (size_t)CACHELINE;

    const BenchConfig cfg{
        .records      = 200'000,                // working set >> L3
        .passes       = 2,
        .reps         = 5,
        .thrash_bytes = 16ull * 1024 * 1024,    // 16 MiB
    };

    const double ws_mb = (double)(cfg.records * stride) / (1024.0 * 1024.0);
    std::cout << "stride=" << stride << " B"
              << "  records=" << cfg.records
              << "  working_set=" << ws_mb << " MiB"
              << "  passes=" << cfg.passes
              << "  reps=" << cfg.reps
              << "\n\n";

    double t_naive   = bench_plan("naive  ", fields, naive,   stride, cfg);
    double t_optimal = bench_plan("optimal", fields, optimal, stride, cfg);

    std::cout << "\nspeedup: " << (t_naive / t_optimal) << "x\n";
}
