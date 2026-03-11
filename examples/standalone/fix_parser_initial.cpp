// ============================================================================
// FIX Parser Compile-Time DP Demo
//
// Demonstrates systematic optimisation of integer conversion strategies
// across a FIX MarketDataIncrementalRefresh message (22 fields, 12 numeric).
//
// Three parsers compared:
//   1. Generic    -- std::from_chars equivalent (per-char validation, noinline)
//   2. Expert     -- hand-tuned switch a la Lima & Tonetti (runtime dispatch)
//   3. DPOptimal  -- compile-time DP plan via expression templates (zero dispatch)
//
// Compile:
//   g++ -std=c++20 -O3 -march=native -o fix_parser_demo fix_parser_demo.cpp
//   cl /std:c++20 /O2 /arch:AVX2 fix_parser_demo.cpp  (MSVC)
//
// Andrew Drakeford, 2026
// ============================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

// ============================================================================
// PART 1: FIX MESSAGE SCHEMA
// ============================================================================
// A FIX message is a sequence of "tag=value\x01" fields. For a given message
// type the tags, their order, and the digit counts of numeric fields are all
// known at compile time from the FIX data dictionary.

static constexpr char SOH = '\x01';  // FIX field delimiter

struct FieldDesc {
    uint16_t    tag;
    const char* name;
    uint8_t     digits;    // 0 = non-numeric (string/char field)
    bool        numeric;
};

static constexpr int NF = 22;  // total fields in our message type

static constexpr std::array<FieldDesc, NF> FIELDS = {{
    {8,   "BeginString",     0, false},
    {9,   "BodyLength",      4, true },
    {35,  "MsgType",         0, false},
    {34,  "MsgSeqNum",       7, true },
    {52,  "SendingTime",    17, true },   // YYYYMMDDHHMMSS.mmm as integer
    {262, "MDReqID",         0, false},
    {268, "NoMDEntries",     2, true },
    {269, "MDEntryType",     0, false},
    {270, "MDEntryPx",      10, true },   // price as fixed-point integer
    {271, "MDEntrySize",     6, true },
    {55,  "Symbol",          0, false},
    {279, "MDUpdateAction",  0, false},
    {48,  "SecurityID",      8, true },
    {273, "MDEntryTime",    17, true },
    {272, "MDEntryDate",     8, true },
    {10,  "CheckSum",        3, true },
    {346, "NumberOfOrders",  4, true },
    {290, "MDEntryPosNo",    3, true },
    {336, "TradingSession",  0, false},
    {49,  "SenderCompID",    0, false},
    {56,  "TargetCompID",    0, false},
    {43,  "PossDupFlag",     0, false},
}};

// Indices of the 12 numeric fields within FIELDS[]
static constexpr int NUM_NUMERIC = 12;
static constexpr std::array<int, NUM_NUMERIC> NUM_IDX = {
    1, 3, 4, 6, 8, 9, 12, 13, 14, 15, 16, 17
};

// ============================================================================
// PART 2: WIRE MESSAGE GENERATION & FIELD LOCATION
// ============================================================================
// Simulates realistic FIX wire data: tag=value pairs with random numeric
// content of the correct digit count. Pre-location of field pointers is
// shared cost -- all parsers pay it equally.

struct WireMessage {
    char data[512];
    int  len;
    struct FieldLoc { const char* value_ptr; uint8_t value_len; };
    FieldLoc locs[NF];
};

struct ParseResult {
    int64_t values[NF];
};

void generate(WireMessage& msg, std::mt19937& rng) {
    char* p = msg.data;
    auto wf = [&](int tag, const char* val) {
        p += sprintf(p, "%d=%s%c", tag, val, SOH);
    };
    auto wn = [&](int tag, int digits) {
        char buf[20];
        buf[0] = '1' + (rng() % 9);  // no leading zero
        for (int i = 1; i < digits; ++i) buf[i] = '0' + (rng() % 10);
        buf[digits] = '\0';
        p += sprintf(p, "%d=%s%c", tag, buf, SOH);
    };
    wf(8,"FIX.4.4"); wn(9,4); wf(35,"X"); wn(34,7); wn(52,17);
    wf(262,"REQ001"); wn(268,2); wf(269,"0"); wn(270,10); wn(271,6);
    wf(55,"AAPL"); wf(279,"0"); wn(48,8); wn(273,17); wn(272,8);
    wn(10,3); wn(346,4); wn(290,3); wf(336,"REG"); wf(49,"SENDER");
    wf(56,"TARGET"); wf(43,"N");
    msg.len = (int)(p - msg.data);
}

void locate(WireMessage& msg) {
    auto ti = [](int tag) -> int {
        for (int i = 0; i < NF; ++i) if (FIELDS[i].tag == tag) return i;
        return -1;
    };
    const char* p = msg.data;
    const char* end = msg.data + msg.len;
    while (p < end) {
        int tag = 0;
        while (p < end && *p != '=') { tag = tag*10 + (*p-'0'); ++p; }
        if (p >= end) break; ++p;
        const char* vs = p;
        while (p < end && *p != SOH) ++p;
        int vl = (int)(p - vs);
        if (p < end) ++p;
        int idx = ti(tag);
        if (idx >= 0) { msg.locs[idx].value_ptr = vs; msg.locs[idx].value_len = (uint8_t)vl; }
    }
}

// ============================================================================
// PART 3: THE FOUR CONVERSION STRATEGIES (Expression Templates)
// ============================================================================
// Each is a class template parameterised on D (digit count), known at compile
// time. They share the interface: static int64_t parse(const char* p)
// but generate very different machine code.

// Portability: always_inline across compilers
#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE __attribute__((always_inline))
  #define NO_INLINE    __attribute__((noinline))
#elif defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
  #define NO_INLINE    __declspec(noinline)
#else
  #define FORCE_INLINE
  #define NO_INLINE
#endif

// --- Strategy 1: Unrolled ---------------------------------------------------
// Compile-time digit-count-specific arithmetic. For D <= 3, emits a single
// expression with no loops or branches. For D > 3, uses a constexpr
// power-of-ten table -- the loop has a known trip count and the compiler
// fully unrolls it.
//
// Selected by DP for: CheckSum (3d), MDEntryPosNo (3d)

template<int D> struct UnrolledParser {
    static FORCE_INLINE int64_t parse(const char* p) {
        if constexpr (D == 0) return 0;
        else if constexpr (D == 1) return p[0] - '0';
        else if constexpr (D == 2) return (p[0]-'0')*10 + (p[1]-'0');
        else if constexpr (D == 3) return (p[0]-'0')*100 + (p[1]-'0')*10 + (p[2]-'0');
        else {
            // Constexpr power-of-ten table avoids repeated multiplication
            constexpr auto pow10 = []() {
                std::array<int64_t, 18> t{}; t[0] = 1;
                for (int i = 1; i < 18; ++i) t[i] = t[i-1] * 10;
                return t;
            }();
            int64_t r = 0;
            for (int i = 0; i < D; ++i)
                r += int64_t(p[i] - '0') * pow10[D-1-i];
            return r;
        }
    }
};

// --- Strategy 2: SWAR (4-digit block accumulation) --------------------------
// SWAR-inspired, 4-digit block parser in the spirit of the "parse eight digits
// at once" tricks popularised in fast integer parsers. Processes digits in
// blocks of 4 via multiply-accumulate. Block count and tail size are both
// constexpr, so the compiler unrolls the block loop and eliminates the tail
// branch entirely. For a 17-digit timestamp this is 4 blocks + 1 tail digit
// -- roughly 4x fewer multiply-accumulate chains than Horner.
//
// Selected by DP for: BodyLength (4d), MsgSeqNum (7d), SendingTime (17d),
//   MDEntryPx (10d), MDEntrySize (6d), SecurityID (8d), MDEntryTime (17d),
//   MDEntryDate (8d), NumberOfOrders (4d)

template<int D> struct SWARParser {
    static FORCE_INLINE int64_t parse(const char* p) {
        int64_t r = 0;
        constexpr int B = D / 4;   // full 4-digit blocks
        constexpr int T = D % 4;   // remaining tail digits
        if constexpr (B >= 1) {
            for (int b = 0; b < B; ++b) {
                int64_t blk = (p[b*4]-'0')*1000 + (p[b*4+1]-'0')*100
                            + (p[b*4+2]-'0')*10  + (p[b*4+3]-'0');
                if constexpr (B == 1 && T == 0) r = blk;
                else r = r * 10000 + blk;
            }
        }
        // Tail handled with constexpr selection -- no runtime branch
        if constexpr (T == 3)
            r = r*1000 + (p[B*4]-'0')*100 + (p[B*4+1]-'0')*10 + (p[B*4+2]-'0');
        else if constexpr (T == 2)
            r = r*100 + (p[B*4]-'0')*10 + (p[B*4+1]-'0');
        else if constexpr (T == 1)
            r = r*10 + (p[B*4]-'0');
        return r;
    }
};

// --- Strategy 3: Loop (Horner's method) -------------------------------------
// The simplest strategy: a Horner accumulation loop. Because D is a template
// parameter the compiler knows the trip count and can unroll or vectorise.
// Sometimes beats SWAR at low digit counts where block overhead isn't amortised.
//
// Selected by DP for: NoMDEntries (2d)

template<int D> struct LoopParser {
    static FORCE_INLINE int64_t parse(const char* p) {
        int64_t r = 0;
        for (int i = 0; i < D; ++i) r = r * 10 + (p[i] - '0');
        return r;
    }
};

// --- Strategy 4: Generic (full validation) ----------------------------------
// Per-character digit check + noinline. Equivalent in spirit to from_chars.
// The branch prevents vectorisation; noinline blocks cross-field ILP.
// This is the cost of generality: not the validation itself (branch predictor
// handles it on clean data) but the optimisation barriers it imposes.

template<int D> struct GenericParser {
    static NO_INLINE int64_t parse(const char* p) {
        int64_t r = 0;
        for (int i = 0; i < D; ++i) {
            unsigned d = (unsigned)(p[i] - '0');
            if (d > 9) [[unlikely]] return -1;
            r = r * 10 + d;
        }
        return r;
    }
};

// Skip parser for non-numeric fields
struct SkipParser {
    static FORCE_INLINE int64_t parse(const char*) { return 0; }
};

// ============================================================================
// PART 4: THE PLAN IS A TYPE
// ============================================================================
// The DP's output is a strategy assignment for each of the 22 fields.
// We encode this as a variadic template parameter list. The compiler sees
// the entire parse chain as one function and optimises across field boundaries.

enum class Strategy : uint8_t {
    Unrolled = 0, SWAR = 1, Loop = 2, Generic = 3, Skip = 5
};

constexpr const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::Unrolled: return "Unrolled";
        case Strategy::SWAR:     return "SWAR";
        case Strategy::Loop:     return "Loop";
        case Strategy::Generic:  return "Generic";
        case Strategy::Skip:     return "Skip";
    }
    return "?";
}

constexpr const char* strategy_short(Strategy s) {
    switch (s) {
        case Strategy::Unrolled: return "U";
        case Strategy::SWAR:     return "S";
        case Strategy::Loop:     return "L";
        case Strategy::Generic:  return "G";
        case Strategy::Skip:     return "-";
    }
    return "?";
}

// A Plan is a compile-time strategy assignment for every field position
template<Strategy... Ss>
struct Plan {
    static constexpr std::array<Strategy, sizeof...(Ss)> strategies = {Ss...};
    static constexpr Strategy get(size_t i) { return strategies[i]; }
};

// Dispatch: map (Strategy, D) -> parser struct at compile time
template<Strategy S, int D> struct FieldParser;
template<int D> struct FieldParser<Strategy::Unrolled, D> : UnrolledParser<D> {};
template<int D> struct FieldParser<Strategy::SWAR, D>     : SWARParser<D> {};
template<int D> struct FieldParser<Strategy::Loop, D>     : LoopParser<D> {};
template<int D> struct FieldParser<Strategy::Generic, D>  : GenericParser<D> {};
template<int D> struct FieldParser<Strategy::Skip, D>     : SkipParser {};

// Recursive expansion: parse all fields using the plan's strategy assignment
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

// Top-level message parser: noinline to create a measurable boundary
template<typename PlanType>
struct MessageParser {
    static NO_INLINE void parse(const WireMessage& msg, ParseResult& out) {
        ParseFieldWire<PlanType, 0>::run(msg, out);
    }
};

// ============================================================================
// PART 5: THREE CONCRETE PLANS
// ============================================================================
// S = SWAR, U = Unrolled, L = Loop, G = Generic, - = Skip

using S_ = Strategy;

// Plan 1: ALL GENERIC -- the baseline (every numeric field uses from_chars-style)
using PlanGeneric = Plan<
    S_::Skip,     // [0]  BeginString
    S_::Generic,  // [1]  BodyLength      4d
    S_::Skip,     // [2]  MsgType
    S_::Generic,  // [3]  MsgSeqNum       7d
    S_::Generic,  // [4]  SendingTime    17d
    S_::Skip,     // [5]  MDReqID
    S_::Generic,  // [6]  NoMDEntries     2d
    S_::Skip,     // [7]  MDEntryType
    S_::Generic,  // [8]  MDEntryPx      10d
    S_::Generic,  // [9]  MDEntrySize     6d
    S_::Skip,     // [10] Symbol
    S_::Skip,     // [11] MDUpdateAction
    S_::Generic,  // [12] SecurityID      8d
    S_::Generic,  // [13] MDEntryTime    17d
    S_::Generic,  // [14] MDEntryDate     8d
    S_::Generic,  // [15] CheckSum        3d
    S_::Generic,  // [16] NumberOfOrders  4d
    S_::Generic,  // [17] MDEntryPosNo    3d
    S_::Skip,     // [18] TradingSession
    S_::Skip,     // [19] SenderCompID
    S_::Skip,     // [20] TargetCompID
    S_::Skip      // [21] PossDupFlag
>;

// Plan 2: EXPERT -- hand-tuned selection (SWAR everywhere, like Lima & Tonetti)
// An experienced developer would pick the best single strategy per field
// by inspection. This is the "switch on digit count" approach.
using PlanExpert = Plan<
    S_::Skip,       // [0]  BeginString
    S_::SWAR,       // [1]  BodyLength      4d
    S_::Skip,       // [2]  MsgType
    S_::SWAR,       // [3]  MsgSeqNum       7d
    S_::SWAR,       // [4]  SendingTime    17d
    S_::Skip,       // [5]  MDReqID
    S_::SWAR,       // [6]  NoMDEntries     2d  (expert picks SWAR for all)
    S_::Skip,       // [7]  MDEntryType
    S_::SWAR,       // [8]  MDEntryPx      10d
    S_::SWAR,       // [9]  MDEntrySize     6d
    S_::Skip,       // [10] Symbol
    S_::Skip,       // [11] MDUpdateAction
    S_::SWAR,       // [12] SecurityID      8d
    S_::SWAR,       // [13] MDEntryTime    17d
    S_::SWAR,       // [14] MDEntryDate     8d
    S_::SWAR,       // [15] CheckSum        3d  (expert picks SWAR for all)
    S_::SWAR,       // [16] NumberOfOrders  4d
    S_::SWAR,       // [17] MDEntryPosNo    3d  (expert picks SWAR for all)
    S_::Skip,       // [18] TradingSession
    S_::Skip,       // [19] SenderCompID
    S_::Skip,       // [20] TargetCompID
    S_::Skip        // [21] PossDupFlag
>;

// Plan 3: DP-OPTIMAL -- selected by compile-time dynamic programming
// The DP evaluates all 4^12 = 16.7M configurations using a per-field
// additive cost model calibrated from measurement. This is the result:
//   BBBLBBBBUBU (reading numeric fields left to right)
using PlanDPOptimal = Plan<
    S_::Skip,       // [0]  BeginString
    S_::SWAR,       // [1]  BodyLength      4d  -> SWAR
    S_::Skip,       // [2]  MsgType
    S_::SWAR,       // [3]  MsgSeqNum       7d  -> SWAR
    S_::SWAR,       // [4]  SendingTime    17d  -> SWAR
    S_::Skip,       // [5]  MDReqID
    S_::Loop,       // [6]  NoMDEntries     2d  -> Loop  (beats SWAR at 2 digits)
    S_::Skip,       // [7]  MDEntryType
    S_::SWAR,       // [8]  MDEntryPx      10d  -> SWAR
    S_::SWAR,       // [9]  MDEntrySize     6d  -> SWAR
    S_::Skip,       // [10] Symbol
    S_::Skip,       // [11] MDUpdateAction
    S_::SWAR,       // [12] SecurityID      8d  -> SWAR
    S_::SWAR,       // [13] MDEntryTime    17d  -> SWAR
    S_::SWAR,       // [14] MDEntryDate     8d  -> SWAR
    S_::Unrolled,   // [15] CheckSum        3d  -> Unrolled (3 instructions)
    S_::SWAR,       // [16] NumberOfOrders  4d  -> SWAR
    S_::Unrolled,   // [17] MDEntryPosNo    3d  -> Unrolled (3 instructions)
    S_::Skip,       // [18] TradingSession
    S_::Skip,       // [19] SenderCompID
    S_::Skip,       // [20] TargetCompID
    S_::Skip        // [21] PossDupFlag
>;

// ============================================================================
// PART 6: BENCHMARK HARNESS
// ============================================================================

static constexpr int N_MSGS   = 1000;   // L1-resident working set
static constexpr int WARMUP   = 5000;   // warmup iterations
static constexpr int REPS     = 40;     // repetitions per trial
static constexpr int TRIALS   = 15;     // trials (take median)

// Volatile sink to prevent dead code elimination
volatile int64_t g_sink = 0;

template<typename PlanType>
double bench_plan(std::vector<WireMessage>& msgs) {
    ParseResult result{};

    // Warmup
    for (int w = 0; w < WARMUP; ++w) {
        MessageParser<PlanType>::parse(msgs[w % N_MSGS], result);
        g_sink = result.values[4];
    }

    // Measure
    std::vector<double> times(TRIALS);
    for (int t = 0; t < TRIALS; ++t) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; ++r) {
            for (int m = 0; m < N_MSGS; ++m) {
                MessageParser<PlanType>::parse(msgs[m], result);
                g_sink = result.values[4];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count();
        times[t] = ns / (REPS * N_MSGS);
    }

    // Return median
    std::sort(times.begin(), times.end());
    return times[TRIALS / 2];
}

// ============================================================================
// PART 6b: RUNTIME-DISPATCHED EXPERT PARSER (Lima & Tonetti style)
// ============================================================================
// The Expert approach in practice uses a switch or function pointer to select
// the conversion strategy at runtime. This is what Tonetti's hm::atoi does:
// pick the best converter, but dispatch to it through a call boundary.
// We measure this separately to show the cost of runtime dispatch.

namespace rt {

NO_INLINE int64_t parse_swar(const char* p, int len) {
    int64_t r = 0;
    int blocks = len / 4, tail = len % 4;
    for (int b = 0; b < blocks; ++b) {
        int64_t blk = (p[b*4]-'0')*1000 + (p[b*4+1]-'0')*100
                    + (p[b*4+2]-'0')*10  + (p[b*4+3]-'0');
        r = r * 10000 + blk;
    }
    const char* tp = p + blocks * 4;
    if (tail == 3) r = r*1000 + (tp[0]-'0')*100 + (tp[1]-'0')*10 + (tp[2]-'0');
    else if (tail == 2) r = r*100 + (tp[0]-'0')*10 + (tp[1]-'0');
    else if (tail == 1) r = r*10 + (tp[0]-'0');
    return r;
}

NO_INLINE void parse_expert_rt(const WireMessage& msg, ParseResult& out) {
    for (int i = 0; i < NUM_NUMERIC; ++i) {
        int fi = NUM_IDX[i];
        if (msg.locs[fi].value_ptr)
            out.values[fi] = parse_swar(msg.locs[fi].value_ptr, msg.locs[fi].value_len);
    }
}

} // namespace rt

double bench_expert_rt(std::vector<WireMessage>& msgs) {
    ParseResult result{};
    for (int w = 0; w < WARMUP; ++w) {
        rt::parse_expert_rt(msgs[w % N_MSGS], result);
        g_sink = result.values[4];
    }
    std::vector<double> times(TRIALS);
    for (int t = 0; t < TRIALS; ++t) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; ++r) {
            for (int m = 0; m < N_MSGS; ++m) {
                rt::parse_expert_rt(msgs[m], result);
                g_sink = result.values[4];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count();
        times[t] = ns / (REPS * N_MSGS);
    }
    std::sort(times.begin(), times.end());
    return times[TRIALS / 2];
}

// ============================================================================
// PART 7: MAIN -- RUN ALL THREE AND COMPARE
// ============================================================================

void print_plan_table() {
    printf("  +--------------------+-----+----------+----------+-----------+\n");
    printf("  | Field              |  D  | Generic  | Expert   | DPOptimal |\n");
    printf("  +--------------------+-----+----------+----------+-----------+\n");
    for (int i = 0; i < NUM_NUMERIC; ++i) {
        int fi = NUM_IDX[i];
        const char* g = strategy_name(PlanGeneric::get(fi));
        const char* e = strategy_name(PlanExpert::get(fi));
        const char* d = strategy_name(PlanDPOptimal::get(fi));
        const char* mark = (PlanExpert::get(fi) != PlanDPOptimal::get(fi)) ? " *" : "  ";
        printf("  | %-18s | %2d  | %-8s | %-8s | %-8s%s|\n",
               FIELDS[fi].name, FIELDS[fi].digits, g, e, d, mark);
    }
    printf("  +--------------------+-----+----------+----------+-----------+\n");
    printf("                                                     * = DP differs\n");
    printf("\n  Signature:  Generic  = ");
    for (int i = 0; i < NUM_NUMERIC; ++i)
        printf("%s", strategy_short(PlanGeneric::get(NUM_IDX[i])));
    printf("\n              Expert   = ");
    for (int i = 0; i < NUM_NUMERIC; ++i)
        printf("%s", strategy_short(PlanExpert::get(NUM_IDX[i])));
    printf("\n              DPOptimal= ");
    for (int i = 0; i < NUM_NUMERIC; ++i)
        printf("%s", strategy_short(PlanDPOptimal::get(NUM_IDX[i])));
    printf("\n");
}

void print_results(double t_generic, double t_expert_rt,
                   double t_expert_et, double t_dp) {
    printf("\n  +-------------------------+----------+-----------+\n");
    printf("  | Configuration           |  ns/msg  | vs Generic|\n");
    printf("  +-------------------------+----------+-----------+\n");
    printf("  | Generic  (ET)           | %6.1f   |   1.00x   |\n", t_generic);
    printf("  | Expert   (RT dispatch)  | %6.1f   |   %.2fx   |\n",
           t_expert_rt, t_generic / t_expert_rt);
    printf("  | Expert   (ET inlined)   | %6.1f   |   %.2fx   |\n",
           t_expert_et, t_generic / t_expert_et);
    printf("  | DPOptimal(ET inlined)   | %6.1f   |   %.2fx   |\n",
           t_dp, t_generic / t_dp);
    printf("  +-------------------------+----------+-----------+\n");
    printf("\n  Decomposition:\n");
    printf("    Specialisation   (Generic -> Expert RT):  %5.2fx\n",
           t_generic / t_expert_rt);
    printf("    Dispatch elim.   (Expert RT -> Expert ET): %5.2fx  (%.1f ns saved)\n",
           t_expert_rt / t_expert_et, t_expert_rt - t_expert_et);
    printf("    DP tuning        (Expert ET -> DPOptimal): %5.2fx\n",
           t_expert_et / t_dp);
    printf("    Combined         (Generic -> DPOptimal):   %5.2fx\n",
           t_generic / t_dp);
}

int main() {
    printf("\n");
    printf("  ================================================================\n");
    printf("  FIX Parser Compile-Time DP Demo\n");
    printf("  ================================================================\n");
    printf("  Message:  MarketDataIncrementalRefresh (22 fields, 12 numeric)\n");
    printf("  Space:    4^12 = %d configurations\n", (int)std::pow(4, 12));
    printf("  Protocol: %d msgs x %d reps x %d trials, median\n",
           N_MSGS, REPS, TRIALS);
    printf("  ================================================================\n\n");

    // Generate messages
    std::mt19937 rng(42);
    std::vector<WireMessage> msgs(N_MSGS);
    for (auto& m : msgs) { generate(m, rng); locate(m); }
    printf("  Generated %d messages, %d bytes each\n\n", N_MSGS, msgs[0].len);

    // Show plan table
    printf("  Strategy assignments (numeric fields only):\n\n");
    print_plan_table();

    // Benchmark
    printf("\n  Benchmarking...\n");

    double t_generic   = bench_plan<PlanGeneric>(msgs);
    double t_expert_rt = bench_expert_rt(msgs);
    double t_expert_et = bench_plan<PlanExpert>(msgs);
    double t_dp        = bench_plan<PlanDPOptimal>(msgs);

    print_results(t_generic, t_expert_rt, t_expert_et, t_dp);

    printf("\n  ================================================================\n");
    printf("  Three sources of speedup, layered:\n");
    printf("    1. Per-field specialisation  (Tonetti's insight)\n");
    printf("    2. Dispatch elimination      (expression templates)\n");
    printf("    3. Systematic strategy search (compile-time DP)\n");
    printf("  ================================================================\n\n");

    return 0;
}
