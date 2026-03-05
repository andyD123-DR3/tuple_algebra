// test_phase2_strategies.cpp  —  40 tests for Phase 2 strategy headers
//
// Group A: StrategyGeneric                    T01–T06
// Group B: StrategyLoop                       T07–T11
// Group C: StrategyUnrolled                   T12–T16
// Group D: StrategySWAR                       T17–T22
// Group E: StrategyHybrid                     T23–T27
// Group F: CompositeParser composition        T28–T33
// Group G: Cross-strategy oracle agreement    T34–T40
//   (all 5 strategies must agree with StrategyGeneric on a corpus)

#include <ctdp/calibrator/fix/composite_parser.h>
#include <ctdp/calibrator/fix_et_parser.h>   // generate_message_pool

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ctdp::calibrator::fix;

// ── test infrastructure ───────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(expr, msg) do { \
    if (!(expr)) { \
        std::cerr << "FAIL [" << (msg) << "]: " #expr "\n"; \
        ++g_failed; \
    } \
} while(0)

// Build a SOH-terminated field string: digits + SOH + padding
static std::string make_field(const char* digits) {
    std::string s = digits;
    s += '\x01';
    return s;
}

// Parse with generic for reference value — used in oracle tests
[[maybe_unused]] static std::uint64_t ref_value(const char* digits) {
    return StrategyGeneric<20>::parse(digits).value;
}

// Generate a random N-digit string
static std::string rand_digits(int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0, 9);
    std::string s;
    s.reserve(static_cast<std::size_t>(n) + 1);
    for (int i = 0; i < n; ++i) s += static_cast<char>('0' + d(rng));
    s += '\x01';
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP A: StrategyGeneric
// ─────────────────────────────────────────────────────────────────────────────

static void test_A() {
    // T01: 3-digit field
    {
        auto f = make_field("123");
        auto r = StrategyGeneric<10>::parse(f.c_str());
        CHECK(r.value == 123, "T01-a");
        CHECK(r.length == 3,  "T01-b");
        CHECK(r.ok,           "T01-c");
        ++g_passed;
        std::cout << "T01 PASS: StrategyGeneric basic 3-digit\n";
    }

    // T02: leading zeros preserved as numeric value
    {
        auto f = make_field("007");
        auto r = StrategyGeneric<10>::parse(f.c_str());
        CHECK(r.value == 7,  "T02-a");
        CHECK(r.length == 3, "T02-b");
        CHECK(r.ok,          "T02-c");
        ++g_passed;
        std::cout << "T02 PASS: StrategyGeneric leading zeros\n";
    }

    // T03: single digit
    {
        auto f = make_field("9");
        auto r = StrategyGeneric<1>::parse(f.c_str());
        CHECK(r.value == 9, "T03-a");
        CHECK(r.length == 1, "T03-b");
        ++g_passed;
        std::cout << "T03 PASS: StrategyGeneric single digit\n";
    }

    // T04: non-digit byte stops parse — ok=false
    {
        const char bad[] = "12A\x01";
        auto r = StrategyGeneric<10>::parse(bad);
        CHECK(r.length == 2, "T04-a");  // stopped at 'A'
        CHECK(!r.ok,         "T04-b");
        ++g_passed;
        std::cout << "T04 PASS: StrategyGeneric stops on non-digit (ok=false)\n";
    }

    // T05: MaxDigits cap — doesn't read past limit
    {
        // "12345678" but MaxDigits=4 → reads "1234"
        const char f[] = "12345678\x01";
        auto r = StrategyGeneric<4>::parse(f);
        CHECK(r.value == 1234, "T05-a");
        CHECK(r.length == 4,   "T05-b");
        ++g_passed;
        std::cout << "T05 PASS: StrategyGeneric MaxDigits cap\n";
    }

    // T06: SOH-only (empty field) → value=0, length=0, ok=true
    {
        const char f[] = "\x01";
        auto r = StrategyGeneric<10>::parse(f);
        CHECK(r.value == 0,  "T06-a");
        CHECK(r.length == 0, "T06-b");
        CHECK(r.ok,          "T06-c");
        ++g_passed;
        std::cout << "T06 PASS: StrategyGeneric empty field (SOH first)\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP B: StrategyLoop
// ─────────────────────────────────────────────────────────────────────────────

static void test_B() {
    // T07: basic match with Generic
    {
        auto f = make_field("456");
        auto rg = StrategyGeneric<10>::parse(f.c_str());
        auto rl = StrategyLoop<10>::parse(f.c_str());
        CHECK(rl.value == rg.value,   "T07-a");
        CHECK(rl.length == rg.length, "T07-b");
        CHECK(rl.ok,                  "T07-c");
        ++g_passed;
        std::cout << "T07 PASS: StrategyLoop matches Generic on \"456\"\n";
    }

    // T08: 8-digit field
    {
        auto f = make_field("20240115");
        auto rg = StrategyGeneric<10>::parse(f.c_str());
        auto rl = StrategyLoop<8>::parse(f.c_str());
        CHECK(rl.value == rg.value, "T08-a");
        ++g_passed;
        std::cout << "T08 PASS: StrategyLoop 8-digit field\n";
    }

    // T09: MaxDigits == exact field length (no tail)
    {
        auto f = make_field("999");
        auto r = StrategyLoop<3>::parse(f.c_str());
        CHECK(r.value == 999, "T09-a");
        CHECK(r.length == 3,  "T09-b");
        ++g_passed;
        std::cout << "T09 PASS: StrategyLoop exact MaxDigits\n";
    }

    // T10: field shorter than MaxDigits — SOH before limit
    {
        auto f = make_field("42");
        auto r = StrategyLoop<8>::parse(f.c_str());
        CHECK(r.value == 42, "T10-a");
        CHECK(r.length == 2, "T10-b");
        CHECK(r.ok,          "T10-c");
        ++g_passed;
        std::cout << "T10 PASS: StrategyLoop short field with MaxDigits=8\n";
    }

    // T11: single-digit field
    {
        auto f = make_field("7");
        auto r = StrategyLoop<6>::parse(f.c_str());
        CHECK(r.value == 7,  "T11-a");
        CHECK(r.length == 1, "T11-b");
        ++g_passed;
        std::cout << "T11 PASS: StrategyLoop single digit\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP C: StrategyUnrolled
// ─────────────────────────────────────────────────────────────────────────────

static void test_C() {
    // T12: 1-digit
    {
        const char f[] = "5\x01";
        auto r = StrategyUnrolled<1>::parse(f);
        CHECK(r.value == 5,  "T12-a");
        CHECK(r.length == 1, "T12-b");
        CHECK(r.ok,          "T12-c");
        ++g_passed;
        std::cout << "T12 PASS: StrategyUnrolled<1>\n";
    }

    // T13: 6-digit
    {
        auto f = make_field("987654");
        auto r = StrategyUnrolled<6>::parse(f.c_str());
        CHECK(r.value == 987654, "T13-a");
        CHECK(r.length == 6,     "T13-b");
        CHECK(r.ok,              "T13-c");
        ++g_passed;
        std::cout << "T13 PASS: StrategyUnrolled<6>\n";
    }

    // T14: 8-digit (common for UTCTimestamp date)
    {
        auto f = make_field("20240115");
        auto ru = StrategyUnrolled<8>::parse(f.c_str());
        auto rg = StrategyGeneric<8>::parse(f.c_str());
        CHECK(ru.value == rg.value, "T14-a");
        CHECK(ru.ok,                "T14-b");
        ++g_passed;
        std::cout << "T14 PASS: StrategyUnrolled<8> matches Generic\n";
    }

    // T15: always returns length == Digits and ok == true
    {
        const char f[] = "000000\x01";
        auto r = StrategyUnrolled<6>::parse(f);
        CHECK(r.value == 0,  "T15-a");
        CHECK(r.length == 6, "T15-b");
        CHECK(r.ok,          "T15-c");
        ++g_passed;
        std::cout << "T15 PASS: StrategyUnrolled always ok=true, length=Digits\n";
    }

    // T16: 10-digit field (max in unrolled chain)
    {
        auto f = make_field("1234567890");
        auto ru = StrategyUnrolled<10>::parse(f.c_str());
        auto rg = StrategyGeneric<10>::parse(f.c_str());
        CHECK(ru.value == rg.value,   "T16-a");
        CHECK(ru.length == rg.length, "T16-b");
        ++g_passed;
        std::cout << "T16 PASS: StrategyUnrolled<10> matches Generic\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP D: StrategySWAR
// ─────────────────────────────────────────────────────────────────────────────

static void test_D() {
    // T17: 8-digit field — single word read
    {
        auto f = make_field("20240115");
        auto rs = StrategySWAR<8>::parse(f.c_str());
        auto rg = StrategyGeneric<8>::parse(f.c_str());
        CHECK(rs.value == rg.value,   "T17-a");
        CHECK(rs.length == rg.length, "T17-b");
        CHECK(rs.ok,                  "T17-c");
        ++g_passed;
        std::cout << "T17 PASS: StrategySWAR<8> 8-digit (single word)\n";
    }

    // T18: 3-digit field — shorter than one word
    {
        auto f = make_field("123");
        auto rs = StrategySWAR<10>::parse(f.c_str());
        auto rg = StrategyGeneric<10>::parse(f.c_str());
        CHECK(rs.value == rg.value,   "T18-a");
        CHECK(rs.length == rg.length, "T18-b");
        ++g_passed;
        std::cout << "T18 PASS: StrategySWAR<10> 3-digit (tail only)\n";
    }

    // T19: 12-digit field — one full word + 4-byte tail
    {
        auto f = make_field("123456789012");
        auto rs = StrategySWAR<12>::parse(f.c_str());
        auto rg = StrategyGeneric<12>::parse(f.c_str());
        CHECK(rs.value == rg.value,   "T19-a");
        CHECK(rs.length == rg.length, "T19-b");
        ++g_passed;
        std::cout << "T19 PASS: StrategySWAR<12> 12-digit (word + tail)\n";
    }

    // T20: SOH terminates mid-word
    {
        // Field has 5 digits then SOH — SOH appears in the first 8-byte word
        auto f = make_field("54321");
        auto rs = StrategySWAR<10>::parse(f.c_str());
        auto rg = StrategyGeneric<10>::parse(f.c_str());
        CHECK(rs.value == rg.value,   "T20-a");
        CHECK(rs.length == rg.length, "T20-b");
        CHECK(rs.length == 5,         "T20-c");
        ++g_passed;
        std::cout << "T20 PASS: StrategySWAR SOH mid-word (5 digits)\n";
    }

    // T21: 16-digit field — two full words
    {
        auto f = make_field("1234567890123456");
        auto rs = StrategySWAR<16>::parse(f.c_str());
        auto rg = StrategyGeneric<16>::parse(f.c_str());
        CHECK(rs.value == rg.value,   "T21-a");
        CHECK(rs.length == rg.length, "T21-b");
        ++g_passed;
        std::cout << "T21 PASS: StrategySWAR<16> 16-digit (two full words)\n";
    }

    // T22: single digit
    {
        auto f = make_field("7");
        auto rs = StrategySWAR<8>::parse(f.c_str());
        CHECK(rs.value == 7,  "T22-a");
        CHECK(rs.length == 1, "T22-b");
        CHECK(rs.ok,          "T22-c");
        ++g_passed;
        std::cout << "T22 PASS: StrategySWAR single digit\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP E: StrategyHybrid
// ─────────────────────────────────────────────────────────────────────────────

static void test_E() {
    // T23: 8+4 = 12-digit field (UTCTimestamp-like)
    {
        auto f = make_field("202401150830");
        auto rh = StrategyHybrid<8,4>::parse(f.c_str());
        auto rg = StrategyGeneric<12>::parse(f.c_str());
        CHECK(rh.value == rg.value,   "T23-a");
        CHECK(rh.length == 12,        "T23-b");
        CHECK(rh.ok,                  "T23-c");
        ++g_passed;
        std::cout << "T23 PASS: StrategyHybrid<8,4> 12-digit matches Generic\n";
    }

    // T24: 4+3 = 7-digit field
    {
        auto f = make_field("1234567");
        auto rh = StrategyHybrid<4,3>::parse(f.c_str());
        auto rg = StrategyGeneric<7>::parse(f.c_str());
        CHECK(rh.value == rg.value, "T24-a");
        CHECK(rh.length == 7,       "T24-b");
        ++g_passed;
        std::cout << "T24 PASS: StrategyHybrid<4,3> matches Generic\n";
    }

    // T25: prefix SOH early (short field) — suffix not parsed
    {
        // Only 3 digits then SOH, PrefixDigits=8 → prefix stops early
        auto f = make_field("123");
        auto rh = StrategyHybrid<8,4>::parse(f.c_str());
        CHECK(rh.value == 123, "T25-a");
        CHECK(rh.length == 3,  "T25-b");
        CHECK(rh.ok,           "T25-c");
        ++g_passed;
        std::cout << "T25 PASS: StrategyHybrid prefix SOH early\n";
    }

    // T26: 1+1 = 2-digit minimal hybrid
    {
        auto f = make_field("42");
        auto rh = StrategyHybrid<1,1>::parse(f.c_str());
        CHECK(rh.value == 42, "T26-a");
        CHECK(rh.length == 2, "T26-b");
        ++g_passed;
        std::cout << "T26 PASS: StrategyHybrid<1,1> minimal\n";
    }

    // T27: combined value arithmetic: prefix=1234, suffix=567
    //   expected = 1234 * 1000 + 567 = 1234567
    {
        auto f = make_field("1234567");
        auto rh = StrategyHybrid<4,3>::parse(f.c_str());
        CHECK(rh.value == 1234567ULL, "T27-a");
        ++g_passed;
        std::cout << "T27 PASS: StrategyHybrid combined value arithmetic\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP F: CompositeParser
// ─────────────────────────────────────────────────────────────────────────────

static void test_F() {
    // T28: 4-field reference parser matches expected values
    {
        const char msg[] = "123\x01""456789\x01""1234\x01""12345678\x01";
        using P = ReferenceParser<3,6,4,8>;
        auto r = P::parse_message(msg);
        CHECK(r.values[0] == 123,      "T28-a");
        CHECK(r.values[1] == 456789,   "T28-b");
        CHECK(r.values[2] == 1234,     "T28-c");
        CHECK(r.values[3] == 12345678, "T28-d");
        CHECK(r.ok,                    "T28-e");
        ++g_passed;
        std::cout << "T28 PASS: ReferenceParser<3,6,4,8> 4-field message\n";
    }

    // T29: total_bytes = sum(lengths) + N_fields (one SOH each)
    {
        const char msg[] = "12\x01""345\x01""6789\x01";
        using P = ReferenceParser<2,3,4>;
        auto r = P::parse_message(msg);
        // lengths: 2,3,4  SOHs: 3  total = 9+3 = 12... but field 3 is 4 digits
        CHECK(r.total_bytes == 2+1 + 3+1 + 4+1, "T29-a");  // 12
        CHECK(r.lengths[0] == 2, "T29-b");
        CHECK(r.lengths[1] == 3, "T29-c");
        CHECK(r.lengths[2] == 4, "T29-d");
        ++g_passed;
        std::cout << "T29 PASS: CompositeParser total_bytes accounting\n";
    }

    // T30: mixed strategies — Unrolled+SWAR+Generic
    {
        const char msg[] = "999\x01""12345678\x01""42\x01";
        using P = CompositeParser<
            Field<StrategyUnrolled<3>, 3>,
            Field<StrategySWAR<8>,     8>,
            Field<StrategyGeneric<4>,  4>
        >;
        auto r = P::parse_message(msg);
        CHECK(r.values[0] == 999,      "T30-a");
        CHECK(r.values[1] == 12345678, "T30-b");
        CHECK(r.values[2] == 42,       "T30-c");
        CHECK(r.ok,                    "T30-d");
        ++g_passed;
        std::cout << "T30 PASS: CompositeParser mixed strategies (U+SWAR+G)\n";
    }

    // T31: num_fields static constexpr correct
    {
        using P3 = ReferenceParser<1,2,3>;
        using P5 = ReferenceParser<1,2,3,4,5>;
        static_assert(P3::num_fields == 3);
        static_assert(P5::num_fields == 5);
        ++g_passed;
        std::cout << "T31 PASS: CompositeParser::num_fields static constexpr\n";
    }

    // T32: ok=false propagated when one field has non-digit
    {
        // field 1 has a non-digit 'X'
        const char msg[] = "123\x01""X56\x01""789\x01";
        using P = ReferenceParser<3,3,3>;
        auto r = P::parse_message(msg);
        CHECK(!r.ok, "T32-a");
        ++g_passed;
        std::cout << "T32 PASS: CompositeParser propagates ok=false\n";
    }

    // T33: 1-field parser — minimal composition
    {
        const char msg[] = "42\x01";
        using P = CompositeParser<Field<StrategyUnrolled<2>, 2>>;
        static_assert(P::num_fields == 1, "T33-static");
        auto r = P::parse_message(msg);
        CHECK(r.values[0] == 42,  "T33-b");
        CHECK(r.ok,               "T33-c");
        ++g_passed;
        std::cout << "T33 PASS: CompositeParser single field\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROUP G: Cross-strategy oracle agreement (10,000-message corpus)
//
//  For each of the 4 digit widths used in trivial_schema (3,6,4,8),
//  generate 10,000 random fields and verify that all 5 strategies agree
//  with StrategyGeneric on (value, length) for every valid field.
// ─────────────────────────────────────────────────────────────────────────────

template<int D>
static void check_all_strategies_agree(const std::string& fld, int test_num) {
    const char* p = fld.c_str();
    auto rg = StrategyGeneric<D>::parse(p);
    auto rl = StrategyLoop<D>::parse(p);
    auto ru = StrategyUnrolled<D>::parse(p);
    auto rs = StrategySWAR<D>::parse(p);

    if (rg.value != rl.value || rg.length != rl.length)
        std::cerr << "MISMATCH Loop vs Generic at T" << test_num
                  << " field=\"" << fld.substr(0,fld.size()-1)
                  << "\" g=" << rg.value << " l=" << rl.value << "\n";

    if (rg.value != ru.value)
        std::cerr << "MISMATCH Unrolled vs Generic at T" << test_num
                  << " field=\"" << fld.substr(0,fld.size()-1)
                  << "\" g=" << rg.value << " u=" << ru.value << "\n";

    if (rg.value != rs.value || rg.length != rs.length)
        std::cerr << "MISMATCH SWAR vs Generic at T" << test_num
                  << " field=\"" << fld.substr(0,fld.size()-1)
                  << "\" g=" << rg.value << " s=" << rs.value << "\n";

    CHECK(rg.value  == rl.value,   "Loop-value");
    CHECK(rg.length == rl.length,  "Loop-length");
    CHECK(rg.value  == ru.value,   "Unrolled-value");
    CHECK(rg.value  == rs.value,   "SWAR-value");
    CHECK(rg.length == rs.length,  "SWAR-length");
}

static void test_G() {
    constexpr int CORPUS_SIZE = 10000;
    std::mt19937 rng(20240115);

    // T34: 3-digit fields
    {
        int failures_before = g_failed;
        for (int i = 0; i < CORPUS_SIZE; ++i)
            check_all_strategies_agree<3>(rand_digits(3, rng), 34);
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T34 PASS: all strategies agree — 3-digit x" << CORPUS_SIZE << "\n";
        }
    }

    // T35: 6-digit fields
    {
        int failures_before = g_failed;
        for (int i = 0; i < CORPUS_SIZE; ++i)
            check_all_strategies_agree<6>(rand_digits(6, rng), 35);
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T35 PASS: all strategies agree — 6-digit x" << CORPUS_SIZE << "\n";
        }
    }

    // T36: 4-digit fields
    {
        int failures_before = g_failed;
        for (int i = 0; i < CORPUS_SIZE; ++i)
            check_all_strategies_agree<4>(rand_digits(4, rng), 36);
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T36 PASS: all strategies agree — 4-digit x" << CORPUS_SIZE << "\n";
        }
    }

    // T37: 8-digit fields
    {
        int failures_before = g_failed;
        for (int i = 0; i < CORPUS_SIZE; ++i)
            check_all_strategies_agree<8>(rand_digits(8, rng), 37);
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T37 PASS: all strategies agree — 8-digit x" << CORPUS_SIZE << "\n";
        }
    }

    // T38: 12-digit fields (full_schema's longest)
    {
        int failures_before = g_failed;
        for (int i = 0; i < CORPUS_SIZE; ++i)
            check_all_strategies_agree<12>(rand_digits(12, rng), 38);
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T38 PASS: all strategies agree — 12-digit x" << CORPUS_SIZE << "\n";
        }
    }

    // T39: CompositeParser — 4-field oracle agreement on 10,000 messages
    //   All-generic reference vs All-unrolled vs All-SWAR
    {
        using RefP  = ReferenceParser<3,6,4,8>;
        using UnrP  = CompositeParser<
            Field<StrategyUnrolled<3>,3>,
            Field<StrategyUnrolled<6>,6>,
            Field<StrategyUnrolled<4>,4>,
            Field<StrategyUnrolled<8>,8>>;
        using SwarP = CompositeParser<
            Field<StrategySWAR<3>,3>,
            Field<StrategySWAR<6>,6>,
            Field<StrategySWAR<4>,4>,
            Field<StrategySWAR<8>,8>>;

        // Build synthetic messages matching the trivial_schema layout
        // "DDD\x01DDDDDD\x01DDDD\x01DDDDDDDD\x01" (3+6+4+8 digits, 4 SOHs)
        std::mt19937 msg_rng(42);
        int failures_before = g_failed;
        for (int m = 0; m < CORPUS_SIZE; ++m) {
            std::string msg;
            msg += rand_digits(3, msg_rng);
            msg += rand_digits(6, msg_rng);
            msg += rand_digits(4, msg_rng);
            msg += rand_digits(8, msg_rng);

            auto ref  = RefP::parse_message(msg.c_str());
            auto unr  = UnrP::parse_message(msg.c_str());
            auto swar = SwarP::parse_message(msg.c_str());

            for (int fi = 0; fi < 4; ++fi) {
                CHECK(unr.values[fi]  == ref.values[fi], "T39-unr");
                CHECK(swar.values[fi] == ref.values[fi], "T39-swar");
            }
        }
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T39 PASS: CompositeParser oracle agreement "
                         "(Ref vs Unrolled vs SWAR, 4-field x" << CORPUS_SIZE << ")\n";
        }
    }

    // T40: StrategyHybrid oracle agreement — 8+4 vs Generic<12>
    //   on 10,000 random 12-digit fields
    {
        std::mt19937 hyb_rng(2024);
        int failures_before = g_failed;
        for (int i = 0; i < CORPUS_SIZE; ++i) {
            auto f  = rand_digits(12, hyb_rng);
            auto rh = StrategyHybrid<8,4>::parse(f.c_str());
            auto rg = StrategyGeneric<12>::parse(f.c_str());
            CHECK(rh.value == rg.value,   "T40-value");
            CHECK(rh.length == rg.length, "T40-length");
            CHECK(rh.ok,                  "T40-ok");
        }
        if (g_failed == failures_before) {
            ++g_passed;
            std::cout << "T40 PASS: StrategyHybrid<8,4> agrees with Generic<12> "
                         "on " << CORPUS_SIZE << " random fields\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Group A: StrategyGeneric ===\n";
    test_A();
    std::cout << "\n=== Group B: StrategyLoop ===\n";
    test_B();
    std::cout << "\n=== Group C: StrategyUnrolled ===\n";
    test_C();
    std::cout << "\n=== Group D: StrategySWAR ===\n";
    test_D();
    std::cout << "\n=== Group E: StrategyHybrid ===\n";
    test_E();
    std::cout << "\n=== Group F: CompositeParser ===\n";
    test_F();
    std::cout << "\n=== Group G: Cross-strategy oracle agreement ===\n";
    test_G();

    std::cout << "\n──────────────────────────────────\n";
    std::cout << g_passed << "/40 tests passed";
    if (g_failed > 0)
        std::cout << "  (" << g_failed << " FAILED)";
    std::cout << "\n";
    return (g_failed > 0) ? 1 : 0;
}
