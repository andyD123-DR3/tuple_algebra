#ifndef CTDP_CALIBRATOR_FIX_COMPOSITE_PARSER_H
#define CTDP_CALIBRATOR_FIX_COMPOSITE_PARSER_H

// ============================================================
//  composite_parser.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  CompositeParser<FieldPolicy...>
//
//  Zero-overhead sequential composition of per-field strategies.
//  No virtual dispatch.  The compiler sees the entire field chain
//  as one function, enabling cross-field register allocation and ILP.
//
//  FieldPolicy concept:
//    A type with a static member function:
//      static FieldResult parse(const char* p) noexcept
//    and a static constexpr int digits (nominal field width, used
//    to populate MessageResult::values[]).
//
//  Use the Field<Strategy, Digits> alias to build FieldPolicies:
//
//    using MyParser = CompositeParser<
//        Field<StrategyUnrolled<3>, 3>,   // field 0: 3-digit fixed
//        Field<StrategyGeneric<6>,  6>,   // field 1: up to 6 digits
//        Field<StrategySWAR<8>,     8>,   // field 2: up to 8 digits
//    >;
//    auto result = MyParser::parse_message(msg_ptr);
//
//  MessageResult<N>:
//    values[i]  — parsed uint64 for field i
//    lengths[i] — bytes consumed for field i (not including SOH)
//    ok         — true if all fields parsed without error
//    total_bytes — total bytes consumed including all SOH delimiters
//
//  parse_message advances through the message buffer:
//    for each field: call strategy.parse(p), advance p by length+1 (+1 for SOH)
//
//  Correctness contract (enforced by test suite):
//    All five strategy types must produce identical MessageResult::values[]
//    on a 10,000-message corpus when wired with the same digit widths.
//
//  Dependencies: all strategy headers (via fix_parser_strategies.h)
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/strategy_generic.h>
#include <ctdp/calibrator/fix/strategy_loop.h>
#include <ctdp/calibrator/fix/strategy_unrolled.h>
#include <ctdp/calibrator/fix/strategy_swar.h>
#include <ctdp/calibrator/fix/strategy_hybrid.h>

#include <array>
#include <cstddef>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  MessageResult<N>
// ─────────────────────────────────────────────────────────────────────────────

template<std::size_t N>
struct MessageResult {
    std::array<std::uint64_t, static_cast<std::size_t>(N)> values{};
    std::array<int,           static_cast<std::size_t>(N)> lengths{};
    bool ok{true};
    int  total_bytes{0};

    [[nodiscard]] constexpr bool all_ok() const noexcept { return ok; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Field<StrategyT, NominalDigits> — wraps a strategy as a FieldPolicy
//
//  StrategyT must be a concrete strategy type, e.g. StrategyGeneric<6>.
//  NominalDigits is stored for documentation / test validation; the
//  strategy's template arg sets the actual parse limit.
// ─────────────────────────────────────────────────────────────────────────────

template<typename StrategyT, int NominalDigits>
struct Field {
    using strategy_type = StrategyT;
    static constexpr int digits = NominalDigits;

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static FieldResult parse(const char* p) noexcept {
        return StrategyT::parse(p);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  CompositeParser<FieldPolicies...>
// ─────────────────────────────────────────────────────────────────────────────

template<typename... FieldPolicies>
struct CompositeParser {
    static constexpr std::size_t num_fields = sizeof...(FieldPolicies);

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static MessageResult<num_fields> parse_message(const char* msg) noexcept
    {
        MessageResult<num_fields> result{};
        const char* p = msg;
        parse_impl<0, FieldPolicies...>(p, result);
        return result;
    }

private:
    // Base case: no more fields
    template<int I>
    CTDP_STRATEGY_INLINE
    static void parse_impl(const char*&, MessageResult<num_fields>&) noexcept {}

    // Recursive case: parse one field, advance pointer, recurse
    template<int I, typename FP, typename... Rest>
    CTDP_STRATEGY_INLINE
    static void parse_impl(const char*& p,
                           MessageResult<num_fields>& result) noexcept
    {
        FieldResult fr = FP::parse(p);

        result.values [static_cast<std::size_t>(I)] = fr.value;
        result.lengths[static_cast<std::size_t>(I)] = fr.length;
        result.ok    &= fr.ok;
        result.total_bytes += fr.length + 1;  // +1 for SOH delimiter

        p += fr.length + 1;  // advance past field + SOH

        parse_impl<I + 1, Rest...>(p, result);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience alias: ReferenceParser<Digits...>
//
//  CompositeParser wired entirely with StrategyGeneric — the correctness
//  oracle.  All other composite parsers must agree with this on every message.
//
//  Usage:
//    using Ref4 = ReferenceParser<3, 6, 4, 8>;
//    auto r = Ref4::parse_message(msg);
// ─────────────────────────────────────────────────────────────────────────────

template<int... Digits>
using ReferenceParser = CompositeParser<
    Field<StrategyGeneric<Digits>, Digits>...
>;

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_COMPOSITE_PARSER_H
