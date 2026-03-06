#ifndef CTDP_CALIBRATOR_FIX_STRATEGY_HYBRID_H
#define CTDP_CALIBRATOR_FIX_STRATEGY_HYBRID_H

// ============================================================
//  strategy_hybrid.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  StrategyHybrid<PrefixDigits, SuffixDigits>
//
//  SWAR prefix scan + Unrolled suffix.  Designed for fields
//  that have a known-structure suffix following a variable-
//  length or separator-bounded prefix.
//
//  Primary use case — UTCTimestamp (FIX tag 60):
//    20240115-08:30:00.000
//    |----8--| |---12---|
//    YYYYMMDD  HH:MM:SS.mmm
//
//  For the benchmark harness (pure integer fields), Hybrid
//  treats the field as:
//    PrefixDigits: scanned via SWAR (stops at SOH or after
//                  PrefixDigits bytes, whichever comes first)
//    SuffixDigits: parsed via Unrolled immediately after
//                  the prefix (no separator consumed)
//
//  The combined parsed value is:
//    value = prefix_value * 10^SuffixDigits + suffix_value
//
//  This is numerically equivalent to parsing the whole field
//  as a single (PrefixDigits + SuffixDigits)-digit integer,
//  which makes the correctness check against StrategyGeneric
//  straightforward: they should agree on the same integer value.
//
//  Note: In real UTCTimestamp parsing the '-' separator would
//  be consumed between prefix and suffix; that variant is left
//  to application-level code above CompositeParser.
// ============================================================

#include <ctdp/calibrator/fix/strategy_swar.h>
#include <ctdp/calibrator/fix/strategy_unrolled.h>

namespace ctdp::calibrator::fix {

namespace detail {
    // Raise 10 to a compile-time power
    template<int N>
    inline constexpr std::uint64_t pow10_ct =
        (N == 0) ? 1ULL : 10ULL * pow10_ct<N-1>;

    template<>
    inline constexpr std::uint64_t pow10_ct<0> = 1ULL;
} // namespace detail

template<int PrefixDigits, int SuffixDigits>
    requires (PrefixDigits >= 1 && SuffixDigits >= 1
           && PrefixDigits + SuffixDigits <= 20)
struct StrategyHybrid {

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static FieldResult parse(const char* p) noexcept
    {
        // Prefix: SWAR scan up to PrefixDigits bytes
        auto prefix = StrategySWAR<PrefixDigits>::parse(p);
        if (!prefix.ok) return { 0, prefix.length, false };

        // If prefix hit SOH early (short field), no suffix
        if (prefix.length < PrefixDigits) {
            return { prefix.value, prefix.length, true };
        }

        // Suffix: fixed-length unrolled, starts immediately after prefix
        auto suffix = StrategyUnrolled<SuffixDigits>::parse(p + PrefixDigits);

        const int    total = PrefixDigits + SuffixDigits;
        const std::uint64_t combined =
            prefix.value * detail::pow10_ct<SuffixDigits> + suffix.value;

        return { combined, total, true };
    }
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_STRATEGY_HYBRID_H
