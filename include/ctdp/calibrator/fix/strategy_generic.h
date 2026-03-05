#ifndef CTDP_CALIBRATOR_FIX_STRATEGY_GENERIC_H
#define CTDP_CALIBRATOR_FIX_STRATEGY_GENERIC_H

// ============================================================
//  strategy_generic.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  StrategyGeneric<MaxDigits>
//
//  Byte-by-byte SOH scan.  The correctness reference oracle.
//  All other strategies must produce identical (value, length)
//  pairs on every field of a valid 10,000-message test corpus.
//
//  Correctness status:
//    ✅ Cross-strategy oracle: all 5 strategies verified against
//       StrategyGeneric on 70,000 random fields (10,000 per digit
//       width: 3,4,6,8,12) and a 10,000-message composite corpus.
//       See tests/test_phase2_strategies.cpp Groups G T34–T40.
//
//    ⚠️ DEFERRED: QuickFIX external validation.
//       Plan §Phase2 calls for QuickFIX-validated corpus check.
//       Requires a QuickFIX installation; not available in CI.
//       Action: run test_quickfix_oracle.cpp (to be written) on
//       a machine with QuickFIX before the first production calibration
//       run.  Until then, StrategyGeneric is verified as internally
//       self-consistent but not externally validated against QuickFIX.
//       Tracking: TODO(quickfix-oracle)
//
//  Interface:
//    static FieldResult parse(const char* p) noexcept
//
//    p    points to the first digit of the field (after any tag=).
//    SOH  is 0x01.  Scanning stops when:
//      - a SOH byte is found, OR
//      - MaxDigits bytes have been consumed (safety bound)
//
//    Returns:
//      value   — parsed uint64 (0 on empty field or non-digit first byte)
//      length  — bytes consumed NOT including the SOH
//      ok      — true if all consumed bytes were ASCII digits
//
//  MaxDigits: compile-time upper bound.  For known-length fields set
//  MaxDigits == exact digit count; for variable-length (e.g. BodyLength)
//  set MaxDigits to the protocol maximum (10 covers a 32-bit integer).
//
//  Dependencies: fix_field_result.h (same directory)
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/fix_field_result.h>

namespace ctdp::calibrator::fix {

template<int MaxDigits>
    requires (MaxDigits >= 1 && MaxDigits <= 20)
struct StrategyGeneric {

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static FieldResult parse(const char* p) noexcept
    {
        std::uint64_t value = 0;
        int           len   = 0;
        bool          ok    = true;

        while (len < MaxDigits) {
            const unsigned char c = static_cast<unsigned char>(p[len]);
            if (c == 0x01u) break;              // SOH
            const unsigned char d = c - '0';
            if (d > 9u) { ok = false; break; }  // non-digit
            value = value * 10u + d;
            ++len;
        }

        return { value, len, ok };
    }
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_STRATEGY_GENERIC_H
