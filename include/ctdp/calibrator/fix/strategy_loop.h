#ifndef CTDP_CALIBRATOR_FIX_STRATEGY_LOOP_H
#define CTDP_CALIBRATOR_FIX_STRATEGY_LOOP_H

// ============================================================
//  strategy_loop.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  StrategyLoop<MaxDigits>
//
//  Vectorisation-friendly sequential scan.  Good for numeric
//  fields of moderate length (3–8 digits).  The compiler can
//  auto-vectorise the inner loop across multiple messages in a
//  batch context; no branch inside the digit loop.
//
//  Unlike StrategyGeneric, the loop runs exactly MaxDigits
//  iterations — SOH detection is done by zeroing the
//  contribution when a non-digit is found (branchless body).
//  The first non-digit byte terminates accumulation: that
//  position is recorded as `length`.
//
//  Trade-off: always executes MaxDigits iterations, so it
//  over-scans short fields.  Prefer StrategyUnrolled for
//  known fixed-length fields.
// ============================================================

#include <ctdp/calibrator/fix/fix_field_result.h>

namespace ctdp::calibrator::fix {

template<int MaxDigits>
    requires (MaxDigits >= 1 && MaxDigits <= 20)
struct StrategyLoop {

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static FieldResult parse(const char* p) noexcept
    {
        std::uint64_t value = 0;
        int           len   = 0;
        bool          ok    = true;
        bool          ended = false;

        for (int i = 0; i < MaxDigits; ++i) {
            const unsigned char c = static_cast<unsigned char>(p[i]);
            const unsigned char d = c - '0';
            const bool is_digit   = (d <= 9u);
            const bool is_soh     = (c == 0x01u);

            if (!ended && (is_soh || !is_digit)) {
                len   = i;
                ok    = is_soh || (i > 0);  // non-digit on first byte → not ok
                if (!is_soh && i == 0) ok = false;
                if (!is_soh && i  > 0) ok = true;
                ended = true;
            }

            // Branchless accumulation: contribute only while not ended
            if (!ended) {
                value = value * 10u + static_cast<std::uint64_t>(d);
            }
        }

        if (!ended) len = MaxDigits;
        return { value, len, ok };
    }
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_STRATEGY_LOOP_H
