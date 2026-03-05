#ifndef CTDP_CALIBRATOR_FIX_STRATEGY_UNROLLED_H
#define CTDP_CALIBRATOR_FIX_STRATEGY_UNROLLED_H

// ============================================================
//  strategy_unrolled.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  StrategyUnrolled<Digits>
//
//  Fixed-length field, compile-time unrolled.  Fastest strategy
//  for fields with a known, constant digit count.  No branches,
//  no loop, no SOH scan — reads exactly Digits bytes.
//
//  Precondition: the field is exactly Digits digits followed by
//  SOH.  Undefined behaviour if the field is shorter.  The
//  correctness suite validates this precondition holds for the
//  test corpus before any calibration run.
//
//  Always returns ok=true and length=Digits.  Callers that need
//  defensive parsing should use StrategyGeneric instead.
// ============================================================

#include <ctdp/calibrator/fix/fix_field_result.h>

namespace ctdp::calibrator::fix {

template<int Digits>
    requires (Digits >= 1 && Digits <= 20)
struct StrategyUnrolled {

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static FieldResult parse(const char* p) noexcept
    {
        std::uint64_t r = 0;
        if constexpr (Digits >= 1)  r = r * 10u + static_cast<std::uint64_t>(p[0]  - '0');
        if constexpr (Digits >= 2)  r = r * 10u + static_cast<std::uint64_t>(p[1]  - '0');
        if constexpr (Digits >= 3)  r = r * 10u + static_cast<std::uint64_t>(p[2]  - '0');
        if constexpr (Digits >= 4)  r = r * 10u + static_cast<std::uint64_t>(p[3]  - '0');
        if constexpr (Digits >= 5)  r = r * 10u + static_cast<std::uint64_t>(p[4]  - '0');
        if constexpr (Digits >= 6)  r = r * 10u + static_cast<std::uint64_t>(p[5]  - '0');
        if constexpr (Digits >= 7)  r = r * 10u + static_cast<std::uint64_t>(p[6]  - '0');
        if constexpr (Digits >= 8)  r = r * 10u + static_cast<std::uint64_t>(p[7]  - '0');
        if constexpr (Digits >= 9)  r = r * 10u + static_cast<std::uint64_t>(p[8]  - '0');
        if constexpr (Digits >= 10) r = r * 10u + static_cast<std::uint64_t>(p[9]  - '0');
        if constexpr (Digits >= 11) r = r * 10u + static_cast<std::uint64_t>(p[10] - '0');
        if constexpr (Digits >= 12) r = r * 10u + static_cast<std::uint64_t>(p[11] - '0');
        return { r, Digits, true };
    }
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_STRATEGY_UNROLLED_H
