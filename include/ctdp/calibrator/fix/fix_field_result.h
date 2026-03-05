#ifndef CTDP_CALIBRATOR_FIX_FIELD_RESULT_H
#define CTDP_CALIBRATOR_FIX_FIELD_RESULT_H

// ============================================================
//  fix_field_result.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  FieldResult — common return type for all per-field strategies.
//  CTDP_STRATEGY_INLINE — portable always-inline for strategy hot paths.
//
//  All five strategy headers (Generic, Loop, Unrolled, SWAR, Hybrid)
//  and composite_parser.h depend on this header.  No other dependencies.
// ============================================================

#include <cstdint>

#if defined(_MSC_VER)
#  define CTDP_STRATEGY_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define CTDP_STRATEGY_INLINE [[gnu::always_inline]] inline
#else
#  define CTDP_STRATEGY_INLINE inline
#endif

namespace ctdp::calibrator::fix {

/// Result of parsing one SOH-terminated FIX field.
struct FieldResult {
    std::uint64_t value;   ///< Parsed integer value
    int           length;  ///< Bytes consumed (NOT including the SOH delimiter)
    bool          ok;      ///< false if a non-digit byte was encountered before SOH
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_FIELD_RESULT_H
