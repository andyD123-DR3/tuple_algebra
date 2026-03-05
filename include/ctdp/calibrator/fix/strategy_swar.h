#ifndef CTDP_CALIBRATOR_FIX_STRATEGY_SWAR_H
#define CTDP_CALIBRATOR_FIX_STRATEGY_SWAR_H

// ============================================================
//  strategy_swar.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  StrategySWAR<MaxDigits>
//
//  SWAR (SIMD Within A Register): loads 8 bytes at a time into
//  a uint64_t via std::memcpy (handles unaligned loads portably),
//  converts the full word from ASCII to binary, then detects the
//  first SOH byte using a parallel comparison trick.
//
//  Good for long variable-length fields (8+ digits): BodyLength,
//  MsgSeqNum, UTCTimestamp numeric prefix.
//
//  SOH detection (the SWAR trick):
//    Let W = loaded word XOR 0x0101010101010101 (SOH broadcast).
//    A byte that was SOH becomes 0x00 after XOR.
//    Then (W - LO) & ~W & HI detects zero bytes, where:
//      LO = 0x0101010101010101
//      HI = 0x8080808080808080
//    The position of the first zero byte gives the field length.
//
//  Little-endian assumption: the first byte of the field is
//  in the least-significant byte of the loaded word.  On
//  big-endian systems this strategy produces incorrect results;
//  a compile-time check guards against accidental BE use.
//
//  Fallback: when MaxDigits < 8 or the remaining field is short,
//  falls back to StrategyUnrolled for the tail.
// ============================================================

#include <ctdp/calibrator/fix/fix_field_result.h>
#include <ctdp/calibrator/fix/strategy_generic.h>  // tail fallback

#include <cstring>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  error "StrategySWAR requires a little-endian platform"
#endif

namespace ctdp::calibrator::fix {

// ── pow10 lookup table used by StrategySWAR ──────────────────────────────────
namespace detail {
inline constexpr std::uint64_t pow10_table[9] = {
    1ULL,          // 10^0
    10ULL,         // 10^1
    100ULL,        // 10^2
    1000ULL,       // 10^3
    10000ULL,      // 10^4
    100000ULL,     // 10^5
    1000000ULL,    // 10^6
    10000000ULL,   // 10^7
    100000000ULL,  // 10^8
};

/// Detect the position of the first SOH (0x01) byte in a uint64_t word.
/// Returns the byte index [0,7] of the first SOH, or 8 if none found.
/// Word is assumed to be loaded in native (little-endian) byte order:
/// byte 0 of the input string is in bits [7:0].
[[nodiscard]] CTDP_STRATEGY_INLINE
int first_soh_pos(std::uint64_t w) noexcept
{
    // XOR with SOH broadcast: bytes that were SOH become 0x00
    constexpr std::uint64_t SOH_BROADCAST = 0x0101010101010101ULL;
    constexpr std::uint64_t LO            = 0x0101010101010101ULL;
    constexpr std::uint64_t HI            = 0x8080808080808080ULL;

    std::uint64_t x = w ^ SOH_BROADCAST;
    // Zero-byte detection: ((x - LO) & ~x & HI) != 0 for zero bytes
    std::uint64_t zero_mask = (x - LO) & ~x & HI;
    if (zero_mask == 0) return 8;  // no SOH in this word
    // Lowest set bit in zero_mask is at bit 7 + 8*pos of the first zero byte
    // __builtin_ctzll counts trailing zeros; divide by 8 gives byte index
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<int>(__builtin_ctzll(zero_mask)) >> 3;
#else
    // Portable fallback: scan bytes
    for (int i = 0; i < 8; ++i)
        if (((zero_mask >> (i * 8)) & 0xFFu) != 0) return i;
    return 8;
#endif
}

/// Parse up to 8 ASCII digits from a uint64_t word (little-endian byte order).
/// `count` is the number of valid bytes to process [1,8].
[[nodiscard]] CTDP_STRATEGY_INLINE
std::uint64_t parse_word_digits(std::uint64_t w, int count) noexcept
{
    std::uint64_t val = 0;
    for (int i = 0; i < count; ++i) {
        const unsigned char d = static_cast<unsigned char>(w & 0xFFu) - '0';
        w >>= 8;
        val = val * 10u + d;
    }
    return val;
}

} // namespace detail

template<int MaxDigits>
    requires (MaxDigits >= 1 && MaxDigits <= 20)
struct StrategySWAR {

    [[nodiscard]] CTDP_STRATEGY_INLINE
    static FieldResult parse(const char* p) noexcept
    {
        std::uint64_t value = 0;
        int           pos   = 0;

        // Process 8-byte words while at least 8 bytes remain
        while (pos + 8 <= MaxDigits) {
            std::uint64_t w;
            std::memcpy(&w, p + pos, 8);

            int soh = detail::first_soh_pos(w);
            if (soh < 8) {
                // SOH found within this word
                value = value * detail::pow10_table[soh] +
                        detail::parse_word_digits(w, soh);
                return { value, pos + soh, true };
            }
            // All 8 bytes are digits
            value = value * 100000000ULL + detail::parse_word_digits(w, 8);
            pos += 8;
        }

        // Tail: remaining bytes < 8 — use byte-by-byte scan
        while (pos < MaxDigits) {
            const unsigned char c = static_cast<unsigned char>(p[pos]);
            if (c == 0x01u) break;
            const unsigned char d = c - '0';
            if (d > 9u) return { value, pos, false };
            value = value * 10u + d;
            ++pos;
        }

        return { value, pos, true };
    }
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_STRATEGY_SWAR_H
