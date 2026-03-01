#ifndef CTDP_CALIBRATOR_FEATURE_ENCODER_H
#define CTDP_CALIBRATOR_FEATURE_ENCODER_H

// ctdp::calibrator::feature_encoder — Point-to-numeric-vector encoding
//
// Converts a point_type into a fixed-width std::array<float, W> for
// the ML pipeline. Column names are static constexpr to avoid lifetime
// issues with dynamically constructed views.
//
// Design v2.2 §5.3:
//   Returns std::array<float, W>, not char* buffer.
//   Enum dimensions get one-hot encoding; scalars pass through directly.
//   The CSV writer composes the feature encoder with the snapshot formatter.
//
// The FeatureEncoder concept is deliberately simple — users implement it
// for each point_type.  Helper functions (encode_scalar, encode_enum)
// provide common building blocks.

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>

namespace ctdp::calibrator {

// ─── FeatureEncoder concept ──────────────────────────────────────

/// A FeatureEncoder converts a point_type into a fixed-width numeric
/// vector suitable for cost model training.
///
/// Required:
///   E::width          — the number of output features (std::size_t)
///   E::column_names   — static constexpr array of width string_views
///   enc.encode(pt)    — returns std::array<float, width>
///
template <typename E, typename PointType>
concept FeatureEncoder = requires(E const& enc, PointType const& pt) {
    { E::width } -> std::convertible_to<std::size_t>;
    { E::column_names };
    { enc.encode(pt) } -> std::convertible_to<std::array<float, E::width>>;
};

// ─── Encoding helpers ────────────────────────────────────────────

/// Encode a scalar value as a single float feature.
/// Usage: features[i] = encode_scalar(pt.field_count);
template <typename T>
    requires std::integral<T> || std::floating_point<T>
[[nodiscard]] constexpr float encode_scalar(T value) noexcept {
    return static_cast<float>(value);
}

/// One-hot encode an enum value into a sub-array.
/// Cardinality is the number of enum values (must cover [0, Cardinality)).
///
/// Usage:
///   auto bits = encode_one_hot<3>(static_cast<int>(pt.strategy));
///   // strategy::A=0 → {1,0,0}  strategy::B=1 → {0,1,0}  etc.
///
template <std::size_t Cardinality>
[[nodiscard]] constexpr auto encode_one_hot(int value) noexcept
    -> std::array<float, Cardinality>
{
    std::array<float, Cardinality> result{};
    auto idx = static_cast<std::size_t>(value);
    if (idx < Cardinality) {
        result[idx] = 1.0f;
    }
    return result;
}

/// Copy a sub-array of features into a destination array at a given offset.
/// Returns the new offset (past the end of the copied range).
///
/// Usage:
///   std::size_t pos = 0;
///   pos = copy_features(features, pos, encode_one_hot<3>(strategy));
///   pos = copy_features(features, pos, std::array{encode_scalar(size)});
///
template <std::size_t DstN, std::size_t SrcN>
constexpr std::size_t copy_features(
    std::array<float, DstN>& dst,
    std::size_t offset,
    std::array<float, SrcN> const& src) noexcept
{
    for (std::size_t i = 0; i < SrcN && (offset + i) < DstN; ++i) {
        dst[offset + i] = src[i];
    }
    return offset + SrcN;
}

// ─── PointFormatter adapter from FeatureEncoder ──────────────────

/// Generates a PointFormatter (for csv_writer) from a FeatureEncoder.
/// This bridges Phase 2's csv_writer with Phase 3's feature encoding.
///
/// CSV columns are the feature column names; values are the encoded floats.
///
template <typename Encoder, typename PointType>
    requires FeatureEncoder<Encoder, PointType>
struct feature_point_formatter {
    Encoder encoder;

    [[nodiscard]] auto csv_header() const -> std::string {
        std::string hdr;
        for (std::size_t i = 0; i < Encoder::width; ++i) {
            if (i > 0) hdr += ',';
            hdr += Encoder::column_names[i];
        }
        return hdr;
    }

    [[nodiscard]] auto to_csv(PointType const& pt) const -> std::string {
        auto features = encoder.encode(pt);
        std::string row;
        for (std::size_t i = 0; i < Encoder::width; ++i) {
            if (i > 0) row += ',';
            // Use enough precision to round-trip floats
            auto val = features[i];
            if (val == static_cast<float>(static_cast<int>(val))) {
                row += std::to_string(static_cast<int>(val));
            } else {
                // 6 significant digits for non-integer floats
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(val));
                row += buf;
            }
        }
        return row;
    }
};

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_FEATURE_ENCODER_H
