// ctdp/space/partition.h — Partition ordinate type
//
// A partition of N items groups them into non-empty, non-overlapping subsets.
// The canonical representation is a restricted growth string (RGS):
//   [0, 0, 1, 1, 2] means items 0,1 in group 0; items 2,3 in group 1; item 4 in group 2.
//
// Restriction: the first occurrence of label k must precede the first occurrence
// of label k+1. This avoids equivalent representations like [1,1,0,0,2].
//
// Cardinality = Bell(N): Bell(1)=1, Bell(2)=2, Bell(3)=5, Bell(4)=15,
//   Bell(5)=52, ..., Bell(12)=4,213,597.
//
// Feature encoding: pairwise co-membership — N*(N-1)/2 binary features,
// one per pair (i,j), indicating whether items i and j share a group.
//
// Usage:
//   auto part = make_partition<5>("lane_grouping");
//   // part.cardinality() == 52 (Bell(5))
//   // part.value_at(0) == [0,0,0,0,0] (all one group)
//   // part.value_at(51) == [0,1,2,3,4] (all singletons)
//   // Feature width = 5*4/2 = 10 binary features
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_PARTITION_H
#define CTDP_SPACE_PARTITION_H

#include "ctdp/space/concepts.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// partition_value<N> — a partition of N items as a restricted growth string
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
struct partition_value {
    static_assert(N > 0 && N <= 16, "partition_value: N must be in [1, 16]");

    std::array<std::uint8_t, N> labels{};

    constexpr bool operator==(const partition_value&) const = default;
    constexpr auto operator<=>(const partition_value&) const = default;

    /// Number of distinct groups in this partition.
    constexpr std::size_t num_groups() const {
        std::uint8_t mx = 0;
        for (auto l : labels) if (l > mx) mx = l;
        return static_cast<std::size_t>(mx) + 1;
    }

    /// Are items i and j in the same group?
    constexpr bool same_group(std::size_t i, std::size_t j) const {
        return labels[i] == labels[j];
    }

    /// Is this a valid restricted growth string?
    constexpr bool is_canonical() const {
        if (N == 0) return true;
        if (labels[0] != 0) return false;
        std::uint8_t max_seen = 0;
        for (std::size_t i = 1; i < N; ++i) {
            if (labels[i] > max_seen + 1) return false;
            if (labels[i] > max_seen) max_seen = labels[i];
        }
        return true;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Partition tables — precomputed T(m, k) for ranking/unranking
//
// T(m, k) = number of restricted growth strings of length m where
//           k groups have been used so far (entering the sequence).
//
// Recurrence: T(0, k) = 1
//             T(m, k) = k * T(m-1, k) + T(m-1, k+1)
//
// Bell(n) = T(n-1, 1) for n >= 1.
// ═══════════════════════════════════════════════════════════════════════

namespace detail {

// Maximum supported partition size
static constexpr std::size_t partition_max_n = 16;

// T[m][k] for m in [0, partition_max_n], k in [0, partition_max_n+1]
// k=0 is unused but simplifies indexing.
struct partition_tables {
    // T[m][k]: restricted growth string count
    // m = remaining positions (0..16), k = groups used so far (1..17)
    std::uint64_t T[partition_max_n + 1][partition_max_n + 2]{};

    constexpr partition_tables() {
        // Base case: T(0, k) = 1 for all k
        for (std::size_t k = 0; k <= partition_max_n + 1; ++k)
            T[0][k] = 1;

        // Fill row by row: T(m, k) = k * T(m-1, k) + T(m-1, k+1)
        for (std::size_t m = 1; m <= partition_max_n; ++m) {
            for (std::size_t k = partition_max_n + 1; k >= 1; --k) {
                std::uint64_t next_k = (k + 1 <= partition_max_n + 1) ? T[m-1][k+1] : 0;
                T[m][k] = k * T[m-1][k] + next_k;
            }
        }
    }

    /// Bell number: total partitions of n items.
    constexpr std::uint64_t bell(std::size_t n) const {
        if (n == 0) return 1;
        return T[n - 1][1];
    }
};

// Single constexpr instance
inline constexpr partition_tables tables{};

// ── Ranking: partition_value → index ─────────────────────────────────
//
// Lexicographic rank of a restricted growth string.
//
// At each position i (after the fixed a[0]=0), we count how many
// complete RGS strings have a smaller label at position i — these
// all precede the target in lexicographic order.
//
// Worked example: N=3, rank of [0, 1, 0]
//   Bell(3) = 5 partitions: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [0,1,2]
//   Expected rank = 2.
//
//   k=1 (one group used after a[0]=0)
//
//   i=1: label=1, remaining=1
//     label == k (new group): skip k=1 existing-group choices,
//     each with T(1,1)=2 continuations → rank += 1 × 2 = 2.  k→2.
//
//   i=2: label=0, remaining=0
//     label < k: 0 existing-group choices precede → rank += 0 × T(0,2) = 0.
//
//   Total rank = 2. ✓

template <std::size_t N>
constexpr std::size_t rank_partition(const partition_value<N>& pv) {
    if constexpr (N <= 1) return 0;

    std::size_t rank = 0;
    std::uint8_t k = 1;  // groups used so far (after a[0] = 0)

    for (std::size_t i = 1; i < N; ++i) {
        std::size_t remaining = N - i - 1;
        std::uint8_t label = pv.labels[i];

        // Count all RGS strings that have a smaller label at position i.
        // Labels 0..k-1 are existing groups; label k would open a new group.
        // Each preceding label contributes T(remaining, k) continuations.
        if (label < k) {
            rank += static_cast<std::size_t>(label) * tables.T[remaining][k];
        } else {
            // label == k (new group): all k existing labels precede it
            rank += static_cast<std::size_t>(k) * tables.T[remaining][k];
        }

        if (label == k) ++k;
    }
    return rank;
}

// ── Unranking: index → partition_value ───────────────────────────────
//
// Inverse of ranking: given a lexicographic index, reconstruct the RGS.
//
// At each position, try labels in order (0, 1, ..., k). Each existing
// group label (0..k-1) accounts for T(remaining, k) strings. If the
// rank is smaller than that count, we've found the label; otherwise
// subtract and try the next.
//
// Worked example: N=3, unrank index 2 → [0, 1, 0]
//
//   k=1, a[0]=0, rank=2
//
//   i=1: remaining=1
//     Try label=0: T(1,1)=2 continuations. rank=2 ≥ 2 → rank -= 2 → rank=0. Next.
//     Try label=1: label==k → new group. Accept.  a[1]=1, k→2.
//
//   i=2: remaining=0
//     Try label=0: T(0,2)=1 continuation. rank=0 < 1 → accept.  a[2]=0.
//
//   Result: [0, 1, 0]. ✓

template <std::size_t N>
constexpr partition_value<N> unrank_partition(std::size_t rank) {
    partition_value<N> pv{};
    pv.labels[0] = 0;

    if constexpr (N <= 1) return pv;

    std::uint8_t k = 1;  // groups used after a[0] = 0

    for (std::size_t i = 1; i < N; ++i) {
        std::size_t remaining = N - i - 1;

        // Try labels 0, 1, ..., k in order
        // Labels 0..k-1 are existing (each costs T(remaining, k))
        // Label k is new group (costs T(remaining, k+1))
        std::uint8_t label = 0;

        // First, try existing group labels
        while (label < k) {
            std::uint64_t count = tables.T[remaining][k];
            if (rank < count) break;
            rank -= static_cast<std::size_t>(count);
            ++label;
        }

        // If we exhausted existing groups, label == k (new group)
        // No need to subtract further — the remaining rank indexes within this choice

        pv.labels[i] = label;
        if (label == k) ++k;
    }
    return pv;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// partition_desc<N> — dimension descriptor for partitions of N items
//
// Satisfies dimension_descriptor. The value_type is partition_value<N>.
// Feature encoding: pairwise co-membership, N*(N-1)/2 binary features.
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
struct partition_desc {
    static_assert(N > 0 && N <= 16, "partition_desc: N must be in [1, 16]");

    std::string_view name;
    encoding_hint encoding = encoding_hint::pairwise;
    static constexpr dim_kind kind = dim_kind::partition;
    static constexpr std::size_t item_count = N;
    using value_type = partition_value<N>;

    constexpr explicit partition_desc(std::string_view n) : name(n) {}

    /// Override encoding. Only pairwise is supported for partitions.
    /// One-hot over Bell(N) would be infeasible; raw has no meaning.
    constexpr partition_desc encoded_as(encoding_hint e) const {
        if (e != encoding_hint::pairwise)
            throw std::invalid_argument(
                "partition_desc only supports pairwise encoding");
        return *this;
    }
    constexpr encoding_hint default_encoding() const { return encoding; }

    /// Cardinality = Bell(N).
    constexpr std::size_t cardinality() const {
        return static_cast<std::size_t>(detail::tables.bell(N));
    }

    /// i-th partition in canonical (lexicographic RGS) order.
    constexpr value_type value_at(std::size_t i) const {
        return detail::unrank_partition<N>(i);
    }

    /// Is this partition value in canonical form?
    constexpr bool contains(value_type v) const {
        return v.is_canonical();
    }

    /// Lexicographic rank of a partition value.
    constexpr std::size_t index_of(value_type v) const {
        if (!v.is_canonical()) return cardinality();  // out-of-range sentinel
        return detail::rank_partition<N>(v);
    }

    /// Feature width for pairwise co-membership encoding.
    /// This is NOT the cardinality — it's the number of pair indicators.
    constexpr std::size_t feature_width() const {
        return N * (N - 1) / 2;
    }

    /// Write pairwise co-membership features to output buffer.
    /// Buffer must have at least feature_width() doubles.
    void write_features(value_type val, double* out) const {
        std::size_t k = 0;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j)
                out[k++] = val.same_group(i, j) ? 1.0 : 0.0;
    }
};

// Factory
template <std::size_t N>
constexpr auto make_partition(std::string_view name) {
    return partition_desc<N>(name);
}

// Concept check
static_assert(dimension_descriptor<partition_desc<3>>);
static_assert(dimension_descriptor<partition_desc<5>>);

// ═══════════════════════════════════════════════════════════════════════
// Bell number convenience — useful for tests and static_asserts
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
inline constexpr std::uint64_t bell_number = detail::tables.bell(N);

static_assert(bell_number<1> == 1);
static_assert(bell_number<2> == 2);
static_assert(bell_number<3> == 5);
static_assert(bell_number<4> == 15);
static_assert(bell_number<5> == 52);
static_assert(bell_number<6> == 203);
static_assert(bell_number<7> == 877);
static_assert(bell_number<8> == 4140);

} // namespace ctdp::space

#endif // CTDP_SPACE_PARTITION_H
