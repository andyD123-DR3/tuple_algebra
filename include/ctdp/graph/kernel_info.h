// graph/annotation/kernel_info.h - Computation kernel metadata
// Part of the compile-time DP library (C++20)
//
// Each computation graph node has cost properties needed for optimization:
// flops, memory traffic, fusability, arithmetic intensity.
// kernel_info is stored in property_map<kernel_info, MaxV> and consumed
// by fusion_legal, coarsen, and cost model analysis.

#ifndef CTDP_GRAPH_KERNEL_INFO_H
#define CTDP_GRAPH_KERNEL_INFO_H

#include "graph_concepts.h"
#include "property_map.h"

#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// Kernel tag: lightweight identifier for operation type
// =============================================================================

/// Lightweight operation-type tag for kernel classification.
/// Used by fusion_legal to determine which kernel pairs can merge.
struct kernel_tag {
    std::uint16_t value{};

    friend constexpr bool operator==(kernel_tag, kernel_tag) = default;
    friend constexpr auto operator<=>(kernel_tag, kernel_tag) = default;
};

/// Sentinel: unclassified kernel.
inline constexpr kernel_tag unclassified_kernel{0};

// =============================================================================
// Kernel info: per-node cost model metadata
// =============================================================================

/// Cost model metadata for a computation kernel.
///
/// Describes the computational profile of a graph node. Used for:
/// - Fusion legality: can two kernels share a loop body?
/// - Coarsening cost: aggregate cost of fused groups
/// - Roofline analysis: compute-bound or memory-bound?
/// - Scheduling: critical-path bottlenecks
struct kernel_info {
    kernel_tag tag{};
    std::size_t flops = 0;
    std::size_t bytes_read = 0;
    std::size_t bytes_written = 0;
    bool is_fusable = true;

    [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
        return bytes_read + bytes_written;
    }

    [[nodiscard]] constexpr double arithmetic_intensity() const noexcept {
        auto const tb = total_bytes();
        if (tb == 0) return 0.0;
        return static_cast<double>(flops) / static_cast<double>(tb);
    }

    [[nodiscard]] constexpr bool
    is_compute_bound(double machine_balance) const noexcept {
        return arithmetic_intensity() >= machine_balance;
    }

    /// Merge two kernel_info values (for fused kernels).
    /// Flops/bytes sum. Tag preserved if matching, else unclassified.
    /// Fusability AND'd.
    [[nodiscard]] constexpr kernel_info
    merged_with(kernel_info const& other) const noexcept {
        return kernel_info{
            .tag = (tag == other.tag) ? tag : unclassified_kernel,
            .flops = flops + other.flops,
            .bytes_read = bytes_read + other.bytes_read,
            .bytes_written = bytes_written + other.bytes_written,
            .is_fusable = is_fusable && other.is_fusable
        };
    }

    friend constexpr bool
    operator==(kernel_info const&, kernel_info const&) = default;
};

inline constexpr kernel_info default_kernel_info{};

// =============================================================================
// Type alias and factories
// =============================================================================

template<std::size_t MaxV>
using kernel_map = property_map<kernel_info, MaxV>;

template<std::size_t MaxV, graph_queryable G>
[[nodiscard]] constexpr kernel_map<MaxV>
make_uniform_kernel_map(G const& g, kernel_info const& info) {
    return make_uniform_property_map<kernel_info, MaxV>(g, info);
}

template<std::size_t MaxV, graph_queryable G, typename Fn>
[[nodiscard]] constexpr kernel_map<MaxV>
make_kernel_map(G const& g, Fn&& fn) {
    return make_property_map<kernel_info, MaxV>(g, static_cast<Fn&&>(fn));
}

// =============================================================================
// Aggregate queries
// =============================================================================

template<std::size_t MaxV>
[[nodiscard]] constexpr std::size_t
total_flops(kernel_map<MaxV> const& kmap) noexcept {
    std::size_t sum = 0;
    for (std::size_t i = 0; i < kmap.size(); ++i) {
        sum += kmap[i].flops;
    }
    return sum;
}

template<std::size_t MaxV>
[[nodiscard]] constexpr std::size_t
total_bytes(kernel_map<MaxV> const& kmap) noexcept {
    std::size_t sum = 0;
    for (std::size_t i = 0; i < kmap.size(); ++i) {
        sum += kmap[i].total_bytes();
    }
    return sum;
}

template<std::size_t MaxV>
[[nodiscard]] constexpr double
overall_arithmetic_intensity(kernel_map<MaxV> const& kmap) noexcept {
    auto const tb = total_bytes(kmap);
    if (tb == 0) return 0.0;
    return static_cast<double>(total_flops(kmap)) /
           static_cast<double>(tb);
}

template<std::size_t MaxV>
[[nodiscard]] constexpr std::size_t
fusable_count(kernel_map<MaxV> const& kmap) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < kmap.size(); ++i) {
        if (kmap[i].is_fusable) ++count;
    }
    return count;
}

template<std::size_t MaxV>
[[nodiscard]] constexpr std::size_t
count_by_tag(kernel_map<MaxV> const& kmap, kernel_tag tag) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < kmap.size(); ++i) {
        if (kmap[i].tag == tag) ++count;
    }
    return count;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_KERNEL_INFO_H
