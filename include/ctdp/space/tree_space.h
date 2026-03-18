// ctdp/space/tree_space.h — Partition-rooted hierarchy
//
// Composes a partition_desc<N> root with per-group child spaces.
// For each legal partition value, the factory generates one child
// space per group; tree_space enumerates the Cartesian product of
// all groups' child spaces to produce composite tree_points.
//
// This layer is partition-rooted, not a fully abstract root type.
// It assumes: canonical labels 0..K-1, lane indices per group via
// group_lanes(), maximum group count = N. A generic root abstraction
// (grouping-traits) is deferred.
//
// Usage:
//   auto space = tree_space<3>(
//       "my_tree", make_partition<3>("grouping"),
//       my_child_factory, my_filter);
//   space.enumerate([&](const auto& pt) {
//       // pt.root is a partition_value<3>
//       // pt.group_plans is a vector of child points
//   });
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_TREE_SPACE_H
#define CTDP_SPACE_TREE_SPACE_H

#include "ctdp/space/concepts.h"
#include "ctdp/space/partition.h"
#include "ctdp/space/feature_bridge.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// group_lanes — extract lane indices for one group
//
// Span version: writes into caller buffer, returns count.
// Callback version: zero allocation.
// Both are O(N), single pass over labels.
// ═══════════════════════════════════════════════════════════════════════

/// Write the lane indices belonging to group_label into out_indices.
/// Returns the number of lanes in this group.
template <std::size_t N>
std::size_t group_lanes(
    const partition_value<N>& pv,
    std::size_t group_label,
    std::span<std::size_t> out_indices)
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (pv.labels[i] == static_cast<std::uint8_t>(group_label)) {
            if (count < out_indices.size())
                out_indices[count] = i;
            ++count;
        }
    }
    return count;
}

/// Call fn(lane_index) for each lane in the given group.
template <std::size_t N, typename F>
void for_each_lane_in_group(
    const partition_value<N>& pv,
    std::size_t group_label,
    F&& fn)
{
    for (std::size_t i = 0; i < N; ++i)
        if (pv.labels[i] == static_cast<std::uint8_t>(group_label))
            fn(i);
}

// ═══════════════════════════════════════════════════════════════════════
// tree_point — composite point: root partition + per-group child plans
//
// group_plans[k] is the plan for group k (canonical label order).
// group_plans.size() == root.num_groups().
//
// Requires ChildPoint to be equality-comparable (for deduplication).
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N, typename ChildPoint>
struct tree_point {
    partition_value<N> root;
    std::vector<ChildPoint> group_plans;

    std::size_t num_groups() const { return group_plans.size(); }

    bool operator==(const tree_point&) const = default;
    auto operator<=>(const tree_point&) const = default;
};

// ═══════════════════════════════════════════════════════════════════════
// tree_space — partition-rooted composite space
//
// Template parameters:
//   N             — partition size (number of items/lanes)
//   ChildFactory  — callable(partition_value<N>, size_t group, span<size_t> lanes)
//                   → child_space_type. Must be pure and deterministic.
//   Filter        — callable(partition_value<N>) → bool.
//                   Must be pure and deterministic.
//
// name_ must point to static or program-lifetime storage.
//
// The point reference passed to enumerate callbacks is a temporary
// constructed by copying. Callers may freely store it.
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N, typename ChildFactory, typename Filter>
struct tree_space {
    static constexpr std::size_t partition_size = N;

    // Deduce child space type from the factory's return type
    using child_space_type = std::invoke_result_t<
        const ChildFactory&,
        const partition_value<N>&,
        std::size_t,
        std::span<const std::size_t>>;
    using child_point_type = typename child_space_type::point_type;
    using point_type = tree_point<N, child_point_type>;

    std::string_view name_;
    partition_desc<N> root_desc_;
    ChildFactory child_factory_;
    Filter filter_;

    tree_space(std::string_view name,
               partition_desc<N> root,
               ChildFactory factory,
               Filter filter)
        : name_(name)
        , root_desc_(std::move(root))
        , child_factory_(std::move(factory))
        , filter_(std::move(filter))
    {}

    std::string_view space_name() const { return name_; }

    /// Cardinality: sum over legal roots of product of per-group
    /// child cardinalities. Child cardinality may vary between groups.
    std::size_t cardinality() const {
        std::size_t total = 0;
        for (std::size_t i = 0; i < root_desc_.cardinality(); ++i) {
            auto rv = root_desc_.value_at(i);
            if (!filter_(rv)) continue;

            std::size_t product = 1;
            auto K = rv.num_groups();
            for (std::size_t g = 0; g < K; ++g) {
                std::array<std::size_t, N> lane_buf{};
                auto count = group_lanes(rv, g, lane_buf);
                auto lanes = std::span<const std::size_t>{lane_buf.data(), count};
                auto child = child_factory_(rv, g, lanes);
                auto cc = child.cardinality();
                if (cc == 0) { product = 0; break; }
                product *= cc;  // TODO: saturating_mul if N grows
            }
            total += product;
        }
        return total;
    }

    /// Enumerate all composite points. For each legal root, produces
    /// the Cartesian product of all groups' child spaces.
    template <typename F>
    void enumerate(F&& fn) const {
        for (std::size_t i = 0; i < root_desc_.cardinality(); ++i) {
            auto rv = root_desc_.value_at(i);
            if (!filter_(rv)) continue;

            auto K = rv.num_groups();
            std::vector<child_point_type> partial;
            partial.reserve(K);
            enumerate_groups(rv, 0, K, partial, fn);
        }
    }

private:
    /// Recursive Cartesian product over groups.
    /// Builds partial plan vector, calls callback when all groups filled.
    template <typename F>
    void enumerate_groups(
        const partition_value<N>& rv,
        std::size_t group_idx,
        std::size_t num_groups,
        std::vector<child_point_type>& partial,
        F& fn) const
    {
        if (group_idx == num_groups) {
            fn(point_type{rv, partial});  // copies partial
            return;
        }

        std::array<std::size_t, N> lane_buf{};
        auto count = group_lanes(rv, group_idx, lane_buf);
        auto lanes = std::span<const std::size_t>{lane_buf.data(), count};
        auto child = child_factory_(rv, group_idx, lanes);

        child.enumerate([&](const auto& child_pt) {
            partial.push_back(child_pt);
            enumerate_groups(rv, group_idx + 1, num_groups, partial, fn);
            partial.pop_back();
        });
    }
};

/// Factory function for tree_space (deduces template args).
template <std::size_t N, typename ChildFactory, typename Filter>
auto make_tree_space(
    std::string_view name,
    partition_desc<N> root,
    ChildFactory factory,
    Filter filter)
{
    return tree_space<N, ChildFactory, Filter>(
        name, std::move(root), std::move(factory), std::move(filter));
}

// ═══════════════════════════════════════════════════════════════════════
// tree_bridge — feature encoding for composite points
//
// Root features: pairwise co-membership from partition_desc<N>.
// Child features: N fixed-width blocks, each encoded through a
// per-group bridge built from that group's actual child space.
// Unused blocks (groups K..N-1) are zero-filled.
//
// ChildBridgeFactory: callable(child_space) → child_bridge.
// For reduction spaces, this is simply default_bridge.
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N, typename ChildFactory, typename ChildBridgeFactory>
struct tree_bridge {
    partition_desc<N> root_desc_;
    ChildFactory child_factory_;
    ChildBridgeFactory bridge_factory_;
    std::size_t child_feature_width_;

    tree_bridge(partition_desc<N> root,
                ChildFactory child_factory,
                ChildBridgeFactory bridge_factory,
                std::size_t child_feature_width)
        : root_desc_(std::move(root))
        , child_factory_(std::move(child_factory))
        , bridge_factory_(std::move(bridge_factory))
        , child_feature_width_(child_feature_width)
    {}

    /// Total features = root pairwise + N * child block width.
    std::size_t num_features() const {
        return root_desc_.feature_width() + N * child_feature_width_;
    }

    /// Encode a composite point into a feature vector.
    template <typename ChildPoint>
    std::vector<double> encode(const tree_point<N, ChildPoint>& pt) const {
        std::vector<double> out(num_features(), 0.0);
        auto* cursor = out.data();

        // Root: pairwise co-membership features
        auto root_width = root_desc_.feature_width();
        root_desc_.write_features(pt.root, cursor);
        cursor += root_width;

        // Per-group child blocks
        auto K = pt.num_groups();
        assert(K <= N);

        for (std::size_t g = 0; g < K; ++g) {
            // Reconstruct child space and bridge for this group
            std::array<std::size_t, N> lane_buf{};
            auto count = group_lanes(pt.root, g, lane_buf);
            auto lanes = std::span<const std::size_t>{lane_buf.data(), count};
            auto child_space = child_factory_(pt.root, g, lanes);
            auto child_bridge = bridge_factory_(child_space);

            // Encode this group's plan into its block
            child_bridge.write_features(
                pt.group_plans[g],
                std::span<double>{cursor, child_feature_width_});
            cursor += child_feature_width_;
        }
        // Blocks K..N-1 are already zero-filled from vector init.

        return out;
    }
};

/// Factory function for tree_bridge.
template <std::size_t N, typename ChildFactory, typename ChildBridgeFactory>
auto make_tree_bridge(
    partition_desc<N> root,
    ChildFactory child_factory,
    ChildBridgeFactory bridge_factory,
    std::size_t child_feature_width)
{
    return tree_bridge<N, ChildFactory, ChildBridgeFactory>(
        std::move(root), std::move(child_factory),
        std::move(bridge_factory), child_feature_width);
}

// ═══════════════════════════════════════════════════════════════════════
// make_flat_tree_space — degenerate: wraps a child space as trivial tree
//
// Root: partition_desc<1> (one item, one group, Bell(1)=1).
// Factory: ignores arguments, returns the stored child space.
// Filter: always true.
// Cardinality = child.cardinality().
//
// Behavioural equivalence under projection (tree_point.group_plans[0]
// bijects with inner points). NOT representational identity — the point
// type is tree_point<1, ChildPoint>, not ChildPoint directly.
// NOT feature-compatible with a partition_desc<M>-rooted tree for M>1.
// ═══════════════════════════════════════════════════════════════════════

template <typename ChildSpace>
auto make_flat_tree_space(std::string_view name, ChildSpace child) {
    // Factory: always returns the stored child, ignoring all arguments.
    auto factory = [child = std::move(child)](
        const partition_value<1>&,
        std::size_t,
        std::span<const std::size_t>) {
            return child;
    };

    auto filter = [](const partition_value<1>&) { return true; };

    return make_tree_space<1>(
        name, make_partition<1>("flat"), std::move(factory), filter);
}

} // namespace ctdp::space

#endif // CTDP_SPACE_TREE_SPACE_H
