#pragma once

#include "spmv_design_space/facts.hpp"
#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ctdp::spmv_dsl {

struct region_descriptor {
    std::size_t first = 0;
    std::size_t last = 0;

    [[nodiscard]] std::size_t rows() const noexcept {
        return last >= first ? last - first : 0;
    }
};

struct recursive_search_options {
    std::size_t min_leaf_rows = 512;
    std::size_t preferred_leaf_rows = 1024;
    std::size_t max_depth = 3;
    double split_overhead = 64.0;
};

struct recursive_search_node {
    region_descriptor region{};
    std::size_t depth = 0;
    bool selected_split = false;
    double stop_cost = 0.0;
    double split_cost = 0.0;
    double selected_cost = 0.0;
    std::string selected_reason;
    std::vector<recursive_search_node> children;

    [[nodiscard]] bool is_leaf() const noexcept {
        return !selected_split;
    }
};

inline double estimate_stop_cost(region_descriptor region,
                                 std::size_t depth,
                                 const recursive_search_options& options) noexcept {
    const auto rows = static_cast<double>(region.rows());
    const auto preferred = static_cast<double>(std::max<std::size_t>(1, options.preferred_leaf_rows));
    const double locality_penalty = rows > preferred ? 0.35 * (rows - preferred) : 0.0;
    const double depth_discount = 1.0 - std::min<double>(0.08 * static_cast<double>(depth), 0.24);
    return rows * depth_discount + locality_penalty;
}

inline recursive_search_node search_region(region_descriptor region,
                                           const sparse_facts& facts,
                                           const recursive_search_options& options,
                                           std::size_t depth = 0) {
    recursive_search_node node{};
    node.region = region;
    node.depth = depth;
    node.stop_cost = estimate_stop_cost(region, depth, options);

    const bool can_split = facts.stencil_like &&
                           depth < options.max_depth &&
                           region.rows() >= 2 * std::max<std::size_t>(1, options.min_leaf_rows);

    if (!can_split) {
        node.selected_split = false;
        node.selected_cost = node.stop_cost;
        if (!facts.stencil_like) {
            node.selected_reason = "stop: no recognised stencil facts for recursive split";
        } else if (depth >= options.max_depth) {
            node.selected_reason = "stop: max recursive depth reached";
        } else {
            node.selected_reason = "stop: region below split threshold";
        }
        return node;
    }

    const auto mid = region.first + region.rows() / 2;
    auto left = search_region({region.first, mid}, facts, options, depth + 1);
    auto right = search_region({mid, region.last}, facts, options, depth + 1);
    node.split_cost = options.split_overhead + left.selected_cost + right.selected_cost;

    if (node.split_cost < node.stop_cost) {
        node.selected_split = true;
        node.selected_cost = node.split_cost;
        node.selected_reason = "split: recursive children beat stop cost";
        node.children.push_back(std::move(left));
        node.children.push_back(std::move(right));
    } else {
        node.selected_split = false;
        node.selected_cost = node.stop_cost;
        node.selected_reason = "stop: leaf beats recursive split";
    }

    return node;
}

inline recursive_search_node search_recursive_spmv_plan(const sparse_facts& facts,
                                                        const recursive_search_options& options = {}) {
    return search_region({0, facts.rows}, facts, options, 0);
}

inline void print_recursive_tree(std::ostream& os,
                                 const recursive_search_node& node,
                                 int indent = 0) {
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    if (node.selected_split) {
        os << "split<row_bisection> rows[" << node.region.first << "," << node.region.last
           << ") cost=" << std::fixed << std::setprecision(1) << node.selected_cost << "\n";
        for (const auto& child : node.children) {
            print_recursive_tree(os, child, indent + 1);
        }
    } else {
        os << "leaf<matrix_free_vectorised_rows> rows[" << node.region.first << "," << node.region.last
           << ") cost=" << std::fixed << std::setprecision(1) << node.selected_cost << "\n";
    }
}

inline void print_recursive_trace(std::ostream& os,
                                  const recursive_search_node& node,
                                  int indent = 0) {
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "depth " << node.depth << " rows[" << node.region.first << "," << node.region.last << ")"
       << " stop=" << std::fixed << std::setprecision(1) << node.stop_cost;
    if (node.split_cost > 0.0) {
        os << " split=" << std::fixed << std::setprecision(1) << node.split_cost;
    }
    os << " selected=" << (node.selected_split ? "split" : "stop")
       << " (" << node.selected_reason << ")\n";
    for (const auto& child : node.children) {
        print_recursive_trace(os, child, indent + 1);
    }
}

inline std::string recursive_tree_string(const recursive_search_node& node) {
    std::ostringstream os;
    print_recursive_tree(os, node, 0);
    return os.str();
}

inline std::string recursive_trace_string(const recursive_search_node& node) {
    std::ostringstream os;
    print_recursive_trace(os, node, 0);
    return os.str();
}

} // namespace ctdp::spmv_dsl
