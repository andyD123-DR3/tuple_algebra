// ctdp/solver/interval_rooted_candidate.h
// Stage 2 Phase A: public interval-rooted solution value.

#ifndef CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H
#define CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H

#include "../core/plan.h"
#include "interval_context.h"
#include "spaces/interval_split_space.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

namespace ctdp::solver {

template<std::size_t MaxN>
struct interval_rooted_node_ref;

template<std::size_t MaxN>
struct interval_rooted_preorder_range;

template<std::size_t MaxN>
struct interval_rooted_inorder_range;

template<std::size_t MaxN>
struct interval_rooted_postorder_range;

template<std::size_t MaxN>
struct interval_rooted_candidate {
    using interval_type = interval_context;
    using node_ref = interval_rooted_node_ref<MaxN>;
    using preorder_range = interval_rooted_preorder_range<MaxN>;
    using inorder_range = interval_rooted_inorder_range<MaxN>;
    using postorder_range = interval_rooted_postorder_range<MaxN>;

    static constexpr std::size_t max_size = MaxN;
    static constexpr std::size_t absent_code = 0;
    static constexpr std::size_t leaf_code = 1;

    std::size_t n{};
    std::array<std::size_t, (MaxN + 1) * (MaxN + 1)> split_or_tag{};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return n == 0;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return n;
    }

    [[nodiscard]] constexpr std::size_t leaf_count() const noexcept {
        return n;
    }

    [[nodiscard]] constexpr interval_type root_interval() const noexcept {
        assert(n > 0 && n <= MaxN && "root_interval requires 0 < n <= MaxN");
        return interval_type{0, n};
    }

    [[nodiscard]] constexpr bool contains(std::size_t i, std::size_t j) const noexcept {
        if (!valid_interval(i, j) || n == 0 || !root_represented()) {
            return false;
        }
        return reachable_contains(0, n, i, j);
    }

    [[nodiscard]] constexpr bool contains(interval_type ctx) const noexcept {
        return contains(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr bool is_leaf(std::size_t i, std::size_t j) const noexcept {
        return contains(i, j) && raw_code(i, j) == leaf_code;
    }

    [[nodiscard]] constexpr bool is_leaf(interval_type ctx) const noexcept {
        return is_leaf(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr bool is_internal(std::size_t i, std::size_t j) const noexcept {
        return contains(i, j) && raw_code(i, j) >= leaf_code + 1;
    }

    [[nodiscard]] constexpr bool is_internal(interval_type ctx) const noexcept {
        return is_internal(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr std::size_t split(std::size_t i, std::size_t j) const noexcept {
        assert(is_internal(i, j) && "split requires a represented internal interval");
        return raw_code(i, j) - 2;
    }

    [[nodiscard]] constexpr std::size_t split(interval_type ctx) const noexcept {
        return split(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr interval_type left_interval(std::size_t i, std::size_t j) const noexcept {
        return interval_type{i, split(i, j)};
    }

    [[nodiscard]] constexpr interval_type left_interval(interval_type ctx) const noexcept {
        return left_interval(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr interval_type right_interval(std::size_t i, std::size_t j) const noexcept {
        return interval_type{split(i, j), j};
    }

    [[nodiscard]] constexpr interval_type right_interval(interval_type ctx) const noexcept {
        return right_interval(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr std::optional<node_ref> find_node(std::size_t i, std::size_t j) const noexcept {
        if (!contains(i, j)) {
            return std::nullopt;
        }
        return node_ref{this, interval_type{i, j}};
    }

    [[nodiscard]] constexpr std::optional<node_ref> find_node(interval_type ctx) const noexcept {
        return find_node(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr node_ref root() const noexcept {
        assert(n > 0 && contains(0, n) && "root requires a non-empty represented root interval");
        return node_ref{this, interval_type{0, n}};
    }

    [[nodiscard]] constexpr preorder_range preorder() const noexcept;
    [[nodiscard]] constexpr inorder_range inorder() const noexcept;
    [[nodiscard]] constexpr postorder_range postorder() const noexcept;

    [[nodiscard]] constexpr bool is_legal() const noexcept {
        if (n > MaxN) {
            return false;
        }
        if (n == 0) {
            return true;
        }
        if (!root_represented()) {
            return false;
        }
        return legal_subtree(0, n);
    }

    [[nodiscard]] constexpr bool is_canonical() const noexcept {
        if (!is_legal()) {
            return false;
        }

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j <= n; ++j) {
                if (raw_code(i, j) != absent_code && !contains(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] constexpr bool operator==(interval_rooted_candidate const& other) const noexcept {
        if (n != other.n) {
            return false;
        }
        if (n == 0) {
            return true;
        }
        if (!root_represented() || !other.root_represented()) {
            return false;
        }
        return equal_subtree(0, n, other);
    }

private:
    [[nodiscard]] static constexpr std::size_t index(std::size_t i, std::size_t j) noexcept {
        return i * (MaxN + 1) + j;
    }

    [[nodiscard]] constexpr bool valid_interval(std::size_t i, std::size_t j) const noexcept {
        return i < j && j <= n;
    }

    [[nodiscard]] constexpr bool root_represented() const noexcept {
        return n > 0 && n <= MaxN && raw_code(0, n) != absent_code;
    }

    [[nodiscard]] constexpr std::size_t raw_code(std::size_t i, std::size_t j) const noexcept {
        assert(i < j && j <= MaxN && "raw_code requires interval inside storage bounds");
        return split_or_tag[index(i, j)];
    }

    [[nodiscard]] constexpr bool raw_is_leaf(std::size_t i, std::size_t j) const noexcept {
        return raw_code(i, j) == leaf_code;
    }

    [[nodiscard]] constexpr bool raw_is_internal(std::size_t i, std::size_t j) const noexcept {
        return raw_code(i, j) >= leaf_code + 1;
    }

    [[nodiscard]] constexpr std::size_t raw_split(std::size_t i, std::size_t j) const noexcept {
        return raw_code(i, j) - 2;
    }

    [[nodiscard]] constexpr bool reachable_contains(std::size_t cur_i,
                                                    std::size_t cur_j,
                                                    std::size_t want_i,
                                                    std::size_t want_j) const noexcept {
        if (cur_i == want_i && cur_j == want_j) {
            return raw_code(cur_i, cur_j) != absent_code;
        }
        if (raw_is_leaf(cur_i, cur_j) || !raw_is_internal(cur_i, cur_j)) {
            return false;
        }

        auto k = raw_split(cur_i, cur_j);
        if (!(cur_i < k && k < cur_j)) {
            return false;
        }

        return reachable_contains(cur_i, k, want_i, want_j)
            || reachable_contains(k, cur_j, want_i, want_j);
    }

    [[nodiscard]] constexpr bool legal_subtree(std::size_t i, std::size_t j) const noexcept {
        auto code = raw_code(i, j);
        if (code == absent_code) {
            return false;
        }

        if (j == i + 1) {
            return code == leaf_code;
        }

        if (!raw_is_internal(i, j)) {
            return false;
        }

        auto k = raw_split(i, j);
        if (!(i < k && k < j)) {
            return false;
        }

        return legal_subtree(i, k) && legal_subtree(k, j);
    }

    [[nodiscard]] constexpr bool equal_subtree(std::size_t i,
                                               std::size_t j,
                                               interval_rooted_candidate const& other) const noexcept {
        bool mine = contains(i, j);
        bool theirs = other.contains(i, j);
        if (mine != theirs) {
            return false;
        }
        if (!mine) {
            return true;
        }
        if (is_leaf(i, j) || other.is_leaf(i, j)) {
            return is_leaf(i, j) && other.is_leaf(i, j);
        }
        if (!is_internal(i, j) || !other.is_internal(i, j)) {
            return false;
        }
        auto my_split = split(i, j);
        auto other_split = other.split(i, j);
        if (my_split != other_split) {
            return false;
        }
        return equal_subtree(i, my_split, other)
            && equal_subtree(my_split, j, other);
    }
};

template<std::size_t MaxN>
struct interval_rooted_node_ref {
    using candidate_type = interval_rooted_candidate<MaxN>;
    using interval_type = interval_context;

    candidate_type const* candidate{};
    interval_type ctx{0, 1};

    [[nodiscard]] constexpr interval_type interval() const noexcept {
        return ctx;
    }

    [[nodiscard]] constexpr bool is_leaf() const noexcept {
        return candidate->is_leaf(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr bool is_internal() const noexcept {
        return candidate->is_internal(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr std::size_t split() const noexcept {
        return candidate->split(ctx.i, ctx.j);
    }

    [[nodiscard]] constexpr interval_rooted_node_ref left() const noexcept {
        auto next = candidate->left_interval(ctx.i, ctx.j);
        return interval_rooted_node_ref{candidate, next};
    }

    [[nodiscard]] constexpr interval_rooted_node_ref right() const noexcept {
        auto next = candidate->right_interval(ctx.i, ctx.j);
        return interval_rooted_node_ref{candidate, next};
    }
};

template<std::size_t MaxN>
struct interval_rooted_preorder_range {
    using node_ref = interval_rooted_node_ref<MaxN>;
    static constexpr std::size_t capacity = (MaxN > 0) ? (2 * MaxN - 1) : 1;

    std::array<node_ref, capacity> nodes{};
    std::size_t count{};

    [[nodiscard]] constexpr node_ref const* begin() const noexcept {
        return nodes.data();
    }

    [[nodiscard]] constexpr node_ref const* end() const noexcept {
        return nodes.data() + count;
    }
};

template<std::size_t MaxN>
struct interval_rooted_inorder_range {
    using node_ref = interval_rooted_node_ref<MaxN>;
    static constexpr std::size_t capacity = interval_rooted_preorder_range<MaxN>::capacity;

    std::array<node_ref, capacity> nodes{};
    std::size_t count{};

    [[nodiscard]] constexpr node_ref const* begin() const noexcept {
        return nodes.data();
    }

    [[nodiscard]] constexpr node_ref const* end() const noexcept {
        return nodes.data() + count;
    }
};

template<std::size_t MaxN>
struct interval_rooted_postorder_range {
    using node_ref = interval_rooted_node_ref<MaxN>;
    static constexpr std::size_t capacity = interval_rooted_preorder_range<MaxN>::capacity;

    std::array<node_ref, capacity> nodes{};
    std::size_t count{};

    [[nodiscard]] constexpr node_ref const* begin() const noexcept {
        return nodes.data();
    }

    [[nodiscard]] constexpr node_ref const* end() const noexcept {
        return nodes.data() + count;
    }
};

template<std::size_t MaxN>
using interval_rooted_plan = ctdp::plan<interval_rooted_candidate<MaxN>>;

namespace detail {

template<std::size_t MaxN>
constexpr void build_interval_rooted_preorder(
    interval_rooted_candidate<MaxN> const& c,
    std::size_t i,
    std::size_t j,
    std::array<interval_rooted_node_ref<MaxN>, interval_rooted_preorder_range<MaxN>::capacity>& out,
    std::size_t& count)
{
    out[count++] = interval_rooted_node_ref<MaxN>{&c, interval_context{i, j}};

    if (c.is_leaf(i, j)) {
        return;
    }

    auto k = c.split(i, j);
    build_interval_rooted_preorder(c, i, k, out, count);
    build_interval_rooted_preorder(c, k, j, out, count);
}

template<std::size_t MaxN>
constexpr void build_interval_rooted_inorder(
    interval_rooted_candidate<MaxN> const& c,
    std::size_t i,
    std::size_t j,
    std::array<interval_rooted_node_ref<MaxN>, interval_rooted_inorder_range<MaxN>::capacity>& out,
    std::size_t& count)
{
    if (c.is_leaf(i, j)) {
        out[count++] = interval_rooted_node_ref<MaxN>{&c, interval_context{i, j}};
        return;
    }

    auto k = c.split(i, j);
    build_interval_rooted_inorder(c, i, k, out, count);
    out[count++] = interval_rooted_node_ref<MaxN>{&c, interval_context{i, j}};
    build_interval_rooted_inorder(c, k, j, out, count);
}

template<std::size_t MaxN>
constexpr void build_interval_rooted_postorder(
    interval_rooted_candidate<MaxN> const& c,
    std::size_t i,
    std::size_t j,
    std::array<interval_rooted_node_ref<MaxN>, interval_rooted_postorder_range<MaxN>::capacity>& out,
    std::size_t& count)
{
    if (c.is_leaf(i, j)) {
        out[count++] = interval_rooted_node_ref<MaxN>{&c, interval_context{i, j}};
        return;
    }

    auto k = c.split(i, j);
    build_interval_rooted_postorder(c, i, k, out, count);
    build_interval_rooted_postorder(c, k, j, out, count);
    out[count++] = interval_rooted_node_ref<MaxN>{&c, interval_context{i, j}};
}

template<std::size_t MaxN, class SplitFn>
constexpr void build_interval_rooted_subtree(interval_rooted_candidate<MaxN>& c,
                                             std::size_t i,
                                             std::size_t j,
                                             SplitFn const& split_for) {
    assert(i < j && j <= c.n && "Subtree interval must lie inside candidate bounds");

    if (j == i + 1) {
        c.split_or_tag[i * (MaxN + 1) + j] = interval_rooted_candidate<MaxN>::leaf_code;
        return;
    }

    auto k = static_cast<std::size_t>(split_for(i, j));
    assert(i < k && k < j && "Reconstruction split must be strictly interior");

    c.split_or_tag[i * (MaxN + 1) + j] = k + 2;
    build_interval_rooted_subtree(c, i, k, split_for);
    build_interval_rooted_subtree(c, k, j, split_for);
}

} // namespace detail

template<std::size_t MaxN>
constexpr auto interval_rooted_candidate<MaxN>::preorder() const noexcept -> preorder_range {
    preorder_range out{};

    if (n == 0) {
        return out;
    }

    assert(is_legal() && "preorder requires a legal interval-rooted candidate");
    detail::build_interval_rooted_preorder(*this, 0, n, out.nodes, out.count);
    return out;
}

template<std::size_t MaxN>
constexpr auto interval_rooted_candidate<MaxN>::inorder() const noexcept -> inorder_range {
    inorder_range out{};

    if (n == 0) {
        return out;
    }

    assert(is_legal() && "inorder requires a legal interval-rooted candidate");
    detail::build_interval_rooted_inorder(*this, 0, n, out.nodes, out.count);
    return out;
}

template<std::size_t MaxN>
constexpr auto interval_rooted_candidate<MaxN>::postorder() const noexcept -> postorder_range {
    postorder_range out{};

    if (n == 0) {
        return out;
    }

    assert(is_legal() && "postorder requires a legal interval-rooted candidate");
    detail::build_interval_rooted_postorder(*this, 0, n, out.nodes, out.count);
    return out;
}

template<std::size_t MaxN>
[[nodiscard]] constexpr auto make_empty_interval_rooted_candidate()
    -> interval_rooted_candidate<MaxN>
{
    return {};
}

template<std::size_t MaxN>
[[nodiscard]] constexpr auto make_single_leaf_interval_rooted_candidate()
    -> interval_rooted_candidate<MaxN>
{
    static_assert(MaxN >= 1, "Single-leaf candidate requires MaxN >= 1");

    interval_rooted_candidate<MaxN> c{};
    c.n = 1;
    c.split_or_tag[0 * (MaxN + 1) + 1] = interval_rooted_candidate<MaxN>::leaf_code;
    return c;
}

template<std::size_t MaxN, class SplitFn>
[[nodiscard]] constexpr auto reconstruct_interval_rooted_candidate(std::size_t n,
                                                                   SplitFn&& split_for)
    -> interval_rooted_candidate<MaxN>
{
    assert(n <= MaxN && "Reconstruction size exceeds candidate capacity");

    if (n == 0) {
        return make_empty_interval_rooted_candidate<MaxN>();
    }
    if (n == 1) {
        return make_single_leaf_interval_rooted_candidate<MaxN>();
    }

    interval_rooted_candidate<MaxN> c{};
    c.n = n;
    detail::build_interval_rooted_subtree(c, 0, n, split_for);
    return c;
}

template<std::size_t MaxN>
[[nodiscard]] constexpr auto reconstruct_interval_rooted_candidate(
    ::ctdp::interval_split_candidate<MaxN> const& splits)
    -> interval_rooted_candidate<MaxN>
{
    return reconstruct_interval_rooted_candidate<MaxN>(
        splits.n,
        [&splits](std::size_t i, std::size_t j) constexpr -> std::size_t {
            assert(j >= i + 2 && "Legacy split lookup requires an internal interval");
            return splits.split(i, j - 1) + 1;
        });
}

template<std::size_t MaxN>
[[nodiscard]] constexpr auto reconstruct_interval_rooted_candidate(
    ::ctdp::plan<::ctdp::interval_split_candidate<MaxN>> const& result)
    -> interval_rooted_candidate<MaxN>
{
    return reconstruct_interval_rooted_candidate(result.params);
}

} // namespace ctdp::solver

#endif // CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H






