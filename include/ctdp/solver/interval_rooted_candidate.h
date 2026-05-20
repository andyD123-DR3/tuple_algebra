// ctdp/solver/interval_rooted_candidate.h
// Stage 2 Phase A: public interval-rooted solution value.

#ifndef CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H
#define CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H

#include "../core/plan.h"
#include "interval_context.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>

namespace ctdp::solver {

template<std::size_t MaxN>
struct interval_rooted_node_ref;

template<std::size_t MaxN>
struct interval_rooted_candidate {
    using interval_type = interval_context;
    using node_ref = interval_rooted_node_ref<MaxN>;

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

    [[nodiscard]] constexpr bool is_leaf(std::size_t i, std::size_t j) const noexcept {
        return contains(i, j) && raw_code(i, j) == leaf_code;
    }

    [[nodiscard]] constexpr bool is_internal(std::size_t i, std::size_t j) const noexcept {
        return contains(i, j) && raw_code(i, j) >= leaf_code + 1;
    }

    [[nodiscard]] constexpr std::size_t split(std::size_t i, std::size_t j) const noexcept {
        assert(is_internal(i, j) && "split requires a represented internal interval");
        return raw_code(i, j) - 2;
    }

    [[nodiscard]] constexpr interval_type left_interval(std::size_t i, std::size_t j) const noexcept {
        return interval_type{i, split(i, j)};
    }

    [[nodiscard]] constexpr interval_type right_interval(std::size_t i, std::size_t j) const noexcept {
        return interval_type{split(i, j), j};
    }

    [[nodiscard]] constexpr std::optional<node_ref> find_node(std::size_t i, std::size_t j) const noexcept {
        if (!contains(i, j)) {
            return std::nullopt;
        }
        return node_ref{this, interval_type{i, j}};
    }

    [[nodiscard]] constexpr node_ref root() const noexcept {
        assert(n > 0 && contains(0, n) && "root requires a non-empty represented root interval");
        return node_ref{this, interval_type{0, n}};
    }

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
using interval_rooted_plan = ctdp::plan<interval_rooted_candidate<MaxN>>;

} // namespace ctdp::solver

#endif // CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H

