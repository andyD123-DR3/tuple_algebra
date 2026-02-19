// ctdp/solver/memo/candidate_cache.h
// Compile-time dynamic programming framework — Analytics: Solver
// Memoisation cache for cost evaluations within DP algorithms.
// Wraps constexpr_map with hit/miss statistics.
//
// std::optional<Value> is fully constexpr in C++20 (P0848R3, P2231R1),
// supported by GCC 13.3.

#ifndef CTDP_SOLVER_MEMO_CANDIDATE_CACHE_H
#define CTDP_SOLVER_MEMO_CANDIDATE_CACHE_H

#include "../../core/constexpr_map.h"
#include <cstddef>
#include <optional>

namespace ctdp {

template<typename Key, typename Value, std::size_t MaxEntries>
struct candidate_cache {
    constexpr_map<Key, Value, MaxEntries> store{};

    // Mutable: observational side-channel, not semantic state.
    // mutable in constexpr is permitted in C++20 (P1330R0) as long as
    // the object itself was not declared const at point of creation.
    mutable std::size_t hit_count{0};
    mutable std::size_t miss_count{0};

    [[nodiscard]] constexpr auto lookup(Key const& k) const -> std::optional<Value> {
        auto it = store.find(k);
        if (it != store.end()) {
            ++hit_count;
            return it->second;
        }
        ++miss_count;
        return std::nullopt;
    }

    constexpr void insert(Key const& k, Value v) {
        auto it = store.find(k);
        if (it != store.end()) {
            it->second = v;  // update existing
        } else {
            store.insert(k, v);
        }
    }

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        return store.size();
    }

    [[nodiscard]] constexpr auto hits() const -> std::size_t {
        return hit_count;
    }

    [[nodiscard]] constexpr auto misses() const -> std::size_t {
        return miss_count;
    }

    [[nodiscard]] constexpr auto hit_rate() const -> double {
        auto total = hit_count + miss_count;
        if (total == 0) return 0.0;
        return static_cast<double>(hit_count) / static_cast<double>(total);
    }

    constexpr void clear() {
        store.clear();
        hit_count = 0;
        miss_count = 0;
    }
};

} // namespace ctdp

#endif // CTDP_SOLVER_MEMO_CANDIDATE_CACHE_H
