// examples/standalone/cache_size_opt.cpp
// CT-DP standalone example: compile-time cache size optimisation
//
// 1 dimension: cache_size (1–8 entries)
// Cost model: compile-time LRU simulation sharing engine with runtime cache.
// Teaches: cost-as-simulation, and why analytical models ≠ runtime reality.

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <utility>

// ============================================================================
// 1. LRU cache — shared between cost model (constexpr) and executor (runtime)
// ============================================================================

template<std::size_t Capacity>
struct lru_cache {
    std::array<int, Capacity> entries{};
    std::size_t count = 0;

    constexpr bool lookup(int key) {
        for (std::size_t i = 0; i < count; ++i) {
            if (entries[i] == key) {
                // Move to front (MRU)
                int found = entries[i];
                for (std::size_t j = i; j > 0; --j) entries[j] = entries[j-1];
                entries[0] = found;
                return true;
            }
        }
        // Miss: insert at front, evict LRU if full
        if (count < Capacity) ++count;
        for (std::size_t j = count - 1; j > 0; --j) entries[j] = entries[j-1];
        entries[0] = key;
        return false;
    }
};

// ============================================================================
// 2. Instance — access pattern
// ============================================================================

template<std::size_t N>
struct access_pattern {
    std::array<int, N> sequence;
    int num_functions;
};

constexpr auto make_alternating() {
    access_pattern<20> p{};
    p.num_functions = 5;
    // Hot: functions 2 and 3 alternate. Cold: 0, 1, 4 occasional.
    constexpr std::array<int, 20> seq = {
        2,3,2,3,2,3,2,3,0,2,3,2,3,1,2,3,2,3,4,2
    };
    p.sequence = seq;
    return p;
}

constexpr auto make_sequential() {
    access_pattern<20> p{};
    p.num_functions = 6;
    constexpr std::array<int, 20> seq = {
        0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1
    };
    p.sequence = seq;
    return p;
}

constexpr auto make_phase_shift() {
    access_pattern<20> p{};
    p.num_functions = 6;
    // Phase 1: {0,1,2} hot. Phase 2: {3,4,5} hot.
    constexpr std::array<int, 20> seq = {
        0,1,2,0,1,2,0,1,2,0,  3,4,5,3,4,5,3,4,5,3
    };
    p.sequence = seq;
    return p;
}

constexpr auto alternating_pattern = make_alternating();
constexpr auto sequential_pattern  = make_sequential();
constexpr auto phase_shift_pattern = make_phase_shift();

// ============================================================================
// 3. Cost model — constexpr LRU simulation (SAME engine as runtime)
// ============================================================================

template<std::size_t CacheSize, std::size_t SeqLen>
constexpr int simulate_misses(access_pattern<SeqLen> const& pattern) {
    lru_cache<CacheSize> cache{};
    int misses = 0;
    for (std::size_t i = 0; i < SeqLen; ++i)
        if (!cache.lookup(pattern.sequence[i])) ++misses;
    return misses;
}

// ============================================================================
// 4. Solve — exhaustive over cache sizes 1..MaxSize
// ============================================================================

struct cache_result {
    std::size_t best_size = 0;
    int best_misses = std::numeric_limits<int>::max();
};

template<std::size_t MaxSize, std::size_t SeqLen, std::size_t... Is>
constexpr cache_result solve_impl(access_pattern<SeqLen> const& pattern,
                                   std::index_sequence<Is...>) {
    cache_result result{};
    auto try_size = [&](auto size_tag) {
        constexpr std::size_t S = decltype(size_tag)::value;
        int misses = simulate_misses<S>(pattern);
        if (misses < result.best_misses) {
            result.best_misses = misses;
            result.best_size = S;
        }
    };
    (try_size(std::integral_constant<std::size_t, Is + 1>{}), ...);
    return result;
}

template<std::size_t MaxSize, std::size_t SeqLen>
constexpr cache_result solve_cache_opt(access_pattern<SeqLen> const& pattern) {
    return solve_impl<MaxSize>(pattern, std::make_index_sequence<MaxSize>{});
}

// Compile-time solve
static constexpr auto r_alt   = solve_cache_opt<8>(alternating_pattern);
static constexpr auto r_seq   = solve_cache_opt<8>(sequential_pattern);
static constexpr auto r_phase = solve_cache_opt<8>(phase_shift_pattern);

static_assert(r_alt.best_size == 3);    // 2 hot + 1 buffer absorbs cold evictions
static_assert(r_seq.best_size == 6);    // needs all 6
static_assert(r_phase.best_size == 3);  // 3 covers each phase

// ============================================================================
// 5. Runtime executor — uses the SAME lru_cache
// ============================================================================

// Simulated "expensive computation" per function
static double compute_function(int func_id, double input) {
    double result = input;
    for (int i = 0; i < 100; ++i)
        result = result * 0.999 + static_cast<double>(func_id) * 0.001;
    return result;
}

template<std::size_t CacheSize>
struct cached_executor {
    struct cache_entry { int func_id; double result; };
    lru_cache<CacheSize> lru{};
    std::array<cache_entry, CacheSize> data{};

    double execute(int func_id, double input) {
        // Check cache via LRU
        // Simple linear scan (matches the LRU structure)
        for (std::size_t i = 0; i < lru.count; ++i) {
            if (lru.entries[i] == func_id) {
                // Hit: move to front in LRU, return cached
                lru.lookup(func_id);
                return data[i].result;
            }
        }
        // Miss
        double result = compute_function(func_id, input);
        lru.lookup(func_id);
        // Store at front
        for (std::size_t j = lru.count - 1; j > 0; --j) data[j] = data[j-1];
        data[0] = {func_id, result};
        return result;
    }
};

// ============================================================================
// 6. Benchmark
// ============================================================================

int main() {
    constexpr int ITERS = 50;

    auto bench = [](auto& executor, auto const& pattern, const char* label) {
        double total = 0.0;
        // Warm up
        for (auto fid : pattern.sequence) total += executor.execute(fid, 1.0);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < ITERS; ++rep)
            for (auto fid : pattern.sequence)
                total += executor.execute(fid, 1.0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::cout << "    " << std::setw(20) << label << ": "
                  << std::fixed << std::setprecision(0) << us << " us\n";
        return us;
    };

    auto run_instance = [&](const char* name, auto const& pattern, auto const& result) {
        std::cout << "\n  " << name << ": optimal cache_size = " << result.best_size
                  << " (" << result.best_misses << " misses/pass)\n";

        cached_executor<1> c1{};
        cached_executor<2> c2{};
        cached_executor<4> c4{};
        cached_executor<6> c6{};
        cached_executor<8> c8{};

        bench(c1, pattern, "size=1");
        bench(c2, pattern, "size=2");
        bench(c4, pattern, "size=4");
        bench(c6, pattern, "size=6");
        bench(c8, pattern, "size=8");
    };

    std::cout << "Cache Size Optimisation Demo\n";
    std::cout << "  Cost model: compile-time LRU simulation (same engine as runtime)\n";

    run_instance("Alternating (hot=2,3)", alternating_pattern, r_alt);
    run_instance("Sequential (0-5)",      sequential_pattern,  r_seq);
    run_instance("Phase shift (0-2→3-5)", phase_shift_pattern, r_phase);

    std::cout << "\n  Note: phase_shift cost model says size=3 suffices.\n"
              << "  But runtime size=6 is faster because it stays warm across phases.\n"
              << "  This is why calibration replaces analytical cost models.\n";

    return 0;
}
