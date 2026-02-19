// tests/test_memo.cpp
// Tests for solver/memo/ (candidate_cache, cost_purity)

#include "ctdp/solver/memo/candidate_cache.h"
#include "ctdp/solver/memo/cost_purity.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =====================================================================
// candidate_cache
// =====================================================================

constexpr auto cache_basic_test() {
    candidate_cache<int, double, 16> cache{};

    cache.insert(42, 3.14);
    cache.insert(99, 2.71);

    auto r1 = cache.lookup(42);
    auto r2 = cache.lookup(99);
    auto r3 = cache.lookup(7);  // miss

    return std::tuple{
        r1.has_value(), r1.value_or(0.0),
        r2.has_value(), r2.value_or(0.0),
        r3.has_value(),
        cache.hits(), cache.misses(), cache.size()
    };
}

static_assert(std::get<0>(cache_basic_test()) == true);   // 42 found
static_assert(std::get<1>(cache_basic_test()) == 3.14);
static_assert(std::get<2>(cache_basic_test()) == true);   // 99 found
static_assert(std::get<3>(cache_basic_test()) == 2.71);
static_assert(std::get<4>(cache_basic_test()) == false);  // 7 not found
static_assert(std::get<5>(cache_basic_test()) == 2);      // 2 hits
static_assert(std::get<6>(cache_basic_test()) == 1);      // 1 miss
static_assert(std::get<7>(cache_basic_test()) == 2);      // 2 entries

TEST(CandidateCache, BasicLookup) {
    constexpr auto result = cache_basic_test();
    EXPECT_TRUE(std::get<0>(result));
    EXPECT_DOUBLE_EQ(std::get<1>(result), 3.14);
    EXPECT_TRUE(std::get<2>(result));
    EXPECT_DOUBLE_EQ(std::get<3>(result), 2.71);
    EXPECT_FALSE(std::get<4>(result));
}

TEST(CandidateCache, HitMissStats) {
    constexpr auto result = cache_basic_test();
    EXPECT_EQ(std::get<5>(result), 2u);  // hits
    EXPECT_EQ(std::get<6>(result), 1u);  // misses
    EXPECT_EQ(std::get<7>(result), 2u);  // size
}

// --- Hit rate ---
constexpr auto cache_hit_rate_test() {
    candidate_cache<int, double, 16> cache{};
    cache.insert(1, 1.0);
    (void)cache.lookup(1);  // hit
    (void)cache.lookup(1);  // hit
    (void)cache.lookup(2);  // miss
    return cache.hit_rate();
}

// 2 hits / 3 total ≈ 0.666...
static_assert(cache_hit_rate_test() > 0.66);
static_assert(cache_hit_rate_test() < 0.67);

TEST(CandidateCache, HitRate) {
    constexpr auto rate = cache_hit_rate_test();
    EXPECT_NEAR(rate, 2.0 / 3.0, 1e-10);
}

// --- Update existing key ---
constexpr auto cache_update_test() {
    candidate_cache<int, double, 16> cache{};
    cache.insert(1, 1.0);
    cache.insert(1, 2.0);  // update
    auto r = cache.lookup(1);
    return std::pair{r.value_or(0.0), cache.size()};
}

static_assert(cache_update_test().first == 2.0);
static_assert(cache_update_test().second == 1);  // still one entry

TEST(CandidateCache, UpdateExisting) {
    constexpr auto result = cache_update_test();
    EXPECT_DOUBLE_EQ(result.first, 2.0);
    EXPECT_EQ(result.second, 1u);
}

// --- Clear ---
constexpr auto cache_clear_test() {
    candidate_cache<int, double, 16> cache{};
    cache.insert(1, 1.0);
    (void)cache.lookup(1);  // hit
    cache.clear();
    return std::tuple{cache.size(), cache.hits(), cache.misses()};
}

static_assert(std::get<0>(cache_clear_test()) == 0);
static_assert(std::get<1>(cache_clear_test()) == 0);
static_assert(std::get<2>(cache_clear_test()) == 0);

TEST(CandidateCache, Clear) {
    constexpr auto result = cache_clear_test();
    EXPECT_EQ(std::get<0>(result), 0u);
    EXPECT_EQ(std::get<1>(result), 0u);
    EXPECT_EQ(std::get<2>(result), 0u);
}

// =====================================================================
// cost_purity
// =====================================================================

struct PureCost {};
struct ImpureCost {};

// Opt in: declare PureCost as pure
template<> struct ctdp::cost_purity<PureCost> : std::true_type {};

// Default: impure (safe default — no opt-in means no caching)
static_assert(!cost_is_pure_v<ImpureCost>);

// Opted in: pure
static_assert(cost_is_pure_v<PureCost>);

TEST(CostPurity, DefaultImpure) {
    EXPECT_FALSE(cost_is_pure_v<ImpureCost>);
}

TEST(CostPurity, OptedInPure) {
    EXPECT_TRUE(cost_is_pure_v<PureCost>);
}
