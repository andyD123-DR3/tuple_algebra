// tests/core/core_test.cpp
//
// Google Tests for ctdp core library.
//
// Tests constexpr_vector, constexpr_sort, constexpr_map, ct_limits, and concepts.
// Every test exercises both runtime correctness (EXPECT_*) and compile-time
// correctness (static_assert) where applicable.
//
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#include "constexpr_vector.h"
#include "constexpr_sort.h"
#include "constexpr_map.h"
#include "ct_limits.h"
#include "concepts.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>

using namespace ctdp;

// ============================================================================
// constexpr_vector — Construction
// ============================================================================

TEST(ConstexprVector, DefaultConstruction) {
    constexpr constexpr_vector<int, 10> v;
    static_assert(v.empty());
    static_assert(v.size() == 0);
    static_assert(v.capacity() == 10);
    static_assert(v.max_size() == 10);

    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.size(), 0u);
}

TEST(ConstexprVector, FillConstruction) {
    constexpr constexpr_vector<int, 10> v(5, 42);
    static_assert(v.size() == 5);
    static_assert(v[0] == 42);
    static_assert(v[4] == 42);

    EXPECT_EQ(v.size(), 5u);
    EXPECT_EQ(v[0], 42);
    EXPECT_EQ(v[4], 42);
}

TEST(ConstexprVector, InitializerList) {
    constexpr constexpr_vector<int, 10> v{1, 2, 3, 4, 5};
    static_assert(v.size() == 5);
    static_assert(v[0] == 1);
    static_assert(v[4] == 5);

    EXPECT_EQ(v.size(), 5u);
    EXPECT_EQ(v.front(), 1);
    EXPECT_EQ(v.back(), 5);
}

// ============================================================================
// constexpr_vector — Element Access
// ============================================================================

TEST(ConstexprVector, At) {
    constexpr constexpr_vector<int, 10> v{10, 20, 30};
    static_assert(v.at(0) == 10);
    static_assert(v.at(1) == 20);
    static_assert(v.at(2) == 30);

    EXPECT_EQ(v.at(0), 10);
    EXPECT_EQ(v.at(2), 30);
}

TEST(ConstexprVector, AtThrowsOutOfRange) {
    constexpr_vector<int, 10> v{1, 2, 3};
    EXPECT_THROW(v.at(3), std::out_of_range);
    EXPECT_THROW(v.at(100), std::out_of_range);
}

TEST(ConstexprVector, FrontBack) {
    constexpr constexpr_vector<int, 10> v{7, 8, 9};
    static_assert(v.front() == 7);
    static_assert(v.back() == 9);

    EXPECT_EQ(v.front(), 7);
    EXPECT_EQ(v.back(), 9);
}

TEST(ConstexprVector, Data) {
    constexpr constexpr_vector<int, 5> v{1, 2, 3};
    static_assert(v.data() != nullptr);

    EXPECT_NE(v.data(), nullptr);
    EXPECT_EQ(*v.data(), 1);
}

// ============================================================================
// constexpr_vector — Modifiers
// ============================================================================

TEST(ConstexprVector, PushBack) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(3);
        return v;
    }();

    static_assert(v.size() == 3);
    static_assert(v[0] == 1);
    static_assert(v[1] == 2);
    static_assert(v[2] == 3);

    EXPECT_EQ(v.size(), 3u);
}

TEST(ConstexprVector, PushBackCapacityExceeded) {
    constexpr_vector<int, 2> v{1, 2};
    EXPECT_THROW(v.push_back(3), std::length_error);
}

TEST(ConstexprVector, PopBack) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3};
        v.pop_back();
        return v;
    }();

    static_assert(v.size() == 2);
    static_assert(v.back() == 2);

    EXPECT_EQ(v.size(), 2u);
}

TEST(ConstexprVector, PopBackEmpty) {
    constexpr_vector<int, 10> v;
    EXPECT_THROW(v.pop_back(), std::out_of_range);
}

TEST(ConstexprVector, EmplaceBack) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v;
        v.emplace_back(42);
        v.emplace_back(99);
        return v;
    }();

    static_assert(v.size() == 2);
    static_assert(v[0] == 42);
    static_assert(v[1] == 99);
}

TEST(ConstexprVector, Clear) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3};
        v.clear();
        return v;
    }();

    static_assert(v.empty());
    EXPECT_TRUE(v.empty());
}

TEST(ConstexprVector, Resize) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3};
        v.resize(5, 99);
        return v;
    }();

    static_assert(v.size() == 5);
    static_assert(v[2] == 3);
    static_assert(v[3] == 99);
    static_assert(v[4] == 99);

    EXPECT_EQ(v.size(), 5u);
}

TEST(ConstexprVector, ResizeShrink) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3, 4, 5};
        v.resize(2);
        return v;
    }();

    static_assert(v.size() == 2);
    static_assert(v[1] == 2);
}

TEST(ConstexprVector, Insert) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 3, 5};
        v.insert(v.begin() + 1, 2);
        return v;
    }();

    static_assert(v.size() == 4);
    static_assert(v[0] == 1);
    static_assert(v[1] == 2);
    static_assert(v[2] == 3);
    static_assert(v[3] == 5);
}

TEST(ConstexprVector, Erase) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3, 4, 5};
        v.erase(v.begin() + 2);
        return v;
    }();

    static_assert(v.size() == 4);
    static_assert(v[0] == 1);
    static_assert(v[1] == 2);
    static_assert(v[2] == 4);
    static_assert(v[3] == 5);
}

TEST(ConstexprVector, EraseRange) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3, 4, 5};
        v.erase(v.begin() + 1, v.begin() + 4);
        return v;
    }();

    static_assert(v.size() == 2);
    static_assert(v[0] == 1);
    static_assert(v[1] == 5);
}

// ============================================================================
// constexpr_vector — Iterators
// ============================================================================

TEST(ConstexprVector, IteratorRange) {
    constexpr auto sum = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3, 4, 5};
        int total = 0;
        for (auto it = v.begin(); it != v.end(); ++it) {
            total += *it;
        }
        return total;
    }();

    static_assert(sum == 15);
    EXPECT_EQ(sum, 15);
}

TEST(ConstexprVector, ReverseIterators) {
    constexpr auto first_from_rbegin = []() constexpr {
        constexpr_vector<int, 10> v{10, 20, 30};
        return *v.rbegin();
    }();

    static_assert(first_from_rbegin == 30);
    EXPECT_EQ(first_from_rbegin, 30);
}

// ============================================================================
// constexpr_vector — Comparison
// ============================================================================

TEST(ConstexprVector, Equality) {
    constexpr constexpr_vector<int, 10> a{1, 2, 3};
    constexpr constexpr_vector<int, 10> b{1, 2, 3};
    constexpr constexpr_vector<int, 10> c{1, 2, 4};
    constexpr constexpr_vector<int, 10> d{1, 2};

    static_assert(a == b);
    static_assert(!(a == c));
    static_assert(!(a == d));

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
}

TEST(ConstexprVector, ThreeWayComparison) {
    constexpr constexpr_vector<int, 10> a{1, 2, 3};
    constexpr constexpr_vector<int, 10> b{1, 2, 4};
    constexpr constexpr_vector<int, 10> c{1, 2};

    static_assert(a < b);
    static_assert(c < a);

    EXPECT_LT(a, b);
    EXPECT_LT(c, a);
}

// ============================================================================
// constexpr_sort
// ============================================================================

TEST(ConstexprSort, SortAscending) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{5, 3, 1, 4, 2};
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(v[0] == 1);
    static_assert(v[1] == 2);
    static_assert(v[2] == 3);
    static_assert(v[3] == 4);
    static_assert(v[4] == 5);

    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[4], 5);
}

TEST(ConstexprSort, SortDescending) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{5, 3, 1, 4, 2};
        constexpr_sort(v.begin(), v.end(), std::greater<>{});
        return v;
    }();

    static_assert(v[0] == 5);
    static_assert(v[4] == 1);
}

TEST(ConstexprSort, SortAlreadySorted) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 2, 3, 4, 5};
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(v[0] == 1);
    static_assert(v[4] == 5);
}

TEST(ConstexprSort, SortReversed) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{5, 4, 3, 2, 1};
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(v[0] == 1);
    static_assert(v[4] == 5);
}

TEST(ConstexprSort, SortSingleElement) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{42};
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(v[0] == 42);
    static_assert(v.size() == 1);
}

TEST(ConstexprSort, SortEmpty) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v;
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(v.empty());
}

TEST(ConstexprSort, SortDuplicates) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{3, 1, 3, 1, 2};
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(v[0] == 1);
    static_assert(v[1] == 1);
    static_assert(v[2] == 2);
    static_assert(v[3] == 3);
    static_assert(v[4] == 3);
}

TEST(ConstexprSort, SortLargeRange) {
    // Exercise heap sort (> 32 elements)
    constexpr auto sorted = []() constexpr {
        constexpr_vector<int, 50> v;
        for (int i = 50; i > 0; --i) v.push_back(i);
        constexpr_sort(v.begin(), v.end());
        return v;
    }();

    static_assert(sorted[0] == 1);
    static_assert(sorted[49] == 50);
    static_assert(constexpr_is_sorted(sorted.begin(), sorted.end()));
}

TEST(ConstexprSort, StableSortVec) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{5, 3, 1, 4, 2};
        constexpr_stable_sort_vec(v);
        return v;
    }();

    static_assert(v[0] == 1);
    static_assert(v[4] == 5);
    static_assert(constexpr_is_sorted(v.begin(), v.end()));
}

TEST(ConstexprSort, StableSortPreservesOrder) {
    // Pairs sorted by first element; equal firsts should preserve second order
    constexpr auto v = []() constexpr {
        constexpr_vector<std::pair<int,int>, 10> v;
        v.push_back({2, 1});
        v.push_back({1, 1});
        v.push_back({2, 2});
        v.push_back({1, 2});
        constexpr_stable_sort_vec(v, [](auto const& a, auto const& b) {
            return a.first < b.first;
        });
        return v;
    }();

    static_assert(v[0].first == 1 && v[0].second == 1);
    static_assert(v[1].first == 1 && v[1].second == 2);
    static_assert(v[2].first == 2 && v[2].second == 1);
    static_assert(v[3].first == 2 && v[3].second == 2);
}

TEST(ConstexprSort, IsSorted) {
    constexpr constexpr_vector<int, 10> sorted{1, 2, 3, 4, 5};
    constexpr constexpr_vector<int, 10> unsorted{1, 3, 2, 4, 5};
    constexpr constexpr_vector<int, 10> empty;

    static_assert(constexpr_is_sorted(sorted.begin(), sorted.end()));
    static_assert(!constexpr_is_sorted(unsorted.begin(), unsorted.end()));
    static_assert(constexpr_is_sorted(empty.begin(), empty.end()));
}

// ============================================================================
// constexpr_unique
// ============================================================================

TEST(ConstexprSort, Unique) {
    constexpr auto v = []() constexpr {
        constexpr_vector<int, 10> v{1, 1, 2, 3, 3, 3, 4, 5, 5};
        auto new_end = constexpr_unique(v.begin(), v.end());
        v.resize(static_cast<size_t>(new_end - v.begin()));
        return v;
    }();

    static_assert(v.size() == 5);
    static_assert(v[0] == 1);
    static_assert(v[1] == 2);
    static_assert(v[2] == 3);
    static_assert(v[3] == 4);
    static_assert(v[4] == 5);
}

// ============================================================================
// constexpr_map — Basic Operations
// ============================================================================

TEST(ConstexprMap, DefaultConstruction) {
    constexpr constexpr_map<int, double, 10> m;
    static_assert(m.empty());
    static_assert(m.size() == 0);
    static_assert(m.capacity() == 10);

    EXPECT_TRUE(m.empty());
}

TEST(ConstexprMap, InsertAndFind) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, double, 10> m;
        m.insert(3, 3.0);
        m.insert(1, 1.0);
        m.insert(2, 2.0);
        return m;
    }();

    static_assert(m.size() == 3);
    static_assert(m.find(1) != m.end());
    static_assert(m.find(2) != m.end());
    static_assert(m.find(3) != m.end());
    static_assert(m.find(4) == m.end());

    EXPECT_EQ(m.size(), 3u);
}

TEST(ConstexprMap, InsertDuplicate) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, double, 10> m;
        auto [it1, ok1] = m.insert(1, 1.0);
        auto [it2, ok2] = m.insert(1, 2.0);
        return std::pair{m.size(), ok2};
    }();

    static_assert(m.first == 1);   // size unchanged
    static_assert(m.second == false); // duplicate rejected

    EXPECT_EQ(m.first, 1u);
    EXPECT_FALSE(m.second);
}

TEST(ConstexprMap, InsertOrAssign) {
    constexpr auto val = []() constexpr {
        constexpr_map<int, double, 10> m;
        m.insert(1, 1.0);
        m.insert_or_assign(1, 99.0);
        return m.at(1);
    }();

    static_assert(val == 99.0);
    EXPECT_DOUBLE_EQ(val, 99.0);
}

TEST(ConstexprMap, Contains) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, int, 10> m;
        m.insert(5, 50);
        m.insert(10, 100);
        return m;
    }();

    static_assert(m.contains(5));
    static_assert(m.contains(10));
    static_assert(!m.contains(7));

    EXPECT_TRUE(m.contains(5));
    EXPECT_FALSE(m.contains(7));
}

TEST(ConstexprMap, Count) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, int, 10> m;
        m.insert(1, 10);
        return m;
    }();

    static_assert(m.count(1) == 1);
    static_assert(m.count(2) == 0);
}

TEST(ConstexprMap, At) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, double, 10> m;
        m.insert(42, 3.14);
        return m;
    }();

    static_assert(m.at(42) == 3.14);
    EXPECT_DOUBLE_EQ(m.at(42), 3.14);
}

TEST(ConstexprMap, AtThrows) {
    constexpr_map<int, int, 10> m;
    EXPECT_THROW(m.at(1), std::out_of_range);
}

TEST(ConstexprMap, SubscriptOperator) {
    constexpr auto val = []() constexpr {
        constexpr_map<int, int, 10> m;
        m[5] = 50;
        return m[5];
    }();

    static_assert(val == 50);
}

// ============================================================================
// constexpr_map — Erase
// ============================================================================

TEST(ConstexprMap, EraseByKey) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, int, 10> m;
        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        m.erase(2);
        return m;
    }();

    static_assert(m.size() == 2);
    static_assert(m.contains(1));
    static_assert(!m.contains(2));
    static_assert(m.contains(3));
}

TEST(ConstexprMap, EraseNonexistent) {
    constexpr auto count = []() constexpr {
        constexpr_map<int, int, 10> m;
        m.insert(1, 10);
        return m.erase(99);
    }();

    static_assert(count == 0);
}

TEST(ConstexprMap, Clear) {
    constexpr auto m = []() constexpr {
        constexpr_map<int, int, 10> m;
        m.insert(1, 10);
        m.insert(2, 20);
        m.clear();
        return m;
    }();

    static_assert(m.empty());
}

// ============================================================================
// constexpr_map — Iteration Order (sorted by key)
// ============================================================================

TEST(ConstexprMap, SortedIteration) {
    constexpr auto keys = []() constexpr {
        constexpr_map<int, int, 10> m;
        m.insert(30, 3);
        m.insert(10, 1);
        m.insert(20, 2);
        constexpr_vector<int, 3> keys;
        for (auto it = m.begin(); it != m.end(); ++it) {
            keys.push_back(it->first);
        }
        return keys;
    }();

    static_assert(keys[0] == 10);
    static_assert(keys[1] == 20);
    static_assert(keys[2] == 30);
}

TEST(ConstexprMap, PairKeyType) {
    // DP memoization with pair keys
    constexpr auto val = []() constexpr {
        using Key = std::pair<size_t, size_t>;
        constexpr_map<Key, double, 20> memo;
        memo.insert({0, 3}, 42.5);
        memo.insert({1, 2}, 7.0);
        return memo.at({0, 3});
    }();

    static_assert(val == 42.5);
}

TEST(ConstexprMap, CapacityExceeded) {
    constexpr_map<int, int, 2> m;
    m.insert(1, 10);
    m.insert(2, 20);
    EXPECT_THROW(m.insert(3, 30), std::length_error);
}

// ============================================================================
// ct_limits
// ============================================================================

TEST(CtLimits, DefaultValues) {
    static_assert(ct_limits::topo_sort_max == 512);
    static_assert(ct_limits::coloring_max == 128);
    static_assert(ct_limits::scc_max == 512);
    static_assert(ct_limits::connected_components_max == 512);
    static_assert(ct_limits::min_cut_max == 64);
    static_assert(ct_limits::shortest_path_max == 256);
    static_assert(ct_limits::pareto_front_max == 50);
    static_assert(ct_limits::exhaustive_max == 10'000);
    static_assert(ct_limits::sequence_dp_max == 1000);
    static_assert(ct_limits::interval_dp_max == 500);
    static_assert(ct_limits::permutation_max == 20);
    static_assert(ct_limits::memo_table_max == 10'000);
    static_assert(ct_limits::vector_max == 10'000);
    static_assert(ct_limits::map_max == 10'000);
    static_assert(ct_limits::max_iterations == 100'000);
    static_assert(ct_limits::max_recursion_depth == 256);
}

TEST(CtLimits, WithinLimit) {
    static_assert(ct_limits::within_limit(100, 512));
    static_assert(ct_limits::within_limit(512, 512));
    static_assert(!ct_limits::within_limit(513, 512));
    static_assert(ct_limits::within_limit(0, 512));

    EXPECT_TRUE(ct_limits::within_limit(100, 512));
    EXPECT_FALSE(ct_limits::within_limit(513, 512));
}

// ============================================================================
// Concepts
// ============================================================================

TEST(Concepts, CostValue) {
    static_assert(cost_value<double>);
    static_assert(cost_value<float>);
    static_assert(cost_value<int64_t>);
    static_assert(cost_value<int32_t>);
    static_assert(cost_value<int>);

    // string actually satisfies cost_value (has <, +, default ctor)
    // A non-conforming type:
    struct no_add { auto operator<=>(no_add const&) const = default; };
    static_assert(!cost_value<no_add>);
}

TEST(Concepts, CostValueCustomType) {
    struct my_cost {
        double val;
        constexpr auto operator<=>(my_cost const&) const = default;
        constexpr my_cost operator+(my_cost const& other) const {
            return {val + other.val};
        }
    };

    static_assert(cost_value<my_cost>);
}

TEST(Concepts, CostFunction) {
    auto fn = [](int const& desc, int const& cand) -> double {
        return static_cast<double>(desc + cand);
    };

    static_assert(cost_function<decltype(fn), int, int>);
}

TEST(Concepts, DescriptorRange) {
    static_assert(descriptor_range<std::array<int, 5>>);
    static_assert(descriptor_range<std::vector<int>>);
}

TEST(Concepts, Candidate) {
    static_assert(candidate<int>);
    static_assert(candidate<double>);
    static_assert(candidate<std::pair<int, int>>);
}

TEST(Concepts, SearchSpace) {
    struct test_space {
        using candidate_type = int;
        constexpr size_t size() const { return 10; }
    };

    static_assert(search_space<test_space>);
}

TEST(Concepts, SearchSpaceNonConforming) {
    struct no_candidate_type {
        constexpr size_t size() const { return 10; }
    };

    struct no_size {
        using candidate_type = int;
    };

    static_assert(!search_space<no_candidate_type>);
    static_assert(!search_space<no_size>);
}
