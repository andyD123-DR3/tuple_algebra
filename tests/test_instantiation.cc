// tests/test_instantiation.cc
// Tests for the instantiation tier: strategy_map, dispatch_table, plan_executor.
// Covers: binding, lookup, dispatch, execution, constexpr validation,
//         and end-to-end solve → dispatch → execute pipeline.

#include "ctdp/engine/instantiation/plan_executor.h"
#include "ctdp/solver/algorithms/beam_search.h"
#include "ctdp/solver/algorithms/per_element_argmin.h"
#include "ctdp/solver/spaces/per_element_space.h"
#include "ctdp/solver/spaces/heterogeneous_per_element_space.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =============================================================================
// Test domain: 3 strategies for 4 positions
// =============================================================================

enum class Strat { Fast, Medium, Safe };

// "Implementation" is just a double cost multiplier for testing.
using Impl = double;

constexpr Impl fast_impl   = 1.0;
constexpr Impl medium_impl = 3.0;
constexpr Impl safe_impl   = 5.0;

// =============================================================================
// strategy_map tests
// =============================================================================

TEST(StrategyMap, BindAndLookup) {
    constexpr auto smap = make_strategy_map<Strat, Impl>(
        std::pair{Strat::Fast,   fast_impl},
        std::pair{Strat::Medium, medium_impl},
        std::pair{Strat::Safe,   safe_impl}
    );

    static_assert(smap[Strat::Fast]   == 1.0);
    static_assert(smap[Strat::Medium] == 3.0);
    static_assert(smap[Strat::Safe]   == 5.0);
    static_assert(smap.size() == 3);
}

TEST(StrategyMap, Contains) {
    constexpr auto smap = make_strategy_map<Strat, Impl>(
        std::pair{Strat::Fast, fast_impl},
        std::pair{Strat::Safe, safe_impl}
    );

    static_assert(smap.contains(Strat::Fast));
    static_assert(!smap.contains(Strat::Medium));
    static_assert(smap.contains(Strat::Safe));
}

TEST(StrategyMap, Overwrite) {
    strategy_map<Strat, Impl, 3> smap;
    smap.bind(Strat::Fast, 1.0);
    smap.bind(Strat::Fast, 99.0);  // overwrite
    EXPECT_DOUBLE_EQ(smap[Strat::Fast], 99.0);
    EXPECT_EQ(smap.size(), 1u);  // no duplicate entry
}

TEST(StrategyMap, FunctionPointerImpl) {
    using FnPtr = int(*)(int);
    auto add1 = +[](int x) -> int { return x + 1; };
    auto add2 = +[](int x) -> int { return x + 2; };

    auto smap = make_strategy_map<Strat, FnPtr>(
        std::pair{Strat::Fast, add1},
        std::pair{Strat::Safe, add2}
    );

    EXPECT_EQ(smap[Strat::Fast](10), 11);
    EXPECT_EQ(smap[Strat::Safe](10), 12);
}

// =============================================================================
// uniform_dispatch tests
// =============================================================================

TEST(UniformDispatch, ConceptSatisfied) {
    using UD = uniform_dispatch<Strat, Impl, 3>;
    static_assert(dispatchable<UD, Strat>);
}

TEST(UniformDispatch, DispatchIgnoresPosition) {
    constexpr auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast,   fast_impl},
        std::pair{Strat::Medium, medium_impl},
        std::pair{Strat::Safe,   safe_impl}
    );

    // Same result regardless of position.
    static_assert(dt.dispatch(0, Strat::Fast) == 1.0);
    static_assert(dt.dispatch(1, Strat::Fast) == 1.0);
    static_assert(dt.dispatch(99, Strat::Fast) == 1.0);
    static_assert(dt.dispatch(0, Strat::Safe) == 5.0);
}

// =============================================================================
// positional_dispatch tests
// =============================================================================

TEST(PositionalDispatch, ConceptSatisfied) {
    using PD = positional_dispatch<Strat, Impl, 3, 4>;
    static_assert(dispatchable<PD, Strat>);
}

TEST(PositionalDispatch, PerPositionMapping) {
    // Position 0: Fast=10, Medium=20
    // Position 1: Fast=30, Safe=40
    auto map0 = make_strategy_map<Strat, Impl>(
        std::pair{Strat::Fast, 10.0},
        std::pair{Strat::Medium, 20.0}
    );
    auto map1 = make_strategy_map<Strat, Impl>(
        std::pair{Strat::Fast, 30.0},
        std::pair{Strat::Safe, 40.0}
    );

    auto dt = make_positional_dispatch(map0, map1);

    EXPECT_DOUBLE_EQ(dt.dispatch(0, Strat::Fast), 10.0);
    EXPECT_DOUBLE_EQ(dt.dispatch(0, Strat::Medium), 20.0);
    EXPECT_DOUBLE_EQ(dt.dispatch(1, Strat::Fast), 30.0);
    EXPECT_DOUBLE_EQ(dt.dispatch(1, Strat::Safe), 40.0);
}

TEST(PositionalDispatch, BindAfterConstruction) {
    positional_dispatch<Strat, Impl, 3, 2> dt;
    dt.bind(0, Strat::Fast, 1.0);
    dt.bind(0, Strat::Safe, 5.0);
    dt.bind(1, Strat::Medium, 3.0);

    EXPECT_DOUBLE_EQ(dt.dispatch(0, Strat::Fast), 1.0);
    EXPECT_DOUBLE_EQ(dt.dispatch(0, Strat::Safe), 5.0);
    EXPECT_DOUBLE_EQ(dt.dispatch(1, Strat::Medium), 3.0);
    EXPECT_TRUE(dt.contains(0, Strat::Fast));
    EXPECT_FALSE(dt.contains(1, Strat::Fast));
}

// =============================================================================
// execute_plan tests: per_element plan + uniform dispatch
// =============================================================================

TEST(ExecutePlan, UniformDispatchAccumulates) {
    // Build a plan: positions 0,1,2 assigned Fast,Medium,Safe.
    plan<per_element_candidate<Strat, 3>> p;
    p.params[0] = Strat::Fast;
    p.params[1] = Strat::Medium;
    p.params[2] = Strat::Safe;
    p.predicted_cost = 9.0;

    auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast,   fast_impl},
        std::pair{Strat::Medium, medium_impl},
        std::pair{Strat::Safe,   safe_impl}
    );

    double total = 0.0;
    execute_plan(p, dt, [&total](std::size_t, Impl const& impl) {
        total += impl;
    });

    EXPECT_DOUBLE_EQ(total, 9.0);  // 1 + 3 + 5
}

TEST(ExecutePlan, PositionalDispatchDifferentImpls) {
    plan<per_element_candidate<Strat, 2>> p;
    p.params[0] = Strat::Fast;
    p.params[1] = Strat::Fast;
    p.predicted_cost = 0.0;

    // Fast at pos 0 → 10, Fast at pos 1 → 30.
    auto map0 = make_strategy_map<Strat, Impl>(
        std::pair{Strat::Fast, 10.0});
    auto map1 = make_strategy_map<Strat, Impl>(
        std::pair{Strat::Fast, 30.0});
    auto dt = make_positional_dispatch(map0, map1);

    double total = 0.0;
    execute_plan(p, dt, [&total](std::size_t, Impl const& impl) {
        total += impl;
    });

    EXPECT_DOUBLE_EQ(total, 40.0);  // 10 + 30
}

// =============================================================================
// execute_plan with descriptors (candidate_traits path)
// =============================================================================

TEST(ExecutePlan, WithDescriptors) {
    plan<per_element_candidate<Strat, 3>> p;
    p.params[0] = Strat::Fast;
    p.params[1] = Strat::Safe;
    p.params[2] = Strat::Medium;
    p.predicted_cost = 9.0;

    // Descriptors: just position names for this test.
    std::array<std::string, 3> descs = {"field_a", "field_b", "field_c"};

    auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast,   fast_impl},
        std::pair{Strat::Medium, medium_impl},
        std::pair{Strat::Safe,   safe_impl}
    );

    std::string log;
    execute_plan(p, descs, dt,
        [&log](std::size_t /*pos*/, std::string const& desc, Impl const& impl) {
            log += desc + ":" + std::to_string(impl) + " ";
        });

    EXPECT_TRUE(log.find("field_a:1") != std::string::npos);
    EXPECT_TRUE(log.find("field_b:5") != std::string::npos);
    EXPECT_TRUE(log.find("field_c:3") != std::string::npos);
}

// =============================================================================
// execute_plan_with_context
// =============================================================================

TEST(ExecutePlan, WithContext) {
    plan<per_element_candidate<Strat, 3>> p;
    p.params[0] = Strat::Fast;
    p.params[1] = Strat::Fast;
    p.params[2] = Strat::Safe;

    auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast,  fast_impl},
        std::pair{Strat::Safe,  safe_impl}
    );

    struct Ctx { double sum = 0.0; std::size_t calls = 0; };
    Ctx ctx;

    execute_plan_with_context(p, dt, ctx,
        [](std::size_t, Impl const& impl, Ctx& c) {
            c.sum += impl;
            c.calls++;
        });

    EXPECT_DOUBLE_EQ(ctx.sum, 7.0);  // 1 + 1 + 5
    EXPECT_EQ(ctx.calls, 3u);
}

// =============================================================================
// collect_implementations
// =============================================================================

TEST(CollectImplementations, FlatArray) {
    plan<per_element_candidate<Strat, 3>> p;
    p.params[0] = Strat::Safe;
    p.params[1] = Strat::Fast;
    p.params[2] = Strat::Medium;

    auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast,   fast_impl},
        std::pair{Strat::Medium, medium_impl},
        std::pair{Strat::Safe,   safe_impl}
    );

    auto impls = collect_implementations<Impl>(p, dt);

    EXPECT_DOUBLE_EQ(impls[0], 5.0);  // Safe
    EXPECT_DOUBLE_EQ(impls[1], 1.0);  // Fast
    EXPECT_DOUBLE_EQ(impls[2], 3.0);  // Medium
}

// =============================================================================
// execute_plan_checked: early exit
// =============================================================================

TEST(ExecutePlanChecked, EarlyExit) {
    plan<per_element_candidate<Strat, 4>> p;
    p.params[0] = Strat::Fast;
    p.params[1] = Strat::Safe;   // will fail check
    p.params[2] = Strat::Fast;
    p.params[3] = Strat::Fast;

    auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast,  fast_impl},
        std::pair{Strat::Safe,  safe_impl}
    );

    // Stop if impl > 3.0 (Safe = 5.0 fails).
    auto stats = execute_plan_checked(p, dt,
        [](std::size_t, Impl const& impl) -> bool {
            return impl <= 3.0;
        });

    EXPECT_EQ(stats.positions_executed, 1u);  // only pos 0
    EXPECT_EQ(stats.positions_skipped, 3u);   // pos 1,2,3
}

TEST(ExecutePlanChecked, AllPass) {
    plan<per_element_candidate<Strat, 3>> p;
    p.params[0] = Strat::Fast;
    p.params[1] = Strat::Fast;
    p.params[2] = Strat::Fast;

    auto dt = make_uniform_dispatch<Strat, Impl>(
        std::pair{Strat::Fast, fast_impl}
    );

    auto stats = execute_plan_checked(p, dt,
        [](std::size_t, Impl const&) -> bool { return true; });

    EXPECT_EQ(stats.positions_executed, 3u);
    EXPECT_EQ(stats.positions_skipped, 0u);
}

// =============================================================================
// Constexpr validation: full pipeline
// =============================================================================

constexpr auto constexpr_pipeline_test() {
    // 1. Build space
    auto space = make_anonymous_space<Strat, 3>(
        std::array{Strat::Fast, Strat::Medium, Strat::Safe});

    // 2. Solve
    auto cost = [](auto const& c) constexpr -> double {
        double t = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            switch (c[i]) {
                case Strat::Fast:   t += 1.0; break;
                case Strat::Medium: t += 3.0; break;
                case Strat::Safe:   t += 5.0; break;
            }
        }
        return t;
    };
    auto p = beam_search(space, cost);

    // 3. Build dispatch
    auto dt = make_uniform_dispatch<Strat, double>(
        std::pair{Strat::Fast,   1.0},
        std::pair{Strat::Medium, 3.0},
        std::pair{Strat::Safe,   5.0}
    );

    // 4. Execute and accumulate
    double total = 0.0;
    execute_plan(p, dt, [&total](std::size_t, double const& v) {
        total += v;
    });

    return total;
}

// Full solve → dispatch → execute at compile time.
static_assert(constexpr_pipeline_test() == 3.0);  // all-Fast = 1+1+1

// =============================================================================
// Integration: constrained solve → dispatch → execute
// =============================================================================

constexpr auto constrained_pipeline_test() {
    auto space = make_anonymous_space<Strat, 3>(
        std::array{Strat::Fast, Strat::Medium, Strat::Safe});

    auto cost = [](auto const& c) constexpr -> double {
        double t = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            switch (c[i]) {
                case Strat::Fast:   t += 1.0; break;
                case Strat::Medium: t += 3.0; break;
                case Strat::Safe:   t += 5.0; break;
            }
        }
        return t;
    };

    // Constraint: position 0 must not be Fast.
    auto no_fast_0 = [](auto const& c) constexpr -> bool {
        return c[0] != Strat::Fast;
    };

    auto p = beam_search(space, cost, no_fast_0);

    auto dt = make_uniform_dispatch<Strat, double>(
        std::pair{Strat::Fast,   1.0},
        std::pair{Strat::Medium, 3.0},
        std::pair{Strat::Safe,   5.0}
    );

    double total = 0.0;
    execute_plan(p, dt, [&total](std::size_t, double const& v) {
        total += v;
    });

    return total;
}

// Constrained: Medium(3) + Fast(1) + Fast(1) = 5.0
static_assert(constrained_pipeline_test() == 5.0);

// =============================================================================
// Integration: collect_implementations at compile time
// =============================================================================

constexpr auto collect_test() {
    plan<per_element_candidate<Strat, 3>> p;
    p.params[0] = Strat::Safe;
    p.params[1] = Strat::Fast;
    p.params[2] = Strat::Medium;

    auto dt = make_uniform_dispatch<Strat, double>(
        std::pair{Strat::Fast,   1.0},
        std::pair{Strat::Medium, 3.0},
        std::pair{Strat::Safe,   5.0}
    );

    return collect_implementations<double>(p, dt);
}

static_assert(collect_test()[0] == 5.0);
static_assert(collect_test()[1] == 1.0);
static_assert(collect_test()[2] == 3.0);
