#pragma once

// ============================================================================
// CT-DP Phase 5 — Plan library public entry point
// ============================================================================
//
// This is the only header users should include. Direct inclusion of
// constructors.h, wrappers.h, or traits.h produces a compile error because
// partial inclusion can cause is_plan_v to give different answers in
// different translation units (an is_wrapper ODR hazard).
//
// Recommended usage:
//   #include <ctdp/plan/plan.h>
//   namespace cp = ct_dp::plan;
//   namespace cw = ct_dp::plan::wrap;
//
// so plan types read as, for example:
//   cp::seq_t<cp::leaf<K1>, cw::vectorise<8, cp::leaf<K2>>>
//
// Future major revisions will wrap the ct_dp::plan namespace in an inline
// version namespace (plan_v1, plan_v2, ...). This is not yet in place for
// a one-user research codebase but is anticipated.
// ============================================================================

#if __cplusplus < 202002L
#  error "CT-DP Phase 5 plan library requires C++20 or later"
#endif

#define CTDP_PLAN_UMBRELLA_INCLUDE 1

#include "constructors.hpp"
#include "wrappers.hpp"
#include "traits.hpp"

#undef CTDP_PLAN_UMBRELLA_INCLUDE
