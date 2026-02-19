// ctdp/solver/solver.h
// Compile-time dynamic programming framework — Analytics: Solver
// Umbrella header.

#ifndef CTDP_SOLVER_SOLVER_H
#define CTDP_SOLVER_SOLVER_H

// Core vocabulary types (re-exported for convenience)
#include "../core/plan.h"
#include "../core/plan_set.h"
#include "../core/plan_compose.h"
#include "../core/plan_traversal.h"
#include "../core/candidate_traits.h"

// Concepts
#include "concepts.h"

// Spaces
#include "spaces/per_element_space.h"
#include "spaces/heterogeneous_per_element_space.h"
#include "spaces/interval_split_space.h"
#include "spaces/cartesian_space.h"
#include "spaces/permutation_space.h"

// Cost models
#include "cost_models/additive.h"
#include "cost_models/chain.h"
#include "cost_models/weighted.h"

// Algorithms
#include "algorithms/per_element_argmin.h"
#include "algorithms/interval_dp.h"
#include "algorithms/exhaustive_search.h"
#include "algorithms/beam_search.h"
#include "algorithms/local_search.h"
#include "algorithms/select_and_run.h"

// Constraints
#include "constraints/constraint.h"

// Memoisation
#include "memo/candidate_cache.h"
#include "memo/cost_purity.h"

#endif // CTDP_SOLVER_SOLVER_H
