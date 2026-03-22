// interval_partition_plan.hpp - Plan term for binary interval partition
// Sprint 8: Integration layer for descriptor + context
// Author: Andrew Drakeford

#ifndef CT_DP_PLAN_INTERVAL_PARTITION_PLAN_HPP
#define CT_DP_PLAN_INTERVAL_PARTITION_PLAN_HPP

#include "ct_dp/solver/interval_context.hpp"
#include "ct_dp/space/binary_cut_desc.hpp"
#include <cassert>
#include <cstddef>

namespace ct_dp {
    namespace plan {

<<<<<<< HEAD
        struct interval_partition_plan {
            solver::interval_context whole;
            size_t split;
            solver::interval_context left_ctx;
            solver::interval_context right_ctx;

            template<size_t Len>
            [[nodiscard]] static constexpr interval_partition_plan make(
                solver::interval_context ctx,
                space::binary_cut_desc<Len> desc,
                size_t ordinal
            ) noexcept {
                assert(ctx.size() == Len &&
                    "Context size must match descriptor length (PRECONDITION)");
                size_t rel = desc.ordinal_to_relative_cut(ordinal);
                size_t k = ctx.start() + rel;
                return { ctx, k, ctx.left(k), ctx.right(k) };
            }

            // Sprint 9: Runtime plan construction from split point
            [[nodiscard]] static constexpr interval_partition_plan from_split(
                solver::interval_context ctx,
                size_t k
            ) noexcept {
                assert(k > ctx.i && k < ctx.j && "Invalid split point");
                return { ctx, k, solver::interval_context{ctx.i, k}, solver::interval_context{k, ctx.j} };
            }

            constexpr bool is_legal() const noexcept {
                return whole.i < split && split < whole.j &&
                    left_ctx.i == whole.i && left_ctx.j == split &&
                    right_ctx.i == split && right_ctx.j == whole.j;
            }

            constexpr size_t left_size() const noexcept { return left_ctx.size(); }
            constexpr size_t right_size() const noexcept { return right_ctx.size(); }

            constexpr bool preserves_size() const noexcept {
                return left_size() + right_size() == whole.size();
            }
        };

    } // namespace plan
=======
/**
 * @brief Plan term representing a binary partition of an interval
 * 
 * Semantic Law: Applying a valid cut choice to an interval_context yields
 * a partition plan node consisting of the whole interval, an absolute cut point,
 * and the derived left/right subintervals.
 * 
 * This is the INTEGRATION LAYER - the named home for descriptor+context composition.
 * 
 * NOTE: interval_partition_plan is a plain value term (public aggregate);
 * callers may construct illegal values directly. Use make(...) for
 * invariant-preserving construction. The is_legal() method can verify
 * whether a plan satisfies structural invariants.
 */
struct interval_partition_plan {
    solver::interval_context whole;       ///< Full interval [i, j)
    size_t split;                         ///< Absolute cut point k where i < k < j
    solver::interval_context left_ctx;    ///< Left subinterval [i, k)
    solver::interval_context right_ctx;   ///< Right subinterval [k, j)
    
    // Sprint 9: Runtime plan construction from split point
    [[nodiscard]] static constexpr interval_partition_plan from_split(
        solver::interval_context ctx, 
        size_t k
    ) noexcept {
        assert(k > ctx.i && k < ctx.j && "Invalid split point");
        return {ctx, k, solver::interval_context{ctx.i, k}, solver::interval_context{k, ctx.j}};
    }
    
    /**
     * @brief Factory: Apply descriptor choice to context
     * 
     * This is the integration layer - the semantic composition of
     * descriptor (choice space) and context (runtime state).
     * 
     * Preconditions (enforced by assertion):
     * - ctx.size() == Len (context size must match descriptor length)
     * - ordinal < desc.size() (ordinal must be valid)
     * 
     * Failure Policy:
     * - Debug builds: assert() checks preconditions
     * - Production builds: undefined behavior on violation
     * - No runtime error channel provided
     * - Rationale: Low-level library design; caller responsibility
     * 
     * Construction Rule:
     * Applying ordinal from binary_cut_desc<Len> to interval_context{i,j}
     * with j-i == Len yields interval_partition_plan with:
     * - whole = {i, j}
     * - split = i + rel (where rel = desc.ordinal_to_relative_cut(ordinal))
     * - left_ctx = {i, split}
     * - right_ctx = {split, j}
     * 
     * @tparam Len Descriptor length parameter
     * @param ctx Runtime interval context
     * @param desc Descriptor providing choice space
     * @param ordinal Choice index [0, Len-2]
     * @return Partition plan with derived subintervals
     */
    template<size_t Len>
    [[nodiscard]] static constexpr interval_partition_plan make(
        solver::interval_context ctx,
        space::binary_cut_desc<Len> desc,
        size_t ordinal
    ) noexcept {
        // Precondition: Context size must match descriptor length
        assert(ctx.size() == Len && 
               "Context size must match descriptor length (PRECONDITION)");
        
        // Apply descriptor to context
        size_t rel = desc.ordinal_to_relative_cut(ordinal);
        size_t k = ctx.start() + rel;  // Absolute cut point
        
        return {
            ctx,              // whole
            k,                // split
            ctx.left(k),      // left_ctx
            ctx.right(k)      // right_ctx
        };
    }
    
    /**
     * @brief Verify legality invariant
     * 
     * A partition plan is legal iff:
     * - whole.i < split < whole.j (split is strictly interior)
     * - left_ctx = [whole.i, split)
     * - right_ctx = [split, whole.j)
     * 
     * @return true if plan is legal
     */
    constexpr bool is_legal() const noexcept {
        return whole.i < split && split < whole.j &&
               left_ctx.i == whole.i && left_ctx.j == split &&
               right_ctx.i == split && right_ctx.j == whole.j;
    }
    
    /**
     * @brief Get left subproblem size
     * @return left_ctx.size()
     */
    constexpr size_t left_size() const noexcept { 
        return left_ctx.size(); 
    }
    
    /**
     * @brief Get right subproblem size
     * @return right_ctx.size()
     */
    constexpr size_t right_size() const noexcept { 
        return right_ctx.size(); 
    }
    
    /**
     * @brief Verify size preservation invariant
     * 
     * Law: Partition preserves interval coverage
     * left_size() + right_size() == whole.size()
     * 
     * @return true if sizes sum to whole
     */
    constexpr bool preserves_size() const noexcept {
        return left_size() + right_size() == whole.size();
    }
};

} // namespace plan
>>>>>>> 91070352b362068fb3adf745db68327a0cc24878
} // namespace ct_dp

#endif // CT_DP_PLAN_INTERVAL_PARTITION_PLAN_HPP