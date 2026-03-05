#ifndef CTDP_CALIBRATOR_FIX_FIELD_DESCRIPTOR_H
#define CTDP_CALIBRATOR_FIX_FIELD_DESCRIPTOR_H

// ============================================================
//  fix_field_descriptor.h  –  CT-DP FIX Parser Optimiser
//
//  FieldDescriptor: compile-time per-field schema record.
//  Schema<N>:       array of N FieldDescriptors + helpers.
//
//  Two concrete schemas are provided:
//
//    trivial_schema  — 4 fields, 4^4 = 256 plan space
//      BeginString(8)  2 digits  U/S/L/G
//      BodyLength(9)   4 digits  U/S/L/G
//      MsgType(35)     1 digit   U only   (single char, no loop needed)
//      CheckSum(10)    3 digits  U/S/L/G
//
//    full_schema     — 12 fields, 4^12 = 16.7M plan space
//      NewOrderSingle required integer fields (tag ordering matches
//      field_digits[] in fix_et_parser.h and the existing 12-field
//      calibration harness).
//
//  FieldDescriptor records:
//    name           — human-readable label for diagnostics / CSV output
//    tag            — FIX tag number
//    digit_count    — number of decimal digits in the field value
//    allowed_mask   — bitmask of valid Strategy values (bit i = Strategy(i))
//    is_variable    — true if digit_count may vary per-message
//                     (affects corpus-level alignment; see calibrator_design_v2 §corpus)
//
//  allowed_mask helpers:
//    all_strategies()      — 0b1111 (U|S|L|G)
//    strategies({U,S,L})   — builds mask from initializer_list
//    is_allowed(mask, s)   — test one strategy
//
//  Plan enumeration:
//    Schema<N>::enumerate_plans() returns all valid plans as
//    vector<array<Strategy,N>>, respecting per-field allowed_mask.
//    For trivial_schema this is 3×4×1×4 = 48 plans (MsgType locked to U).
//    For full_schema with all fields unlocked it is 4^12 = 16,777,216.
//
//  Dependencies: data_point.h (for Strategy / NUM_STRATEGIES)
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/data_point.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  allowed_mask helpers
// ─────────────────────────────────────────────────────────────────────────────

using AllowedMask = uint8_t;

[[nodiscard]] constexpr AllowedMask all_strategies() noexcept {
    return (AllowedMask{1} << NUM_STRATEGIES) - 1;  // 0b00001111
}

[[nodiscard]] constexpr AllowedMask strategies(
    std::initializer_list<Strategy> strats) noexcept
{
    AllowedMask mask = 0;
    for (Strategy s : strats)
        mask |= static_cast<AllowedMask>(1u << static_cast<int>(s));
    return mask;
}

[[nodiscard]] constexpr bool is_allowed(AllowedMask mask, Strategy s) noexcept {
    return (mask >> static_cast<int>(s)) & 1u;
}

[[nodiscard]] constexpr int count_allowed(AllowedMask mask) noexcept {
    int n = 0;
    for (int i = 0; i < NUM_STRATEGIES; ++i)
        n += (mask >> i) & 1;
    return n;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FieldDescriptor
// ─────────────────────────────────────────────────────────────────────────────

struct FieldDescriptor {
    std::string_view name;          // e.g. "BeginString"
    int              tag;           // FIX tag number
    int              digit_count;   // nominal decimal digits
    AllowedMask      allowed_mask;  // valid strategies for this field
    bool             is_variable;   // digit_count may vary per-message

    [[nodiscard]] constexpr bool allows(Strategy s) const noexcept {
        return is_allowed(allowed_mask, s);
    }

    [[nodiscard]] constexpr int n_allowed() const noexcept {
        return count_allowed(allowed_mask);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Schema<N> — array of N FieldDescriptors + plan-enumeration helpers
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
    requires (N >= 1 && N <= 16)
struct Schema {
    std::array<FieldDescriptor, N> fields;

    // Total plan-space size = Π allowed[i] over all fields.
    [[nodiscard]] constexpr int64_t plan_space_size() const noexcept {
        int64_t sz = 1;
        for (auto& f : fields) sz *= f.n_allowed();
        return sz;
    }

    // digit_count array — matches the layout expected by fix_et_parser.h
    [[nodiscard]] constexpr std::array<int, N> digit_counts() const noexcept {
        std::array<int, N> dc{};
        for (int i = 0; i < N; ++i) dc[i] = fields[i].digit_count;
        return dc;
    }

    // Enumerate every valid plan as an array<Strategy,N>.
    // Order: field[0] varies slowest, field[N-1] varies fastest
    // (consistent with plan_id base-4 encoding in DataPoint).
    [[nodiscard]] std::vector<std::array<Strategy, N>> enumerate_plans() const {
        std::vector<std::array<Strategy, N>> plans;
        plans.reserve(static_cast<std::size_t>(plan_space_size()));
        std::array<Strategy, N> current{};
        enumerate_impl(plans, current, 0);
        return plans;
    }

    // Validate that a plan respects all per-field allowed_masks.
    [[nodiscard]] constexpr bool is_valid_plan(
        const std::array<Strategy, N>& plan) const noexcept
    {
        for (int i = 0; i < N; ++i)
            if (!fields[i].allows(plan[i])) return false;
        return true;
    }

private:
    void enumerate_impl(std::vector<std::array<Strategy, N>>& out,
                        std::array<Strategy, N>&               current,
                        int                                    field_idx) const
    {
        if (field_idx == N) {
            out.push_back(current);
            return;
        }
        for (int si = 0; si < NUM_STRATEGIES; ++si) {
            auto s = static_cast<Strategy>(si);
            if (fields[field_idx].allows(s)) {
                current[field_idx] = s;
                enumerate_impl(out, current, field_idx + 1);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  trivial_schema — 4 fields; proof-of-concept / March 13 talk target
//
//  Field         Tag   Digits  Allowed       Rationale
//  BeginString    8      2     U/S/L/G       Fixed 2-digit version "11"
//  BodyLength     9      4     U/S/L/G       Variable (is_variable=true)
//  MsgType       35      1     U only        Single char — no loop benefit
//  CheckSum      10      3     U/S/L/G       Fixed 3-digit value
//
//  Plan space: 4 × 4 × 1 × 4 = 64 valid plans (not 256 — MsgType locked).
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr Schema<4> trivial_schema = {{
    FieldDescriptor{ "BeginString", 8,  2, all_strategies(),          false },
    FieldDescriptor{ "BodyLength",  9,  4, all_strategies(),          true  },
    FieldDescriptor{ "MsgType",    35,  1, strategies({Strategy::Unrolled}), false },
    FieldDescriptor{ "CheckSum",   10,  3, all_strategies(),          false },
}};

// ─────────────────────────────────────────────────────────────────────────────
//  full_schema — 12 fields; NewOrderSingle required integer fields.
//
//  Digit counts match field_digits[] in fix_et_parser.h (the existing
//  12-field calibration harness).  All fields allow all strategies; the
//  DP search will select based on measured cost, not hard exclusions.
//
//  Plan space: 4^12 = 16,777,216 (beam search required above N=6).
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr Schema<12> full_schema = {{
    FieldDescriptor{ "BeginString",    8,   2, all_strategies(), false },
    FieldDescriptor{ "BodyLength",     9,   4, all_strategies(), true  },
    FieldDescriptor{ "MsgType",       35,   1, all_strategies(), false },
    FieldDescriptor{ "SenderCompID",  49,   4, all_strategies(), false },
    FieldDescriptor{ "TargetCompID",  56,   4, all_strategies(), false },
    FieldDescriptor{ "MsgSeqNum",     34,   6, all_strategies(), false },
    FieldDescriptor{ "ClOrdID",       11,   6, all_strategies(), false },
    FieldDescriptor{ "Symbol",        55,   4, all_strategies(), false },
    FieldDescriptor{ "Side",          54,   1, all_strategies(), false },
    FieldDescriptor{ "TransactTime",  60,   8, all_strategies(), false },
    FieldDescriptor{ "OrderQty",      38,   6, all_strategies(), false },
    FieldDescriptor{ "CheckSum",      10,   3, all_strategies(), false },
}};

// ─────────────────────────────────────────────────────────────────────────────
//  make_data_points_from_schema — build the full corpus skeleton for a schema.
//
//  Enumerates all valid plans, creates a DataPoint<N> per plan with
//  measurements left zeroed (status = HardDeleted until runs are filled).
//  Call update_status() on each after measurement.
//
//  assign_folds() should be called after all measurements are populated.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] std::vector<DataPoint<N>>
make_data_points_from_schema(const Schema<N>& schema) {
    auto plans = schema.enumerate_plans();
    std::vector<DataPoint<N>> corpus;
    corpus.reserve(plans.size());
    for (auto& plan : plans) {
        DataPoint<N> dp{};
        dp.plan = plan;
        dp.update_plan_id();
        // status starts as HardDeleted (zero runs) until measurement fills it
        dp.status = DataPointStatus::HardDeleted;
        corpus.push_back(dp);
    }
    return corpus;
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_FIELD_DESCRIPTOR_H
