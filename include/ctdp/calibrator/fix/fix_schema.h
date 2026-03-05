#ifndef CTDP_CALIBRATOR_FIX_SCHEMA_H
#define CTDP_CALIBRATOR_FIX_SCHEMA_H

// ============================================================
//  fix_schema.h  –  CT-DP FIX Parser Optimiser
//
//  Top-level convenience header for the fix/ calibrator stack.
//
//  Including this one file gives access to the entire Phase 1
//  type hierarchy:
//
//    data_point.h          → Strategy, DataPoint<N>, RunTriple, ...
//    fix_field_descriptor.h→ FieldDescriptor, Schema<N>,
//                            trivial_schema, full_schema
//    fix_strategy_ids.h    → SchemaIndex<N>, CostTable<N,T>,
//                            trivial_index, full_index
//    counter_preprocessor.h→ CounterPreprocessor<N>, assign_folds
//
//  ET parser bridge:
//    schema_to_fix_config<N>(schema, plan)
//      Converts an array<Strategy,N> plan into fix_config for
//      direct use with fix_et_parser.h template instantiations.
//      Pads with Strategy::Generic if N < num_fields.
//
//    fix_config_to_plan<N>(cfg)
//      Inverse: extracts the first N entries of a fix_config.
//
//  Schema validation:
//    validate_schema_vs_parser<N>(schema)
//      Asserts that schema.digit_counts() matches the first N
//      entries of field_digits[] in fix_et_parser.h.
//      Throws std::logic_error on mismatch (debug builds only).
//
//  Dependencies: all fix/ headers + fix_et_parser.h
//  C++ standard: C++20
// ============================================================

// fix/ calibrator stack
#include <ctdp/calibrator/fix/data_point.h>
#include <ctdp/calibrator/fix/fix_field_descriptor.h>
#include <ctdp/calibrator/fix/fix_strategy_ids.h>
#include <ctdp/calibrator/fix/counter_preprocessor.h>

// ET parser (Strategy already canonical in data_point.h; no ODR issue)
#include <ctdp/calibrator/fix_et_parser.h>

#include <stdexcept>
#include <string>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  schema_to_fix_config — bridge from Schema<N> plan to fix_et_parser fix_config
//
//  fix_config is array<Strategy, num_fields> (always 12 elements).
//  For Trivial (N=4), positions [4..11] are padded with Generic
//  — the ET parser generates dead code for those fields which the
//  compiler eliminates; only fields [0..N) contribute to parse time.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] constexpr fix_config
schema_to_fix_config(const std::array<Strategy, N>& plan) noexcept {
    fix_config cfg{};
    cfg.fill(Strategy::Generic);            // default: Generic for unused fields
    for (int i = 0; i < N && i < num_fields; ++i)
        cfg[static_cast<std::size_t>(i)] = plan[i];
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
//  fix_config_to_plan — extract first N fields from a fix_config
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] constexpr std::array<Strategy, N>
fix_config_to_plan(const fix_config& cfg) noexcept {
    std::array<Strategy, N> plan{};
    for (int i = 0; i < N && i < num_fields; ++i)
        plan[i] = cfg[static_cast<std::size_t>(i)];
    return plan;
}

// ─────────────────────────────────────────────────────────────────────────────
//  validate_schema_vs_parser — check digit_count alignment at runtime.
//
//  The ET parser's field_digits[] is the ground truth for digit widths.
//  This function confirms that Schema<N>.digit_counts() agrees with the
//  first N entries.  Call once at startup in debug builds.
//
//  Throws std::logic_error describing the first mismatch found.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
void validate_schema_vs_parser(const Schema<N>& schema) {
    auto dc = schema.digit_counts();
    for (int i = 0; i < N && i < num_fields; ++i) {
        if (dc[i] != field_digits[static_cast<std::size_t>(i)]) {
            throw std::logic_error(
                "fix_schema: digit_count mismatch at field " + std::to_string(i) +
                ": schema says " + std::to_string(dc[i]) +
                ", parser expects " + std::to_string(field_digits[static_cast<std::size_t>(i)]));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  plan_string — round-trip string for a plan array (e.g. "USGL")
//  Convenience wrapper over strategy_char; avoids spelling out the loop.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] std::string plan_string(const std::array<Strategy, N>& plan) noexcept {
    std::string s;
    s.reserve(N);
    for (auto st : plan) s += strategy_char(st);
    return s;
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_SCHEMA_H
