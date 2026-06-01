#pragma once

#include "spmv_design_space/enums.hpp"

#include <string>

namespace ctdp::spmv_dsl {

struct environment_contract {
    bool round_to_nearest_even = true;
    bool implicit_fma_allowed = false;
    bool ftz_daz_allowed = false;
    bool deterministic_scope_required = true;
};

struct observation_contract {
    bool observe_x_next = true;
    bool observe_rho = true;
    bool observe_residual = false;
};

struct expression_contract {
    contract_level level = contract_level::strict_expression;
    environment_contract environment{};
    observation_contract observation{};
    bool fixed_pointwise_expression = true;
    bool fixed_preconditioner_binding = true;
    bool fixed_observed_reduction_tree = true;
};

struct capabilities {
    bool preserves_expression_identity = true;
    bool preserves_environment = true;
    bool preserves_observed_reduction = true;
    bool supports_bitwise_check = true;
    bool uses_backend_black_box = false;
    bool may_use_fast_math = false;
    bool may_use_unordered_parallelism = false;
    bool may_use_mixed_precision = false;
};

struct legality_result {
    bool structurally_legal = true;
    bool numerically_legal = true;
    bool target_legal = true;
    bool observation_legal = true;
    std::string reason = "legal";

    [[nodiscard]] bool legal() const noexcept {
        return structurally_legal && numerically_legal && target_legal && observation_legal;
    }
};

constexpr capabilities capabilities_for(reduction_kind reduction) noexcept {
    capabilities c{};
    if (reduction == reduction_kind::thread_local_unordered_witness) {
        c.preserves_observed_reduction = false;
        c.supports_bitwise_check = false;
        c.may_use_unordered_parallelism = true;
    }
    return c;
}

constexpr bool strict_requires_fixed_preconditioner(contract_level level) noexcept {
    return level == contract_level::strict_expression;
}

} // namespace ctdp::spmv_dsl
