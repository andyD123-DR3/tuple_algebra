#pragma once

#include "spmv_design_space/plan.hpp"

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ctdp::spmv_dsl::plan_tree {

// A tiny self-contained plan-tree vocabulary for the standalone SpMV design
// space demonstrator.  It deliberately mirrors the intended CT-DP framework
// vocabulary, but keeps this project buildable as a standalone CMake project.
//
// There are two forms of leaf:
//   * leaf<Component> is a compile-time marker used by the older recursive
//     report code.
//   * leaf<Tag, Value> is the new selected-domain-component form.  It stores
//     the enum value chosen by the search, e.g. leaf<role::storage>{csr}.
//
// The current executors still consume plan_descriptor.  This layer is the
// first explicit shape of the target plan-tree architecture: selected domain
// components become leaves that can later acquire requirements, capabilities,
// cost contributions and lowering behaviour.

template<class Tag, class Value = void>
struct leaf;

template<class Component>
struct leaf<Component, void> {
    using component_t = Component;
};

template<class Tag, class Value>
struct leaf {
    using tag_type = Tag;
    using value_type = Value;

    Value value;
};

template<class... Children>
struct sequence {
    using children_t = std::tuple<Children...>;

    std::tuple<Children...> children;
};

template<class... Children>
struct seq {
    using children_t = std::tuple<Children...>;
};

template<class Tessellation, class Inner>
struct nest {
    using tessellation_t = Tessellation;
    using inner_t = Inner;
};

template<class Key, class... Groups>
struct split {
    using key_t = Key;
    using groups_t = std::tuple<Groups...>;
};

namespace role {

struct contract { static constexpr std::string_view name = "contract"; };
struct expression { static constexpr std::string_view name = "expression"; };
struct environment { static constexpr std::string_view name = "environment"; };
struct storage { static constexpr std::string_view name = "storage"; };
struct decomposition { static constexpr std::string_view name = "decomposition"; };
struct ordering { static constexpr std::string_view name = "ordering"; };
struct colouring { static constexpr std::string_view name = "colouring"; };
struct preconditioner { static constexpr std::string_view name = "preconditioner"; };
struct threading { static constexpr std::string_view name = "threading"; };
struct simd { static constexpr std::string_view name = "simd"; };
struct fusion { static constexpr std::string_view name = "fusion"; };
struct reduction { static constexpr std::string_view name = "reduction"; };
struct observation { static constexpr std::string_view name = "observation"; };
struct conformance { static constexpr std::string_view name = "conformance"; };
struct lowering { static constexpr std::string_view name = "lowering"; };

} // namespace role

namespace tag {

struct strict_expression_contract { static constexpr const char* name = "contract::strict_expression"; };
struct residual_jacobi_dot_update { static constexpr const char* name = "expression::residual_jacobi_dot_update"; };
struct strict_ieee_no_implicit_fma { static constexpr const char* name = "environment::strict_ieee_no_implicit_fma"; };
struct matrix_free_stencil { static constexpr const char* name = "storage::matrix_free_stencil"; };
struct fixed_row_order { static constexpr const char* name = "ordering::fixed_row_order"; };
struct fixed_nonzero_order { static constexpr const char* name = "ordering::fixed_nonzero_order"; };
struct row_bisection_search { static constexpr const char* name = "decomposition::row_bisection_search"; };
struct canonical_pairwise_rho { static constexpr const char* name = "reduction::canonical_pairwise_rho"; };
struct observe_x_next { static constexpr const char* name = "observation::observe_x_next"; };
struct observe_rho { static constexpr const char* name = "observation::observe_rho"; };
struct bitwise_observation_check { static constexpr const char* name = "conformance::bitwise_observation_check"; };
struct matrix_free_executor_adapter { static constexpr const char* name = "lowering::matrix_free_executor_adapter"; };

} // namespace tag

template<class T, class = void>
struct component_name {
    static std::string get() { return "<unnamed component>"; }
};

template<class T>
struct component_name<T, std::void_t<decltype(T::name)>> {
    static std::string get() { return std::string(T::name); }
};

inline void indent(std::ostream& os, int depth) {
    for (int i = 0; i < depth; ++i) {
        os << "  ";
    }
}

template<class Value>
std::string value_name(Value value) {
    return std::string(ctdp::spmv_dsl::to_string(value));
}

inline std::string value_name(std::string_view value) {
    return std::string(value);
}

template<class Plan>
void print(std::ostream& os, int depth = 0);

template<class Component>
void print(std::ostream& os, leaf<Component, void>, int depth) {
    indent(os, depth);
    os << "leaf<" << component_name<Component>::get() << ">\n";
}

template<class Tag, class Value>
void print(std::ostream& os, const leaf<Tag, Value>& selected, int depth) {
    indent(os, depth);
    os << "leaf<" << component_name<Tag>::get() << "::" << value_name(selected.value) << ">\n";
}

template<class... Children>
void print(std::ostream& os, seq<Children...>, int depth) {
    indent(os, depth);
    os << "seq\n";
    (print(os, Children{}, depth + 1), ...);
}

template<class... Children>
void print(std::ostream& os, const sequence<Children...>& selected, int depth) {
    indent(os, depth);
    os << "seq\n";
    std::apply([&](const auto&... child) {
        (print(os, child, depth + 1), ...);
    }, selected.children);
}

template<class Tessellation, class Inner>
void print(std::ostream& os, nest<Tessellation, Inner>, int depth) {
    indent(os, depth);
    os << "nest<" << component_name<Tessellation>::get() << ">\n";
    print(os, Inner{}, depth + 1);
}

template<class Key, class... Groups>
void print(std::ostream& os, split<Key, Groups...>, int depth) {
    indent(os, depth);
    os << "split<" << component_name<Key>::get() << ">\n";
    (print(os, Groups{}, depth + 1), ...);
}

template<class Plan>
void print(std::ostream& os, int depth) {
    print(os, Plan{}, depth);
}

template<class Plan>
std::string to_string() {
    std::ostringstream os;
    print<Plan>(os, 0);
    return os.str();
}

template<class... Children>
std::string to_string(const sequence<Children...>& selected) {
    std::ostringstream os;
    print(os, selected, 0);
    return os.str();
}

template<class>
inline constexpr bool always_false_v = false;

template<class Tag, class Candidate>
inline constexpr bool leaf_has_tag_v = false;

template<class Tag, class Value>
inline constexpr bool leaf_has_tag_v<Tag, leaf<Tag, Value>> = true;

template<class Tag, class Tuple, std::size_t I = 0>
decltype(auto) get_leaf_from_tuple(Tuple& tuple) {
    if constexpr (I >= std::tuple_size_v<std::remove_reference_t<Tuple>>) {
        static_assert(always_false_v<Tag>, "requested plan_tree leaf tag is not present in this sequence");
    } else {
        using candidate_t = std::remove_cv_t<std::remove_reference_t<decltype(std::get<I>(tuple))>>;
        if constexpr (leaf_has_tag_v<Tag, candidate_t>) {
            return (std::get<I>(tuple));
        } else {
            return get_leaf_from_tuple<Tag, Tuple, I + 1>(tuple);
        }
    }
}

template<class Tag, class... Children>
decltype(auto) get_leaf(sequence<Children...>& selected) {
    return get_leaf_from_tuple<Tag>(selected.children);
}

template<class Tag, class... Children>
decltype(auto) get_leaf(const sequence<Children...>& selected) {
    return get_leaf_from_tuple<Tag>(selected.children);
}

using spmv_plan_tree = sequence<
    leaf<role::contract, contract_level>,
    leaf<tag::residual_jacobi_dot_update>,
    leaf<tag::strict_ieee_no_implicit_fma>,
    leaf<role::storage, storage_kind>,
    leaf<role::decomposition, decomposition_kind>,
    leaf<role::ordering, ordering_kind>,
    leaf<role::colouring, colouring_kind>,
    leaf<role::preconditioner, preconditioner_kind>,
    leaf<role::threading, threading_kind>,
    leaf<role::simd, simd_kind>,
    leaf<role::fusion, fusion_kind>,
    leaf<role::reduction, reduction_kind>,
    leaf<tag::observe_x_next>,
    leaf<tag::observe_rho>,
    leaf<tag::bitwise_observation_check>,
    leaf<role::lowering, executor_kind>
>;

inline spmv_plan_tree as_plan_tree(const plan_descriptor& p) {
    return {{
        leaf<role::contract, contract_level>{p.contract},
        leaf<tag::residual_jacobi_dot_update>{},
        leaf<tag::strict_ieee_no_implicit_fma>{},
        leaf<role::storage, storage_kind>{p.storage},
        leaf<role::decomposition, decomposition_kind>{p.decomposition},
        leaf<role::ordering, ordering_kind>{p.ordering},
        leaf<role::colouring, colouring_kind>{p.colouring},
        leaf<role::preconditioner, preconditioner_kind>{p.preconditioner},
        leaf<role::threading, threading_kind>{p.threading},
        leaf<role::simd, simd_kind>{p.simd},
        leaf<role::fusion, fusion_kind>{p.fusion},
        leaf<role::reduction, reduction_kind>{p.reduction},
        leaf<tag::observe_x_next>{},
        leaf<tag::observe_rho>{},
        leaf<tag::bitwise_observation_check>{},
        leaf<role::lowering, executor_kind>{p.executor}
    }};
}

using matrix_free_recursive_candidate = seq<
    leaf<tag::strict_expression_contract>,
    leaf<tag::residual_jacobi_dot_update>,
    leaf<tag::strict_ieee_no_implicit_fma>,
    leaf<tag::matrix_free_stencil>,
    leaf<tag::fixed_row_order>,
    leaf<tag::fixed_nonzero_order>,
    nest<
        tag::row_bisection_search,
        leaf<tag::matrix_free_stencil>
    >,
    leaf<tag::canonical_pairwise_rho>,
    leaf<tag::observe_x_next>,
    leaf<tag::observe_rho>,
    leaf<tag::bitwise_observation_check>,
    leaf<tag::matrix_free_executor_adapter>
>;

} // namespace ctdp::spmv_dsl::plan_tree
