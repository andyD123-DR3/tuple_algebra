#pragma once

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

namespace ctdp::spmv_dsl::plan_tree {

template<class Component>
struct leaf {
    using component_t = Component;
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
    static std::string get() { return T::name; }
};

inline void indent(std::ostream& os, int depth) {
    for (int i = 0; i < depth; ++i) {
        os << "  ";
    }
}

template<class Plan>
void print(std::ostream& os, int depth = 0);

template<class Component>
void print(std::ostream& os, leaf<Component>, int depth) {
    indent(os, depth);
    os << "leaf<" << component_name<Component>::get() << ">\n";
}

template<class... Children>
void print(std::ostream& os, seq<Children...>, int depth) {
    indent(os, depth);
    os << "seq\n";
    (print(os, Children{}, depth + 1), ...);
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
