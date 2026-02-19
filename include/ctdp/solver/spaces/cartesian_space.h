// ctdp/solver/spaces/cartesian_space.h
// Compile-time dynamic programming framework — Analytics: Solver
// Product of K finite enumeration sets.
// Small enough for exhaustive search.
//
// Satisfies: search_space, enumerable_space, neighbourhood_space.

#ifndef CTDP_SOLVER_SPACES_CARTESIAN_SPACE_H
#define CTDP_SOLVER_SPACES_CARTESIAN_SPACE_H

#include "../concepts.h"
#include "../../core/constexpr_vector.h"
#include <cstddef>
#include <tuple>

namespace ctdp {

template<std::size_t MaxPerAxis = 16, typename... Enums>
struct cartesian_space {
    using candidate_type = std::tuple<Enums...>;

    static constexpr std::size_t num_axes = sizeof...(Enums);

    std::tuple<constexpr_vector<Enums, MaxPerAxis>...> axes{};

    // Product of axis sizes
    [[nodiscard]] constexpr auto size() const -> std::size_t {
        std::size_t result = 1;
        std::apply([&](auto const&... axis) {
            ((result *= axis.size()), ...);
        }, axes);
        return result;
    }

    // Enumerate all candidates via callback
    template<typename F>
    constexpr void enumerate(F fn) const {
        candidate_type c{};
        enumerate_impl<0>(fn, c);
    }

    // Neighbours: flip one axis value at a time
    template<typename F>
    constexpr void neighbours(candidate_type const& c, F fn) const {
        neighbours_impl<0>(c, fn);
    }

private:
    template<std::size_t Axis, typename F>
    constexpr void enumerate_impl(F& fn, candidate_type& c) const {
        if constexpr (Axis == num_axes) {
            fn(static_cast<candidate_type const&>(c));
        } else {
            auto const& axis = std::get<Axis>(axes);
            for (std::size_t i = 0; i < axis.size(); ++i) {
                std::get<Axis>(c) = axis[i];
                enumerate_impl<Axis + 1>(fn, c);
            }
        }
    }

    template<std::size_t Axis, typename F>
    constexpr void neighbours_impl(candidate_type const& c, F& fn) const {
        if constexpr (Axis < num_axes) {
            auto const& axis = std::get<Axis>(axes);
            auto const current = std::get<Axis>(c);
            for (std::size_t i = 0; i < axis.size(); ++i) {
                if (axis[i] != current) {
                    candidate_type neighbour = c;
                    std::get<Axis>(neighbour) = axis[i];
                    fn(static_cast<candidate_type const&>(neighbour));
                }
            }
            neighbours_impl<Axis + 1>(c, fn);
        }
    }
};

} // namespace ctdp

// Verify concept satisfaction
namespace ctdp::detail {
    enum class CartColor_ { R, G };
    enum class CartSize_  { S, M };
    static_assert(search_space<cartesian_space<4, CartColor_, CartSize_>>);
}

#endif // CTDP_SOLVER_SPACES_CARTESIAN_SPACE_H
