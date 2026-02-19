// graph/construction/from_stencil.h - Stencil-to-implicit-graph factory
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Stencil patterns on regular grids are a primary domain for SpMV and PDE
// problems. from_stencil converts a dimensions array + offset list into an
// implicit_graph whose generator maps:
//   node_id → grid coordinate → offset application → neighbor list
//
// Boundary nodes produce fewer neighbors (offsets that fall outside the
// grid are excluded). The generator is deterministic: offsets are applied
// in the order provided, yielding consistent neighbor ordering.
//
// Example: 2D 5-point stencil on 100×100 grid:
//   auto g = from_stencil(
//       std::array<std::size_t, 2>{100, 100},
//       std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
//   );
//   // g.node_count() == 10000
//   // Interior node: 4 neighbors
//   // Corner node: 2 neighbors

#ifndef CTDP_GRAPH_FROM_STENCIL_H
#define CTDP_GRAPH_FROM_STENCIL_H

#include "implicit_graph.h"
#include "graph_concepts.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

namespace detail {

/// Compute total number of grid points from dimensions.
template<std::size_t Dim>
[[nodiscard]] constexpr std::size_t
grid_size(std::array<std::size_t, Dim> const& dims) noexcept {
    std::size_t total = 1;
    for (std::size_t d = 0; d < Dim; ++d) {
        total *= dims[d];
    }
    return total;
}

/// Convert flat index to multi-dimensional coordinate.
template<std::size_t Dim>
[[nodiscard]] constexpr std::array<std::size_t, Dim>
to_coord(std::size_t flat, std::array<std::size_t, Dim> const& dims) noexcept {
    std::array<std::size_t, Dim> coord{};
    for (std::size_t d = Dim; d > 0; --d) {
        auto const i = d - 1;
        coord[i] = flat % dims[i];
        flat /= dims[i];
    }
    return coord;
}

/// Convert multi-dimensional coordinate to flat index.
template<std::size_t Dim>
[[nodiscard]] constexpr std::size_t
to_flat(std::array<std::size_t, Dim> const& coord,
        std::array<std::size_t, Dim> const& dims) noexcept {
    std::size_t flat = 0;
    for (std::size_t d = 0; d < Dim; ++d) {
        flat = flat * dims[d] + coord[d];
    }
    return flat;
}

/// Stencil generator: maps node_id to neighbor list via grid coordinates.
///
/// Template parameters:
/// - Dim: number of grid dimensions
/// - StencilSize: number of stencil offsets (max possible neighbors)
template<std::size_t Dim, std::size_t StencilSize>
struct stencil_generator {
    std::array<std::size_t, Dim> dims;
    std::array<std::array<int, Dim>, StencilSize> offsets;

    [[nodiscard]] constexpr ctdp::constexpr_vector<node_id, StencilSize>
    operator()(node_id u) const {
        ctdp::constexpr_vector<node_id, StencilSize> result{};
        auto const coord = to_coord<Dim>(
            static_cast<std::size_t>(u.value), dims);

        for (std::size_t s = 0; s < StencilSize; ++s) {
            bool valid = true;
            std::array<std::size_t, Dim> nbr_coord{};

            for (std::size_t d = 0; d < Dim; ++d) {
                auto const c = static_cast<int>(coord[d]);
                auto const offset = offsets[s][d];
                auto const nc = c + offset;

                // Bounds check: nc must be in [0, dims[d])
                if (nc < 0 || static_cast<std::size_t>(nc) >= dims[d]) {
                    valid = false;
                    break;
                }
                nbr_coord[d] = static_cast<std::size_t>(nc);
            }

            if (valid) {
                auto const flat = to_flat<Dim>(nbr_coord, dims);
                result.push_back(node_id{
                    static_cast<std::uint16_t>(flat)});
            }
        }

        return result;
    }
};

} // namespace detail

/// Create an implicit graph from a grid stencil pattern.
///
/// The resulting graph has one node per grid point. Each node's neighbors
/// are determined by applying the stencil offsets to its grid coordinate.
/// Boundary nodes have fewer neighbors (out-of-bounds offsets are excluded).
///
/// Template parameters (deduced):
/// - Dim: number of grid dimensions
/// - StencilSize: number of stencil offsets
///
/// Parameters:
/// - dimensions: grid size in each dimension (e.g., {100, 100} for 2D)
/// - offsets: stencil offsets (e.g., {{-1,0},{1,0},{0,-1},{0,1}} for 5-point)
///
/// Returns: implicit_graph whose generator returns
///          constexpr_vector<node_id, StencilSize>
///
/// Example:
/// ```cpp
/// // 2D 5-point stencil on 10×10 grid
/// constexpr auto g = from_stencil(
///     std::array<std::size_t, 2>{10, 10},
///     std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
/// );
/// static_assert(g.node_count() == 100);
/// ```
template<std::size_t Dim, std::size_t StencilSize>
[[nodiscard]] constexpr auto
from_stencil(
    std::array<std::size_t, Dim> dimensions,
    std::array<std::array<int, Dim>, StencilSize> offsets) {
    auto const total = detail::grid_size<Dim>(dimensions);
    return implicit_graph{
        total,
        detail::stencil_generator<Dim, StencilSize>{dimensions, offsets}
    };
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_FROM_STENCIL_H
