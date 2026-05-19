#include "ctdp/space/descriptor.h"
#include "ctdp/space/permutation.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace ctdp::space;

namespace {

std::size_t kendall_tau_distance(const permutation_value<4>& a,
                                 const permutation_value<4>& b) {
    std::array<std::size_t, 4> pos_a{};
    std::array<std::size_t, 4> pos_b{};
    for (std::size_t i = 0; i < 4; ++i) {
        pos_a[a[i]] = i;
        pos_b[b[i]] = i;
    }

    std::size_t inversions = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            bool a_before = pos_a[i] < pos_a[j];
            bool b_before = pos_b[i] < pos_b[j];
            if (a_before != b_before) ++inversions;
        }
    }
    return inversions;
}

void print_perm(const permutation_value<4>& p) {
    std::cerr << "[";
    for (std::size_t i = 0; i < 4; ++i) {
        if (i) std::cerr << ",";
        std::cerr << p[i];
    }
    std::cerr << "]";
}

} // namespace

int main() {
    auto space = descriptor_space("order_search",
        make_permutation<4>("loop_order"),
        bool_flag("vectorize"));
    auto bridge = default_bridge(space);

    constexpr permutation_value<4> target(std::array<std::size_t, 4>{2, 0, 3, 1});

    auto result = exhaustive_search_with_cost(space, [&](auto const& pt) {
        const auto& order = std::get<0>(pt);
        bool vectorize = std::get<1>(pt);
        double cost = static_cast<double>(kendall_tau_distance(order, target));
        if (vectorize) cost -= 0.25;
        return cost;
    });

    std::cerr << "Permutation descriptor example\n";
    std::cerr << "  space cardinality: " << space.cardinality() << "\n";
    std::cerr << "  feature width:     " << bridge.num_features() << "\n";
    std::cerr << "  best order:        ";
    print_perm(std::get<0>(result.best));
    std::cerr << "\n";
    std::cerr << "  vectorize:         " << std::boolalpha << std::get<1>(result.best) << "\n";
    std::cerr << "  best cost:         " << std::fixed << std::setprecision(2)
              << result.best_cost << "\n";
    std::cerr << "  candidates tested: " << result.evaluated << "\n";

    std::vector<double> features(bridge.num_features(), 0.0);
    bridge.write_features(result.best, features);
    std::cerr << "  encoded features:  ";
    for (std::size_t i = 0; i < features.size(); ++i) {
        if (i) std::cerr << " ";
        std::cerr << static_cast<int>(features[i]);
    }
    std::cerr << "\n";

    return 0;
}

