// examples/graph/example_rcm_spmv_benchmark.cpp
// Demonstrates RCM as a graph-derived ordering for an SpMV-like stencil.
//
// This version deliberately shows two RCM rebuild policies:
//   1. standard: rebuild CSR rows and sort entries by the new column index
//   2. careful:  rebuild CSR rows but preserve the original per-row term order
//
// Both use the same RCM permutation and both improve locality.  The standard
// rebuild is a normal sparse-matrix implementation choice, but sorting row
// entries changes the floating-point reduction expression.  The careful rebuild
// preserves the expression and therefore can be bitwise identical to the
// original computation after undoing the permutation.

#include <ctdp/graph/symmetric_graph.h>
#include <ctdp/graph/rcm.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

using namespace ctdp::graph;

namespace {

constexpr std::size_t N = 4096;       // <= 65535 because graph node ids are uint16_t.
constexpr unsigned Bits = 12;          // log2(N).
constexpr int Radius = 7;             // 24 neighbours plus the diagonal term.
constexpr std::size_t TermsPerInteriorRow = 1 + 2 * Radius;
constexpr std::size_t Payload = 128;     // floats per logical row; expands the locality window.
constexpr int Iterations = 16;
constexpr int Trials = 7;

static_assert((std::size_t{1} << Bits) == N);

constexpr std::uint16_t bit_reverse(std::uint16_t x, unsigned bits) {
    std::uint16_t r = 0;
    for (unsigned i = 0; i < bits; ++i) {
        r = static_cast<std::uint16_t>((r << 1u) | (x & 1u));
        x = static_cast<std::uint16_t>(x >> 1u);
    }
    return r;
}

using Graph = symmetric_graph<cap_from<N, N * Radius>>;

Graph make_badly_numbered_banded_graph() {
    symmetric_graph_builder<cap_from<N, N * Radius>> b;
    static_cast<void>(b.add_nodes(N));

    // Logical banded path: each row connects to the next Radius rows.
    // Physical label: bit_reverse(logical_position), which makes nearby
    // logical nodes far apart in the identity ordering.
    for (std::uint32_t logical = 0; logical < N; ++logical) {
        auto u = bit_reverse(static_cast<std::uint16_t>(logical), Bits);
        for (int d = 1; d <= Radius && logical + static_cast<std::uint32_t>(d) < N; ++d) {
            auto v = bit_reverse(static_cast<std::uint16_t>(logical + static_cast<std::uint32_t>(d)), Bits);
            b.add_edge(node_id{u}, node_id{v});
        }
    }
    return b.finalise();
}

struct Layout {
    std::vector<std::uint16_t> old_at_physical;   // physical row -> old node id
    std::vector<std::uint16_t> physical_of_old;   // old node id -> physical row
};

Layout make_identity_layout() {
    Layout l;
    l.old_at_physical.resize(N);
    l.physical_of_old.resize(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        l.old_at_physical[i] = static_cast<std::uint16_t>(i);
        l.physical_of_old[i] = static_cast<std::uint16_t>(i);
    }
    return l;
}

template<std::size_t MaxV>
Layout make_rcm_layout(rcm_result<MaxV> const& r) {
    Layout l;
    l.old_at_physical.resize(N);
    l.physical_of_old.resize(N);
    for (std::uint32_t new_pos = 0; new_pos < N; ++new_pos) {
        auto old = r.inverse[new_pos];
        l.old_at_physical[new_pos] = old;
    }
    for (std::uint32_t old = 0; old < N; ++old) {
        l.physical_of_old[old] = r.permutation[old];
    }
    return l;
}

std::vector<std::uint16_t> make_logical_position_of_old() {
    std::vector<std::uint16_t> logical_of_old(N);
    for (std::uint32_t logical = 0; logical < N; ++logical) {
        auto old = bit_reverse(static_cast<std::uint16_t>(logical), Bits);
        logical_of_old[old] = static_cast<std::uint16_t>(logical);
    }
    return logical_of_old;
}

struct Term {
    std::uint16_t old_col{};
    float value{};
};

std::vector<int> expression_offsets() {
    std::vector<int> offsets;
    offsets.reserve(TermsPerInteriorRow);

    // Deliberately non-column-sorted expression order.  The large/small/large
    // pattern makes reassociation visible in float arithmetic.
    offsets.push_back(+1);
    offsets.push_back(-1);
    offsets.push_back(+2);
    offsets.push_back(-2);
    for (int d = 3; d <= Radius; ++d) {
        offsets.push_back((d % 2) ? +d : -d);
        offsets.push_back((d % 2) ? -d : +d);
    }
    offsets.push_back(0); // diagonal term last, again not sorted by column.
    return offsets;
}

float coefficient_for_expression_index(std::size_t k) {
    // Groups of four are: +large, +small, -large, +small.
    // With float left-folding, (+large + small - large + small) differs from
    // (+large - large + small + small).  This makes a row-sort visibly change
    // the bit pattern while keeping the algebraic stencil equivalent.
    switch (k % 4) {
        case 0: return  100000000.0f;
        case 1: return  1.0f;
        case 2: return -100000000.0f;
        default: return 1.0f;
    }
}

std::vector<Term> original_row_terms(std::uint16_t old_row,
                                     std::vector<std::uint16_t> const& logical_of_old) {
    auto const logical = static_cast<int>(logical_of_old[old_row]);
    auto offsets = expression_offsets();

    std::vector<Term> terms;
    terms.reserve(TermsPerInteriorRow);
    for (std::size_t k = 0; k < offsets.size(); ++k) {
        int col_logical = logical + offsets[k];
        if (col_logical < 0 || col_logical >= static_cast<int>(N)) {
            continue;
        }
        auto old_col = bit_reverse(static_cast<std::uint16_t>(col_logical), Bits);
        terms.push_back(Term{old_col, coefficient_for_expression_index(k)});
    }
    return terms;
}

struct Csr {
    std::vector<std::uint32_t> row_ptr;
    std::vector<std::uint16_t> col;
    std::vector<float> val;
};

enum class RebuildPolicy {
    preserve_expression_order,
    sort_by_new_column
};

Csr build_csr(Layout const& layout,
              std::vector<std::uint16_t> const& logical_of_old,
              RebuildPolicy policy) {
    Csr csr;
    csr.row_ptr.resize(N + 1);
    csr.col.reserve(N * TermsPerInteriorRow);
    csr.val.reserve(N * TermsPerInteriorRow);

    for (std::uint32_t phys_row = 0; phys_row < N; ++phys_row) {
        csr.row_ptr[phys_row] = static_cast<std::uint32_t>(csr.col.size());

        auto old_row = layout.old_at_physical[phys_row];
        auto terms = original_row_terms(old_row, logical_of_old);

        std::vector<std::pair<std::uint16_t, float>> rebuilt;
        rebuilt.reserve(terms.size());
        for (auto const& t : terms) {
            rebuilt.emplace_back(layout.physical_of_old[t.old_col], t.value);
        }

        if (policy == RebuildPolicy::sort_by_new_column) {
            std::sort(rebuilt.begin(), rebuilt.end(), [](auto a, auto b) {
                return a.first < b.first;
            });
        }

        for (auto const& [c, v] : rebuilt) {
            csr.col.push_back(c);
            csr.val.push_back(v);
        }
    }
    csr.row_ptr[N] = static_cast<std::uint32_t>(csr.col.size());
    return csr;
}

std::vector<float> make_original_x() {
    std::vector<float> x(N * Payload);
    for (std::uint32_t old = 0; old < N; ++old) {
        for (std::uint32_t p = 0; p < Payload; ++p) {
            // Keep values close to one so the coefficient pattern drives the
            // non-associativity demonstration, but still force real memory loads.
            x[old * Payload + p] = 1.0f + static_cast<float>((old * 17u + p * 13u) & 15u) * 1.0e-6f;
        }
    }
    return x;
}

std::vector<float> permute_x_to_layout(std::vector<float> const& x_old, Layout const& l) {
    std::vector<float> x_new(N * Payload);
    for (std::uint32_t old = 0; old < N; ++old) {
        auto phys = l.physical_of_old[old];
        std::copy_n(x_old.data() + old * Payload, Payload,
                    x_new.data() + static_cast<std::size_t>(phys) * Payload);
    }
    return x_new;
}

void spmv_payload(Csr const& csr,
                  std::vector<float> const& x,
                  std::vector<float>& y) {
    for (std::uint32_t row = 0; row < N; ++row) {
        auto* yy = y.data() + static_cast<std::size_t>(row) * Payload;
        for (std::uint32_t p = 0; p < Payload; ++p) {
            yy[p] = 0.0f;
        }

        auto begin = csr.row_ptr[row];
        auto end = csr.row_ptr[row + 1];
        for (std::uint32_t e = begin; e < end; ++e) {
            auto const* xx = x.data() + static_cast<std::size_t>(csr.col[e]) * Payload;
            float const a = csr.val[e];
            for (std::uint32_t p = 0; p < Payload; ++p) {
                yy[p] += a * xx[p];
            }
        }
    }
}

volatile float sink = 0.0f;

float checksum(std::vector<float> const& y) {
    float s = 0.0f;
    for (float v : y) s += v;
    sink = s;
    return s;
}

double time_kernel_ms(Csr const& csr,
                      std::vector<float> const& x,
                      std::vector<float>& y) {
    for (int i = 0; i < 2; ++i) {
        spmv_payload(csr, x, y);
    }

    std::array<double, Trials> samples{};
    for (std::size_t t = 0; t < samples.size(); ++t) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < Iterations; ++i) {
            spmv_payload(csr, x, y);
        }
        auto t1 = std::chrono::steady_clock::now();
        checksum(y);
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        samples[t] = dt.count() / static_cast<double>(Iterations);
    }

    std::sort(samples.begin(), samples.end());
    return samples[Trials / 2];
}

std::uint32_t float_bits(float x) {
    std::uint32_t bits{};
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

struct CompareResult {
    std::size_t mismatches = 0;
    float max_abs_error = 0.0f;
    std::uint32_t first_original_bits = 0;
    std::uint32_t first_reordered_bits = 0;
    float first_original = 0.0f;
    float first_reordered = 0.0f;
};

CompareResult compare_under_permutation(std::vector<float> const& y_identity,
                                         std::vector<float> const& y_reordered,
                                         Layout const& reordered_layout) {
    CompareResult r{};
    bool recorded = false;
    for (std::uint32_t old = 0; old < N; ++old) {
        auto reordered_phys = reordered_layout.physical_of_old[old];
        for (std::uint32_t p = 0; p < Payload; ++p) {
            auto a = y_identity[old * Payload + p];
            auto b = y_reordered[static_cast<std::size_t>(reordered_phys) * Payload + p];
            auto err = std::fabs(a - b);
            if (err > r.max_abs_error) r.max_abs_error = err;
            if (float_bits(a) != float_bits(b)) {
                ++r.mismatches;
                if (!recorded) {
                    r.first_original = a;
                    r.first_reordered = b;
                    r.first_original_bits = float_bits(a);
                    r.first_reordered_bits = float_bits(b);
                    recorded = true;
                }
            }
        }
    }
    return r;
}

} // namespace

int main() {
    std::cout << "=== RCM SpMV expression-preservation benchmark ===\n";
    std::cout << "N=" << N << ", radius=" << Radius
              << ", payload=" << Payload << " floats/row"
              << ", iterations=" << Iterations
              << ", trials=" << Trials << "\n\n";

    auto graph = make_badly_numbered_banded_graph();
    auto r = rcm(graph);

    std::cout << "Graph bandwidth: " << r.bandwidth_before
              << " -> " << r.bandwidth_after << "\n";
    std::cout << "RCM verified: " << (r.verified ? "yes" : "no") << "\n\n";

    auto logical_of_old = make_logical_position_of_old();
    auto identity = make_identity_layout();
    auto rcm_layout = make_rcm_layout(r);

    std::cout << "CSR rebuild policies:\n";
    std::cout << "  original:      bad numbering, original row-term order\n";
    std::cout << "  RCM standard:  RCM numbering, row entries sorted by new column index\n";
    std::cout << "  RCM careful:   RCM numbering, original row-term order preserved\n\n";

    auto csr_identity = build_csr(identity, logical_of_old,
                                  RebuildPolicy::preserve_expression_order);
    auto csr_rcm_standard = build_csr(rcm_layout, logical_of_old,
                                      RebuildPolicy::sort_by_new_column);
    auto csr_rcm_careful = build_csr(rcm_layout, logical_of_old,
                                     RebuildPolicy::preserve_expression_order);

    auto x_old = make_original_x();
    auto x_identity = permute_x_to_layout(x_old, identity);
    auto x_rcm = permute_x_to_layout(x_old, rcm_layout);

    std::vector<float> y_identity(N * Payload);
    std::vector<float> y_standard(N * Payload);
    std::vector<float> y_careful(N * Payload);

    spmv_payload(csr_identity, x_identity, y_identity);
    spmv_payload(csr_rcm_standard, x_rcm, y_standard);
    spmv_payload(csr_rcm_careful, x_rcm, y_careful);

    auto standard_cmp = compare_under_permutation(y_identity, y_standard, rcm_layout);
    auto careful_cmp = compare_under_permutation(y_identity, y_careful, rcm_layout);

    auto t_identity = time_kernel_ms(csr_identity, x_identity, y_identity);
    auto t_standard = time_kernel_ms(csr_rcm_standard, x_rcm, y_standard);
    auto t_careful = time_kernel_ms(csr_rcm_careful, x_rcm, y_careful);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Correctness after undoing the permutation:\n";
    std::cout << "  RCM standard bitwise identical: "
              << (standard_cmp.mismatches == 0 ? "yes" : "no")
              << "  mismatches=" << standard_cmp.mismatches
              << "  max_abs_error=" << standard_cmp.max_abs_error << "\n";
    if (standard_cmp.mismatches != 0) {
        std::cout << std::hex << std::showbase;
        std::cout << "      first differing bits: original=" << standard_cmp.first_original_bits
                  << " standard=" << standard_cmp.first_reordered_bits << "\n";
        std::cout << std::dec << std::noshowbase << std::fixed << std::setprecision(3);
        std::cout << "      first differing values: original=" << standard_cmp.first_original
                  << " standard=" << standard_cmp.first_reordered << "\n";
    }
    std::cout << "  RCM careful  bitwise identical: "
              << (careful_cmp.mismatches == 0 ? "yes" : "no")
              << "  mismatches=" << careful_cmp.mismatches
              << "  max_abs_error=" << careful_cmp.max_abs_error << "\n\n";

    std::cout << "Median time per SpMV-like pass:\n";
    std::cout << "  original bad ordering: " << t_identity << " ms\n";
    std::cout << "  RCM standard:          " << t_standard
              << " ms  speedup=" << (t_identity / t_standard) << "x\n";
    std::cout << "  RCM careful:           " << t_careful
              << " ms  speedup=" << (t_identity / t_careful) << "x\n\n";

    std::cout << "Expression contract:\n";
    std::cout << "  standard RCM changes the row-local addition order: yes\n";
    std::cout << "  careful RCM preserves the row-local addition order: yes\n";
    std::cout << "  careful RCM keeps the same observable float expression: "
              << (careful_cmp.mismatches == 0 ? "yes" : "no") << "\n";

    return r.verified && careful_cmp.mismatches == 0 ? 0 : 1;
}
