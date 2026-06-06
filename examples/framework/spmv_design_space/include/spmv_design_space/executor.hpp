#pragma once

#include "spmv_design_space/plan.hpp"
#include "spmv_design_space/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ctdp::spmv_dsl {

struct execution_result {
    std::vector<double> residual;
    std::vector<double> z;
    std::vector<double> q;
    std::vector<double> x_next;
    std::vector<std::size_t> visited_rows;
    std::string execution_path;
    std::size_t worker_count = 1;
    std::size_t simd_lanes = 1;
    double rho = 0.0;
    double sigma = 0.0;
    double alpha = 0.0;
};

struct observation_fingerprint {
    std::uint64_t rho_bits = 0;
    std::uint64_t sigma_bits = 0;
    std::uint64_t alpha_bits = 0;
    std::uint64_t residual_hash = 0;
    std::uint64_t z_hash = 0;
    std::uint64_t q_hash = 0;
    std::uint64_t x_next_hash = 0;
    std::uint64_t observation_hash = 0;
};

inline std::uint64_t double_bits(double x) noexcept {
    std::uint64_t out = 0;
    static_assert(sizeof(out) == sizeof(x));
    std::memcpy(&out, &x, sizeof(out));
    return out;
}

inline std::uint64_t fnv1a_mix(std::uint64_t h, std::uint64_t x) noexcept {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (int i = 0; i < 8; ++i) {
        const auto byte = static_cast<unsigned char>((x >> (8 * i)) & 0xffu);
        h ^= byte;
        h *= prime;
    }
    return h;
}

inline std::uint64_t hash_double_vector(const std::vector<double>& xs) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    h = fnv1a_mix(h, static_cast<std::uint64_t>(xs.size()));
    for (const auto x : xs) {
        h = fnv1a_mix(h, double_bits(x));
    }
    return h;
}

inline observation_fingerprint fingerprint_observation(const execution_result& r) noexcept {
    observation_fingerprint f;
    f.rho_bits = double_bits(r.rho);
    f.sigma_bits = double_bits(r.sigma);
    f.alpha_bits = double_bits(r.alpha);
    f.residual_hash = hash_double_vector(r.residual);
    f.z_hash = hash_double_vector(r.z);
    f.q_hash = hash_double_vector(r.q);
    f.x_next_hash = hash_double_vector(r.x_next);

    std::uint64_t h = 1469598103934665603ull;
    h = fnv1a_mix(h, f.rho_bits);
    h = fnv1a_mix(h, f.sigma_bits);
    h = fnv1a_mix(h, f.alpha_bits);
    h = fnv1a_mix(h, f.residual_hash);
    h = fnv1a_mix(h, f.z_hash);
    h = fnv1a_mix(h, f.q_hash);
    h = fnv1a_mix(h, f.x_next_hash);
    f.observation_hash = h;
    return f;
}

inline std::string hex64(std::uint64_t x) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setfill('0') << std::setw(16) << x;
    return os.str();
}

inline std::size_t lanes_for(simd_kind simd) noexcept {
    switch (simd) {
    case simd_kind::scalar: return 1;
    case simd_kind::lanes4: return 4;
    case simd_kind::lanes8: return 8;
    }
    return 1;
}

inline bool uses_parallel_threading(threading_kind threading) noexcept {
    return threading == threading_kind::static_blocks ||
           threading == threading_kind::recursive_tasks ||
           threading == threading_kind::colour_phases;
}

inline void append_natural_rows(std::vector<std::size_t>& rows, std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) {
        rows.push_back(i);
    }
}

inline void append_recursive_rows(std::vector<std::size_t>& rows, std::size_t lo, std::size_t hi, std::size_t leaf) {
    const auto n = hi - lo;
    if (n <= leaf) {
        append_natural_rows(rows, lo, hi);
        return;
    }
    const auto mid = lo + n / 2;
    append_recursive_rows(rows, lo, mid, leaf);
    append_recursive_rows(rows, mid, hi, leaf);
}

inline std::vector<std::size_t> row_visit_order(const stencil_problem& problem, const plan_descriptor& plan) {
    std::vector<std::size_t> rows;
    rows.reserve(problem.size());

    if (plan.colouring == colouring_kind::red_black_stencil) {
        const auto colours = red_black_colours(problem);
        for (std::size_t colour = 0; colour < 2; ++colour) {
            for (std::size_t i = 0; i < problem.size(); ++i) {
                if (colours[i] == colour) {
                    rows.push_back(i);
                }
            }
        }
        return rows;
    }

    switch (plan.decomposition) {
    case decomposition_kind::flat_rows:
        append_natural_rows(rows, 0, problem.size());
        break;
    case decomposition_kind::blocked_rows: {
        constexpr std::size_t block = 64;
        for (std::size_t lo = 0; lo < problem.size(); lo += block) {
            append_natural_rows(rows, lo, std::min(problem.size(), lo + block));
        }
        break;
    }
    case decomposition_kind::recursive_grid_bisection:
        append_recursive_rows(rows, 0, problem.size(), 64);
        break;
    }
    return rows;
}

inline std::string execution_path_for(const plan_descriptor& plan,
                                      std::size_t workers,
                                      std::size_t lanes) {
    std::string out;
    out += std::string(to_string(plan.executor));
    out += "/";
    out += std::string(to_string(plan.storage));
    out += "/";
    out += std::string(to_string(plan.decomposition));
    if (plan.colouring != colouring_kind::none) {
        out += "/";
        out += std::string(to_string(plan.colouring));
    }
    out += "/";
    out += std::string(to_string(plan.preconditioner));
    out += "/";
    out += std::string(to_string(plan.threading));
    out += "(" + std::to_string(workers) + "T)";
    out += "/";
    out += std::string(to_string(plan.simd));
    out += "(" + std::to_string(lanes) + " lanes)";
    out += "/";
    out += std::string(to_string(plan.fusion));
    out += "/";
    out += std::string(to_string(plan.reduction));
    return out;
}

inline double canonical_pairwise_sum(const std::vector<double>& xs, std::size_t lo, std::size_t hi) {
    const auto n = hi - lo;
    if (n == 0) {
        return 0.0;
    }
    if (n == 1) {
        return xs[lo];
    }
    const auto mid = lo + n / 2;
    return canonical_pairwise_sum(xs, lo, mid) + canonical_pairwise_sum(xs, mid, hi);
}

inline double thread_local_unordered_sum_witness(const std::vector<double>& terms) {
    constexpr std::size_t chunks = 4;
    std::vector<double> partials(chunks, 0.0);
    for (std::size_t i = 0; i < terms.size(); ++i) {
        partials[i % chunks] += terms[i];
    }
    double total = 0.0;
    for (std::size_t i = chunks; i > 0; --i) {
        total += partials[i - 1];
    }
    return total;
}

inline double reduce_terms_for_plan(const plan_descriptor& plan,
                                    const std::vector<double>& terms) {
    switch (plan.reduction) {
    case reduction_kind::canonical_pairwise:
        return canonical_pairwise_sum(terms, 0, terms.size());
    case reduction_kind::thread_local_unordered_witness:
        return thread_local_unordered_sum_witness(terms);
    }
    throw std::invalid_argument("unknown reduction");
}

template<std::size_t W>
inline void fill_products_blocked(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::vector<double>& products,
    std::size_t first,
    std::size_t last) {
    std::size_t i = first;
    for (; i + W <= last; i += W) {
        fixed_block<double, W> aa{};
        fixed_block<double, W> bb{};
        for (std::size_t lane = 0; lane < W; ++lane) {
            aa[lane] = a[i + lane];
            bb[lane] = b[i + lane];
        }
        const auto pp = fixed_binary(aa, bb, [](double x, double y) noexcept { return x * y; });
        for (std::size_t lane = 0; lane < W; ++lane) {
            products[i + lane] = pp[lane];
        }
    }
    for (; i < last; ++i) {
        products[i] = a[i] * b[i];
    }
}

inline double canonical_pairwise_dot(const std::vector<double>& a,
                                     const std::vector<double>& b,
                                     std::size_t lanes) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("canonical_pairwise_dot size mismatch");
    }
    std::vector<double> products(a.size());
    if (lanes >= 8) {
        fill_products_blocked<8>(a, b, products, 0, products.size());
    } else if (lanes >= 4) {
        fill_products_blocked<4>(a, b, products, 0, products.size());
    } else {
        fill_products_blocked<1>(a, b, products, 0, products.size());
    }
    return canonical_pairwise_sum(products, 0, products.size());
}

inline double thread_local_unordered_dot_witness(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("thread_local_unordered_dot_witness size mismatch");
    }
    constexpr std::size_t chunks = 4;
    std::vector<double> partials(chunks, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        partials[i % chunks] += a[i] * b[i];
    }
    double total = 0.0;
    for (std::size_t i = chunks; i > 0; --i) {
        total += partials[i - 1];
    }
    return total;
}

inline double dot_for_plan(const plan_descriptor& plan,
                           const std::vector<double>& a,
                           const std::vector<double>& b,
                           std::size_t lanes) {
    switch (plan.reduction) {
    case reduction_kind::canonical_pairwise:
        return canonical_pairwise_dot(a, b, lanes);
    case reduction_kind::thread_local_unordered_witness:
        return thread_local_unordered_dot_witness(a, b);
    }
    throw std::invalid_argument("unknown reduction");
}

inline double preconditioner_value(const plan_descriptor& plan, double residual) {
    switch (plan.preconditioner) {
    case preconditioner_kind::fixed_diagonal_jacobi:
        return residual / 4.0;
    case preconditioner_kind::none_solver_family:
        return residual;
    case preconditioner_kind::coloured_smoother_solver_family:
        return 0.75 * residual / 4.0;
    }
    return residual;
}

template<std::size_t W>
inline void apply_preconditioner_range(
    const plan_descriptor& plan,
    const std::vector<double>& residual,
    std::vector<double>& z,
    std::size_t first,
    std::size_t last) {
    std::size_t i = first;
    for (; i + W <= last; i += W) {
        fixed_block<double, W> rr{};
        for (std::size_t lane = 0; lane < W; ++lane) {
            rr[lane] = residual[i + lane];
        }

        fixed_block<double, W> zz{};
        switch (plan.preconditioner) {
        case preconditioner_kind::fixed_diagonal_jacobi:
            for (std::size_t lane = 0; lane < W; ++lane) {
                zz[lane] = rr[lane] / 4.0;
            }
            break;
        case preconditioner_kind::none_solver_family:
            zz = rr;
            break;
        case preconditioner_kind::coloured_smoother_solver_family:
            for (std::size_t lane = 0; lane < W; ++lane) {
                zz[lane] = 0.75 * rr[lane] / 4.0;
            }
            break;
        }

        for (std::size_t lane = 0; lane < W; ++lane) {
            z[i + lane] = zz[lane];
        }
    }

    for (; i < last; ++i) {
        switch (plan.preconditioner) {
        case preconditioner_kind::fixed_diagonal_jacobi:
            z[i] = residual[i] / 4.0;
            break;
        case preconditioner_kind::none_solver_family:
            z[i] = residual[i];
            break;
        case preconditioner_kind::coloured_smoother_solver_family:
            z[i] = 0.75 * residual[i] / 4.0;
            break;
        }
    }
}

inline void apply_preconditioner(
    const plan_descriptor& plan,
    const std::vector<double>& residual,
    std::vector<double>& z,
    std::size_t lanes,
    execution_context* ctx) {
    z.resize(residual.size());

    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            apply_preconditioner_range<8>(plan, residual, z, first, last);
        } else if (lanes >= 4) {
            apply_preconditioner_range<4>(plan, residual, z, first, last);
        } else {
            apply_preconditioner_range<1>(plan, residual, z, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        const auto tasks = make_row_tasks(residual.size(), ctx->task_grain);
        ctx->pool->run(tasks, body);
    } else {
        body(0, residual.size());
    }
}

inline double apply_operator_at(const stencil_problem& problem,
                                const plan_descriptor& plan,
                                const std::vector<double>& x,
                                std::size_t row) {
    if (plan.executor == executor_kind::reference ||
        plan.executor == executor_kind::csr_executor) {
        return apply_csr_at(problem.csr, x, row);
    }
    if (plan.executor == executor_kind::dia_executor) {
        return apply_dia_at(problem, x, row);
    }
    if (plan.executor == executor_kind::hybrid_dia_csr_executor) {
        return apply_hybrid_dia_csr_at(problem, x, row);
    }
    if (plan.executor == executor_kind::matrix_free_executor) {
        return apply_five_point_at(problem, x, row);
    }
    throw std::invalid_argument("unknown executor");
}

inline double apply_operator_at(const stencil_problem& problem, const plan_descriptor& plan, std::size_t row) {
    return apply_operator_at(problem, plan, problem.x, row);
}

template<std::size_t W>
inline void compute_residual_range(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::size_t first,
    std::size_t last) {
    std::size_t k = first;
    for (; k + W <= last; k += W) {
        fixed_block<double, W> bb{};
        fixed_block<double, W> ax{};
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            bb[lane] = problem.b[row];
            ax[lane] = apply_operator_at(problem, plan, row);
        }
        const auto rr = fixed_binary(bb, ax, [](double x, double y) noexcept { return x - y; });
        for (std::size_t lane = 0; lane < W; ++lane) {
            residual[rows[k + lane]] = rr[lane];
        }
    }
    for (; k < last; ++k) {
        const auto row = rows[k];
        residual[row] = problem.b[row] - apply_operator_at(problem, plan, row);
    }
}

inline void compute_residual(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::size_t lanes,
    execution_context* ctx) {
    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            compute_residual_range<8>(problem, plan, rows, residual, first, last);
        } else if (lanes >= 4) {
            compute_residual_range<4>(problem, plan, rows, residual, first, last);
        } else {
            compute_residual_range<1>(problem, plan, rows, residual, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        if (plan.threading == threading_kind::colour_phases &&
            plan.colouring == colouring_kind::red_black_stencil) {
            const auto colours = red_black_colours(problem);
            auto first_black = rows.size();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (colours[rows[i]] == 1) {
                    first_black = i;
                    break;
                }
            }
            const auto red_tasks = make_row_tasks(first_black, ctx->task_grain);
            ctx->pool->run(red_tasks, body);

            auto black_tasks = make_row_tasks(rows.size() - first_black, ctx->task_grain);
            for (auto& task : black_tasks) {
                task.first += first_black;
                task.last += first_black;
            }
            ctx->pool->run(black_tasks, body);
        } else {
            const auto tasks = make_row_tasks(rows.size(), ctx->task_grain);
            ctx->pool->run(tasks, body);
        }
    } else {
        body(0, rows.size());
    }
}

template<std::size_t W>
inline void compute_operator_image_range(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<double>& x,
    std::vector<double>& y,
    std::size_t first,
    std::size_t last) {
    std::size_t i = first;
    for (; i + W <= last; i += W) {
        for (std::size_t lane = 0; lane < W; ++lane) {
            y[i + lane] = apply_operator_at(problem, plan, x, i + lane);
        }
    }
    for (; i < last; ++i) {
        y[i] = apply_operator_at(problem, plan, x, i);
    }
}

inline void compute_operator_image(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<double>& x,
    std::vector<double>& y,
    std::size_t lanes,
    execution_context* ctx) {
    y.resize(problem.size());
    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            compute_operator_image_range<8>(problem, plan, x, y, first, last);
        } else if (lanes >= 4) {
            compute_operator_image_range<4>(problem, plan, x, y, first, last);
        } else {
            compute_operator_image_range<1>(problem, plan, x, y, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        const auto tasks = make_row_tasks(problem.size(), ctx->task_grain);
        ctx->pool->run(tasks, body);
    } else {
        body(0, problem.size());
    }
}

template<std::size_t W>
inline void update_x_range(
    const stencil_problem& problem,
    const std::vector<double>& z,
    std::vector<double>& x_next,
    double alpha,
    std::size_t first,
    std::size_t last) {
    std::size_t i = first;
    for (; i + W <= last; i += W) {
        fixed_block<double, W> xx{};
        fixed_block<double, W> zz{};
        for (std::size_t lane = 0; lane < W; ++lane) {
            xx[lane] = problem.x[i + lane];
            zz[lane] = z[i + lane];
        }
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto scaled_step = alpha * zz[lane];
            x_next[i + lane] = xx[lane] + scaled_step;
        }
    }
    for (; i < last; ++i) {
        const auto scaled_step = alpha * z[i];
        x_next[i] = problem.x[i] + scaled_step;
    }
}

inline void update_x(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<double>& z,
    std::vector<double>& x_next,
    double alpha,
    std::size_t lanes,
    execution_context* ctx) {
    x_next.resize(problem.size());
    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            update_x_range<8>(problem, z, x_next, alpha, first, last);
        } else if (lanes >= 4) {
            update_x_range<4>(problem, z, x_next, alpha, first, last);
        } else {
            update_x_range<1>(problem, z, x_next, alpha, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        const auto tasks = make_row_tasks(problem.size(), ctx->task_grain);
        ctx->pool->run(tasks, body);
    } else {
        body(0, problem.size());
    }
}

template<std::size_t W>
inline void compute_fused_range(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::vector<double>& z,
    std::vector<double>& x_next,
    std::vector<double>& products,
    double alpha,
    std::size_t first,
    std::size_t last) {
    std::size_t k = first;
    for (; k + W <= last; k += W) {
        fixed_block<double, W> xx{};
        fixed_block<double, W> bb{};
        fixed_block<double, W> ax{};
        fixed_block<double, W> rr{};
        fixed_block<double, W> zz{};
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            xx[lane] = problem.x[row];
            bb[lane] = problem.b[row];
            ax[lane] = apply_operator_at(problem, plan, row);
        }
        rr = fixed_binary(bb, ax, [](double x, double y) noexcept { return x - y; });
        for (std::size_t lane = 0; lane < W; ++lane) {
            zz[lane] = preconditioner_value(plan, rr[lane]);
        }
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            residual[row] = rr[lane];
            z[row] = zz[lane];
            products[row] = rr[lane] * zz[lane];
            const auto scaled_step = alpha * zz[lane];
            x_next[row] = xx[lane] + scaled_step;
        }
    }
    for (; k < last; ++k) {
        const auto row = rows[k];
        const auto ri = problem.b[row] - apply_operator_at(problem, plan, row);
        const auto zi = preconditioner_value(plan, ri);
        residual[row] = ri;
        z[row] = zi;
        products[row] = ri * zi;
        const auto scaled_step = alpha * zi;
        x_next[row] = problem.x[row] + scaled_step;
    }
}

inline void compute_fused(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::vector<double>& z,
    std::vector<double>& x_next,
    std::vector<double>& products,
    double alpha,
    std::size_t lanes,
    execution_context* ctx) {
    residual.resize(problem.size());
    z.resize(problem.size());
    x_next.resize(problem.size());
    products.assign(problem.size(), 0.0);

    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            compute_fused_range<8>(problem, plan, rows, residual, z, x_next, products, alpha, first, last);
        } else if (lanes >= 4) {
            compute_fused_range<4>(problem, plan, rows, residual, z, x_next, products, alpha, first, last);
        } else {
            compute_fused_range<1>(problem, plan, rows, residual, z, x_next, products, alpha, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        if (plan.threading == threading_kind::colour_phases &&
            plan.colouring == colouring_kind::red_black_stencil) {
            const auto colours = red_black_colours(problem);
            auto first_black = rows.size();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (colours[rows[i]] == 1) {
                    first_black = i;
                    break;
                }
            }
            const auto red_tasks = make_row_tasks(first_black, ctx->task_grain);
            ctx->pool->run(red_tasks, body);

            auto black_tasks = make_row_tasks(rows.size() - first_black, ctx->task_grain);
            for (auto& task : black_tasks) {
                task.first += first_black;
                task.last += first_black;
            }
            ctx->pool->run(black_tasks, body);
        } else {
            const auto tasks = make_row_tasks(rows.size(), ctx->task_grain);
            ctx->pool->run(tasks, body);
        }
    } else {
        body(0, rows.size());
    }
}


struct fusion_group_descriptor {
    int first_stage = 0;
    int last_stage = 0;
};

inline bool fusion_cut_after(fusion_kind fusion, int stage) noexcept {
    switch (fusion) {
    case fusion_kind::r_z_p_u:
        return stage == 0 || stage == 1 || stage == 2;
    case fusion_kind::rz_p_u:
        return stage == 1 || stage == 2;
    case fusion_kind::r_zp_u:
        return stage == 0 || stage == 2;
    case fusion_kind::r_z_pu:
        return stage == 0 || stage == 1;
    case fusion_kind::rzp_u:
        return stage == 2;
    case fusion_kind::rz_pu:
        return stage == 1;
    case fusion_kind::r_zpu:
        return stage == 0;
    case fusion_kind::rzpu:
        return false;
    }
    return true;
}

inline std::vector<fusion_group_descriptor> fusion_groups(fusion_kind fusion) {
    std::vector<fusion_group_descriptor> groups;
    int first = 0;
    // The solver update uses alpha = step_scale * (rho / sigma).  That scalar
    // is not available until after the observed dot products have been reduced,
    // so U is always separated by a hard alpha barrier.  The current legacy
    // fusion enum therefore only controls materialisation barriers around R,
    // Z, and rho; every effective plan then performs [(Q,sigma)][alpha][U].
    for (int stage = 0; stage < 2; ++stage) {
        if (fusion_cut_after(fusion, stage)) {
            groups.push_back({first, stage});
            first = stage + 1;
        }
    }
    groups.push_back({first, 2});
    return groups;
}

inline bool fusion_group_contains(const fusion_group_descriptor& group, int stage) noexcept {
    return group.first_stage <= stage && stage <= group.last_stage;
}

template<std::size_t W>
inline void compute_fusion_group_range(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const fusion_group_descriptor& group,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::vector<double>& z,
    std::vector<double>& x_next,
    std::vector<double>& products,
    double alpha,
    std::size_t first,
    std::size_t last) {

    const bool do_residual = fusion_group_contains(group, 0);
    const bool do_preconditioner = fusion_group_contains(group, 1);
    const bool do_product = fusion_group_contains(group, 2);
    const bool do_update = fusion_group_contains(group, 3);

    std::size_t k = first;
    for (; k + W <= last; k += W) {
        fixed_block<double, W> xx{};
        fixed_block<double, W> bb{};
        fixed_block<double, W> ax{};
        fixed_block<double, W> rr{};
        fixed_block<double, W> zz{};

        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            xx[lane] = problem.x[row];
            rr[lane] = residual[row];
            zz[lane] = z[row];
            if (do_residual) {
                bb[lane] = problem.b[row];
                ax[lane] = apply_operator_at(problem, plan, row);
            }
        }

        if (do_residual) {
            rr = fixed_binary(bb, ax, [](double x, double y) noexcept { return x - y; });
        }
        if (do_preconditioner) {
            for (std::size_t lane = 0; lane < W; ++lane) {
                zz[lane] = preconditioner_value(plan, rr[lane]);
            }
        }

        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            if (do_residual) {
                residual[row] = rr[lane];
            }
            if (do_preconditioner) {
                z[row] = zz[lane];
            }
            if (do_product) {
                products[row] = rr[lane] * zz[lane];
            }
            if (do_update) {
                const auto scaled_step = alpha * zz[lane];
                x_next[row] = xx[lane] + scaled_step;
            }
        }
    }

    for (; k < last; ++k) {
        const auto row = rows[k];
        auto ri = residual[row];
        auto zi = z[row];
        if (do_residual) {
            ri = problem.b[row] - apply_operator_at(problem, plan, row);
            residual[row] = ri;
        }
        if (do_preconditioner) {
            zi = preconditioner_value(plan, ri);
            z[row] = zi;
        }
        if (do_product) {
            products[row] = ri * zi;
        }
        if (do_update) {
            const auto scaled_step = alpha * zi;
            x_next[row] = problem.x[row] + scaled_step;
        }
    }
}

inline void compute_fusion_group(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const fusion_group_descriptor& group,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::vector<double>& z,
    std::vector<double>& x_next,
    std::vector<double>& products,
    double alpha,
    std::size_t lanes,
    execution_context* ctx) {

    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            compute_fusion_group_range<8>(problem, plan, group, rows, residual, z, x_next, products, alpha, first, last);
        } else if (lanes >= 4) {
            compute_fusion_group_range<4>(problem, plan, group, rows, residual, z, x_next, products, alpha, first, last);
        } else {
            compute_fusion_group_range<1>(problem, plan, group, rows, residual, z, x_next, products, alpha, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        if (plan.threading == threading_kind::colour_phases &&
            plan.colouring == colouring_kind::red_black_stencil) {
            const auto colours = red_black_colours(problem);
            auto first_black = rows.size();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (colours[rows[i]] == 1) {
                    first_black = i;
                    break;
                }
            }
            const auto red_tasks = make_row_tasks(first_black, ctx->task_grain);
            ctx->pool->run(red_tasks, body);

            auto black_tasks = make_row_tasks(rows.size() - first_black, ctx->task_grain);
            for (auto& task : black_tasks) {
                task.first += first_black;
                task.last += first_black;
            }
            ctx->pool->run(black_tasks, body);
        } else {
            const auto tasks = make_row_tasks(rows.size(), ctx->task_grain);
            ctx->pool->run(tasks, body);
        }
    } else {
        body(0, rows.size());
    }
}

inline void compute_fusion_partition(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& residual,
    std::vector<double>& z,
    std::vector<double>& x_next,
    std::vector<double>& products,
    double alpha,
    std::size_t lanes,
    execution_context* ctx) {

    residual.assign(problem.size(), 0.0);
    z.assign(problem.size(), 0.0);
    x_next.assign(problem.size(), 0.0);
    products.assign(problem.size(), 0.0);

    for (const auto& group : fusion_groups(plan.fusion)) {
        compute_fusion_group(problem, plan, group, rows, residual, z, x_next, products, alpha, lanes, ctx);
    }
}

inline double rho_from_products_or_witness(const plan_descriptor& plan,
                                           const std::vector<double>& residual,
                                           const std::vector<double>& z,
                                           const std::vector<double>& products) {
    switch (plan.reduction) {
    case reduction_kind::canonical_pairwise:
        return canonical_pairwise_sum(products, 0, products.size());
    case reduction_kind::thread_local_unordered_witness:
        return thread_local_unordered_dot_witness(residual, z);
    }
    throw std::invalid_argument("unknown reduction");
}


template<std::size_t W>
inline void compute_z_and_rho_terms_range(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& z,
    std::vector<double>& rho_terms,
    std::size_t first,
    std::size_t last) {

    std::size_t k = first;
    for (; k + W <= last; k += W) {
        fixed_block<double, W> bb{};
        fixed_block<double, W> ax{};
        fixed_block<double, W> rr{};
        fixed_block<double, W> zz{};
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            bb[lane] = problem.b[row];
            ax[lane] = apply_operator_at(problem, plan, row);
        }
        rr = fixed_binary(bb, ax, [](double x, double y) noexcept { return x - y; });
        for (std::size_t lane = 0; lane < W; ++lane) {
            zz[lane] = preconditioner_value(plan, rr[lane]);
        }
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = rows[k + lane];
            z[row] = zz[lane];
            rho_terms[row] = rr[lane] * zz[lane];
        }
    }

    for (; k < last; ++k) {
        const auto row = rows[k];
        const auto ri = problem.b[row] - apply_operator_at(problem, plan, row);
        const auto zi = preconditioner_value(plan, ri);
        z[row] = zi;
        rho_terms[row] = ri * zi;
    }
}

inline void compute_z_and_rho_terms(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<std::size_t>& rows,
    std::vector<double>& z,
    std::vector<double>& rho_terms,
    std::size_t lanes,
    execution_context* ctx) {

    z.assign(problem.size(), 0.0);
    rho_terms.assign(problem.size(), 0.0);

    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            compute_z_and_rho_terms_range<8>(problem, plan, rows, z, rho_terms, first, last);
        } else if (lanes >= 4) {
            compute_z_and_rho_terms_range<4>(problem, plan, rows, z, rho_terms, first, last);
        } else {
            compute_z_and_rho_terms_range<1>(problem, plan, rows, z, rho_terms, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        if (plan.threading == threading_kind::colour_phases &&
            plan.colouring == colouring_kind::red_black_stencil) {
            const auto colours = red_black_colours(problem);
            auto first_black = rows.size();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (colours[rows[i]] == 1) {
                    first_black = i;
                    break;
                }
            }
            const auto red_tasks = make_row_tasks(first_black, ctx->task_grain);
            ctx->pool->run(red_tasks, body);

            auto black_tasks = make_row_tasks(rows.size() - first_black, ctx->task_grain);
            for (auto& task : black_tasks) {
                task.first += first_black;
                task.last += first_black;
            }
            ctx->pool->run(black_tasks, body);
        } else {
            const auto tasks = make_row_tasks(rows.size(), ctx->task_grain);
            ctx->pool->run(tasks, body);
        }
    } else {
        body(0, rows.size());
    }
}

template<std::size_t W>
inline void compute_sigma_terms_range(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<double>& z,
    std::vector<double>& sigma_terms,
    std::size_t first,
    std::size_t last) {

    std::size_t i = first;
    for (; i + W <= last; i += W) {
        for (std::size_t lane = 0; lane < W; ++lane) {
            const auto row = i + lane;
            const auto qi = apply_operator_at(problem, plan, z, row);
            sigma_terms[row] = z[row] * qi;
        }
    }
    for (; i < last; ++i) {
        const auto qi = apply_operator_at(problem, plan, z, i);
        sigma_terms[i] = z[i] * qi;
    }
}

inline void compute_sigma_terms(
    const stencil_problem& problem,
    const plan_descriptor& plan,
    const std::vector<double>& z,
    std::vector<double>& sigma_terms,
    std::size_t lanes,
    execution_context* ctx) {

    sigma_terms.assign(problem.size(), 0.0);
    const auto body = [&](std::size_t first, std::size_t last) {
        if (lanes >= 8) {
            compute_sigma_terms_range<8>(problem, plan, z, sigma_terms, first, last);
        } else if (lanes >= 4) {
            compute_sigma_terms_range<4>(problem, plan, z, sigma_terms, first, last);
        } else {
            compute_sigma_terms_range<1>(problem, plan, z, sigma_terms, first, last);
        }
    };

    if (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading)) {
        const auto tasks = make_row_tasks(problem.size(), ctx->task_grain);
        ctx->pool->run(tasks, body);
    } else {
        body(0, problem.size());
    }
}

inline execution_result execute_plan_solver_state_only(const stencil_problem& problem,
                                                       const plan_descriptor& plan,
                                                       double step_scale,
                                                       execution_context* ctx = nullptr) {
    execution_result r;
    r.visited_rows = row_visit_order(problem, plan);
    r.simd_lanes = lanes_for(plan.simd);
    r.worker_count = (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading))
        ? ctx->pool->size()
        : 1;
    r.execution_path = execution_path_for(plan, r.worker_count, r.simd_lanes);

    std::vector<double> rho_terms;
    std::vector<double> sigma_terms;
    compute_z_and_rho_terms(problem, plan, r.visited_rows, r.z, rho_terms, r.simd_lanes, ctx);
    r.rho = reduce_terms_for_plan(plan, rho_terms);
    compute_sigma_terms(problem, plan, r.z, sigma_terms, r.simd_lanes, ctx);
    r.sigma = reduce_terms_for_plan(plan, sigma_terms);
    r.alpha = (r.sigma == 0.0) ? 0.0 : step_scale * (r.rho / r.sigma);
    update_x(problem, plan, r.z, r.x_next, r.alpha, r.simd_lanes, ctx);

    return r;
}

inline execution_result execute_plan(const stencil_problem& problem,
                                     const plan_descriptor& plan,
                                     double step_scale,
                                     execution_context* ctx = nullptr) {
    execution_result r;
    r.residual.resize(problem.size());
    r.visited_rows = row_visit_order(problem, plan);
    r.simd_lanes = lanes_for(plan.simd);
    r.worker_count = (ctx != nullptr && ctx->pool != nullptr && uses_parallel_threading(plan.threading))
        ? ctx->pool->size()
        : 1;
    r.execution_path = execution_path_for(plan, r.worker_count, r.simd_lanes);

    std::vector<double> products;
    compute_fusion_partition(problem, plan, r.visited_rows, r.residual, r.z, r.x_next, products, step_scale, r.simd_lanes, ctx);
    r.rho = rho_from_products_or_witness(plan, r.residual, r.z, products);
    compute_operator_image(problem, plan, r.z, r.q, r.simd_lanes, ctx);
    r.sigma = dot_for_plan(plan, r.z, r.q, r.simd_lanes);
    r.alpha = (r.sigma == 0.0) ? 0.0 : step_scale * (r.rho / r.sigma);
    update_x(problem, plan, r.z, r.x_next, r.alpha, r.simd_lanes, ctx);

    return r;
}

inline execution_result execute_strict_reference(const stencil_problem& problem, double alpha) {
    plan_descriptor reference;
    reference.name = "reference/matrix-free/strict";
    reference.contract = contract_level::strict_expression;
    reference.storage = storage_kind::matrix_free_stencil;
    reference.preconditioner = preconditioner_kind::fixed_diagonal_jacobi;
    reference.reduction = reduction_kind::canonical_pairwise;
    reference.executor = executor_kind::reference;
    return execute_plan(problem, reference, alpha);
}

inline bool nearly_equal(double a, double b, double tolerance = 0.0) {
    if (tolerance == 0.0) {
        return a == b;
    }
    return std::abs(a - b) <= tolerance;
}

inline bool same_vector(const std::vector<double>& a, const std::vector<double>& b, double tolerance = 0.0) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!nearly_equal(a[i], b[i], tolerance)) {
            return false;
        }
    }
    return true;
}

inline bool conforms_to_reference(
    const execution_result& candidate,
    const execution_result& reference,
    const plan_descriptor& plan) {
    if (plan.contract != contract_level::strict_expression) {
        return true;
    }
    return candidate.rho == reference.rho &&
           candidate.sigma == reference.sigma &&
           candidate.alpha == reference.alpha &&
           same_vector(candidate.residual, reference.residual) &&
           same_vector(candidate.z, reference.z) &&
           same_vector(candidate.q, reference.q) &&
           same_vector(candidate.x_next, reference.x_next);
}

} // namespace ctdp::spmv_dsl
