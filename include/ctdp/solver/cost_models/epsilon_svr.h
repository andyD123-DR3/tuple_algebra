#pragma once
// epsilon_svr.h  —  Proper Epsilon-SVR via Sequential Minimal Optimization
//
// Dual formulation (Smola & Schölkopf, 1998):
//   min  0.5*(a-a*)^T Q (a-a*) + eps*sum(a+a*) - y^T(a-a*)
//   s.t. sum(a_i - a*_i) = 0,   0 <= a_i, a*_i <= C
//
// libSVM 2n-variable encoding:
//   u[i]   = a_i   i in [0,n)   y2[i] = +1
//   u[i+n] = a*_i  i in [0,n)   y2[i+n] = -1
//   Q_ij   = y2[i]*y2[j]*K(x_{i%n}, x_{j%n})
//   p[i]   = eps - y[i]   (i <  n)
//   p[i+n] = eps + y[i]   (i >= n)
//   G[i]   = y2[i]*f[i%n] + p[i]   where f[k] = sum_j coef[j]*K[j,k]
//
// Working sets (Fan, Chen, Lin, JMLR 2005):
//   I_up  = {i: y2=+1,u<C} U {i: y2=-1,u>0}   <- "can increase contribution"
//   I_lo  = {i: y2=+1,u>0} U {i: y2=-1,u<C}   <- "can decrease contribution"
//   KKT:  max_{I_up}(-y2*G) = min_{I_lo}(-y2*G) = -lambda   at optimum
//
// Update for pair (t in I_up, q in I_lo):
//   u[t] += y2[t]*step,   u[q] -= y2[q]*step
//   ΔG[k_all] = y2[k]*step*(K_{t%n,k%n} - K_{q%n,k%n})
//   step = (-y2[t]*G[t] + y2[q]*G[q]) / H,  H = K_tt + K_qq - 2*K_tq
//
// Bias:
//   free a_i  (0<u[i<n]<C): b = -G[i]     = y[i] - eps - f[i]
//   free a*_j (0<u[j+n]<C): b =  G[j+n]   = y[j] + eps - f[j]
//
// References:
//   libSVM: https://github.com/cjlin1/libsvm (BSD-3)
//   Fan, Chen, Lin. JMLR 6:1889-1918, 2005.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace ctdp::cost_models {

// ---------------------------------------------------------------------------
// Hyper-parameters
// ---------------------------------------------------------------------------
struct EpsilonSVRParams {
    double C        = 1.0;
    double epsilon  = 0.1;
    double gamma    = 0.1;
    double tol      = 1e-3;
    int    max_iter = 100'000;
};

// ---------------------------------------------------------------------------
// RBF kernel
// ---------------------------------------------------------------------------
namespace detail {
inline double rbf(std::span<const double> xi,
                  std::span<const double> xj,
                  double gamma) noexcept
{
    double sq = 0.0;
    for (std::size_t k = 0; k < xi.size(); ++k) {
        double d = xi[k] - xj[k]; sq += d * d;
    }
    return std::exp(-gamma * sq);
}
} // namespace detail

// ---------------------------------------------------------------------------
// Trained model
// ---------------------------------------------------------------------------
struct EpsilonSVRModel {
    std::vector<double>              coef;      // a_i - a*_i (n entries)
    double                           bias = 0.0;
    int                              n_sv = 0;
    std::vector<std::vector<double>> sv_x;      // compact SV feature vectors
    std::vector<double>              sv_coef;   // non-zero coef values
    EpsilonSVRParams                 params;

    double predict(std::span<const double> x) const {
        double v = bias;
        for (int i = 0; i < n_sv; ++i)
            v += sv_coef[i] * detail::rbf(sv_x[i], x, params.gamma);
        return v;
    }

    std::vector<double> predict_batch(const std::vector<std::vector<double>>& X) const {
        std::vector<double> out; out.reserve(X.size());
        for (const auto& x : X) out.push_back(predict(x));
        return out;
    }

    double sparsity_ratio() const {
        if (coef.empty()) return 0.0;
        return static_cast<double>(n_sv) / static_cast<double>(coef.size());
    }
};

// ---------------------------------------------------------------------------
// SMO Solver
// ---------------------------------------------------------------------------
class EpsilonSVRSolver {
public:
    static EpsilonSVRModel train(const std::vector<std::vector<double>>& X,
                                 std::span<const double>                  y,
                                 const EpsilonSVRParams&                  p = {})
    {
        const int n = static_cast<int>(X.size());
        if (n == 0) throw std::invalid_argument("EpsilonSVR: empty training set");
        if (static_cast<int>(y.size()) != n)
            throw std::invalid_argument("EpsilonSVR: X/y size mismatch");
        const int N = 2 * n;

        // --- Kernel matrix -------------------------------------------------
        std::vector<double> K(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j) {
                double v = detail::rbf(X[i], X[j], p.gamma);
                K[i*n+j] = K[j*n+i] = v;
            }

        // --- Dual variables ------------------------------------------------
        std::vector<double> u(N, 0.0);

        // y2[i]: sign for variable i
        auto y2 = [](int i, int n_) -> double { return i < n_ ? +1.0 : -1.0; };
        // Physical sample index
        auto ph = [](int i, int n_) -> int { return i < n_ ? i : i - n_; };
        // Linear term p[i]
        auto pt = [&](int i) -> double {
            return i < n ? (p.epsilon - y[i]) : (p.epsilon + y[i - n]);
        };

        // --- f[k] = decision function at x_k (kernel part, no bias) --------
        // f[k] = sum_j (u[j<n] - u[j+n]) * K[j,k]  == sum_j y2[j]*u[j]*K[j%n,k]
        // G[i] = y2[i]*f[i%n] + pt[i]
        std::vector<double> f(n, 0.0);
        std::vector<double> G(N);
        for (int i = 0; i < N; ++i) G[i] = pt(i);  // all u=0 initially

        // --- I_up / I_lo predicates ----------------------------------------
        auto in_up = [&](int i) -> bool {
            return (i < n) ? (u[i] < p.C - 1e-12) : (u[i] > 1e-12);
        };
        auto in_lo = [&](int i) -> bool {
            return (i < n) ? (u[i] > 1e-12) : (u[i] < p.C - 1e-12);
        };
        // gain = -y2[i]*G[i]: larger = more violation
        auto gain = [&](int i) -> double { return -y2(i, n) * G[i]; };

        // --- Main loop -----------------------------------------------------
        for (int iter = 0; iter < p.max_iter; ++iter) {

            // Select t = argmax_{I_up} gain
            int    t    = -1;
            double gmax = -1e300;
            for (int i = 0; i < N; ++i) {
                if (in_up(i)) { double v = gain(i); if (v > gmax) { gmax = v; t = i; } }
            }
            if (t < 0) break;

            // Select q via WSS2 from I_lo
            // Convergence: no j in I_lo satisfies gmax - gain(j) > tol
            // Numerator for step: gmax + y2[j]*G[j] (must be > 0)
            int    q    = -1;
            double best = std::numeric_limits<double>::infinity();
            int    tp   = ph(t, n);
            double Ktt  = K[tp*n+tp];

            bool improving = false;
            for (int j = 0; j < N; ++j) {
                if (!in_lo(j)) continue;
                if (gmax - gain(j) <= p.tol) continue;
                improving = true;
                double num = gmax + y2(j, n) * G[j];
                if (num <= 0.0) continue;
                int    qp  = ph(j, n);
                double H   = std::max(Ktt + K[qp*n+qp] - 2.0*K[tp*n+qp], 1e-12);
                double sc  = H / (num * num);
                if (sc < best) { best = sc; q = j; }
            }
            if (!improving) break;
            if (q < 0) break;

            // Compute step
            // H = Q_tt + Q_qq - 2*Q_tq  where Q_ij = y2[i]*y2[j]*K_{i%n,j%n}
            // Since y2[i]^2 = 1:  H = K_tt + K_qq - 2*y2[t]*y2[q]*K[tp,qp]
            int    qp   = ph(q, n);
            double y2t  = y2(t, n), y2q = y2(q, n);
            double H    = std::max(Ktt + K[qp*n+qp] - 2.0*y2t*y2q*K[tp*n+qp], 1e-12);
            double num  = gmax + y2q * G[q];
            double step = num / H;

            // Clip to box constraints
            // u[t] += y2t*step: if y2t=+1 => step in [0, C-u[t]]; if y2t=-1 => step in [0, u[t]]
            double clip_t = (y2t > 0) ? (p.C - u[t]) : u[t];
            double clip_q = (y2q > 0) ? u[q] : (p.C - u[q]);
            step = std::max(0.0, std::min(step, std::min(clip_t, clip_q)));
            if (step < 1e-15) break;

            // Apply
            u[t] = std::clamp(u[t] + y2t * step, 0.0, p.C);
            u[q] = std::clamp(u[q] - y2q * step, 0.0, p.C);

            // Update f then G
            // Δf[k] = y2[t]²*step*K[tp,k] - y2[q]²*step*K[qp,k]  = step*(K_t - K_q)
            // (y2 factors cancel since y2² = 1)
            for (int k = 0; k < n; ++k)
                f[k] += step * (K[tp*n+k] - K[qp*n+k]);
            for (int i = 0; i < N; ++i)
                G[i] = y2(i, n) * f[ph(i, n)] + pt(i);
        }

        // --- Bias -----------------------------------------------------------
        double b_sum = 0.0; int b_cnt = 0;
        for (int i = 0; i < N; ++i) {
            if (u[i] > 1e-8 && u[i] < p.C - 1e-8) {
                b_sum += (i < n) ? -G[i] : G[i];
                ++b_cnt;
            }
        }
        double bias = (b_cnt > 0) ? (b_sum / b_cnt) : 0.0;
        if (b_cnt == 0) {
            // Use KKT inequalities from bound variables
            double blo = -1e300, bhi = 1e300;
            for (int i = 0; i < N; ++i) {
                double bc = (i < n) ? -G[i] : G[i];
                if (in_up(i)) bhi = std::min(bhi, bc);
                if (in_lo(i)) blo = std::max(blo, bc);
            }
            if (std::isfinite(blo + bhi)) bias = 0.5*(blo + bhi);
        }

        // --- Pack result ----------------------------------------------------
        EpsilonSVRModel m;
        m.params = p; m.bias = bias;
        m.coef.resize(n);
        for (int i = 0; i < n; ++i) m.coef[i] = u[i] - u[i+n];
        for (int i = 0; i < n; ++i) {
            if (std::abs(m.coef[i]) > 1e-8) {
                m.sv_x.push_back(X[i]);
                m.sv_coef.push_back(m.coef[i]);
                ++m.n_sv;
            }
        }
        return m;
    }
};

// ---------------------------------------------------------------------------
// Convenience wrapper
// ---------------------------------------------------------------------------
inline EpsilonSVRModel fit_epsilon_svr(const std::vector<std::vector<double>>& X,
                                       std::span<const double>                  y,
                                       const EpsilonSVRParams&                  p = {})
{
    return EpsilonSVRSolver::train(X, y, p);
}

// ---------------------------------------------------------------------------
// Spearman rank correlation
// ---------------------------------------------------------------------------
inline double spearman_rho(std::span<const double> a, std::span<const double> b)
{
    int n = static_cast<int>(a.size());
    if (n < 2) return 0.0;
    auto rank_vec = [&](std::span<const double> v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int i, int j){ return v[i] < v[j]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = (i + j - 1) / 2.0 + 1.0;
            for (int k = i; k < j; ++k) r[idx[k]] = avg;
            i = j;
        }
        return r;
    };
    auto ra = rank_vec(a); auto rb = rank_vec(b);
    double s = 0, sa = 0, sb = 0;
    double ma = std::accumulate(ra.begin(),ra.end(),0.0)/n;
    double mb = std::accumulate(rb.begin(),rb.end(),0.0)/n;
    for (int i = 0; i < n; ++i) {
        s  += (ra[i]-ma)*(rb[i]-mb);
        sa += (ra[i]-ma)*(ra[i]-ma);
        sb += (rb[i]-mb)*(rb[i]-mb);
    }
    double d = std::sqrt(sa*sb);
    return d < 1e-12 ? 0.0 : s/d;
}

// ---------------------------------------------------------------------------
// LOO grid search
// ---------------------------------------------------------------------------
struct SVRGridResult {
    EpsilonSVRParams best_params;
    double best_cv_rmse     = std::numeric_limits<double>::infinity();
    double best_cv_spearman = 0.0;
    int    best_n_sv        = 0;
    double best_sparsity    = 1.0;
};

inline SVRGridResult svr_grid_search_loo(
    const std::vector<std::vector<double>>& X,
    std::span<const double>                  y,
    const std::vector<double>& C_grid     = {0.01,0.1,1.0,10.0,100.0},
    const std::vector<double>& eps_grid   = {0.5,1.0,2.0,5.0,10.0},
    const std::vector<double>& gamma_grid = {1e-4,1e-3,0.01,0.1,1.0},
    double tol = 1e-3)
{
    int n = static_cast<int>(X.size());
    SVRGridResult best;
    for (double C : C_grid) for (double eps : eps_grid) for (double gamma : gamma_grid) {
        EpsilonSVRParams params{.C=C,.epsilon=eps,.gamma=gamma,.tol=tol};
        std::vector<double> preds(n);
        for (int lo = 0; lo < n; ++lo) {
            std::vector<std::vector<double>> Xtr;
            std::vector<double> ytr;
            for (int i = 0; i < n; ++i) { if (i==lo) continue; Xtr.push_back(X[i]); ytr.push_back(y[i]); }
            preds[lo] = EpsilonSVRSolver::train(Xtr,ytr,params).predict(X[lo]);
        }
        double rmse = 0.0;
        for (int i = 0; i < n; ++i) { double r=preds[i]-y[i]; rmse+=r*r; }
        rmse = std::sqrt(rmse/n);
        double rho = spearman_rho(preds,y);
        if (rmse < best.best_cv_rmse) {
            best.best_cv_rmse=rmse; best.best_cv_spearman=rho; best.best_params=params;
            auto full=EpsilonSVRSolver::train(X,y,params);
            best.best_n_sv=full.n_sv; best.best_sparsity=full.sparsity_ratio();
        }
    }
    return best;
}

} // namespace ctdp::cost_models
