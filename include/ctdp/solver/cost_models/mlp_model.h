#ifndef CTDP_SOLVER_COST_MODELS_MLP_MODEL_H
#define CTDP_SOLVER_COST_MODELS_MLP_MODEL_H

// ctdp v0.7.0 — Multi-layer perceptron performance model
// Architecture: NF → H1 → H2 → 1 (ReLU activations, linear output)
// Training: SGD + momentum, Xavier initialisation
// Quality: in-sample + 5-fold CV OOS metrics

#include "performance_model.h"
#include "feature_extract.h"
#include "cross_validation.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace ctdp::cost_models {

// ─── MLP layer ──────────────────────────────────────────────────────────

struct mlp_layer {
    std::size_t in_dim, out_dim;
    std::vector<double> weights;  // out_dim x in_dim, row-major
    std::vector<double> biases;   // out_dim

    mlp_layer() = default;
    mlp_layer(std::size_t in, std::size_t out)
        : in_dim(in), out_dim(out),
          weights(out * in, 0.0), biases(out, 0.0) {}

    // Xavier uniform initialisation
    void xavier_init(std::mt19937& rng) {
        double limit = std::sqrt(6.0 / static_cast<double>(in_dim + out_dim));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& w : weights) w = dist(rng);
        for (auto& b : biases) b = 0.0;
    }

    // Forward: output = W * input + bias
    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> out(out_dim, 0.0);
        for (std::size_t i = 0; i < out_dim; ++i) {
            double s = biases[i];
            for (std::size_t j = 0; j < in_dim; ++j)
                s += weights[i * in_dim + j] * input[j];
            out[i] = s;
        }
        return out;
    }
};

// ─── mlp_model ──────────────────────────────────────────────────────────

template <typename Point, typename Extractor = log2_interactions_extractor>
class mlp_model {
    mlp_layer layer1_, layer2_, layer3_;
    zscore_params zscore_;
    double y_mean_ = 0.0, y_std_ = 1.0;
    Extractor extractor_;
    model_quality quality_;

    static std::vector<double> relu(const std::vector<double>& x) {
        std::vector<double> r(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            r[i] = std::max(0.0, x[i]);
        return r;
    }

public:
    mlp_model() = default;
    mlp_model(mlp_layer l1, mlp_layer l2, mlp_layer l3,
              zscore_params zs, double ym, double ys,
              Extractor ext, model_quality q)
        : layer1_(std::move(l1)), layer2_(std::move(l2)), layer3_(std::move(l3)),
          zscore_(std::move(zs)), y_mean_(ym), y_std_(ys),
          extractor_(std::move(ext)), quality_(std::move(q)) {}

    double predict(const Point& p) const {
        auto raw = extractor_(p);
        auto feat = apply_zscore(raw, zscore_);
        auto h1 = relu(layer1_.forward(feat));
        auto h2 = relu(layer2_.forward(h1));
        auto out = layer3_.forward(h2);
        return out[0] * y_std_ + y_mean_;  // denormalise
    }

    model_quality quality() const { return quality_; }
    std::string name() const { return "mlp"; }

    const mlp_layer& l1() const { return layer1_; }
    const mlp_layer& l2() const { return layer2_; }
    const mlp_layer& l3() const { return layer3_; }
    const zscore_params& zscore() const { return zscore_; }
    double y_mean() const { return y_mean_; }
    double y_std() const { return y_std_; }
};

// ─── mlp_trainer ────────────────────────────────────────────────────────

template <typename Point, typename Extractor = log2_interactions_extractor>
class mlp_trainer {
    Extractor extractor_;
    std::size_t h1_size_;
    std::size_t h2_size_;
    std::size_t epochs_;
    double lr_;
    double momentum_;
    std::uint64_t seed_;

public:
    explicit mlp_trainer(Extractor ext = {},
                         std::size_t h1 = 16, std::size_t h2 = 8,
                         std::size_t epochs = 500, double lr = 0.01,
                         double momentum = 0.9, std::uint64_t seed = 42)
        : extractor_(std::move(ext)), h1_size_(h1), h2_size_(h2),
          epochs_(epochs), lr_(lr), momentum_(momentum), seed_(seed) {}

    // Build MLP from observations (no CV)
    mlp_model<Point, Extractor> build_raw(
            const std::vector<observation<Point>>& obs) const {
        auto X_raw = extract_features(extractor_, obs);
        auto y_raw = extract_targets(obs);
        const auto n = obs.size();
        const auto nf = X_raw[0].size();

        // Normalise features
        auto zs = compute_zscore(X_raw);
        std::vector<std::vector<double>> X(n);
        for (std::size_t i = 0; i < n; ++i)
            X[i] = apply_zscore(X_raw[i], zs);

        // Normalise targets
        double y_mean = 0.0;
        for (auto v : y_raw) y_mean += v;
        y_mean /= static_cast<double>(n);
        double y_var = 0.0;
        for (auto v : y_raw) { double d = v - y_mean; y_var += d * d; }
        double y_std = std::sqrt(y_var / static_cast<double>(n));
        if (y_std < 1e-15) y_std = 1.0;

        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
            y[i] = (y_raw[i] - y_mean) / y_std;

        // Init layers
        std::mt19937 rng(seed_);
        mlp_layer l1(nf, h1_size_);       l1.xavier_init(rng);
        mlp_layer l2(h1_size_, h2_size_);  l2.xavier_init(rng);
        mlp_layer l3(h2_size_, 1);         l3.xavier_init(rng);

        // SGD + momentum storage
        auto zero_like = [](const mlp_layer& l) {
            return std::make_pair(
                std::vector<double>(l.weights.size(), 0.0),
                std::vector<double>(l.biases.size(), 0.0));
        };
        auto [vw1, vb1] = zero_like(l1);
        auto [vw2, vb2] = zero_like(l2);
        auto [vw3, vb3] = zero_like(l3);

        // Training loop (mini-batch = full batch for small datasets)
        for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
            // Accumulate gradients
            std::vector<double> gw1(l1.weights.size(), 0.0), gb1(l1.biases.size(), 0.0);
            std::vector<double> gw2(l2.weights.size(), 0.0), gb2(l2.biases.size(), 0.0);
            std::vector<double> gw3(l3.weights.size(), 0.0), gb3(l3.biases.size(), 0.0);

            for (std::size_t s = 0; s < n; ++s) {
                // Forward
                auto z1 = l1.forward(X[s]);
                std::vector<double> h1(z1.size());
                for (std::size_t j = 0; j < z1.size(); ++j)
                    h1[j] = std::max(0.0, z1[j]);

                auto z2 = l2.forward(h1);
                std::vector<double> h2(z2.size());
                for (std::size_t j = 0; j < z2.size(); ++j)
                    h2[j] = std::max(0.0, z2[j]);

                auto out = l3.forward(h2);
                double err = out[0] - y[s];

                // Backward: layer 3
                // d_out = err
                for (std::size_t j = 0; j < h2_size_; ++j)
                    gw3[j] += err * h2[j];
                gb3[0] += err;

                // delta2
                std::vector<double> d2(h2_size_, 0.0);
                for (std::size_t j = 0; j < h2_size_; ++j)
                    d2[j] = err * l3.weights[j] * (z2[j] > 0.0 ? 1.0 : 0.0);

                // layer 2 grads
                for (std::size_t j = 0; j < h2_size_; ++j) {
                    for (std::size_t k = 0; k < h1_size_; ++k)
                        gw2[j * h1_size_ + k] += d2[j] * h1[k];
                    gb2[j] += d2[j];
                }

                // delta1
                std::vector<double> d1(h1_size_, 0.0);
                for (std::size_t j = 0; j < h1_size_; ++j) {
                    double s_val = 0.0;
                    for (std::size_t k = 0; k < h2_size_; ++k)
                        s_val += d2[k] * l2.weights[k * h1_size_ + j];
                    d1[j] = s_val * (z1[j] > 0.0 ? 1.0 : 0.0);
                }

                // layer 1 grads
                for (std::size_t j = 0; j < h1_size_; ++j) {
                    for (std::size_t k = 0; k < nf; ++k)
                        gw1[j * nf + k] += d1[j] * X[s][k];
                    gb1[j] += d1[j];
                }
            }

            double inv_n = 1.0 / static_cast<double>(n);

            // Update with momentum
            auto update = [&](std::vector<double>& w, std::vector<double>& g,
                              std::vector<double>& v) {
                for (std::size_t i = 0; i < w.size(); ++i) {
                    v[i] = momentum_ * v[i] - lr_ * g[i] * inv_n;
                    w[i] += v[i];
                }
            };

            update(l1.weights, gw1, vw1);
            update(l1.biases,  gb1, vb1);
            update(l2.weights, gw2, vw2);
            update(l2.biases,  gb2, vb2);
            update(l3.weights, gw3, vw3);
            update(l3.biases,  gb3, vb3);
        }

        // In-sample predictions
        std::vector<double> yhat(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto z1 = l1.forward(X[i]);
            for (auto& v : z1) v = std::max(0.0, v);
            auto z2 = l2.forward(z1);
            for (auto& v : z2) v = std::max(0.0, v);
            auto out = l3.forward(z2);
            yhat[i] = out[0] * y_std + y_mean;
        }

        model_quality q;
        q.r2 = compute_r2(y_raw, yhat);
        q.spearman_rho = compute_spearman(
            std::span<const double>(y_raw), std::span<const double>(yhat));
        q.n_samples = n;
        q.n_params = l1.weights.size() + l1.biases.size()
                   + l2.weights.size() + l2.biases.size()
                   + l3.weights.size() + l3.biases.size();
        q.model_name = "mlp";

        return mlp_model<Point, Extractor>(
            std::move(l1), std::move(l2), std::move(l3),
            std::move(zs), y_mean, y_std, extractor_, std::move(q));
    }

    // Build with full quality metrics (in-sample + 5-fold CV)
    mlp_model<Point, Extractor> build(
            const std::vector<observation<Point>>& obs) const {
        auto model = build_raw(obs);

        if (obs.size() >= 5) {
            auto cv = k_fold_cv<Point>(obs, 5,
                [this](const std::vector<observation<Point>>& train) {
                    return this->build_raw(train);
                });

            auto q = model.quality();
            q.oos_r2 = cv.oos_r2;
            q.oos_spearman_rho = cv.oos_spearman_rho;
            q.oos_rmse = cv.oos_rmse;

            return mlp_model<Point, Extractor>(
                model.l1(), model.l2(), model.l3(),
                model.zscore(), model.y_mean(), model.y_std(),
                extractor_, std::move(q));
        }
        return model;
    }
};

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_MLP_MODEL_H
