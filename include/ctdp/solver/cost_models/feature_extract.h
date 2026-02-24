#ifndef CTDP_SOLVER_COST_MODELS_FEATURE_EXTRACT_H
#define CTDP_SOLVER_COST_MODELS_FEATURE_EXTRACT_H

// ctdp v0.7.0 — Feature extraction for performance models
// Transforms tile-shape points into feature vectors for regression.

#include <cmath>
#include <vector>
#include <concepts>

namespace ctdp::cost_models {

// ─── Point concept ──────────────────────────────────────────────────────
// A Point must expose .dims() returning a span/range of doubles/ints.

template <typename P>
concept point_like = requires(const P& p) {
    { p.dims() };  // returns something iterable of numeric values
};

// ─── Feature extractors ─────────────────────────────────────────────────
// Each extractor takes a Point and returns vector<double> of features.

struct raw_extractor {
    template <point_like Point>
    std::vector<double> operator()(const Point& p) const {
        std::vector<double> f;
        for (auto d : p.dims())
            f.push_back(static_cast<double>(d));
        return f;
    }

    static constexpr const char* feature_name() { return "raw"; }
};

struct log2_extractor {
    template <point_like Point>
    std::vector<double> operator()(const Point& p) const {
        std::vector<double> f;
        for (auto d : p.dims())
            f.push_back(std::log2(static_cast<double>(d)));
        return f;
    }

    static constexpr const char* feature_name() { return "log2"; }
};

struct log2_interactions_extractor {
    template <point_like Point>
    std::vector<double> operator()(const Point& p) const {
        // log2 of each dim + all pairwise products of log2 dims
        std::vector<double> base;
        for (auto d : p.dims())
            base.push_back(std::log2(static_cast<double>(d)));

        std::vector<double> f = base;
        for (std::size_t i = 0; i < base.size(); ++i)
            for (std::size_t j = i; j < base.size(); ++j)
                f.push_back(base[i] * base[j]);
        return f;
    }

    static constexpr const char* feature_name() { return "log2_interactions"; }
};

struct reciprocal_extractor {
    template <point_like Point>
    std::vector<double> operator()(const Point& p) const {
        std::vector<double> f;
        for (auto d : p.dims()) {
            double v = static_cast<double>(d);
            f.push_back(v);
            f.push_back(1.0 / v);
        }
        return f;
    }

    static constexpr const char* feature_name() { return "reciprocal"; }
};

// ─── Convenience: extract feature matrix from observations ──────────────

template <typename Extractor, typename Point>
std::vector<std::vector<double>> extract_features(
        const Extractor& ext,
        const std::vector<observation<Point>>& obs) {
    std::vector<std::vector<double>> X;
    X.reserve(obs.size());
    for (const auto& o : obs)
        X.push_back(ext(o.point));
    return X;
}

template <typename Point>
std::vector<double> extract_targets(
        const std::vector<observation<Point>>& obs) {
    std::vector<double> y;
    y.reserve(obs.size());
    for (const auto& o : obs)
        y.push_back(o.cost);
    return y;
}

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_FEATURE_EXTRACT_H
