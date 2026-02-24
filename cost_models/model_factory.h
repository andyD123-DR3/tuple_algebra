#ifndef CTDP_SOLVER_COST_MODELS_MODEL_FACTORY_H
#define CTDP_SOLVER_COST_MODELS_MODEL_FACTORY_H

// ctdp v0.7.0 — Auto model factory
// Trains linear, MLP, and SVR models.
// Selects winner by oos_spearman_rho (5-fold CV).
// Returns any_model<Point>.

#include "performance_model.h"
#include "linear_model.h"
#include "mlp_model.h"
#include "svr_model.h"
#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace ctdp::cost_models {

template <typename Point, typename Extractor = log2_interactions_extractor>
class auto_model_factory {
    Extractor extractor_;
    bool verbose_;

public:
    explicit auto_model_factory(Extractor ext = {}, bool verbose = false)
        : extractor_(std::move(ext)), verbose_(verbose) {}

    struct build_result {
        any_model<Point> model;
        std::vector<model_quality> all_qualities;  // quality of each candidate
    };

    build_result build(const std::vector<observation<Point>>& obs) const {
        build_result result;

        // Train all three models
        if (verbose_) std::cerr << "[factory] Training linear model...\n";
        linear_trainer<Point, Extractor> lt(extractor_);
        auto lm = lt.build(obs);
        auto lq = lm.quality();
        result.all_qualities.push_back(lq);

        if (verbose_) std::cerr << "[factory] Training MLP model...\n";
        mlp_trainer<Point, Extractor> mt(extractor_);
        auto mm = mt.build(obs);
        auto mq = mm.quality();
        result.all_qualities.push_back(mq);

        if (verbose_) std::cerr << "[factory] Training SVR model...\n";
        svr_trainer<Point, Extractor> st(extractor_);
        auto sm = st.build(obs);
        auto sq = sm.quality();
        result.all_qualities.push_back(sq);

        if (verbose_) {
            std::cerr << "[factory] Results:\n";
            for (const auto& q : result.all_qualities) {
                std::cerr << "  " << q.model_name
                          << ": in-sample ρ=" << q.spearman_rho
                          << ", OOS ρ=" << q.oos_spearman_rho
                          << ", OOS R²=" << q.oos_r2
                          << "\n";
            }
        }

        // Select winner by oos_spearman_rho
        // Use in-sample as fallback if OOS is NaN (e.g. too few samples)
        auto effective_rho = [](const model_quality& q) {
            return std::isfinite(q.oos_spearman_rho)
                ? q.oos_spearman_rho
                : q.spearman_rho;
        };

        double best_rho = effective_rho(lq);
        int winner = 0;

        if (effective_rho(mq) > best_rho) {
            best_rho = effective_rho(mq);
            winner = 1;
        }
        if (effective_rho(sq) > best_rho) {
            best_rho = effective_rho(sq);
            winner = 2;
        }

        switch (winner) {
            case 0:
                if (verbose_) std::cerr << "[factory] Winner: linear\n";
                result.model = any_model<Point>(std::move(lm));
                break;
            case 1:
                if (verbose_) std::cerr << "[factory] Winner: MLP\n";
                result.model = any_model<Point>(std::move(mm));
                break;
            case 2:
                if (verbose_) std::cerr << "[factory] Winner: SVR\n";
                result.model = any_model<Point>(std::move(sm));
                break;
        }

        return result;
    }
};

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_MODEL_FACTORY_H
