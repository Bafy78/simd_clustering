#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>

#include "../../layout/static_soa.hpp"
#include "../../simd.hpp"
#include "point_ops.hpp"

template <eve::product_type PointT>
struct spherical_covariance_model {
    std::vector<float> covariances;
    std::vector<float> precisions;
    std::vector<float> mean_norms;
    std::vector<float> score_constants;
    std::vector<wide_f> sum_x2_w;
    float reg_covar = 1e-6f;

    struct point_cache {
        wide_f norm_sq;
    };

    spherical_covariance_model(
        std::vector<float> precisions_,
        std::size_t n_components,
        float reg_covar_ = 1e-6f
    )
        : covariances(precisions_.size()),
          precisions(std::move(precisions_)),
          mean_norms(n_components),
          score_constants(n_components),
          sum_x2_w(n_components),
          reg_covar(reg_covar_) {}

    void validate_inputs(std::size_t n_components) const {
        if (precisions.size() != n_components) {
            throw std::runtime_error("Spherical GMM precision count must match component count");
        }

        for (float precision : precisions) {
            if (!(precision > 0.0f)) {
                throw std::runtime_error("Spherical GMM precisions must be strictly positive");
            }
        }
    }

    void refresh_covariances_from_precisions() {
        for (std::size_t k = 0; k < precisions.size(); ++k) {
            covariances[k] = 1.0f / precisions[k];
        }
    }

    template <class Weights, class Means>
    void refresh_score_data(const Weights& weights, const Means& means) {
        const float log_2_pi = std::log(2.0f * std::numbers::pi_v<float>);
        constexpr float half_features = 0.5f * static_cast<float>(kumi::size_v<PointT>);

        for (std::size_t k = 0; k < means.size(); ++k) {
            mean_norms[k] = gmm_point_norm_sq(means[k]);

            score_constants[k] =
                std::log(weights[k])
                + half_features * std::log(precisions[k])
                - half_features * log_2_pi
                - 0.5f * precisions[k] * mean_norms[k];
        }
    }

    void reset_simd_accumulators() {
        const auto zero = eve::zero(eve::as<wide_f>());
        std::fill(sum_x2_w.begin(), sum_x2_w.end(), zero);
    }

    template <eve::product_type SimdPointT>
    point_cache make_point_cache(const SimdPointT& point) const {
        return point_cache{gmm_simd_point_norm_sq(point)};
    }

    template <eve::product_type SimdPointT>
    wide_f compute_weighted_log_prob(
        const SimdPointT& point,
        const point_cache& cache,
        const PointT& mean,
        std::size_t k
    ) const {
        const auto dot = gmm_dot_simd_point_with_mean(point, mean);

        return eve::fma(
            wide_f(-0.5f * precisions[k]),
            cache.norm_sq,
            eve::fma(wide_f(precisions[k]), dot, wide_f(score_constants[k]))
        );
    }

    template <eve::product_type SimdPointT, class Ignore>
    void accumulate_second_order(
        std::size_t k,
        wide_f resp,
        const SimdPointT&,
        const point_cache& cache,
        Ignore ignore
    ) {
        sum_x2_w[k] = eve::fma[ignore](resp, cache.norm_sq, sum_x2_w[k]);
    }

    void update_component_from_sufficient_statistics(
        std::size_t k,
        float nk,
        const PointT& mean
    ) {
        constexpr float inv_features = 1.0f / static_cast<float>(kumi::size_v<PointT>);

        const float mean_norm = gmm_point_norm_sq(mean);
        const float avg_x2 = eve::reduce(sum_x2_w[k]) / nk;
        const float covariance = (avg_x2 - mean_norm) * inv_features + reg_covar;

        if (!(covariance > 0.0f)) {
            throw std::runtime_error(
                "Spherical GMM covariance became non-positive; try increasing reg_covar"
            );
        }

        covariances[k] = covariance;
        precisions[k] = 1.0f / covariance;
        mean_norms[k] = mean_norm;
    }
};
