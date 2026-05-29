#pragma once

#include <array>
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
struct diagonal_covariance_model {
    static constexpr std::size_t n_features = kumi::size_v<PointT>;
    using simd_sum_point = std::array<wide_f, n_features>;

    std::vector<float> covariances;
    std::vector<float> precisions;
    std::vector<float> score_constants;
    std::vector<simd_sum_point> sum_x2_w;
    float reg_covar = 1e-6f;

    struct point_cache {
        simd_sum_point x2;
    };

    diagonal_covariance_model(
        std::vector<float> precisions_,
        std::size_t n_components,
        float reg_covar_ = 1e-6f
    )
        : covariances(precisions_.size()),
          precisions(std::move(precisions_)),
          score_constants(n_components),
          sum_x2_w(n_components),
          reg_covar(reg_covar_) {}

    static std::size_t offset(std::size_t component, std::size_t feature) {
        return component * n_features + feature;
    }

    void validate_inputs(std::size_t n_components) const {
        if (precisions.size() != n_components * n_features) {
            throw std::runtime_error(
                "Diagonal GMM precision count must be component count times feature count"
            );
        }

        for (float precision : precisions) {
            if (!(precision > 0.0f)) {
                throw std::runtime_error("Diagonal GMM precisions must be strictly positive");
            }
        }
    }

    void refresh_covariances_from_precisions() {
        for (std::size_t i = 0; i < precisions.size(); ++i) {
            covariances[i] = 1.0f / precisions[i];
        }
    }

    template <class Weights, class Means>
    void refresh_score_data(const Weights& weights, const Means& means) {
        const float log_2_pi = std::log(2.0f * std::numbers::pi_v<float>);
        constexpr float half_features = 0.5f * static_cast<float>(n_features);

        for (std::size_t k = 0; k < means.size(); ++k) {
            float log_precision_det = 0.0f;
            float mean_quadratic = 0.0f;

            kumi::for_each_index(
                [&](auto index, auto mu) {
                    const float precision = precisions[offset(k, index)];
                    log_precision_det += std::log(precision);
                    mean_quadratic += static_cast<float>(mu) * static_cast<float>(mu) * precision;
                },
                means[k]
            );

            score_constants[k] =
                std::log(weights[k])
                + 0.5f * log_precision_det
                - half_features * log_2_pi
                - 0.5f * mean_quadratic;
        }
    }

    void reset_simd_accumulators() {
        const auto zero = eve::zero(eve::as<wide_f>());

        for (auto& row : sum_x2_w) {
            row.fill(zero);
        }
    }

    template <eve::product_type SimdPointT>
    point_cache make_point_cache(const SimdPointT& point) const {
        point_cache cache;

        kumi::for_each_index(
            [&](auto index, auto x) {
                cache.x2[index] = x * x;
            },
            point
        );

        return cache;
    }

    template <eve::product_type SimdPointT>
    wide_f compute_weighted_log_prob(
        const SimdPointT& point,
        const point_cache& cache,
        const PointT& mean,
        std::size_t k
    ) const {
        auto score = wide_f(score_constants[k]);

        kumi::for_each_index(
            [&](auto index, auto x, auto mu) {
                const float mean_times_precision = static_cast<float>(mu)
                    * precisions[offset(k, index)];
                const auto precision = wide_f(precisions[offset(k, index)]);

                score = eve::fma(
                    wide_f(-0.5f) * precision,
                    cache.x2[index],
                    eve::fma(wide_f(mean_times_precision), x, score)
                );
            },
            point,
            mean
        );

        return score;
    }

    template <eve::product_type SimdPointT, class Ignore>
    void accumulate_second_order(
        std::size_t k,
        wide_f resp,
        const SimdPointT& point,
        const point_cache& cache,
        Ignore ignore
    ) {
        kumi::for_each_index(
            [&](auto index, auto) {
                sum_x2_w[k][index] = eve::fma[ignore](
                    resp,
                    cache.x2[index],
                    sum_x2_w[k][index]
                );
            },
            point
        );
    }

    void update_component_from_sufficient_statistics(
        std::size_t k,
        float nk,
        const PointT& mean
    ) {
        kumi::for_each_index(
            [&](auto index, auto mu) {
                const float avg_x2 = eve::reduce(sum_x2_w[k][index]) / nk;
                const float mean_d = static_cast<float>(mu);
                const float covariance = avg_x2 - mean_d * mean_d + reg_covar;

                if (!(covariance > 0.0f)) {
                    throw std::runtime_error(
                        "Diagonal GMM covariance became non-positive; try increasing reg_covar"
                    );
                }

                covariances[offset(k, index)] = covariance;
                precisions[offset(k, index)] = 1.0f / covariance;
            },
            mean
        );
    }
};
