#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>

#include "../../layout/static_soa.hpp"
#include "../../simd.hpp"

template <eve::product_type PointT>
struct spherical_gmm_result {
    std::vector<float> weights;
    std::vector<PointT> means;
    std::vector<float> covariances;
    std::vector<float> precisions;
    std::vector<float> lower_bounds;
    int iterations = 0;
    bool converged = false;
    float lower_bound = -std::numeric_limits<float>::infinity();
};

template <eve::product_type PointT>
float point_norm_sq(const PointT& point) {
    return kumi::inner_product(point, point, 0.0f);
}

template <eve::product_type SimdPointT, eve::product_type PointT>
wide_f dot_simd_point_with_mean(const SimdPointT& point, const PointT& mean) {
    auto dot = eve::zero(eve::as<wide_f>());

    kumi::for_each(
        [&](auto x, auto mu) {
            dot = eve::fma(x, wide_f(mu), dot);
        },
        point,
        mean
    );

    return dot;
}

template <eve::product_type SimdPointT>
wide_f simd_point_norm_sq(const SimdPointT& point) {
    auto norm_sq = eve::zero(eve::as<wide_f>());

    kumi::for_each(
        [&](auto x) {
            norm_sq = eve::fma(x, x, norm_sq);
        },
        point
    );

    return norm_sq;
}

template <eve::product_type PointT>
struct spherical_gmm_em_state {
    using simd_sum_point = std::array<wide_f, kumi::size_v<PointT>>;

    const eve::algo::soa_vector<PointT>& points;
    std::vector<float> weights;
    std::vector<PointT> means;
    std::vector<float> covariances;
    std::vector<float> precisions;
    std::vector<float> mean_norms;
    std::vector<float> score_constants;
    std::vector<wide_f> score_scratch;
    std::vector<wide_f> unnormalized_resp_scratch;
    std::vector<wide_f> nk_w;
    std::vector<simd_sum_point> sum_x_w;
    std::vector<wide_f> sum_x2_w;

    float reg_covar = 1e-6f;

    spherical_gmm_em_state(
        const eve::algo::soa_vector<PointT>& points_,
        std::vector<float> weights_,
        std::vector<PointT> means_,
        std::vector<float> precisions_,
        float reg_covar_
    )
        : points(points_),
          weights(std::move(weights_)),
          means(std::move(means_)),
          covariances(precisions_.size()),
          precisions(std::move(precisions_)),
          mean_norms(means.size()),
          score_constants(means.size()),
          score_scratch(means.size()),
          unnormalized_resp_scratch(means.size()),
          nk_w(means.size()),
          sum_x_w(means.size()),
          sum_x2_w(means.size()),
          reg_covar(reg_covar_) {
        validate_inputs();
        refresh_covariances_from_precisions();
        refresh_score_data();
    }

    std::size_t n_components() const {
        return means.size();
    }

    void validate_inputs() const {
        const std::size_t k = means.size();

        if (points.size() == 0) {
            throw std::runtime_error("GMM requires at least one sample");
        }

        if (k == 0) {
            throw std::runtime_error("GMM requires at least one component");
        }

        if (weights.size() != k) {
            throw std::runtime_error("GMM weights count does not match component count");
        }

        if (precisions.size() != k) {
            throw std::runtime_error("Spherical GMM precision count must match component count");
        }

        for (std::size_t component = 0; component < k; ++component) {
            if (!(weights[component] > 0.0f)) {
                throw std::runtime_error("GMM weights must be strictly positive");
            }

            if (!(precisions[component] > 0.0f)) {
                throw std::runtime_error("Spherical GMM precisions must be strictly positive");
            }
        }
    }

    void refresh_covariances_from_precisions() {
        for (std::size_t k = 0; k < n_components(); ++k) {
            covariances[k] = 1.0f / precisions[k];
        }
    }

    void refresh_score_data() {
        const float log_2_pi = std::log(2.0f * std::numbers::pi_v<float>);
        constexpr float half_features = 0.5f * static_cast<float>(kumi::size_v<PointT>);

        for (std::size_t k = 0; k < n_components(); ++k) {
            mean_norms[k] = point_norm_sq(means[k]);

            // sklearn stores precisions_cholesky_ and computes, for spherical covariance:
            //   log_det = D * log(sqrt(precision)) = 0.5 * D * log(precision)
            //   log_prob = -0.5 * (D * log(2π) + precision * ||x - μ||²) + log_det
            score_constants[k] =
                std::log(weights[k])
                + half_features * std::log(precisions[k])
                - half_features * log_2_pi
                - 0.5f * precisions[k] * mean_norms[k];
        }
    }

    void reset_simd_accumulators() {
        const auto zero = eve::zero(eve::as<wide_f>());

        std::fill(nk_w.begin(), nk_w.end(), zero);
        std::fill(sum_x2_w.begin(), sum_x2_w.end(), zero);

        for (auto& row : sum_x_w) {
            row.fill(zero);
        }
    }

    template <eve::product_type SimdPointT>
    wide_f compute_weighted_log_prob(
        const SimdPointT& point,
        wide_f point_norm,
        std::size_t k
    ) const {
        const auto dot = dot_simd_point_with_mean(point, means[k]);

        return eve::fma(
            wide_f(-0.5f * precisions[k]),
            point_norm,
            eve::fma(wide_f(precisions[k]), dot, wide_f(score_constants[k]))
        );
    }

    float e_step_and_accumulate_sufficient_statistics() {
        reset_simd_accumulators();

        auto lower_bound_sum_w = eve::zero(eve::as<wide_f>());

        eve::algo::for_each[eve::algo::no_unrolling](
            points,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                const auto point = eve::load[ignore](it);
                const auto point_norm = simd_point_norm_sq(point);

                auto max_score = wide_f(-std::numeric_limits<float>::infinity());

                for (std::size_t k = 0; k < n_components(); ++k) {
                    const auto score = compute_weighted_log_prob(point, point_norm, k);
                    score_scratch[k] = score;
                    max_score = eve::max(max_score, score);
                }

                auto denom = eve::zero(eve::as<wide_f>());

                for (std::size_t k = 0; k < n_components(); ++k) {
                    const auto unnormalized_resp = eve::exp(score_scratch[k] - max_score);
                    unnormalized_resp_scratch[k] = unnormalized_resp;
                    denom += unnormalized_resp;
                }

                const auto log_prob_norm = max_score + eve::log(denom);
                lower_bound_sum_w = eve::fma[ignore](
                    log_prob_norm,
                    wide_f(1.0f),
                    lower_bound_sum_w
                );

                const auto inv_denom = wide_f(1.0f) / denom;

                for (std::size_t k = 0; k < n_components(); ++k) {
                    const auto resp = unnormalized_resp_scratch[k] * inv_denom;

                    nk_w[k] = eve::fma[ignore](resp, wide_f(1.0f), nk_w[k]);
                    sum_x2_w[k] = eve::fma[ignore](resp, point_norm, sum_x2_w[k]);

                    kumi::for_each_index(
                        [&](auto index, auto x) {
                            sum_x_w[k][index] = eve::fma[ignore](
                                resp,
                                x,
                                sum_x_w[k][index]
                            );
                        },
                        point
                    );
                }
            }
        );

        return eve::reduce(lower_bound_sum_w) / static_cast<float>(points.size());
    }

    void m_step_from_accumulators() {
        constexpr float eps10 = 10.0f * std::numeric_limits<float>::epsilon();
        constexpr float inv_features = 1.0f / static_cast<float>(kumi::size_v<PointT>);

        std::vector<float> nk(n_components());
        float nk_sum = 0.0f;

        for (std::size_t k = 0; k < n_components(); ++k) {
            nk[k] = eve::reduce(nk_w[k]) + eps10;
            nk_sum += nk[k];
        }

        for (std::size_t k = 0; k < n_components(); ++k) {
            const float inv_nk = 1.0f / nk[k];

            weights[k] = nk[k] / nk_sum;

            kumi::for_each_index(
                [&](auto index, auto& mean_d) {
                    mean_d = eve::reduce(sum_x_w[k][index]) * inv_nk;
                },
                means[k]
            );

            const float mean_norm = point_norm_sq(means[k]);
            const float avg_x2 = eve::reduce(sum_x2_w[k]) * inv_nk;
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

        refresh_score_data();
    }
};

template <eve::product_type PointT>
spherical_gmm_result<PointT> spherical_em(
    const eve::algo::soa_vector<PointT>& points,
    std::vector<float> weights,
    std::vector<PointT> means,
    std::vector<float> precisions,
    int max_iterations = 100,
    float tol = 1e-3f,
    float reg_covar = 1e-6f
) {
    if (max_iterations < 0) {
        throw std::runtime_error("GMM max_iterations must be non-negative");
    }

    spherical_gmm_em_state<PointT> state{
        points,
        std::move(weights),
        std::move(means),
        std::move(precisions),
        reg_covar
    };

    spherical_gmm_result<PointT> result;
    result.lower_bounds.reserve(static_cast<std::size_t>(max_iterations));

    float lower_bound = -std::numeric_limits<float>::infinity();

    for (int iter = 1; iter <= max_iterations; ++iter) {
        const float previous_lower_bound = lower_bound;

        lower_bound = state.e_step_and_accumulate_sufficient_statistics();
        state.m_step_from_accumulators();

        result.lower_bounds.push_back(lower_bound);
        result.iterations = iter;
        result.lower_bound = lower_bound;

        if (std::abs(lower_bound - previous_lower_bound) < tol) {
            result.converged = true;
            break;
        }
    }

    result.weights = std::move(state.weights);
    result.means = std::move(state.means);
    result.covariances = std::move(state.covariances);
    result.precisions = std::move(state.precisions);

    return result;
}
