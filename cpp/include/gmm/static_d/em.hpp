#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>

#include "../../layout/static_soa.hpp"
#include "../../simd.hpp"

template <eve::product_type SampleT>
struct static_gmm_result {
    std::vector<float> weights;
    std::vector<SampleT> means;
    std::vector<float> covariances;
    std::vector<float> precisions;
    std::vector<float> lower_bounds;
    int iterations = 0;
    bool converged = false;
    float lower_bound = -std::numeric_limits<float>::infinity();
};

template <eve::product_type SampleT, class CovarianceModel>
struct static_gmm_em_state {
    using simd_sum_sample = std::array<wide_f, kumi::size_v<SampleT>>;

    const eve::algo::soa_vector<SampleT>& samples;
    std::vector<float> weights;
    std::vector<SampleT> means;
    CovarianceModel covariance;
    std::vector<wide_f> score_scratch;
    std::vector<wide_f> unnormalized_resp_scratch;
    std::vector<wide_f> N_k_w;
    std::vector<simd_sum_sample> sum_x_w;

    static_gmm_em_state(
        const eve::algo::soa_vector<SampleT>& samples_,
        std::vector<float> weights_,
        std::vector<SampleT> means_,
        CovarianceModel covariance_
    )
        : samples(samples_),
          weights(std::move(weights_)),
          means(std::move(means_)),
          covariance(std::move(covariance_)),
          score_scratch(means.size()),
          unnormalized_resp_scratch(means.size()),
          N_k_w(means.size()),
          sum_x_w(means.size()) {
        validate_inputs();
        covariance.refresh_covariances_from_precisions();
        covariance.refresh_score_data(weights, means);
    }

    std::size_t K() const {
        return means.size();
    }

    void validate_inputs() const {
        const std::size_t K = means.size();

        if (samples.size() == 0) {
            throw std::runtime_error("GMM requires at least one sample");
        }

        if (K == 0) {
            throw std::runtime_error("GMM requires at least one cluster");
        }

        if (weights.size() != K) {
            throw std::runtime_error("GMM weights count does not match cluster count");
        }

        for (std::size_t k = 0; k < K; ++k) {
            if (!(weights[k] > 0.0f)) {
                throw std::runtime_error("GMM weights must be strictly positive");
            }
        }

        covariance.validate_inputs(K);
    }

    void reset_simd_accumulators() {
        const auto zero = eve::zero(eve::as<wide_f>());

        std::fill(N_k_w.begin(), N_k_w.end(), zero);

        for (auto& row : sum_x_w) {
            row.fill(zero);
        }

        covariance.reset_simd_accumulators();
    }

    // TODO: Investigate why this is not inlined by the compiler
    __attribute__((always_inline)) float e_step_and_accumulate_sufficient_statistics() {
        reset_simd_accumulators();

        auto lower_bound_sum_w = eve::zero(eve::as<wide_f>());

        eve::algo::for_each[eve::algo::no_unrolling](
            samples,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                const auto sample = eve::load[ignore](it);
                const auto sample_cache = covariance.make_sample_cache(sample);

                auto max_score = wide_f(-std::numeric_limits<float>::infinity());

                for (std::size_t k = 0; k < K(); ++k) {
                    const auto score = covariance.compute_weighted_log_prob(
                        sample,
                        sample_cache,
                        means[k],
                        k
                    );
                    score_scratch[k] = score;
                    max_score = eve::max(max_score, score);
                }

                auto denom = eve::zero(eve::as<wide_f>());

                for (std::size_t k = 0; k < K(); ++k) {
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

                for (std::size_t k = 0; k < K(); ++k) {
                    const auto resp = unnormalized_resp_scratch[k] * inv_denom;

                    N_k_w[k] = eve::fma[ignore](resp, wide_f(1.0f), N_k_w[k]);
                    covariance.accumulate_second_order(k, resp, sample, sample_cache, ignore);

                    kumi::for_each_index(
                        [&](auto index, auto x) {
                            sum_x_w[k][index] = eve::fma[ignore](
                                resp,
                                x,
                                sum_x_w[k][index]
                            );
                        },
                        sample
                    );
                }
            }
        );

        return eve::reduce(lower_bound_sum_w) / static_cast<float>(samples.size());
    }

    void m_step_from_accumulators() {
        constexpr float eps10 = 10.0f * std::numeric_limits<float>::epsilon();

        std::vector<float> N_k(K());
        float N_k_sum = 0.0f;

        for (std::size_t k = 0; k < K(); ++k) {
            N_k[k] = eve::reduce(N_k_w[k]) + eps10;
            N_k_sum += N_k[k];
        }

        for (std::size_t k = 0; k < K(); ++k) {
            const float inv_N_k = 1.0f / N_k[k];

            weights[k] = N_k[k] / N_k_sum;

            kumi::for_each_index(
                [&](auto index, auto& mean_d) {
                    mean_d = eve::reduce(sum_x_w[k][index]) * inv_N_k;
                },
                means[k]
            );

            covariance.update_cluster_from_sufficient_statistics(k, N_k[k], means[k]);
        }

        covariance.refresh_score_data(weights, means);
    }
};

template <eve::product_type SampleT, class CovarianceModel>
static_gmm_result<SampleT> run_static_gmm_em(
    const eve::algo::soa_vector<SampleT>& samples,
    std::vector<float> weights,
    std::vector<SampleT> means,
    CovarianceModel covariance,
    int max_iterations = 100,
    float tol = 1e-3f
) {
    if (max_iterations < 0) {
        throw std::runtime_error("GMM max_iterations must be non-negative");
    }

    static_gmm_em_state<SampleT, CovarianceModel> state{
        samples,
        std::move(weights),
        std::move(means),
        std::move(covariance)
    };

    static_gmm_result<SampleT> result;
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
    result.covariances = std::move(state.covariance.covariances);
    result.precisions = std::move(state.covariance.precisions);

    return result;
}
