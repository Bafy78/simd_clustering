#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>

#include "../../layout/dynamic_soa.hpp"
#include "../../simd.hpp"
#include "../core.hpp"

template<std::size_t D>
struct dynamic_gmm_result {
    std::vector<float> weights;
    aligned_float_vector means_row_major;
    std::vector<float> covariances;
    std::vector<float> precisions;
    std::vector<float> lower_bounds;
    int algorithm_iterations = 0;
    float lower_bound = -std::numeric_limits<float>::infinity();
};

template<
    std::size_t D,
    std::size_t N_VECTORS,
    std::size_t K_TILE,
    class CovariancePolicy
>
struct dynamic_gmm_micro_gemm_em_state {
    static_assert(N_VECTORS > 0);
    static_assert(K_TILE > 0);

    samples_soa_view<D> samples;
    std::vector<float> weights;
    aligned_float_vector means_row_major;
    CovariancePolicy covariance;

    // score_scratch[k][sample_vector] stores either the weighted log-probability
    // or, after normalization pass 1, exp(score - max_score).
    std::vector<wide_f> score_scratch;

    std::vector<wide_f> N_k_w;
    std::vector<wide_f> sum_x_w;

    dynamic_gmm_micro_gemm_em_state(
        samples_soa_view<D> samples_,
        std::vector<float> weights_,
        aligned_float_vector means_row_major_,
        CovariancePolicy covariance_
    )
        : samples(samples_),
          weights(std::move(weights_)),
          means_row_major(std::move(means_row_major_)),
          covariance(std::move(covariance_)),
          score_scratch(weights.size() * N_VECTORS),
          N_k_w(weights.size()),
          sum_x_w(weights.size() * D) {
        validate_inputs();
        covariance.refresh_score_data(weights, means_row_major);
    }

    std::size_t K() const {
        return weights.size();
    }

    std::size_t score_scratch_offset(std::size_t k, std::size_t sample_vector) const {
        return k * N_VECTORS + sample_vector;
    }

    static std::size_t mean_offset(std::size_t k, std::size_t d) {
        return k * D + d;
    }

    void validate_inputs() const {
        const std::size_t K_value = K();

        if (samples.N == 0) {
            throw std::runtime_error("Dynamic GMM requires at least one sample");
        }

        if (K_value == 0) {
            throw std::runtime_error("Dynamic GMM requires at least one cluster");
        }

        if (means_row_major.size() != K_value * D) {
            throw std::runtime_error(
                "Dynamic GMM mean count must be cluster count times dimension count"
            );
        }

        for (float weight : weights) {
            if (!(weight > 0.0f)) {
                throw std::runtime_error("Dynamic GMM weights must be strictly positive");
            }
        }

        covariance.validate_inputs(K_value);
    }

    void reset_simd_accumulators() {
        std::fill(N_k_w.begin(), N_k_w.end(), wide_zero_f);
        std::fill(sum_x_w.begin(), sum_x_w.end(), wide_zero_f);
        covariance.reset_simd_accumulators();
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void score_all_components_for_sample_group(
        std::size_t n,
        Ignore ignore,
        const typename CovariancePolicy::template score_cache<ACTIVE_N_VECTORS>& cache,
        std::array<wide_f, ACTIVE_N_VECTORS>& max_score
    ) {
        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            max_score[sample_vector] = wide_f(-std::numeric_limits<float>::infinity());
        }

        std::size_t k0 = 0;

        for (; k0 + K_TILE <= K(); k0 += K_TILE) {
            score_component_tile<ACTIVE_N_VECTORS>(
                n,
                k0,
                K_TILE,
                cache,
                ignore,
                max_score
            );
        }

        const std::size_t remaining = K() - k0;
        if (remaining != 0) {
            score_component_tile<ACTIVE_N_VECTORS>(
                n,
                k0,
                remaining,
                cache,
                ignore,
                max_score
            );
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void score_component_tile(
        std::size_t n,
        std::size_t k0,
        std::size_t k_count,
        const typename CovariancePolicy::template score_cache<ACTIVE_N_VECTORS>& cache,
        Ignore ignore,
        std::array<wide_f, ACTIVE_N_VECTORS>& max_score
    ) {
        std::array<std::array<wide_f, K_TILE>, ACTIVE_N_VECTORS> scores;

        covariance.template score_tile<ACTIVE_N_VECTORS>(
            samples,
            n,
            k0,
            k_count,
            cache,
            ignore,
            scores
        );

        for (std::size_t t = 0; t < k_count; ++t) {
            const std::size_t k = k0 + t;

            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                const auto score = scores[sample_vector][t];
                score_scratch[score_scratch_offset(k, sample_vector)] = score;
                max_score[sample_vector] = eve::max(max_score[sample_vector], score);
            }
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void normalize_scores_for_sample_group(
        Ignore ignore,
        const std::array<wide_f, ACTIVE_N_VECTORS>& max_score,
        std::array<wide_f, ACTIVE_N_VECTORS>& denom,
        wide_f& lower_bound_sum_w
    ) {
        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            denom[sample_vector] = wide_zero_f;
        }

        for (std::size_t k = 0; k < K(); ++k) {
            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                const std::size_t offset = score_scratch_offset(k, sample_vector);
                const auto unnormalized_resp = eve::exp(
                    score_scratch[offset] - max_score[sample_vector]
                );

                score_scratch[offset] = unnormalized_resp;
                denom[sample_vector] += unnormalized_resp;
            }
        }

        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            const auto log_prob_norm = max_score[sample_vector] + eve::log(denom[sample_vector]);

            lower_bound_sum_w = eve::fma[ignore.else_(lower_bound_sum_w)](
                log_prob_norm,
                wide_f(1.0f),
                lower_bound_sum_w
            );
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void accumulate_sufficient_statistics_for_sample_group(
        std::size_t n,
        Ignore ignore,
        const typename CovariancePolicy::template score_cache<ACTIVE_N_VECTORS>& cache,
        const std::array<wide_f, ACTIVE_N_VECTORS>& denom
    ) {
        std::array<wide_f, ACTIVE_N_VECTORS> inv_denom;
        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            inv_denom[sample_vector] = wide_f(1.0f) / denom[sample_vector];
        }

        for (std::size_t k0 = 0; k0 < K(); k0 += K_TILE) {
            const std::size_t k_count = std::min(K_TILE, K() - k0);

            std::array<std::array<wide_f, ACTIVE_N_VECTORS>, K_TILE> resp_by_component;

            for (std::size_t t = 0; t < k_count; ++t) {
                const std::size_t k = k0 + t;

                for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                    const auto resp =
                        score_scratch[score_scratch_offset(k, sample_vector)]
                        * inv_denom[sample_vector];

                    resp_by_component[t][sample_vector] = resp;

                    N_k_w[k] = eve::fma[ignore.else_(N_k_w[k])](
                        resp,
                        wide_f(1.0f),
                        N_k_w[k]
                    );
                }

                covariance.template accumulate_cluster_sample_cache<ACTIVE_N_VECTORS>(
                    k,
                    resp_by_component[t],
                    cache,
                    ignore
                );
            }

            for (std::size_t d = 0; d < D; ++d) {
                std::array<wide_f, ACTIVE_N_VECTORS> x;
                std::array<wide_f, ACTIVE_N_VECTORS> x2;

                covariance.template load_dimension_values<ACTIVE_N_VECTORS>(
                    samples,
                    n,
                    d,
                    cache,
                    ignore,
                    x,
                    x2
                );

                for (std::size_t t = 0; t < k_count; ++t) {
                    const std::size_t k = k0 + t;
                    wide_f& sum_x = sum_x_w[mean_offset(k, d)];

                    for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                        sum_x = eve::fma[ignore.else_(sum_x)](
                            resp_by_component[t][sample_vector],
                            x[sample_vector],
                            sum_x
                        );
                    }

                    covariance.template accumulate_dimension_second_order<ACTIVE_N_VECTORS>(
                        k,
                        d,
                        resp_by_component[t],
                        x,
                        x2,
                        ignore
                    );
                }
            }
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void process_sample_group(
        std::size_t n,
        Ignore ignore,
        wide_f& lower_bound_sum_w
    ) {
        auto cache = covariance.template make_score_cache<ACTIVE_N_VECTORS>(
            samples,
            n,
            ignore
        );

        std::array<wide_f, ACTIVE_N_VECTORS> max_score;
        std::array<wide_f, ACTIVE_N_VECTORS> denom;

        score_all_components_for_sample_group<ACTIVE_N_VECTORS>(
            n,
            ignore,
            cache,
            max_score
        );

        normalize_scores_for_sample_group<ACTIVE_N_VECTORS>(
            ignore,
            max_score,
            denom,
            lower_bound_sum_w
        );

        accumulate_sufficient_statistics_for_sample_group<ACTIVE_N_VECTORS>(
            n,
            ignore,
            cache,
            denom
        );
    }

    // Only makes a difference (a positive one) in lower dimensions. No drawback noticed.
    __attribute__((always_inline)) float e_step_and_accumulate_sufficient_statistics() {
        constexpr std::size_t card = simd_cardinal();
        constexpr std::size_t group_samples = N_VECTORS * card;

        reset_simd_accumulators();

        wide_f lower_bound_sum_w = wide_zero_f;

        std::size_t n = 0;

        for (; n + group_samples <= samples.N; n += group_samples) {
            process_sample_group<N_VECTORS>(
                n,
                eve::ignore_none,
                lower_bound_sum_w
            );
        }

        for (; n + card <= samples.N; n += card) {
            process_sample_group<1>(
                n,
                eve::ignore_none,
                lower_bound_sum_w
            );
        }

        if (n < samples.N) {
            const std::size_t valid = samples.N - n;
            const std::size_t ignored_lanes = card - valid;

            process_sample_group<1>(
                n,
                eve::ignore_last(ignored_lanes),
                lower_bound_sum_w
            );
        }

        return eve::reduce(lower_bound_sum_w) / static_cast<float>(samples.N);
    }

    float reduced_component_count(std::size_t k) const {
        return eve::reduce(N_k_w[k]);
    }

    float component_weight(std::size_t k) const {
        return weights[k];
    }

    void set_component_weight(std::size_t k, float value) {
        weights[k] = value;
    }

    void write_mean_from_accumulators(std::size_t k, float inv_N_k) {
        for (std::size_t d = 0; d < D; ++d) {
            means_row_major[mean_offset(k, d)] =
                eve::reduce(sum_x_w[mean_offset(k, d)]) * inv_N_k;
        }
    }

    void update_covariance_from_sufficient_statistics(std::size_t k, float N_k) {
        covariance.update_cluster_from_sufficient_statistics(
            k,
            N_k,
            means_row_major
        );
    }

    void recompute_unstable_covariances_from_current_responsibilities() {
        covariance.recompute_unstable_clusters(samples, weights, means_row_major, score_scratch);
    }

    void refresh_score_data() {
        covariance.refresh_score_data(weights, means_row_major);
    }

    void m_step_from_accumulators() {
        gmm::m_step_from_accumulators_common(*this);
    }
};

template<
    std::size_t D,
    std::size_t N_VECTORS,
    std::size_t K_TILE,
    class CovariancePolicy
>
dynamic_gmm_result<D> run_dynamic_gmm_micro_gemm_em(
    samples_soa_view<D> samples,
    std::vector<float> weights,
    aligned_float_vector means_row_major,
    CovariancePolicy covariance,
    int max_iterations = 100,
    float tol = 1e-3f
) {
    if (max_iterations < 0) {
        throw std::runtime_error("Dynamic GMM max_iterations must be non-negative");
    }

    dynamic_gmm_micro_gemm_em_state<D, N_VECTORS, K_TILE, CovariancePolicy> state{
        samples,
        std::move(weights),
        std::move(means_row_major),
        std::move(covariance)
    };

    auto trace = gmm::em_core(state, max_iterations, tol);

    dynamic_gmm_result<D> result;
    result.lower_bounds = std::move(trace.lower_bounds);
    result.algorithm_iterations = trace.algorithm_iterations;
    result.lower_bound = trace.lower_bound;

    result.weights = std::move(state.weights);
    result.means_row_major = std::move(state.means_row_major);
    result.covariances = state.covariance.materialize_covariances();
    result.precisions = state.covariance.materialize_precisions();

    return result;
}
