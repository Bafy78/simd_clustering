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
#include "../covariance_math.hpp"

template <eve::product_type SampleT>
float gmm_sample_norm_sq(const SampleT& sample) {
    return kumi::inner_product(sample, sample, 0.0f);
}

template <eve::product_type SimdSampleT, eve::product_type SampleT>
wide_f gmm_dot_simd_sample_with_mean(const SimdSampleT& sample, const SampleT& mean) {
    auto dot = wide_zero_f;

    kumi::for_each(
        [&](auto x, auto mu) {
            dot = eve::fma(x, wide_f(mu), dot);
        },
        sample,
        mean
    );

    return dot;
}

template <eve::product_type SimdSampleT>
wide_f gmm_simd_sample_norm_sq(const SimdSampleT& sample) {
    auto norm_sq = wide_zero_f;

    kumi::for_each(
        [&](auto x) {
            norm_sq = eve::fma(x, x, norm_sq);
        },
        sample
    );

    return norm_sq;
}


template <eve::product_type SampleT>
struct spherical_covariance_model {
    std::vector<float> precisions;
    std::vector<float> score_constants;
    std::vector<wide_f> sum_x2_w;
    float reg_covar = 1e-6f;

    struct sample_cache {
        wide_f norm_sq;
    };

    spherical_covariance_model(
        std::vector<float> precisions_,
        std::size_t K,
        float reg_covar_ = 1e-6f
    )
        : precisions(std::move(precisions_)),
          score_constants(K),
          sum_x2_w(K),
          reg_covar(reg_covar_) {}

    void validate_inputs(std::size_t K) const {
        if (precisions.size() != K) {
            throw std::runtime_error("Spherical GMM precision count must match cluster count");
        }

        for (float precision : precisions) {
            if (!(precision > 0.0f)) {
                throw std::runtime_error("Spherical GMM precisions must be strictly positive");
            }
        }
    }

    std::vector<float> materialize_covariances() const {
        return gmm::materialize_reciprocal_covariances(precisions);
    }

    template <class Weights, class Means>
    void refresh_score_data(const Weights& weights, const Means& means) {
        const float log_2_pi = gmm::log_2pi();

        for (std::size_t k = 0; k < means.size(); ++k) {
            const float mean_norm = gmm_sample_norm_sq(means[k]);

            score_constants[k] = gmm::weighted_gaussian_score_constant(
                weights[k],
                static_cast<float>(kumi::size_v<SampleT>) * std::log(precisions[k]),
                kumi::size_v<SampleT>,
                precisions[k] * mean_norm,
                log_2_pi
            );
        }
    }

    void reset_simd_accumulators() {
        const auto zero = wide_zero_f;
        std::fill(sum_x2_w.begin(), sum_x2_w.end(), zero);
    }

    template <eve::product_type SimdSampleT>
    sample_cache make_sample_cache(const SimdSampleT& sample) const {
        return sample_cache{gmm_simd_sample_norm_sq(sample)};
    }

    template <eve::product_type SimdSampleT>
    wide_f compute_weighted_log_prob(
        const SimdSampleT& sample,
        const sample_cache& cache,
        const SampleT& mean,
        std::size_t k
    ) const {
        const auto dot = gmm_dot_simd_sample_with_mean(sample, mean);

        return eve::fma(
            wide_f(-0.5f * precisions[k]),
            cache.norm_sq,
            eve::fma(wide_f(precisions[k]), dot, wide_f(score_constants[k]))
        );
    }

    template <eve::product_type SimdSampleT, class Ignore>
    void accumulate_second_order(
        std::size_t k,
        wide_f resp,
        const SimdSampleT&,
        const sample_cache& cache,
        Ignore ignore
    ) {
        sum_x2_w[k] = eve::fma[ignore](resp, cache.norm_sq, sum_x2_w[k]);
    }

    void update_cluster_from_sufficient_statistics(
        std::size_t k,
        float N_k,
        const SampleT& mean
    ) {
        const float mean_norm = gmm_sample_norm_sq(mean);
        const float avg_x2 = eve::reduce(sum_x2_w[k]) / N_k;
        const float covariance = gmm::spherical_covariance_from_raw_norm_moment(
            avg_x2,
            mean_norm,
            kumi::size_v<SampleT>,
            reg_covar
        );

        precisions[k] = gmm::checked_precision_from_covariance(
            covariance,
            "Spherical GMM covariance became non-positive; try increasing reg_covar"
        );
    }

    template <class Samples, class ComponentCounts, class Means, class Scratch>
    void recompute_unstable_clusters(
        const Samples&,
        const ComponentCounts&,
        const Means&,
        Scratch&
    ) {}
};
