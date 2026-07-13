#pragma once

#include <algorithm>
#include <array>
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

#include "../../layout/dynamic_soa.hpp"
#include "../../simd.hpp"
#include "../covariance_math.hpp"

// Score form:
//     log w_k
//   + D/2 log precision_k
//   - D/2 log(2*pi)
//   - 0.5 precision_k ||mu_k||^2
//   + precision_k x.mu_k
//   - 0.5 precision_k ||x||^2
//
// The packed linear term is tile-major:
//     packed_linear[tile][d][k_in_tile] = precision[k] * mean[k,d]
template<std::size_t D, std::size_t K_TILE>
struct dynamic_spherical_gmm_micro_gemm_covariance {
    static constexpr bool requires_sample_norm_sq = true;

    aligned_float_vector precisions;
    aligned_float_vector score_constants;
    aligned_float_vector packed_linear;
    std::vector<wide_f> sum_x2_w;

    std::size_t K = 0;
    std::size_t K_tile_count = 0;
    float reg_covar = 1e-6f;

    template<std::size_t ACTIVE_N_VECTORS>
    struct score_cache {
        std::array<wide_f, ACTIVE_N_VECTORS> norm_sq;
    };

    dynamic_spherical_gmm_micro_gemm_covariance() = default;

    dynamic_spherical_gmm_micro_gemm_covariance(
        std::vector<float> precisions_,
        std::size_t K_,
        float reg_covar_ = 1e-6f
    )
        : precisions(precisions_.begin(), precisions_.end()),
          score_constants(K_),
          sum_x2_w(K_),
          K(K_),
          K_tile_count((K_ + K_TILE - 1) / K_TILE),
          reg_covar(reg_covar_) {
        packed_linear.resize(K_tile_count * D * K_TILE);
    }

    void validate_inputs(std::size_t expected_K) const {
        if (K != expected_K) {
            throw std::runtime_error("Spherical dynamic GMM covariance cluster count mismatch");
        }

        if (precisions.size() != expected_K) {
            throw std::runtime_error("Spherical dynamic GMM precision count must match cluster count");
        }

        if (score_constants.size() != expected_K) {
            throw std::runtime_error("Spherical dynamic GMM score constant count must match cluster count");
        }

        if (sum_x2_w.size() != expected_K) {
            throw std::runtime_error("Spherical dynamic GMM accumulator count must match cluster count");
        }

        for (float precision : precisions) {
            if (!(precision > 0.0f)) {
                throw std::runtime_error("Spherical dynamic GMM precisions must be strictly positive");
            }
        }
    }

    std::vector<float> materialize_covariances() const {
        return gmm::materialize_reciprocal_covariances(precisions);
    }

    std::vector<float> materialize_precisions() const {
        return std::vector<float>(precisions.begin(), precisions.end());
    }

    float* tile_dimension(std::size_t tile, std::size_t d) {
        return packed_linear.data() + (tile * D + d) * K_TILE;
    }

    const float* tile_dimension(std::size_t tile, std::size_t d) const {
        return packed_linear.data() + (tile * D + d) * K_TILE;
    }

    const float* tile_dimension_for_k0(std::size_t k0, std::size_t d) const {
        return tile_dimension(k0 / K_TILE, d);
    }

    void refresh_score_data(
        const std::vector<float>& weights,
        const aligned_float_vector& means_row_major
    ) {
        if (weights.size() != K) {
            throw std::runtime_error("Spherical dynamic GMM weight count must match cluster count");
        }

        if (means_row_major.size() != K * D) {
            throw std::runtime_error("Spherical dynamic GMM mean count must be cluster count times dimension count");
        }

        const float log_2_pi = gmm::log_2pi();

        for (std::size_t tile = 0; tile < K_tile_count; ++tile) {
            const std::size_t k_base = tile * K_TILE;

            for (std::size_t t = 0; t < K_TILE; ++t) {
                const std::size_t k = k_base + t;

                if (k >= K) {
                    continue;
                }

                float mean_norm = 0.0f;

                for (std::size_t d = 0; d < D; ++d) {
                    const float mean = means_row_major[k * D + d];
                    tile_dimension(tile, d)[t] = gmm::spherical_score_linear(mean, precisions[k]);
                    mean_norm += mean * mean;
                }

                score_constants[k] = gmm::weighted_gaussian_score_constant(
                    weights[k],
                    static_cast<float>(D) * std::log(precisions[k]),
                    D,
                    precisions[k] * mean_norm,
                    log_2_pi
                );
            }
        }
    }

    void reset_simd_accumulators() {
        std::fill(sum_x2_w.begin(), sum_x2_w.end(), wide_zero_f);
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    score_cache<ACTIVE_N_VECTORS> make_score_cache(
        samples_soa_view<D> samples,
        std::size_t n,
        Ignore ignore
    ) const {
        constexpr std::size_t card = simd_cardinal();

        score_cache<ACTIVE_N_VECTORS> cache;

        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            cache.norm_sq[sample_vector] = wide_zero_f;
        }

        for (std::size_t d = 0; d < D; ++d) {
            const float* sample_dimension = samples.dimension(d);

            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                const auto x = eve::load[ignore](
                    eve::as_aligned(sample_dimension + n + sample_vector * card)
                );

                cache.norm_sq[sample_vector] = eve::fma(
                    x,
                    x,
                    cache.norm_sq[sample_vector]
                );
            }
        }

        return cache;
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void score_tile(
        samples_soa_view<D> samples,
        std::size_t n,
        std::size_t k0,
        std::size_t k_count,
        const score_cache<ACTIVE_N_VECTORS>& cache,
        Ignore ignore,
        std::array<std::array<wide_f, K_TILE>, ACTIVE_N_VECTORS>& scores
    ) const {
        constexpr std::size_t card = simd_cardinal();

        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            for (std::size_t t = 0; t < k_count; ++t) {
                scores[sample_vector][t] = wide_f(score_constants[k0 + t]);
            }
        }

        for (std::size_t d = 0; d < D; ++d) {
            const float* sample_dimension = samples.dimension(d);
            const float* linear = tile_dimension_for_k0(k0, d);

            std::array<wide_f, ACTIVE_N_VECTORS> x;

            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                x[sample_vector] = eve::load[ignore](
                    eve::as_aligned(sample_dimension + n + sample_vector * card)
                );
            }

            for (std::size_t t = 0; t < k_count; ++t) {
                const wide_f coeff(linear[t]);

                for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                    scores[sample_vector][t] = eve::fma(
                        x[sample_vector],
                        coeff,
                        scores[sample_vector][t]
                    );
                }
            }
        }

        for (std::size_t t = 0; t < k_count; ++t) {
            const wide_f norm_coeff(-0.5f * precisions[k0 + t]);

            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                scores[sample_vector][t] = eve::fma(
                    norm_coeff,
                    cache.norm_sq[sample_vector],
                    scores[sample_vector][t]
                );
            }
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void load_dimension_values(
        samples_soa_view<D> samples,
        std::size_t n,
        std::size_t d,
        const score_cache<ACTIVE_N_VECTORS>&,
        Ignore ignore,
        std::array<wide_f, ACTIVE_N_VECTORS>& x,
        std::array<wide_f, ACTIVE_N_VECTORS>& x2
    ) const {
        constexpr std::size_t card = simd_cardinal();
        const float* sample_dimension = samples.dimension(d);

        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            x[sample_vector] = eve::load[ignore](
                eve::as_aligned(sample_dimension + n + sample_vector * card)
            );
            x2[sample_vector] = x[sample_vector] * x[sample_vector];
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void accumulate_cluster_sample_cache(
        std::size_t k,
        const std::array<wide_f, ACTIVE_N_VECTORS>& resp,
        const score_cache<ACTIVE_N_VECTORS>& cache,
        Ignore ignore
    ) {
        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            sum_x2_w[k] = eve::fma[ignore.else_(sum_x2_w[k])](
                resp[sample_vector],
                cache.norm_sq[sample_vector],
                sum_x2_w[k]
            );
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void accumulate_dimension_second_order(
        std::size_t,
        std::size_t,
        const std::array<wide_f, ACTIVE_N_VECTORS>&,
        const std::array<wide_f, ACTIVE_N_VECTORS>&,
        const std::array<wide_f, ACTIVE_N_VECTORS>&,
        Ignore
    ) {
        // Spherical covariance only needs the weighted sample norm, handled above.
    }

    void update_cluster_from_sufficient_statistics(
        std::size_t k,
        float N_k,
        const aligned_float_vector& means_row_major
    ) {
        const float* mean = means_row_major.data() + k * D;

        float mean_norm = 0.0f;
        for (std::size_t d = 0; d < D; ++d) {
            mean_norm += mean[d] * mean[d];
        }

        const float avg_x2 = eve::reduce(sum_x2_w[k]) / N_k;
        const float covariance = gmm::spherical_covariance_from_raw_norm_moment(
            avg_x2,
            mean_norm,
            D,
            reg_covar
        );

        precisions[k] = gmm::checked_precision_from_covariance(
            covariance,
            "Spherical dynamic GMM covariance became non-positive; try increasing reg_covar"
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
