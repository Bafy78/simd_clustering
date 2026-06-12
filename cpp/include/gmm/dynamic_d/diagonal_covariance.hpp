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
//     constant[k]
//   + sum_d linear[k,d]    * x[d]
//   + sum_d quadratic[k,d] * x[d]^2
//
// where:
//     linear[k,d]    = mean[k,d] * precision[k,d]
//     quadratic[k,d] = -0.5 * precision[k,d]
//
// Both coefficient arrays are tile-major:
//     packed_*[tile][d][k_in_tile]
template<std::size_t D, std::size_t K_TILE>
struct dynamic_diagonal_gmm_micro_gemm_covariance {
    static constexpr bool requires_sample_norm_sq = false;

    aligned_float_vector precisions;
    aligned_float_vector score_constants;
    aligned_float_vector packed_linear;
    aligned_float_vector packed_quadratic;
    std::vector<wide_f> sum_x2_w;

    std::size_t K = 0;
    std::size_t K_tile_count = 0;
    float reg_covar = 1e-6f;

    template<std::size_t ACTIVE_N_VECTORS>
    struct score_cache {};

    dynamic_diagonal_gmm_micro_gemm_covariance() = default;

    dynamic_diagonal_gmm_micro_gemm_covariance(
        std::vector<float> precisions_,
        std::size_t K_,
        float reg_covar_ = 1e-6f
    )
        : precisions(precisions_.begin(), precisions_.end()),
          score_constants(K_),
          sum_x2_w(K_ * D),
          K(K_),
          K_tile_count((K_ + K_TILE - 1) / K_TILE),
          reg_covar(reg_covar_) {
        packed_linear.resize(K_tile_count * D * K_TILE);
        packed_quadratic.resize(K_tile_count * D * K_TILE);
    }

    static std::size_t offset(std::size_t k, std::size_t d) {
        return k * D + d;
    }

    void validate_inputs(std::size_t expected_K) const {
        if (K != expected_K) {
            throw std::runtime_error("Diagonal dynamic GMM covariance cluster count mismatch");
        }

        if (precisions.size() != expected_K * D) {
            throw std::runtime_error(
                "Diagonal dynamic GMM precision count must be cluster count times dimension count"
            );
        }

        if (score_constants.size() != expected_K) {
            throw std::runtime_error("Diagonal dynamic GMM score constant count must match cluster count");
        }

        if (sum_x2_w.size() != expected_K * D) {
            throw std::runtime_error(
                "Diagonal dynamic GMM accumulator count must be cluster count times dimension count"
            );
        }

        for (float precision : precisions) {
            if (!(precision > 0.0f)) {
                throw std::runtime_error("Diagonal dynamic GMM precisions must be strictly positive");
            }
        }
    }

    std::vector<float> materialize_covariances() const {
        return gmm::materialize_reciprocal_covariances(precisions);
    }

    std::vector<float> materialize_precisions() const {
        return std::vector<float>(precisions.begin(), precisions.end());
    }

    float* linear_tile_dimension(std::size_t tile, std::size_t d) {
        return packed_linear.data() + (tile * D + d) * K_TILE;
    }

    const float* linear_tile_dimension(std::size_t tile, std::size_t d) const {
        return packed_linear.data() + (tile * D + d) * K_TILE;
    }

    const float* linear_tile_dimension_for_k0(std::size_t k0, std::size_t d) const {
        return linear_tile_dimension(k0 / K_TILE, d);
    }

    float* quadratic_tile_dimension(std::size_t tile, std::size_t d) {
        return packed_quadratic.data() + (tile * D + d) * K_TILE;
    }

    const float* quadratic_tile_dimension(std::size_t tile, std::size_t d) const {
        return packed_quadratic.data() + (tile * D + d) * K_TILE;
    }

    const float* quadratic_tile_dimension_for_k0(std::size_t k0, std::size_t d) const {
        return quadratic_tile_dimension(k0 / K_TILE, d);
    }

    void refresh_score_data(
        const std::vector<float>& weights,
        const aligned_float_vector& means_row_major
    ) {
        if (weights.size() != K) {
            throw std::runtime_error("Diagonal dynamic GMM weight count must match cluster count");
        }

        if (means_row_major.size() != K * D) {
            throw std::runtime_error("Diagonal dynamic GMM mean count must be cluster count times dimension count");
        }

        const float log_2_pi = gmm::log_2pi();

        for (std::size_t tile = 0; tile < K_tile_count; ++tile) {
            const std::size_t k_base = tile * K_TILE;

            for (std::size_t t = 0; t < K_TILE; ++t) {
                const std::size_t k = k_base + t;

                if (k >= K) {
                    continue;
                }

                float log_precision_det = 0.0f;
                float mean_quadratic = 0.0f;

                for (std::size_t d = 0; d < D; ++d) {
                    const float precision = precisions[offset(k, d)];
                    const float mean = means_row_major[offset(k, d)];
                    const float linear = gmm::diagonal_score_linear(mean, precision);

                    linear_tile_dimension(tile, d)[t] = linear;
                    quadratic_tile_dimension(tile, d)[t] = gmm::diagonal_score_quadratic(precision);

                    log_precision_det += std::log(precision);
                    mean_quadratic += mean * linear;
                }

                score_constants[k] = gmm::weighted_gaussian_score_constant(
                    weights[k],
                    log_precision_det,
                    D,
                    mean_quadratic,
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
        samples_soa_view<D>,
        std::size_t,
        Ignore
    ) const {
        return score_cache<ACTIVE_N_VECTORS>{};
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void score_tile(
        samples_soa_view<D> samples,
        std::size_t n,
        std::size_t k0,
        std::size_t k_count,
        const score_cache<ACTIVE_N_VECTORS>&,
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
            const float* linear = linear_tile_dimension_for_k0(k0, d);
            const float* quadratic = quadratic_tile_dimension_for_k0(k0, d);

            std::array<wide_f, ACTIVE_N_VECTORS> x;
            std::array<wide_f, ACTIVE_N_VECTORS> x2;

            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                x[sample_vector] = eve::load[ignore](
                    eve::as_aligned(sample_dimension + n + sample_vector * card)
                );
                x2[sample_vector] = x[sample_vector] * x[sample_vector];
            }

            for (std::size_t t = 0; t < k_count; ++t) {
                const wide_f linear_coeff(linear[t]);
                const wide_f quadratic_coeff(quadratic[t]);

                for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                    scores[sample_vector][t] = eve::fma(
                        quadratic_coeff,
                        x2[sample_vector],
                        eve::fma(
                            linear_coeff,
                            x[sample_vector],
                            scores[sample_vector][t]
                        )
                    );
                }
            }
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void accumulate_cluster_sample_cache(
        std::size_t,
        const std::array<wide_f, ACTIVE_N_VECTORS>&,
        const score_cache<ACTIVE_N_VECTORS>&,
        Ignore
    ) {
        // Diagonal covariance accumulates second order terms per dimension below.
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void accumulate_dimension_second_order(
        std::size_t k,
        std::size_t d,
        const std::array<wide_f, ACTIVE_N_VECTORS>& resp,
        const std::array<wide_f, ACTIVE_N_VECTORS>&,
        const std::array<wide_f, ACTIVE_N_VECTORS>& x2,
        Ignore ignore
    ) {
        const std::size_t i = offset(k, d);

        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            sum_x2_w[i] = eve::fma[ignore](
                resp[sample_vector],
                x2[sample_vector],
                sum_x2_w[i]
            );
        }
    }

    void update_cluster_from_sufficient_statistics(
        std::size_t k,
        float N_k,
        const aligned_float_vector& means_row_major
    ) {
        for (std::size_t d = 0; d < D; ++d) {
            const std::size_t i = offset(k, d);
            const float avg_x2 = eve::reduce(sum_x2_w[i]) / N_k;
            const float mean = means_row_major[i];
            const float covariance = gmm::diagonal_covariance_from_raw_second_moment(
                avg_x2,
                mean,
                reg_covar
            );

            precisions[i] = gmm::checked_precision_from_covariance(
                covariance,
                "Diagonal dynamic GMM covariance became non-positive; try increasing reg_covar"
            );
        }
    }

    template <class Samples, class ComponentCounts, class Means, class Scratch>
    void recompute_unstable_clusters(
        const Samples&,
        const ComponentCounts&,
        const Means&,
        Scratch&
    ) {}
};
