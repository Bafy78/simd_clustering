#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
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
//   + sum_d linear[k,d] * x[d]
//   + sum_{row,col<=row} quadratic[k,row,col] * x[row] * x[col]
//
// where:
//     linear[k] = precision[k] * mean[k]
//     quadratic[k,row,row] = -0.5 * precision[k,row,row]
//     quadratic[k,row,col] = -precision[k,row,col] for row != col
//
// Coefficients are tile-major to reuse each loaded/cached sample term across K_TILE components:
//     packed_linear[tile][d][k_in_tile]
//     packed_quadratic[tile][lower_triangle_index][k_in_tile]
template<std::size_t D, std::size_t K_TILE>
struct dynamic_full_gmm_micro_gemm_covariance {
    static constexpr bool requires_sample_norm_sq = false;
    static constexpr std::size_t Tri = D * (D + 1) / 2;
    static constexpr float default_reg_covar = 1e-6f;

    using simd_sum_triangle = std::array<wide_f, Tri>;
    using scalar_sample = std::array<float, D>;

    aligned_float_vector precisions;
    aligned_float_vector covariances;
    aligned_float_vector score_constants;
    aligned_float_vector packed_linear;
    aligned_float_vector packed_quadratic;

    std::vector<simd_sum_triangle> sum_xx_w;
    std::vector<float> log_precision_dets;
    std::vector<unsigned char> covariance_current;
    std::vector<unsigned char> covariance_needs_stable_recompute;

    std::size_t K = 0;
    std::size_t K_tile_count = 0;
    float reg_covar = default_reg_covar;

    template<std::size_t ACTIVE_N_VECTORS>
    struct score_cache {
        std::array<std::array<wide_f, ACTIVE_N_VECTORS>, D> x;
    };

    dynamic_full_gmm_micro_gemm_covariance() = default;

    dynamic_full_gmm_micro_gemm_covariance(
        std::vector<float> precisions_,
        std::size_t K_,
        float reg_covar_ = default_reg_covar
    )
        : precisions(precisions_.begin(), precisions_.end()),
          covariances(K_ * D * D),
          score_constants(K_),
          sum_xx_w(K_),
          log_precision_dets(K_, -std::numeric_limits<float>::infinity()),
          covariance_current(K_, 0),
          covariance_needs_stable_recompute(K_, 0),
          K(K_),
          K_tile_count((K_ + K_TILE - 1) / K_TILE),
          reg_covar(reg_covar_) {
        packed_linear.resize(K_tile_count * D * K_TILE);
        packed_quadratic.resize(K_tile_count * Tri * K_TILE);
    }

    static constexpr std::size_t matrix_offset(std::size_t k, std::size_t row, std::size_t col) {
        return k * D * D + row * D + col;
    }

    static constexpr std::size_t mean_offset(std::size_t k, std::size_t d) {
        return k * D + d;
    }

    static constexpr std::size_t triangle_offset(std::size_t row, std::size_t col) {
        return gmm::full_covariance_triangle_offset(row, col);
    }

    static std::span<const float, D * D> matrix_span(
        const aligned_float_vector& matrices,
        std::size_t cluster
    ) {
        return std::span<const float, D * D>(matrices.data() + cluster * D * D, D * D);
    }

    static std::span<float, D * D> matrix_span(
        aligned_float_vector& matrices,
        std::size_t cluster
    ) {
        return std::span<float, D * D>(matrices.data() + cluster * D * D, D * D);
    }

    void validate_inputs(std::size_t expected_K) const {
        if (K != expected_K) {
            throw std::runtime_error("Full dynamic GMM covariance cluster count mismatch");
        }

        if (precisions.size() != expected_K * D * D) {
            throw std::runtime_error(
                "Full dynamic GMM precision count must be cluster count times dimension squared"
            );
        }

        if (covariances.size() != expected_K * D * D) {
            throw std::runtime_error(
                "Full dynamic GMM covariance count must be cluster count times dimension squared"
            );
        }

        if (score_constants.size() != expected_K) {
            throw std::runtime_error("Full dynamic GMM score constant count must match cluster count");
        }

        if (packed_linear.size() != K_tile_count * D * K_TILE) {
            throw std::runtime_error("Full dynamic GMM packed linear coefficient count mismatch");
        }

        if (packed_quadratic.size() != K_tile_count * Tri * K_TILE) {
            throw std::runtime_error("Full dynamic GMM packed quadratic coefficient count mismatch");
        }

        if (sum_xx_w.size() != expected_K) {
            throw std::runtime_error("Full dynamic GMM accumulator count must match cluster count");
        }

        if (log_precision_dets.size() != expected_K) {
            throw std::runtime_error("Full dynamic GMM log determinant count must match cluster count");
        }

        if (covariance_current.size() != expected_K) {
            throw std::runtime_error("Full dynamic GMM covariance state count must match cluster count");
        }

        if (covariance_needs_stable_recompute.size() != expected_K) {
            throw std::runtime_error(
                "Full dynamic GMM stable-recompute state count must match cluster count"
            );
        }
    }

    std::vector<float> materialize_covariances() const {
        std::vector<float> out(K * D * D);

        for (std::size_t k = 0; k < K; ++k) {
            auto out_covariance = std::span<float, D * D>(out.data() + k * D * D, D * D);

            if (covariance_current[k] != 0) {
                const auto covariance = matrix_span(covariances, k);
                std::copy(covariance.begin(), covariance.end(), out_covariance.begin());
            } else {
                gmm::invert_spd_matrix<D>(
                    matrix_span(precisions, k),
                    out_covariance,
                    "Full dynamic GMM precision"
                );
            }
        }

        return out;
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

    float* quadratic_tile_term(std::size_t tile, std::size_t triangle_index) {
        return packed_quadratic.data() + (tile * Tri + triangle_index) * K_TILE;
    }

    const float* quadratic_tile_term(std::size_t tile, std::size_t triangle_index) const {
        return packed_quadratic.data() + (tile * Tri + triangle_index) * K_TILE;
    }

    const float* quadratic_tile_term_for_k0(std::size_t k0, std::size_t triangle_index) const {
        return quadratic_tile_term(k0 / K_TILE, triangle_index);
    }

    void refresh_score_data(
        const std::vector<float>& weights,
        const aligned_float_vector& means_row_major
    ) {
        if (weights.size() != K) {
            throw std::runtime_error("Full dynamic GMM weight count must match cluster count");
        }

        if (means_row_major.size() != K * D) {
            throw std::runtime_error(
                "Full dynamic GMM mean count must be cluster count times dimension count"
            );
        }

        const float log_2_pi = gmm::log_2pi();

        for (std::size_t tile = 0; tile < K_tile_count; ++tile) {
            const std::size_t k_base = tile * K_TILE;

            for (std::size_t t = 0; t < K_TILE; ++t) {
                const std::size_t k = k_base + t;

                if (k >= K) {
                    continue;
                }

                const auto precision = matrix_span(precisions, k);
                const float* mean = means_row_major.data() + k * D;

                if (!std::isfinite(log_precision_dets[k])) {
                    log_precision_dets[k] = gmm::logdet_spd_matrix<D>(
                        precision,
                        "Full dynamic GMM precision"
                    );
                }

                float mean_quadratic = 0.0f;

                for (std::size_t row = 0; row < D; ++row) {
                    float linear = 0.0f;

                    for (std::size_t col = 0; col < D; ++col) {
                        linear += precision[row * D + col] * mean[col];
                    }

                    linear_tile_dimension(tile, row)[t] = linear;
                    mean_quadratic += mean[row] * linear;
                }

                for (std::size_t row = 0; row < D; ++row) {
                    for (std::size_t col = 0; col <= row; ++col) {
                        quadratic_tile_term(tile, triangle_offset(row, col))[t] =
                            gmm::full_covariance_score_quadratic_coefficient(
                                row,
                                col,
                                precision[row * D + col]
                            );
                    }
                }

                score_constants[k] = gmm::weighted_gaussian_score_constant(
                    weights[k],
                    log_precision_dets[k],
                    D,
                    mean_quadratic,
                    log_2_pi
                );
            }
        }
    }

    void reset_simd_accumulators() {
        const auto zero = wide_zero_f;

        for (auto& row : sum_xx_w) {
            row.fill(zero);
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    score_cache<ACTIVE_N_VECTORS> make_score_cache(
        samples_soa_view<D> samples,
        std::size_t n,
        Ignore ignore
    ) const {
        constexpr std::size_t card = simd_cardinal();

        score_cache<ACTIVE_N_VECTORS> cache;

        for (std::size_t d = 0; d < D; ++d) {
            const float* sample_dimension = samples.dimension(d);

            for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                cache.x[d][sample_vector] = eve::load[ignore](
                    eve::as_aligned(sample_dimension + n + sample_vector * card)
                );
            }
        }

        return cache;
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void score_tile(
        samples_soa_view<D>,
        std::size_t,
        std::size_t k0,
        std::size_t k_count,
        const score_cache<ACTIVE_N_VECTORS>& cache,
        Ignore,
        std::array<std::array<wide_f, K_TILE>, ACTIVE_N_VECTORS>& scores
    ) const {
        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            for (std::size_t t = 0; t < k_count; ++t) {
                scores[sample_vector][t] = wide_f(score_constants[k0 + t]);
            }
        }

        for (std::size_t d = 0; d < D; ++d) {
            const float* linear = linear_tile_dimension_for_k0(k0, d);

            for (std::size_t t = 0; t < k_count; ++t) {
                const wide_f linear_coeff(linear[t]);

                for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                    scores[sample_vector][t] = eve::fma(
                        linear_coeff,
                        cache.x[d][sample_vector],
                        scores[sample_vector][t]
                    );
                }
            }
        }

        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                const std::size_t triangle_index = triangle_offset(row, col);
                const float* quadratic = quadratic_tile_term_for_k0(k0, triangle_index);

                for (std::size_t t = 0; t < k_count; ++t) {
                    const wide_f quadratic_coeff(quadratic[t]);

                    for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                        scores[sample_vector][t] = eve::fma(
                            quadratic_coeff,
                            cache.x[row][sample_vector] * cache.x[col][sample_vector],
                            scores[sample_vector][t]
                        );
                    }
                }
            }
        }
    }

    template<std::size_t ACTIVE_N_VECTORS, class Ignore>
    void load_dimension_values(
        samples_soa_view<D>,
        std::size_t,
        std::size_t d,
        const score_cache<ACTIVE_N_VECTORS>& cache,
        Ignore,
        std::array<wide_f, ACTIVE_N_VECTORS>& x,
        std::array<wide_f, ACTIVE_N_VECTORS>& x2
    ) const {
        for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
            x[sample_vector] = cache.x[d][sample_vector];
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
        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                auto& accumulator = sum_xx_w[k][triangle_offset(row, col)];

                for (std::size_t sample_vector = 0; sample_vector < ACTIVE_N_VECTORS; ++sample_vector) {
                    accumulator = eve::fma[ignore](
                        resp[sample_vector],
                        cache.x[row][sample_vector] * cache.x[col][sample_vector],
                        accumulator
                    );
                }
            }
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
        // Full covariance accumulates all lower-triangular cross moments from the score cache above.
    }

    void update_cluster_from_sufficient_statistics(
        std::size_t k,
        float N_k,
        const aligned_float_vector& means_row_major
    ) {
        const float* mean = means_row_major.data() + k * D;
        auto covariance = matrix_span(covariances, k);
        auto precision = matrix_span(precisions, k);

        covariance_needs_stable_recompute[k] = 0;
        bool cancellation_risk = false;

        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                const float avg_xx = eve::reduce(
                    sum_xx_w[k][triangle_offset(row, col)]
                ) / N_k;

                const float mean_product = mean[row] * mean[col];
                float covariance_value = avg_xx - mean_product;

                if (!std::isfinite(covariance_value)) {
                    cancellation_risk = true;
                }

                if (row == col && gmm::full_covariance_diagonal_variance_is_suspicious(
                    avg_xx,
                    mean_product,
                    covariance_value
                )) {
                    cancellation_risk = true;
                }

                if (row == col) {
                    covariance_value += reg_covar;
                }

                covariance[row * D + col] = covariance_value;
                covariance[col * D + row] = covariance_value;
            }
        }

        if (cancellation_risk) {
            covariance_needs_stable_recompute[k] = 1;
            return;
        }

        try {
            const float covariance_logdet = gmm::invert_spd_matrix<D>(
                covariance,
                precision,
                "Full dynamic GMM covariance"
            );
            log_precision_dets[k] = -covariance_logdet;
            covariance_current[k] = 1;
        } catch (const std::runtime_error&) {
            covariance_needs_stable_recompute[k] = 1;
        }
    }

    template <class ComponentCounts>
    void recompute_unstable_clusters(
        samples_soa_view<D> samples,
        const ComponentCounts& component_counts,
        const aligned_float_vector& means_row_major,
        std::vector<wide_f>&
    ) {
        if (!has_clusters_requiring_stable_recompute()) {
            return;
        }

        std::vector<simd_sum_triangle> stable_sum_xx_w(K);
        std::vector<scalar_sample> stable_means(K);
        std::vector<wide_f> fallback_score_scratch(K);
        const auto zero = wide_zero_f;

        for (std::size_t k = 0; k < K; ++k) {
            if (covariance_needs_stable_recompute[k] != 0) {
                stable_sum_xx_w[k].fill(zero);
                stable_means[k] = mean_to_array(means_row_major, k);
            }
        }

        constexpr std::size_t card = simd_cardinal();

        std::size_t n = 0;
        for (; n + card <= samples.N; n += card) {
            recompute_unstable_sample_group(
                samples,
                n,
                eve::ignore_none,
                stable_means,
                stable_sum_xx_w,
                fallback_score_scratch
            );
        }

        if (n < samples.N) {
            const std::size_t valid = samples.N - n;
            const std::size_t ignored_lanes = card - valid;

            recompute_unstable_sample_group(
                samples,
                n,
                eve::ignore_last(ignored_lanes),
                stable_means,
                stable_sum_xx_w,
                fallback_score_scratch
            );
        }

        for (std::size_t k = 0; k < K; ++k) {
            if (covariance_needs_stable_recompute[k] == 0) {
                continue;
            }

            const float N_k = component_counts[k];
            const float inv_N_k = 1.0f / N_k;
            auto covariance = matrix_span(covariances, k);
            auto precision = matrix_span(precisions, k);

            for (std::size_t row = 0; row < D; ++row) {
                for (std::size_t col = 0; col <= row; ++col) {
                    float covariance_value = eve::reduce(
                        stable_sum_xx_w[k][triangle_offset(row, col)]
                    ) * inv_N_k;

                    if (row == col) {
                        covariance_value += reg_covar;
                    }

                    covariance[row * D + col] = covariance_value;
                    covariance[col * D + row] = covariance_value;
                }
            }

            const float covariance_logdet = gmm::invert_spd_matrix<D>(
                covariance,
                precision,
                "Full dynamic GMM stable covariance recompute"
            );
            log_precision_dets[k] = -covariance_logdet;
            covariance_current[k] = 1;
            covariance_needs_stable_recompute[k] = 0;
        }
    }

private:
    bool has_clusters_requiring_stable_recompute() const {
        return std::any_of(
            covariance_needs_stable_recompute.begin(),
            covariance_needs_stable_recompute.end(),
            [](unsigned char value) { return value != 0; }
        );
    }

    static scalar_sample mean_to_array(
        const aligned_float_vector& means_row_major,
        std::size_t k
    ) {
        scalar_sample out{};

        for (std::size_t d = 0; d < D; ++d) {
            out[d] = means_row_major[mean_offset(k, d)];
        }

        return out;
    }

    template<class Ignore>
    void recompute_unstable_sample_group(
        samples_soa_view<D> samples,
        std::size_t n,
        Ignore ignore,
        const std::vector<scalar_sample>& stable_means,
        std::vector<simd_sum_triangle>& stable_sum_xx_w,
        std::vector<wide_f>& fallback_score_scratch
    ) const {
        auto cache = make_score_cache<1>(samples, n, ignore);

        auto max_score = wide_f(-std::numeric_limits<float>::infinity());

        for (std::size_t k0 = 0; k0 < K; k0 += K_TILE) {
            const std::size_t k_count = std::min(K_TILE, K - k0);
            std::array<std::array<wide_f, K_TILE>, 1> scores;

            score_tile<1>(
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
                fallback_score_scratch[k] = scores[0][t];
                max_score = eve::max(max_score, scores[0][t]);
            }
        }

        auto denom = wide_zero_f;

        for (std::size_t k = 0; k < K; ++k) {
            const auto unnormalized_resp = eve::exp(fallback_score_scratch[k] - max_score);
            fallback_score_scratch[k] = unnormalized_resp;
            denom += unnormalized_resp;
        }

        const auto inv_denom = wide_f(1.0f) / denom;

        for (std::size_t k = 0; k < K; ++k) {
            if (covariance_needs_stable_recompute[k] == 0) {
                continue;
            }

            const auto resp = fallback_score_scratch[k] * inv_denom;
            const auto& mean = stable_means[k];
            std::array<wide_f, D> diff{};

            for (std::size_t d = 0; d < D; ++d) {
                diff[d] = cache.x[d][0] - wide_f(mean[d]);
            }

            for (std::size_t row = 0; row < D; ++row) {
                for (std::size_t col = 0; col <= row; ++col) {
                    auto& accumulator = stable_sum_xx_w[k][triangle_offset(row, col)];

                    accumulator = eve::fma[ignore](
                        resp,
                        diff[row] * diff[col],
                        accumulator
                    );
                }
            }
        }
    }
};
