#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
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
#include "../covariance_math.hpp"

template <eve::product_type SampleT>
struct full_covariance_model {
    static constexpr std::size_t D = kumi::size_v<SampleT>;
    static constexpr std::size_t Tri = D * (D + 1) / 2;
    static constexpr float default_reg_covar = 1e-6f;

    using simd_sum_triangle = std::array<wide_f, Tri>;
    using scalar_sample = std::array<float, D>;
    using scalar_matrix = std::array<float, D * D>;

    struct cluster_score_data {
        float constant = 0.0f;
        scalar_sample linear{};
        std::array<float, Tri> quadratic{};
    };

    std::vector<float> precisions;
    std::vector<float> covariances;
    std::vector<cluster_score_data> score_data;
    std::vector<simd_sum_triangle> sum_xx_w;
    std::vector<float> log_precision_dets;
    std::vector<unsigned char> covariance_current;
    std::vector<unsigned char> covariance_needs_stable_recompute;
    float reg_covar = default_reg_covar;

    struct sample_cache {
        simd_sum_triangle xx;
    };

    full_covariance_model(
        std::vector<float> precisions_,
        std::size_t K,
        float reg_covar_ = default_reg_covar
    )
        : precisions(std::move(precisions_)),
          covariances(K * D * D),
          score_data(K),
          sum_xx_w(K),
          log_precision_dets(K, -std::numeric_limits<float>::infinity()),
          covariance_current(K, 0),
          covariance_needs_stable_recompute(K, 0),
          reg_covar(reg_covar_) {}

    static constexpr std::size_t triangle_offset(std::size_t row, std::size_t col) {
        return row * (row + 1) / 2 + col;
    }

    static std::span<const float, D * D> matrix_span(
        const std::vector<float>& matrices,
        std::size_t cluster
    ) {
        return std::span<const float, D * D>(matrices.data() + cluster * D * D, D * D);
    }

    static std::span<float, D * D> matrix_span(
        std::vector<float>& matrices,
        std::size_t cluster
    ) {
        return std::span<float, D * D>(matrices.data() + cluster * D * D, D * D);
    }

    void validate_inputs(std::size_t K) const {
        if (precisions.size() != K * D * D) {
            throw std::runtime_error(
                "Full GMM precision count must be cluster count times dimension squared"
            );
        }

        if (covariances.size() != K * D * D) {
            throw std::runtime_error(
                "Full GMM covariance count must be cluster count times dimension squared"
            );
        }

        if (score_data.size() != K) {
            throw std::runtime_error("Full GMM score data count must match cluster count");
        }

        if (sum_xx_w.size() != K) {
            throw std::runtime_error("Full GMM accumulator count must match cluster count");
        }

        if (log_precision_dets.size() != K) {
            throw std::runtime_error("Full GMM log determinant count must match cluster count");
        }

        if (covariance_current.size() != K) {
            throw std::runtime_error("Full GMM covariance state count must match cluster count");
        }

        if (covariance_needs_stable_recompute.size() != K) {
            throw std::runtime_error(
                "Full GMM stable-recompute state count must match cluster count"
            );
        }
    }

    std::vector<float> materialize_covariances() const {
        std::vector<float> out(K() * D * D);

        for (std::size_t k = 0; k < K(); ++k) {
            auto out_covariance = matrix_span(out, k);

            if (covariance_current[k] != 0) {
                const auto covariance = matrix_span(covariances, k);
                std::copy(covariance.begin(), covariance.end(), out_covariance.begin());
            } else {
                invert_spd_matrix(
                    matrix_span(precisions, k),
                    out_covariance,
                    "Full GMM precision"
                );
            }
        }

        return out;
    }

    template <class Weights, class Means>
    void refresh_score_data(const Weights& weights, const Means& means) {
        const float log_2_pi = gmm::log_2pi();

        for (std::size_t k = 0; k < means.size(); ++k) {
            auto& score_row = score_data[k];
            const auto precision = matrix_span(precisions, k);
            const auto mean = sample_to_array(means[k]);

            if (!std::isfinite(log_precision_dets[k])) {
                log_precision_dets[k] = logdet_spd_matrix(
                    precision,
                    "Full GMM precision"
                );
            }

            float mean_quadratic = 0.0f;

            for (std::size_t row = 0; row < D; ++row) {
                float linear = 0.0f;

                for (std::size_t col = 0; col < D; ++col) {
                    linear += precision[row * D + col] * mean[col];
                }

                score_row.linear[row] = linear;
                mean_quadratic += mean[row] * linear;
            }

            // Lower-triangular quadratic coefficients for -0.5 * x^T P x.
            // Diagonal terms use -0.5 * P_ii; off-diagonal terms use -P_ij because
            // symmetric entries P_ij and P_ji contribute twice to x^T P x.
            for (std::size_t row = 0; row < D; ++row) {
                for (std::size_t col = 0; col <= row; ++col) {
                    const float coefficient = (row == col)
                        ? -0.5f * precision[row * D + col]
                        : -precision[row * D + col];

                    score_row.quadratic[triangle_offset(row, col)] = coefficient;
                }
            }

            score_row.constant = gmm::weighted_gaussian_score_constant(
                weights[k],
                log_precision_dets[k],
                D,
                mean_quadratic,
                log_2_pi
            );
        }
    }

    void reset_simd_accumulators() {
        const auto zero = wide_zero_f;

        for (auto& row : sum_xx_w) {
            row.fill(zero);
        }
    }

    template <eve::product_type SimdSampleT>
    sample_cache make_sample_cache(const SimdSampleT& sample) const {
        sample_cache cache;

        kumi::for_each_index(
            [&](auto row_index, auto x_row) {
                constexpr std::size_t row = decltype(row_index)::value;

                kumi::for_each_index(
                    [&](auto col_index, auto x_col) {
                        constexpr std::size_t col = decltype(col_index)::value;

                        if constexpr (col <= row) {
                            cache.xx[triangle_offset(row, col)] = x_row * x_col;
                        }
                    },
                    sample
                );
            },
            sample
        );

        return cache;
    }

    template <eve::product_type SimdSampleT>
    wide_f compute_weighted_log_prob(
        const SimdSampleT& sample,
        const sample_cache& cache,
        const SampleT&,
        std::size_t k
    ) const {
        const auto& score_row = score_data[k];
        auto score = wide_f(score_row.constant);

        kumi::for_each_index(
            [&](auto index, auto x) {
                score = eve::fma(wide_f(score_row.linear[index]), x, score);
            },
            sample
        );

        for (std::size_t t = 0; t < Tri; ++t) {
            score = eve::fma(wide_f(score_row.quadratic[t]), cache.xx[t], score);
        }

        return score;
    }

    template <eve::product_type SimdSampleT, class Ignore>
    void accumulate_second_order(
        std::size_t k,
        wide_f resp,
        const SimdSampleT&,
        const sample_cache& cache,
        Ignore ignore
    ) {
        for (std::size_t t = 0; t < Tri; ++t) {
            sum_xx_w[k][t] = eve::fma[ignore](resp, cache.xx[t], sum_xx_w[k][t]);
        }
    }

    void update_cluster_from_sufficient_statistics(
        std::size_t k,
        float N_k,
        const SampleT& mean
    ) {
        const auto mean_values = sample_to_array(mean);
        auto covariance = matrix_span(covariances, k);
        auto precision = matrix_span(precisions, k);

        covariance_needs_stable_recompute[k] = 0;
        bool cancellation_risk = false;

        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                const float avg_xx = eve::reduce(
                    sum_xx_w[k][triangle_offset(row, col)]
                ) / N_k;

                const float mean_product = mean_values[row] * mean_values[col];
                float covariance_value = avg_xx - mean_product;

                if (!std::isfinite(covariance_value)) {
                    cancellation_risk = true;
                }

                if (row == col && diagonal_variance_is_suspicious(
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
            const float covariance_logdet = invert_spd_matrix(
                covariance,
                precision,
                "Full GMM covariance"
            );
            log_precision_dets[k] = -covariance_logdet;
            covariance_current[k] = 1;
        } catch (const std::runtime_error&) {
            covariance_needs_stable_recompute[k] = 1;
        }
    }

    template <class Samples, class ComponentCounts, class Means, class Scratch>
    void recompute_unstable_clusters(
        const Samples& samples,
        const ComponentCounts& component_counts,
        const Means& means,
        Scratch& score_scratch
    ) {
        if (!has_clusters_requiring_stable_recompute()) {
            return;
        }

        std::vector<simd_sum_triangle> stable_sum_xx_w(K());
        std::vector<scalar_sample> stable_means(K());
        const auto zero = wide_zero_f;

        for (std::size_t k = 0; k < K(); ++k) {
            if (covariance_needs_stable_recompute[k] != 0) {
                stable_sum_xx_w[k].fill(zero);
                stable_means[k] = sample_to_array(means[k]);
            }
        }

        eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
            samples,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                const auto sample = eve::load[ignore](it);
                const auto sample_cache = make_sample_cache(sample);

                auto max_score = wide_f(-std::numeric_limits<float>::infinity());

                for (std::size_t k = 0; k < K(); ++k) {
                    const auto score = compute_weighted_log_prob(
                        sample,
                        sample_cache,
                        means[k],
                        k
                    );
                    score_scratch[k] = score;
                    max_score = eve::max(max_score, score);
                }

                auto denom = wide_zero_f;

                for (std::size_t k = 0; k < K(); ++k) {
                    const auto unnormalized_resp = eve::exp(score_scratch[k] - max_score);
                    score_scratch[k] = unnormalized_resp;
                    denom += unnormalized_resp;
                }

                const auto inv_denom = wide_f(1.0f) / denom;

                for (std::size_t k = 0; k < K(); ++k) {
                    if (covariance_needs_stable_recompute[k] == 0) {
                        continue;
                    }

                    const auto resp = score_scratch[k] * inv_denom;
                    const auto& mean_values = stable_means[k];
                    std::array<wide_f, D> diff{};

                    kumi::for_each_index(
                        [&](auto index, auto x) {
                            diff[index] = x - wide_f(mean_values[index]);
                        },
                        sample
                    );

                    for (std::size_t row = 0; row < D; ++row) {
                        for (std::size_t col = 0; col <= row; ++col) {
                            const auto diff_product = diff[row] * diff[col];
                            auto& accumulator = stable_sum_xx_w[k][triangle_offset(row, col)];

                            accumulator = eve::fma[ignore](resp, diff_product, accumulator);
                        }
                    }
                }
            }
        );

        for (std::size_t k = 0; k < K(); ++k) {
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

            const float covariance_logdet = invert_spd_matrix(
                covariance,
                precision,
                "Full GMM stable covariance recompute"
            );
            log_precision_dets[k] = -covariance_logdet;
            covariance_current[k] = 1;
            covariance_needs_stable_recompute[k] = 0;
        }
    }

private:
    std::size_t K() const {
        return score_data.size();
    }

    bool has_clusters_requiring_stable_recompute() const {
        return std::any_of(
            covariance_needs_stable_recompute.begin(),
            covariance_needs_stable_recompute.end(),
            [](unsigned char value) { return value != 0; }
        );
    }

    static bool diagonal_variance_is_suspicious(
        float avg_xx,
        float mean_square,
        float raw_variance
    ) {
        if (
            !std::isfinite(avg_xx)
            || !std::isfinite(mean_square)
            || !std::isfinite(raw_variance)
        ) {
            return true;
        }

        if (!(raw_variance > 0.0f)) {
            return true;
        }

        const float cancellation_scale = std::max(std::abs(avg_xx), std::abs(mean_square));

        if (!(cancellation_scale > 1.0f)) {
            return false;
        }

        constexpr float raw_moment_error_factor = 64.0f;
        constexpr float tolerated_covariance_relative_error = 1e-2f;

        constexpr float cancellation_threshold =
            raw_moment_error_factor * std::numeric_limits<float>::epsilon()
            / tolerated_covariance_relative_error;

        return std::abs(raw_variance) / cancellation_scale < cancellation_threshold;
    }

    template <eve::product_type AnySampleT>
    static scalar_sample sample_to_array(const AnySampleT& sample) {
        scalar_sample out{};

        kumi::for_each_index(
            [&](auto index, auto value) {
                out[index] = static_cast<float>(value);
            },
            sample
        );

        return out;
    }

    static float logdet_spd_matrix(std::span<const float, D * D> matrix, const char* label) {
        scalar_matrix cholesky{};
        return cholesky_decompose(matrix, cholesky, label);
    }

    static float invert_spd_matrix(
        std::span<const float, D * D> matrix,
        std::span<float, D * D> inverse,
        const char* label
    ) {
        scalar_matrix cholesky{};
        scalar_matrix inverse_cholesky{};

        const float logdet = cholesky_decompose(matrix, cholesky, label);

        for (std::size_t i = 0; i < D; ++i) {
            inverse_cholesky[i * D + i] = 1.0f / cholesky[i * D + i];

            for (std::size_t j = 0; j < i; ++j) {
                float sum = 0.0f;

                for (std::size_t m = j; m < i; ++m) {
                    sum += cholesky[i * D + m] * inverse_cholesky[m * D + j];
                }

                inverse_cholesky[i * D + j] = -sum / cholesky[i * D + i];
            }
        }

        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                float value = 0.0f;

                for (std::size_t m = row; m < D; ++m) {
                    value += inverse_cholesky[m * D + row] * inverse_cholesky[m * D + col];
                }

                inverse[row * D + col] = value;
                inverse[col * D + row] = value;
            }
        }

        return logdet;
    }

    static float cholesky_decompose(
        std::span<const float, D * D> matrix,
        scalar_matrix& cholesky,
        const char* label
    ) {
        cholesky.fill(0.0f);

        float logdet = 0.0f;

        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                float sum = matrix[row * D + col];

                for (std::size_t m = 0; m < col; ++m) {
                    sum -= cholesky[row * D + m] * cholesky[col * D + m];
                }

                if (row == col) {
                    if (!(sum > 0.0f) || !std::isfinite(sum)) {
                        throw std::runtime_error(
                            std::string(label)
                            + " is not positive definite while computing Cholesky pivot at index "
                            + std::to_string(row)
                            + "; pivot value = "
                            + std::to_string(sum)
                            + "; try increasing reg_covar"
                        );
                    }

                    cholesky[row * D + col] = std::sqrt(sum);
                    logdet += 2.0f * std::log(cholesky[row * D + col]);
                } else {
                    cholesky[row * D + col] = sum / cholesky[col * D + col];
                }
            }
        }

        return logdet;
    }
};
