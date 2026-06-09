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

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>

#include "../../layout/static_soa.hpp"
#include "../../simd.hpp"

template <eve::product_type SampleT>
struct full_covariance_model {
    static constexpr std::size_t D = kumi::size_v<SampleT>;
    static constexpr std::size_t Tri = D * (D + 1) / 2;

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
    float reg_covar = 1e-6f;

    struct sample_cache {
        simd_sum_triangle xx;
    };

    full_covariance_model(
        std::vector<float> precisions_,
        std::size_t K,
        float reg_covar_ = 1e-6f
    )
        : precisions(std::move(precisions_)),
          covariances(K * D * D),
          score_data(K),
          sum_xx_w(K),
          log_precision_dets(K, -std::numeric_limits<float>::infinity()),
          covariance_current(K, 0),
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
        const float log_2_pi = std::log(2.0f * std::numbers::pi_v<float>);
        constexpr float half_dimensions = 0.5f * static_cast<float>(D);

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

            score_row.constant =
                std::log(weights[k])
                + 0.5f * log_precision_dets[k]
                - half_dimensions * log_2_pi
                - 0.5f * mean_quadratic;
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

        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                const float avg_xx = eve::reduce(
                    sum_xx_w[k][triangle_offset(row, col)]
                ) / N_k;

                float covariance_value = avg_xx - mean_values[row] * mean_values[col];

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
            "Full GMM covariance"
        );
        log_precision_dets[k] = -covariance_logdet;
        covariance_current[k] = 1;
    }

private:
    std::size_t K() const {
        return score_data.size();
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
                            + " is not positive definite; try increasing reg_covar"
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
