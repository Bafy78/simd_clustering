#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace gmm {

inline float log_2pi() {
    return std::log(2.0f * std::numbers::pi_v<float>);
}

inline float weighted_gaussian_score_constant(
    float weight,
    float log_precision_det,
    std::size_t dimensions,
    float mean_precision_mean,
    float log_2_pi
) {
    return std::log(weight)
        + 0.5f * log_precision_det
        - 0.5f * static_cast<float>(dimensions) * log_2_pi
        - 0.5f * mean_precision_mean;
}

inline float weighted_gaussian_score_constant(
    float weight,
    float log_precision_det,
    std::size_t dimensions,
    float mean_precision_mean
) {
    return weighted_gaussian_score_constant(
        weight,
        log_precision_det,
        dimensions,
        mean_precision_mean,
        log_2pi()
    );
}

inline float diagonal_score_quadratic(float precision) {
    return -0.5f * precision;
}

inline float diagonal_score_linear(float mean, float precision) {
    return mean * precision;
}

inline float spherical_score_linear(float mean, float precision) {
    return mean * precision;
}

inline float diagonal_covariance_from_raw_second_moment(
    float avg_x2,
    float mean,
    float reg_covar
) {
    return avg_x2 - mean * mean + reg_covar;
}

inline float spherical_covariance_from_raw_norm_moment(
    float avg_norm_x2,
    float mean_norm,
    std::size_t dimensions,
    float reg_covar
) {
    return (avg_norm_x2 - mean_norm) / static_cast<float>(dimensions) + reg_covar;
}

inline float checked_precision_from_covariance(float covariance, const char* error_message) {
    if (!(covariance > 0.0f)) {
        throw std::runtime_error(error_message);
    }

    return 1.0f / covariance;
}

template<class Precisions>
std::vector<float> materialize_reciprocal_covariances(const Precisions& precisions) {
    std::vector<float> out(precisions.size());

    for (std::size_t i = 0; i < precisions.size(); ++i) {
        out[i] = 1.0f / precisions[i];
    }

    return out;
}


constexpr std::size_t full_covariance_triangle_size(std::size_t dimensions) {
    return dimensions * (dimensions + 1) / 2;
}

constexpr std::size_t full_covariance_triangle_offset(std::size_t row, std::size_t col) {
    return row * (row + 1) / 2 + col;
}

inline float full_covariance_score_quadratic_coefficient(
    std::size_t row,
    std::size_t col,
    float precision_value
) {
    return (row == col) ? -0.5f * precision_value : -precision_value;
}

inline bool full_covariance_diagonal_variance_is_suspicious(
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

template<std::size_t D>
float cholesky_decompose_spd_matrix(
    std::span<const float, D * D> matrix,
    std::array<float, D * D>& cholesky,
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

template<std::size_t D>
float logdet_spd_matrix(std::span<const float, D * D> matrix, const char* label) {
    std::array<float, D * D> cholesky{};
    return cholesky_decompose_spd_matrix<D>(matrix, cholesky, label);
}

template<std::size_t D>
float invert_spd_matrix(
    std::span<const float, D * D> matrix,
    std::span<float, D * D> inverse,
    const char* label
) {
    std::array<float, D * D> cholesky{};
    std::array<float, D * D> inverse_cholesky{};

    const float logdet = cholesky_decompose_spd_matrix<D>(matrix, cholesky, label);

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

template<std::size_t D>
float invert_spd_matrix(
    std::span<float, D * D> matrix,
    std::span<float, D * D> inverse,
    const char* label
) {
    return invert_spd_matrix<D>(std::span<const float, D * D>(matrix.data(), matrix.size()), inverse, label);
}

} // namespace gmm
