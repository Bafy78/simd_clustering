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

} // namespace gmm
