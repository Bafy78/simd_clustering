#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

enum class gmm_covariance_type {
    spherical,
    diag,
    full,
    tied
};

inline gmm_covariance_type parse_gmm_covariance_type(std::string_view value) {
    if (value == "spherical") {
        return gmm_covariance_type::spherical;
    }

    if (value == "diag") {
        return gmm_covariance_type::diag;
    }

    if (value == "full") {
        return gmm_covariance_type::full;
    }

    if (value == "tied") {
        return gmm_covariance_type::tied;
    }

    throw std::runtime_error("Unsupported GMM covariance_type: " + std::string(value));
}

inline const char* to_string(gmm_covariance_type covariance_type) {
    switch (covariance_type) {
    case gmm_covariance_type::spherical:
        return "spherical";
    case gmm_covariance_type::diag:
        return "diag";
    case gmm_covariance_type::full:
        return "full";
    case gmm_covariance_type::tied:
        return "tied";
    }

    throw std::runtime_error("Unknown GMM covariance_type");
}

inline std::size_t gmm_precision_value_count(
    gmm_covariance_type covariance_type,
    std::size_t K,
    std::size_t D
) {
    switch (covariance_type) {
    case gmm_covariance_type::spherical:
        return K;
    case gmm_covariance_type::diag:
        return K * D;
    case gmm_covariance_type::full:
        return K * D * D;
    case gmm_covariance_type::tied:
        return D * D;
    }

    throw std::runtime_error("Unknown GMM covariance_type");
}

inline std::size_t gmm_covariance_value_count(
    gmm_covariance_type covariance_type,
    std::size_t K,
    std::size_t D
) {
    return gmm_precision_value_count(covariance_type, K, D);
}
