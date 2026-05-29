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
    std::size_t n_components,
    std::size_t n_features
) {
    switch (covariance_type) {
    case gmm_covariance_type::spherical:
        return n_components;
    case gmm_covariance_type::diag:
        return n_components * n_features;
    case gmm_covariance_type::full:
        return n_components * n_features * n_features;
    case gmm_covariance_type::tied:
        return n_features * n_features;
    }

    throw std::runtime_error("Unknown GMM covariance_type");
}

inline std::size_t gmm_covariance_value_count(
    gmm_covariance_type covariance_type,
    std::size_t n_components,
    std::size_t n_features
) {
    return gmm_precision_value_count(covariance_type, n_components, n_features);
}
