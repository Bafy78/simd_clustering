#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/core.hpp>

#include "covariance_type.hpp"
#include "../io/json.hpp"

template <eve::product_type PointT>
void write_spherical_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t n_components
) {
    if (covariances.size() != n_components) {
        throw std::runtime_error("Spherical GMM covariance count must match component count");
    }

    out << "  \"covariances\": [";
    for (std::size_t k = 0; k < covariances.size(); ++k) {
        if (k != 0) {
            out << ", ";
        }
        out << static_cast<double>(covariances[k]);
    }
    out << "],\n";
}

template <eve::product_type PointT>
void write_diagonal_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t n_components
) {
    constexpr std::size_t n_features = kumi::size_v<PointT>;

    if (covariances.size() != n_components * n_features) {
        throw std::runtime_error(
            "Diagonal GMM covariance count must be component count times feature count"
        );
    }

    out << "  \"covariances\": [\n";
    for (std::size_t k = 0; k < n_components; ++k) {
        out << "    [";
        for (std::size_t d = 0; d < n_features; ++d) {
            if (d != 0) {
                out << ", ";
            }
            out << static_cast<double>(covariances[k * n_features + d]);
        }
        out << "]";

        if (k + 1 != n_components) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ],\n";
}

template <eve::product_type PointT>
void write_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t n_components,
    gmm_covariance_type covariance_type
) {
    switch (covariance_type) {
    case gmm_covariance_type::spherical:
        write_spherical_gmm_covariances_json<PointT>(out, covariances, n_components);
        return;
    case gmm_covariance_type::diag:
        write_diagonal_gmm_covariances_json<PointT>(out, covariances, n_components);
        return;
    case gmm_covariance_type::full:
    case gmm_covariance_type::tied:
        throw std::runtime_error("GMM metrics support only spherical and diag covariance");
    }

    throw std::runtime_error("Unknown GMM covariance_type");
}

template <eve::product_type PointT>
void write_gmm_metrics(
    const std::string& filename,
    const std::vector<float>& weights,
    const std::vector<PointT>& means,
    const std::vector<float>& covariances,
    const std::vector<float>& lower_bounds,
    int iterations,
    bool converged,
    float lower_bound,
    gmm_covariance_type covariance_type
) {
    if (weights.size() != means.size()) {
        throw std::runtime_error("GMM metrics weight and mean counts do not match");
    }

    const std::size_t expected_covariance_count = gmm_covariance_value_count(
        covariance_type,
        weights.size(),
        kumi::size_v<PointT>
    );

    if (covariances.size() != expected_covariance_count) {
        throw std::runtime_error("GMM metrics covariance count does not match covariance type");
    }

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error("Could not open GMM metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"algorithm\": \"gmm\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"covariance_type\": \"" << to_string(covariance_type) << "\",\n";
    out << "  \"iterations\": " << iterations << ",\n";
    out << "  \"converged\": " << (converged ? "true" : "false") << ",\n";
    out << "  \"lower_bound\": " << static_cast<double>(lower_bound) << ",\n";

    out << "  \"lower_bounds\": [";
    for (std::size_t i = 0; i < lower_bounds.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << static_cast<double>(lower_bounds[i]);
    }
    out << "],\n";

    out << "  \"weights\": [";
    for (std::size_t k = 0; k < weights.size(); ++k) {
        if (k != 0) {
            out << ", ";
        }
        out << static_cast<double>(weights[k]);
    }
    out << "],\n";

    out << "  \"means\": [\n";
    for (std::size_t k = 0; k < means.size(); ++k) {
        out << "    ";
        write_point_json(out, means[k]);

        if (k + 1 != means.size()) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ],\n";

    write_gmm_covariances_json<PointT>(out, covariances, weights.size(), covariance_type);

    out << "  \"sklearn_defaults\": {\n";
    out << "    \"tol\": 0.001,\n";
    out << "    \"reg_covar\": 1e-06,\n";
    out << "    \"max_iter\": 100,\n";
    out << "    \"n_init\": 1\n";
    out << "  }\n";
    out << "}\n";
}
