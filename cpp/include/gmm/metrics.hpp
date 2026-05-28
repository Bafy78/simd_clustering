#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/core.hpp>

#include "../io/json.hpp"

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
    const std::string& covariance_type
) {
    if (weights.size() != means.size() || weights.size() != covariances.size()) {
        throw std::runtime_error("GMM metrics parameter sizes do not match");
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
    out << "  \"covariance_type\": \"" << covariance_type << "\",\n";
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

    out << "  \"covariances\": [";
    for (std::size_t k = 0; k < covariances.size(); ++k) {
        if (k != 0) {
            out << ", ";
        }
        out << static_cast<double>(covariances[k]);
    }
    out << "],\n";

    out << "  \"sklearn_defaults\": {\n";
    out << "    \"tol\": 0.001,\n";
    out << "    \"reg_covar\": 1e-06,\n";
    out << "    \"max_iter\": 100,\n";
    out << "    \"n_init\": 1\n";
    out << "  }\n";
    out << "}\n";
}
