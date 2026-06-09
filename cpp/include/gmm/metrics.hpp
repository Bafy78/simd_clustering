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

template <eve::product_type SampleT>
void write_spherical_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t K
) {
    if (covariances.size() != K) {
        throw std::runtime_error("Spherical GMM covariance count must match cluster count");
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

template <eve::product_type SampleT>
void write_diagonal_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t K
) {
    constexpr std::size_t D = kumi::size_v<SampleT>;

    if (covariances.size() != K * D) {
        throw std::runtime_error(
            "Diagonal GMM covariance count must be cluster count times dimension count"
        );
    }

    out << "  \"covariances\": [\n";
    for (std::size_t k = 0; k < K; ++k) {
        out << "    [";
        for (std::size_t d = 0; d < D; ++d) {
            if (d != 0) {
                out << ", ";
            }
            out << static_cast<double>(covariances[k * D + d]);
        }
        out << "]";

        if (k + 1 != K) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ],\n";
}



template <eve::product_type SampleT>
void write_full_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t K
) {
    constexpr std::size_t D = kumi::size_v<SampleT>;

    if (covariances.size() != K * D * D) {
        throw std::runtime_error(
            "Full GMM covariance count must be cluster count times dimension squared"
        );
    }

    out << "  \"covariances\": [\n";
    for (std::size_t k = 0; k < K; ++k) {
        out << "    [\n";

        for (std::size_t row = 0; row < D; ++row) {
            out << "      [";

            for (std::size_t col = 0; col < D; ++col) {
                if (col != 0) {
                    out << ", ";
                }

                out << static_cast<double>(covariances[(k * D + row) * D + col]);
            }

            out << "]";

            if (row + 1 != D) {
                out << ",";
            }

            out << "\n";
        }

        out << "    ]";

        if (k + 1 != K) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ],\n";
}

template <eve::product_type SampleT>
void write_gmm_covariances_json(
    std::ostream& out,
    const std::vector<float>& covariances,
    std::size_t K,
    gmm_covariance_type covariance_type
) {
    switch (covariance_type) {
    case gmm_covariance_type::spherical:
        write_spherical_gmm_covariances_json<SampleT>(out, covariances, K);
        return;
    case gmm_covariance_type::diag:
        write_diagonal_gmm_covariances_json<SampleT>(out, covariances, K);
        return;
    case gmm_covariance_type::full:
        write_full_gmm_covariances_json<SampleT>(out, covariances, K);
        return;
    }
    throw std::runtime_error("Unknown GMM covariance_type");
}

template <eve::product_type SampleT>
void write_gmm_metrics(
    const std::string& filename,
    const std::vector<float>& weights,
    const std::vector<SampleT>& means,
    const std::vector<float>& covariances,
    const std::vector<float>& lower_bounds,
    int algorithm_iterations,
    float lower_bound,
    gmm_covariance_type covariance_type
) {
    if (weights.size() != means.size()) {
        throw std::runtime_error("GMM metrics weight and mean counts do not match");
    }

    const std::size_t expected_covariance_count = gmm_covariance_value_count(
        covariance_type,
        weights.size(),
        kumi::size_v<SampleT>
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
    out << "  \"phase\": \"gmm\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"covariance_type\": \"" << to_string(covariance_type) << "\",\n";
    out << "  \"algorithm_iterations\": " << algorithm_iterations << ",\n";
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
        write_sample_json(out, means[k]);

        if (k + 1 != means.size()) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ],\n";

    write_gmm_covariances_json<SampleT>(out, covariances, weights.size(), covariance_type);

    out << "  \"sklearn_defaults\": {\n";
    out << "    \"tol\": 0.001,\n";
    out << "    \"reg_covar\": 1e-06,\n";
    out << "    \"max_iter\": 100,\n";
    out << "    \"n_init\": 1\n";
    out << "  }\n";
    out << "}\n";
}
