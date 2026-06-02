#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

#include "../io/json.hpp"

// To compute inertia. We could also do it in SIMD by asking assignment to do a sum
// reduction and then keeping the result from the last k-means iteration, but that
// would slow down the algorithm. This stays as a separate scalar verification step.
// Use double so the calculation is precise enough to match scikit-learn outputs.
template <eve::product_type SampleT>
double compute_scalar_dist_sq(const SampleT& sample, const SampleT& centroid) {
    double dist_sq = 0.0;

    kumi::for_each(
        [&dist_sq](auto p, auto c) {
            const double diff = static_cast<double>(p) - static_cast<double>(c);
            dist_sq += diff * diff;
        },
        sample,
        centroid
    );

    return dist_sq;
}

template <eve::product_type SampleT, class Assignments>
void write_lloyd_metrics(
    const std::string& filename,
    const eve::algo::soa_vector<SampleT>& samples,
    const std::vector<SampleT>& centroids,
    const Assignments& assignments,
    int K,
    int algorithm_iterations
) {
    if (K <= 0) {
        throw std::runtime_error("Invalid number of clusters");
    }

    if (static_cast<std::size_t>(K) != centroids.size()) {
        throw std::runtime_error("Centroid count does not match K");
    }

    if (assignments.size() != samples.size()) {
        throw std::runtime_error("Assignment count does not match sample count");
    }

    std::vector<std::size_t> cluster_counts(K, 0);
    std::vector<double> cluster_inertia(K, 0.0);

    double total_inertia = 0.0;

    for (std::size_t n = 0; n < samples.size(); ++n) {
        const std::size_t k = static_cast<std::size_t>(assignments[n]);

        if (k >= static_cast<std::size_t>(K)) {
            throw std::runtime_error("Invalid cluster assignment");
        }

        const double dist_sq = compute_scalar_dist_sq(
            samples.get(n),
            centroids[k]
        );

        cluster_counts[k] += 1;
        cluster_inertia[k] += dist_sq;
        total_inertia += dist_sq;
    }

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error("Could not open Lloyd metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"phase\": \"lloyd\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"algorithm_iterations\": " << algorithm_iterations << ",\n";
    out << "  \"inertia\": " << total_inertia << ",\n";

    out << "  \"cluster_counts\": [";
    for (int k = 0; k < K; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_counts[static_cast<std::size_t>(k)];
    }
    out << "],\n";

    out << "  \"cluster_inertia\": [";
    for (int k = 0; k < K; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_inertia[static_cast<std::size_t>(k)];
    }
    out << "],\n";

    out << "  \"centroids\": [\n";
    for (std::size_t k = 0; k < centroids.size(); ++k) {
        out << "    ";
        write_sample_json(out, centroids[k]);

        if (k + 1 != centroids.size()) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ]\n";

    out << "}\n";
}

