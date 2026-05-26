#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

// To compute inertia. We could also do it in SIMD by asking assignment to do a sum
// reduction and then keeping the result from the last k-means iteration, but that
// would slow down the algorithm. This stays as a separate scalar verification step.
// Use double so the calculation is precise enough to match scikit-learn outputs.
template <eve::product_type PointT>
double compute_scalar_dist_sq(const PointT& point, const PointT& centroid) {
    double dist_sq = 0.0;

    kumi::for_each(
        [&dist_sq](auto p, auto c) {
            const double diff = static_cast<double>(p) - static_cast<double>(c);
            dist_sq += diff * diff;
        },
        point,
        centroid
    );

    return dist_sq;
}

template <eve::product_type PointT>
void write_point_json(std::ostream& out, const PointT& point) {
    out << "[";

    [&] <std::size_t... I>(std::index_sequence<I...>) {
        std::size_t column = 0;

        (
            (
                out << (column++ == 0 ? "" : ", ")
                    << std::setprecision(std::numeric_limits<float>::max_digits10)
                    << static_cast<float>(get<I>(point))
            ),
            ...
        );
    }(std::make_index_sequence<kumi::size_v<PointT>>{});

    out << "]";
}

template <eve::product_type PointT, class Assignments>
void write_lloyd_metrics(
    const std::string& filename,
    const eve::algo::soa_vector<PointT>& points,
    const std::vector<PointT>& centroids,
    const Assignments& assignments,
    int num_clusters,
    int iterations
) {
    if (num_clusters <= 0) {
        throw std::runtime_error("Invalid number of clusters");
    }

    if (static_cast<std::size_t>(num_clusters) != centroids.size()) {
        throw std::runtime_error("Centroid count does not match num_clusters");
    }

    if (assignments.size() != points.size()) {
        throw std::runtime_error("Assignment count does not match point count");
    }

    std::vector<std::size_t> cluster_counts(num_clusters, 0);
    std::vector<double> cluster_inertia(num_clusters, 0.0);

    double total_inertia = 0.0;

    for (std::size_t i = 0; i < points.size(); ++i) {
        const std::size_t cluster_id = static_cast<std::size_t>(assignments[i]);

        if (cluster_id >= static_cast<std::size_t>(num_clusters)) {
            throw std::runtime_error("Invalid cluster assignment");
        }

        const double dist_sq = compute_scalar_dist_sq(
            points.get(i),
            centroids[cluster_id]
        );

        cluster_counts[cluster_id] += 1;
        cluster_inertia[cluster_id] += dist_sq;
        total_inertia += dist_sq;
    }

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error("Could not open Lloyd metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"iterations\": " << iterations << ",\n";
    out << "  \"inertia\": " << total_inertia << ",\n";

    out << "  \"cluster_counts\": [";
    for (int k = 0; k < num_clusters; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_counts[static_cast<std::size_t>(k)];
    }
    out << "],\n";

    out << "  \"cluster_inertia\": [";
    for (int k = 0; k < num_clusters; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_inertia[static_cast<std::size_t>(k)];
    }
    out << "],\n";

    out << "  \"centroids\": [\n";
    for (std::size_t k = 0; k < centroids.size(); ++k) {
        out << "    ";
        write_point_json(out, centroids[k]);

        if (k + 1 != centroids.size()) {
            out << ",";
        }

        out << "\n";
    }
    out << "  ]\n";

    out << "}\n";
}
