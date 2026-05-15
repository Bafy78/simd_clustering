#pragma once
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <span>
#include <string>
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>

#ifndef TUPLE_SIZE
#define TUPLE_SIZE 2
#endif

using PointType = kumi::result::fill_t<TUPLE_SIZE, float>;

// --- Helper: Read Raw Binary Data & Convert to SoA (For Setup) ---
inline eve::algo::soa_vector<PointType> read_dataset_soa(const std::string& filename, std::size_t n_samples) {
    std::vector<float> raw_aos_data(n_samples * TUPLE_SIZE);
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Error: Could not open file " + filename);

    file.read(reinterpret_cast<char*>(raw_aos_data.data()), raw_aos_data.size() * sizeof(float));

    eve::algo::soa_vector<PointType> points(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) {
        PointType pt;
        kumi::for_each_index([&](auto index, auto& element)
        { element = raw_aos_data[i * TUPLE_SIZE + index]; }, pt);
        points.set(i, pt);
    }
    return points;
}

// --- Helper: Read Orchestrator's Initial Centroids ---
inline std::vector<PointType> read_initial_centroids_binary(const std::string& filename, int n_clusters) {
    std::vector<PointType> centroids(n_clusters);
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Error: Could not open initial centroids file " + filename);

    for (auto& c : centroids) {
        [&] <std::size_t... I>(std::index_sequence<I...>) {
            (..., in.read(reinterpret_cast<char*>(&get<I>(c)), sizeof(float)));
        }(std::make_index_sequence<TUPLE_SIZE>{});
    }
    return centroids;
}


// To compute inertia. We could also do it in SIMD by simply asking `assign_points_to_centroid` to
// do a sum reduction and then get the result at the last iteration in k-means, but that would slow down the
// algorithm. So we're keeping it as a separate step for now.
// We're doing double because else the calculation is imprecise and doesn't match scikit
template <eve::product_type PointT>
double compute_scalar_dist_sq(const PointT& point, const PointT& centroid) {
    double dist_sq = 0.0;

    kumi::for_each(
        [&dist_sq](auto p, auto c) {
        double diff = static_cast<double>(p) - static_cast<double>(c);
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
    }(std::make_index_sequence<TUPLE_SIZE>{});

    out << "]";
}

template <eve::product_type PointT>
void write_lloyd_metrics(
    const std::string& filename,
    const eve::algo::soa_vector<PointT>& points,
    const std::vector<PointT>& centroids,
    std::span<const int> assignments,
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
        int cluster_id = assignments[i];

        if (cluster_id < 0 || cluster_id >= num_clusters) {
            throw std::runtime_error("Invalid cluster assignment");
        }

        double dist_sq = compute_scalar_dist_sq(points.get(i), centroids[cluster_id]);

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

        out << cluster_counts[k];
    }
    out << "],\n";

    out << "  \"cluster_inertia\": [";
    for (int k = 0; k < num_clusters; ++k) {
        if (k != 0) {
            out << ", ";
        }

        out << cluster_inertia[k];
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