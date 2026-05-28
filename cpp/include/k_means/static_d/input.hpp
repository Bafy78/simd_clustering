#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../io/binary.hpp"
#include "../../layout/static_soa.hpp"

template<std::size_t D>
std::vector<static_point_type<D>> make_static_centroids_from_aos(
    std::span<const float> aos,
    std::size_t n_clusters
) {
    return make_static_vectors_from_aos<D>(aos, n_clusters, "centroids");
}

inline eve::algo::soa_vector<PointType> read_dataset_soa(
    const std::string& filename,
    std::size_t n_samples
) {
    auto raw_aos_data = read_aos_f32(filename, n_samples, TUPLE_SIZE);
    return make_static_points_from_aos<TUPLE_SIZE>(raw_aos_data, n_samples);
}

inline std::vector<PointType> read_initial_centroids_binary(
    const std::string& filename,
    int n_clusters
) {
    if (n_clusters <= 0) {
        throw std::runtime_error("Invalid number of clusters");
    }

    auto raw_centroids = read_aos_f32(
        filename,
        static_cast<std::size_t>(n_clusters),
        TUPLE_SIZE
    );

    return make_static_centroids_from_aos<TUPLE_SIZE>(
        raw_centroids,
        static_cast<std::size_t>(n_clusters)
    );
}
