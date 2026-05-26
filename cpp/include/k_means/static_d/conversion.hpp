#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "layout.hpp"
#include "../io/binary.hpp"

template<std::size_t D>
void check_static_aos_size(std::span<const float> aos, std::size_t rows, const char* label) {
    if (aos.size() != rows * D) {
        throw std::runtime_error(std::string(label) + " AoS buffer has unexpected size");
    }
}

template<std::size_t D>
void copy_aos_to_static_points(
    std::span<const float> aos,
    std::size_t n_samples,
    static_points_soa_vector<D>& points
) {
    check_static_aos_size<D>(aos, n_samples, "points");

    for (std::size_t i = 0; i < n_samples; ++i) {
        static_point_type<D> pt;

        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[i * D + index];
            },
            pt
        );

        points.set(i, pt);
    }
}

template<std::size_t D>
static_points_soa_vector<D> make_static_points_from_aos(
    std::span<const float> aos,
    std::size_t n_samples
) {
    static_points_soa_vector<D> points(eve::algo::no_init, n_samples);
    copy_aos_to_static_points<D>(aos, n_samples, points);
    return points;
}

template<std::size_t D>
std::vector<static_point_type<D>> make_static_centroids_from_aos(
    std::span<const float> aos,
    std::size_t n_clusters
) {
    check_static_aos_size<D>(aos, n_clusters, "centroids");

    std::vector<static_point_type<D>> centroids(n_clusters);

    for (std::size_t k = 0; k < n_clusters; ++k) {
        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[k * D + index];
            },
            centroids[k]
        );
    }

    return centroids;
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
