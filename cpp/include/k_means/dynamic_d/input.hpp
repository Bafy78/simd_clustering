#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "./layout.hpp"
#include "../../io/binary.hpp"
#include "../../layout/static_soa.hpp"

template<std::size_t D>
void check_dynamic_aos_size(std::span<const float> aos, std::size_t rows, const char* label) {
    if (aos.size() != rows * D) {
        throw std::runtime_error(std::string(label) + " AoS buffer has unexpected size");
    }
}

template<std::size_t D>
void copy_aos_to_dynamic_points(
    std::span<const float> aos,
    std::size_t n_samples,
    points_soa_storage<D>& points
) {
    check_dynamic_aos_size<D>(aos, n_samples, "points");

    if (points.n_samples != n_samples) {
        points.resize(n_samples);
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t d = 0; d < D; ++d) {
            points(i, d) = aos[i * D + d];
        }
    }
}

template<std::size_t D>
points_soa_storage<D> make_dynamic_points_from_aos(
    std::span<const float> aos,
    std::size_t n_samples
) {
    points_soa_storage<D> out(n_samples);
    copy_aos_to_dynamic_points<D>(aos, n_samples, out);
    return out;
}

template<std::size_t D>
centroids_storage<D> make_dynamic_centroids_from_aos(
    std::span<const float> aos,
    std::size_t n_clusters
) {
    check_dynamic_aos_size<D>(aos, n_clusters, "centroids");

    centroids_storage<D> out;
    out.resize(n_clusters);

    for (std::size_t k = 0; k < n_clusters; ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            out.row(k, d) = aos[k * D + d];
        }
    }

    return out;
}

template<std::size_t D>
std::vector<static_point_type<D>> make_static_centroids_from_dynamic(
    const centroids_storage<D>& dynamic_centroids
) {
    std::vector<static_point_type<D>> out(dynamic_centroids.n_clusters);

    for (std::size_t k = 0; k < dynamic_centroids.n_clusters; ++k) {
        static_point_type<D> pt{};

        kumi::for_each_index(
            [&](auto index, auto& value) {
                constexpr std::size_t d = decltype(index)::value;
                value = dynamic_centroids.row(k, d);
            },
            pt
        );

        out[k] = pt;
    }

    return out;
}
