#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

template<std::size_t Dim>
float hdbscan_static_euclidean_distance(
    std::span<const float> aos,
    std::size_t lhs,
    std::size_t rhs
) {
    float squared_distance = 0.0f;

    const std::size_t lhs_offset = lhs * Dim;
    const std::size_t rhs_offset = rhs * Dim;

    for (std::size_t dim = 0; dim < Dim; ++dim) {
        const float diff = aos[lhs_offset + dim] - aos[rhs_offset + dim];
        squared_distance += diff * diff;
    }

    return std::sqrt(squared_distance);
}

template<std::size_t Dim>
void hdbscan_static_euclidean_distance_matrix(
    std::span<const float> aos,
    std::size_t N,
    std::vector<float>& distances
) {
    const std::size_t matrix_size = N * N;
    if (distances.size() != matrix_size) {
        distances.resize(matrix_size);
    }

    for (std::size_t i = 0; i < N; ++i) {
        distances[i * N + i] = 0.0f;
        for (std::size_t j = i + 1; j < N; ++j) {
            const float distance = hdbscan_static_euclidean_distance<Dim>(aos, i, j);
            distances[i * N + j] = distance;
            distances[j * N + i] = distance;
        }
    }
}
