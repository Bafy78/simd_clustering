#pragma once

#include <algorithm>
#include <cstddef>

#include "../../layout/dynamic_soa.hpp"

// Compile-time-D centroid storage for the dynamic/streamed k-means backend.
//
// row_major is the single source of truth:
//     centroids[k][d]
//
// Assignment-specific layouts are owned and refreshed by the assignment
// backends when on_centroids_changed() is called.
template<std::size_t D>
struct centroids_storage {
    aligned_float_vector row_major;

    std::size_t K = 0;

    centroids_storage() = default;

    void resize(std::size_t clusters) {
        K = clusters;
        row_major.resize(K * D);
        std::fill(row_major.begin(), row_major.end(), 0.0f);
    }

    float& row(std::size_t k, std::size_t d) {
        return row_major[k * D + d];
    }

    float row(std::size_t k, std::size_t d) const {
        return row_major[k * D + d];
    }

    template<std::size_t Dimension>
    float& row(std::size_t k) {
        static_assert(Dimension < D);
        return row_major[k * D + Dimension];
    }

    template<std::size_t Dimension>
    float row(std::size_t k) const {
        static_assert(Dimension < D);
        return row_major[k * D + Dimension];
    }
};
