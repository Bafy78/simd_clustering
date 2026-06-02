#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../io/binary.hpp"
#include "../../layout/static_soa.hpp"

template<std::size_t D>
std::vector<static_sample_type<D>> make_static_centroids_from_aos(
    std::span<const float> aos,
    std::size_t K
) {
    return make_static_vectors_from_aos<D>(aos, K, "centroids");
}

inline eve::algo::soa_vector<SampleType> read_dataset_soa(
    const std::string& filename,
    std::size_t N
) {
    auto raw_aos_data = read_aos_f32(filename, N, D);
    return make_static_samples_from_aos<D>(raw_aos_data, N);
}

inline std::vector<SampleType> read_initial_centroids_binary(
    const std::string& filename,
    int K
) {
    if (K <= 0) {
        throw std::runtime_error("Invalid number of clusters");
    }

    auto raw_centroids = read_aos_f32(
        filename,
        static_cast<std::size_t>(K),
        D
    );

    return make_static_centroids_from_aos<D>(
        raw_centroids,
        static_cast<std::size_t>(K)
    );
}
