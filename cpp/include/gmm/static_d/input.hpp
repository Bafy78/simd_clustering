#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../io/binary.hpp"
#include "../../layout/static_soa.hpp"

template<std::size_t D>
std::vector<static_point_type<D>> make_static_means_from_aos(
    std::span<const float> aos,
    std::size_t n_components
) {
    return make_static_vectors_from_aos<D>(aos, n_components, "means");
}

inline eve::algo::soa_vector<PointType> read_static_gmm_points_binary(
    const std::string& filename,
    std::size_t n_samples
) {
    auto raw_points = read_aos_f32(filename, n_samples, TUPLE_SIZE);
    return make_static_points_from_aos<TUPLE_SIZE>(raw_points, n_samples);
}

inline std::vector<PointType> read_static_gmm_means_binary(
    const std::string& filename,
    int n_components
) {
    if (n_components <= 0) {
        throw std::runtime_error("Invalid number of GMM components");
    }

    auto raw_means = read_aos_f32(
        filename,
        static_cast<std::size_t>(n_components),
        TUPLE_SIZE
    );

    return make_static_means_from_aos<TUPLE_SIZE>(
        raw_means,
        static_cast<std::size_t>(n_components)
    );
}
