#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../io/binary.hpp"
#include "../../layout/static_soa.hpp"

template<std::size_t D>
std::vector<static_sample_type<D>> make_static_means_from_aos(
    std::span<const float> aos,
    std::size_t K
) {
    return make_static_vectors_from_aos<D>(aos, K, "means");
}

inline eve::algo::soa_vector<SampleType> read_static_gmm_samples_binary(
    const std::string& filename,
    std::size_t N
) {
    auto raw_samples = read_aos_f32(filename, N, D);
    return make_static_samples_from_aos<D>(raw_samples, N);
}

inline std::vector<SampleType> read_static_gmm_means_binary(
    const std::string& filename,
    int K
) {
    if (K <= 0) {
        throw std::runtime_error("Invalid number of GMM clusters");
    }

    auto raw_means = read_aos_f32(
        filename,
        static_cast<std::size_t>(K),
        D
    );

    return make_static_means_from_aos<D>(
        raw_means,
        static_cast<std::size_t>(K)
    );
}
