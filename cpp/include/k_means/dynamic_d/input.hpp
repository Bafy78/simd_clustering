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
void copy_aos_to_dynamic_samples(
    std::span<const float> aos,
    std::size_t N,
    samples_soa_storage<D>& samples
) {
    check_dynamic_aos_size<D>(aos, N, "samples");

    if (samples.N != N) {
        samples.resize(N);
    }

    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t d = 0; d < D; ++d) {
            samples(n, d) = aos[n * D + d];
        }
    }
}

template<std::size_t D>
samples_soa_storage<D> make_dynamic_samples_from_aos(
    std::span<const float> aos,
    std::size_t N
) {
    samples_soa_storage<D> out(N);
    copy_aos_to_dynamic_samples<D>(aos, N, out);
    return out;
}

template<std::size_t D>
centroids_storage<D> make_dynamic_centroids_from_aos(
    std::span<const float> aos,
    std::size_t K
) {
    check_dynamic_aos_size<D>(aos, K, "centroids");

    centroids_storage<D> out;
    out.resize(K);

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            out.row(k, d) = aos[k * D + d];
        }
    }

    return out;
}

template<std::size_t D>
std::vector<static_sample_type<D>> make_static_centroids_from_dynamic(
    const centroids_storage<D>& dynamic_centroids
) {
    std::vector<static_sample_type<D>> out(dynamic_centroids.K);

    for (std::size_t k = 0; k < dynamic_centroids.K; ++k) {
        static_sample_type<D> sample{};

        kumi::for_each_index(
            [&](auto index, auto& value) {
                constexpr std::size_t d = decltype(index)::value;
                value = dynamic_centroids.row(k, d);
            },
            sample
        );

        out[k] = sample;
    }

    return out;
}
