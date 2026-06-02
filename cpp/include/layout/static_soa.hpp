#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

template<std::size_t D>
using static_sample_type = kumi::result::fill_t<D, float>;

inline constexpr std::size_t D = static_cast<std::size_t>(TUPLE_SIZE);

using SampleType = static_sample_type<D>;

template<std::size_t D>
using static_samples_soa_vector = eve::algo::soa_vector<static_sample_type<D>>;

template<std::size_t D>
void check_static_aos_size(std::span<const float> aos, std::size_t rows, const char* label) {
    if (aos.size() != rows * D) {
        throw std::runtime_error(std::string(label) + " AoS buffer has unexpected size");
    }
}

template<std::size_t D>
void copy_aos_to_static_samples(
    std::span<const float> aos,
    std::size_t N,
    static_samples_soa_vector<D>& samples
) {
    check_static_aos_size<D>(aos, N, "samples");

    for (std::size_t n = 0; n < N; ++n) {
        static_sample_type<D> sample;

        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[n * D + index];
            },
            sample
        );

        samples.set(n, sample);
    }
}

template<std::size_t D>
static_samples_soa_vector<D> make_static_samples_from_aos(
    std::span<const float> aos,
    std::size_t N
) {
    static_samples_soa_vector<D> samples(eve::algo::no_init, N);
    copy_aos_to_static_samples<D>(aos, N, samples);
    return samples;
}

template<std::size_t D>
std::vector<static_sample_type<D>> make_static_vectors_from_aos(
    std::span<const float> aos,
    std::size_t rows,
    const char* label = "vectors"
) {
    check_static_aos_size<D>(aos, rows, label);

    std::vector<static_sample_type<D>> vectors(rows);

    for (std::size_t row = 0; row < rows; ++row) {
        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[row * D + index];
            },
            vectors[row]
        );
    }

    return vectors;
}
