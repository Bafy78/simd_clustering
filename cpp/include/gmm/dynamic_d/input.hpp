#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../io/binary.hpp"
#include "../../layout/dynamic_soa.hpp"
#include "../../layout/static_soa.hpp"

// GMM-specific input helpers for the dynamic-D kernels.
//
// The dynamic GMM backend consumes samples as dimension-major SoA and means as
// row-major [component][dimension] scalars. Metrics still use the static Kumi
// representation, so this header also provides a row-major -> static mean
// conversion helper.

template<std::size_t D>
void check_dynamic_gmm_aos_size(
    std::span<const float> aos,
    std::size_t rows,
    const char* label
) {
    if (aos.size() != rows * D) {
        throw std::runtime_error(std::string(label) + " AoS buffer has unexpected size");
    }
}

template<std::size_t D>
void copy_aos_to_dynamic_gmm_samples(
    std::span<const float> aos,
    std::size_t N,
    samples_soa_storage<D>& samples
) {
    check_dynamic_gmm_aos_size<D>(aos, N, "samples");

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
samples_soa_storage<D> make_dynamic_gmm_samples_from_aos(
    std::span<const float> aos,
    std::size_t N
) {
    samples_soa_storage<D> out(N);
    copy_aos_to_dynamic_gmm_samples<D>(aos, N, out);
    return out;
}

template<std::size_t D>
aligned_float_vector make_dynamic_gmm_means_from_aos(
    std::span<const float> aos,
    std::size_t K
) {
    check_dynamic_gmm_aos_size<D>(aos, K, "means");

    aligned_float_vector out(K * D);

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            out[k * D + d] = aos[k * D + d];
        }
    }

    return out;
}

template<std::size_t D>
std::vector<static_sample_type<D>> make_static_gmm_means_from_dynamic(
    const aligned_float_vector& means_row_major,
    std::size_t K
) {
    if (means_row_major.size() != K * D) {
        throw std::runtime_error("Dynamic GMM mean count must be cluster count times dimension count");
    }

    std::vector<static_sample_type<D>> out(K);

    for (std::size_t k = 0; k < K; ++k) {
        static_sample_type<D> sample{};

        kumi::for_each_index(
            [&](auto index, auto& value) {
                constexpr std::size_t d = decltype(index)::value;
                value = means_row_major[k * D + d];
            },
            sample
        );

        out[k] = sample;
    }

    return out;
}

template<std::size_t D>
samples_soa_storage<D> read_dynamic_gmm_samples_binary(
    const std::string& filename,
    std::size_t N
) {
    auto raw_samples = read_aos_f32(filename, N, D);
    return make_dynamic_gmm_samples_from_aos<D>(raw_samples, N);
}

template<std::size_t D>
aligned_float_vector read_dynamic_gmm_means_binary(
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

    return make_dynamic_gmm_means_from_aos<D>(
        raw_means,
        static_cast<std::size_t>(K)
    );
}
