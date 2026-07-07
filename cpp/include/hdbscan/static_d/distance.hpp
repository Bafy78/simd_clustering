#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "types.hpp"

template<typename SimdSample, typename ScalarSample>
hdbscan_wide squared_euclidean_distance_to_sample(
    const SimdSample& samples,
    const ScalarSample& reference
) {
    auto squared_distance = hdbscan_wide_zero;

    kumi::for_each(
        [&](auto x, auto reference_x) {
            const auto diff = x - hdbscan_wide(reference_x);
            squared_distance = eve::fma(diff, diff, squared_distance);
        },
        samples,
        reference
    );

    return eve::max(squared_distance, hdbscan_wide_zero);
}

template<std::size_t Dim>
void check_hdbscan_aos_size(std::span<const double> aos, std::size_t rows, const char* label) {
    if (aos.size() != rows * Dim) {
        throw std::runtime_error(std::string(label) + " AoS buffer has unexpected size");
    }
}

template<std::size_t Dim>
void copy_aos_to_hdbscan_samples(
    std::span<const double> aos,
    std::size_t N,
    hdbscan_static_samples_soa_vector<Dim>& samples
) {
    check_hdbscan_aos_size<Dim>(aos, N, "HDBSCAN samples");

    for (std::size_t n = 0; n < N; ++n) {
        hdbscan_static_sample_type<Dim> sample;

        kumi::for_each_index(
            [&](auto index, auto& element) {
                element = aos[n * Dim + index];
            },
            sample
        );

        samples.set(n, sample);
    }
}

template<std::size_t Dim>
void squared_euclidean_distance_matrix(
    const hdbscan_static_samples_soa_vector<Dim>& samples,
    std::vector<double, eve::aligned_allocator<double>>& distances
) {
    const std::size_t N = samples.size();
    const std::size_t matrix_size = N * N;
    if (distances.size() != matrix_size) {
        distances.resize(matrix_size);
    }

    for (std::size_t row = 0; row < N; ++row) {
        const auto reference = samples.get(row);
        auto* row_out = distances.data() + row * N;
        auto row_out_range = eve::algo::as_range(row_out, row_out + N);
        auto zipped = eve::views::zip(samples, row_out_range);

        eve::algo::for_each[eve::algo::force_cardinal<hdbscan_cardinal{}()>](
            zipped,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                auto [sample_it, out_it] = it;
                const auto sample = eve::load[ignore](sample_it);
                const auto distance = squared_euclidean_distance_to_sample(sample, reference);
                eve::store[ignore](distance, out_it);
            }
        );

        // Keep the diagonal exactly zero even if fma details change later.
        row_out[row] = 0.0;
    }
}

inline std::uint64_t float64_ordered_bits(double value) {
    return std::bit_cast<std::uint64_t>(value);
}

inline double bits_to_float64(std::uint64_t bits) {
    return std::bit_cast<double>(bits);
}

inline double find_unique_prefix_value(
    const double* row,
    std::uint32_t N,
    std::uint64_t prefix,
    std::uint64_t prefix_mask
) {
    for (std::uint32_t i = 0; i < N; ++i) {
        const std::uint64_t bits = float64_ordered_bits(row[i]);
        if ((bits & prefix_mask) == prefix) {
            return bits_to_float64(bits);
        }
    }

    throw std::runtime_error("Could not find unique HDBSCAN radix-selection prefix");
}

template<std::uint32_t Width>
inline std::uint32_t select_radix_group(
    const double* row,
    std::uint32_t N,
    unsigned shift,
    std::uint64_t& prefix,
    std::uint64_t& prefix_mask,
    std::uint32_t& rank
) {
    static_assert(Width > 0 && Width <= 12);
    constexpr std::uint32_t bucket_count = 1u << Width;
    constexpr std::uint32_t bucket_mask = bucket_count - 1u;

    std::array<std::uint32_t, bucket_count> counts{};

    if (prefix_mask == 0u) {
        for (std::uint32_t i = 0; i < N; ++i) {
            const std::uint64_t bits = float64_ordered_bits(row[i]);
            ++counts[static_cast<std::size_t>((bits >> shift) & bucket_mask)];
        }
    } else {
        for (std::uint32_t i = 0; i < N; ++i) {
            const std::uint64_t bits = float64_ordered_bits(row[i]);
            if ((bits & prefix_mask) == prefix) {
                ++counts[static_cast<std::size_t>((bits >> shift) & bucket_mask)];
            }
        }
    }

    std::uint32_t before = 0u;
    std::uint32_t selected = 0u;
    for (; selected < bucket_count; ++selected) {
        const std::uint32_t count = counts[selected];
        if (rank < before + count) {
            rank -= before;
            break;
        }
        before += count;
    }

    if (selected == bucket_count) {
        throw std::runtime_error("Could not find HDBSCAN radix-selection bucket");
    }

    prefix |= static_cast<std::uint64_t>(selected) << shift;
    prefix_mask |= static_cast<std::uint64_t>(bucket_mask) << shift;
    return counts[selected];
}

inline double kth_smallest_value_radix_float64(
    const double* row,
    std::size_t N,
    std::size_t k_zero_based
) {
    if (N == 0) {
        throw std::runtime_error("Cannot compute HDBSCAN core distances for an empty row");
    }
    if (k_zero_based >= N) {
        throw std::runtime_error("HDBSCAN core-distance order statistic is out of range");
    }
    if (N > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("HDBSCAN radix selection expects N <= UINT32_MAX");
    }

    const auto N_u32 = static_cast<std::uint32_t>(N);
    std::uint32_t rank = static_cast<std::uint32_t>(k_zero_based);
    std::uint64_t prefix = 0u;
    std::uint64_t prefix_mask = 0u;

    if (select_radix_group<12>(row, N_u32, 52, prefix, prefix_mask, rank) == 1u) {
        return find_unique_prefix_value(row, N_u32, prefix, prefix_mask);
    }
    if (select_radix_group<12>(row, N_u32, 40, prefix, prefix_mask, rank) == 1u) {
        return find_unique_prefix_value(row, N_u32, prefix, prefix_mask);
    }
    if (select_radix_group<12>(row, N_u32, 28, prefix, prefix_mask, rank) == 1u) {
        return find_unique_prefix_value(row, N_u32, prefix, prefix_mask);
    }
    if (select_radix_group<12>(row, N_u32, 16, prefix, prefix_mask, rank) == 1u) {
        return find_unique_prefix_value(row, N_u32, prefix, prefix_mask);
    }
    if (select_radix_group<12>(row, N_u32, 4, prefix, prefix_mask, rank) == 1u) {
        return find_unique_prefix_value(row, N_u32, prefix, prefix_mask);
    }
    (void)select_radix_group<4>(row, N_u32, 0, prefix, prefix_mask, rank);

    return bits_to_float64(prefix);
}

inline void core_distances(
    std::span<const double> distance_matrix,
    std::size_t N,
    std::size_t min_samples,
    std::vector<double, eve::aligned_allocator<double>>& core
) {
    if (distance_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN core stage expected a dense N x N squared-distance matrix");
    }
    if (min_samples < 2 || min_samples > N) {
        throw std::runtime_error("HDBSCAN min_samples must be in [2, N]");
    }

    if (core.size() != N) {
        core.resize(N);
    }

    const std::size_t k_zero_based = min_samples - 1;
    for (std::size_t row = 0; row < N; ++row) {
        core[row] = kth_smallest_value_radix_float64(
            distance_matrix.data() + row * N,
            N,
            k_zero_based
        );
    }
}

template<std::size_t Dim>
void squared_euclidean_distance_matrix_and_core_distances(
    const hdbscan_static_samples_soa_vector<Dim>& samples,
    std::vector<double, eve::aligned_allocator<double>>& distances,
    std::vector<double, eve::aligned_allocator<double>>& core,
    std::size_t min_samples
) {
    const std::size_t N = samples.size();
    if (min_samples < 2 || min_samples > N) {
        throw std::runtime_error("HDBSCAN min_samples must be in [2, N]");
    }

    const std::size_t matrix_size = N * N;
    if (distances.size() != matrix_size) {
        distances.resize(matrix_size);
    }
    if (core.size() != N) {
        core.resize(N);
    }

    const std::size_t k_zero_based = min_samples - 1;
    for (std::size_t row = 0; row < N; ++row) {
        const auto reference = samples.get(row);
        auto* row_out = distances.data() + row * N;
        auto row_out_range = eve::algo::as_range(row_out, row_out + N);

        eve::algo::transform_to
        [eve::algo::force_cardinal<hdbscan_cardinal{}()>]
        [eve::algo::no_unrolling](
            samples,
            row_out_range,
            [reference](const auto& sample) {
                return squared_euclidean_distance_to_sample(sample, reference);
            }
        );

        row_out[row] = 0.0;
        core[row] = kth_smallest_value_radix_float64(row_out, N, k_zero_based);
    }
}
