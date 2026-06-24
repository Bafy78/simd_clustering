#pragma once

#include <algorithm>
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

#include "../../layout/static_soa.hpp"
#include "../../simd.hpp"

template<typename SimdSample, typename ScalarSample>
wide_f euclidean_distance_to_sample(
    const SimdSample& samples,
    const ScalarSample& reference
) {
    auto squared_distance = wide_zero_f;

    kumi::for_each(
        [&](auto x, auto reference_x) {
            const auto diff = x - wide_f(reference_x);
            squared_distance = eve::fma(diff, diff, squared_distance);
        },
        samples,
        reference
    );

    return eve::sqrt(eve::max(squared_distance, wide_zero_f));
}

template<std::size_t Dim>
void euclidean_distance_matrix(
    const static_samples_soa_vector<Dim>& samples,
    std::vector<float, eve::aligned_allocator<float>>& distances
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

        eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
            zipped,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                auto [sample_it, out_it] = it;
                const auto sample = eve::load[ignore](sample_it);
                const auto distance = euclidean_distance_to_sample(sample, reference);
                eve::store[ignore](distance, out_it);
            }
        );

        // Keep the diagonal exactly zero even if sqrt/fma details change later.
        row_out[row] = 0.0f;
    }
}

inline float reduce_min_wide(wide_f values) {
    float reduced = std::numeric_limits<float>::infinity();
    for (std::size_t lane = 0; lane < wide_f::size(); ++lane) {
        reduced = std::min(reduced, values.get(lane));
    }
    return reduced;
}

inline int reduce_sum_wide(wide_i values) {
    int reduced = 0;
    for (std::size_t lane = 0; lane < wide_i::size(); ++lane) {
        reduced += values.get(lane);
    }
    return reduced;
}

inline float kth_smallest_value_eve(
    const float* row,
    std::size_t N,
    std::size_t k_zero_based
) {
    if (N == 0) {
        throw std::runtime_error("Cannot compute HDBSCAN core distances for an empty row");
    }
    if (k_zero_based >= N) {
        throw std::runtime_error("HDBSCAN core-distance order statistic is out of range");
    }

    const auto row_range = eve::algo::as_range(row, row + N);
    const float infinity = std::numeric_limits<float>::infinity();
    float lower = -infinity;
    std::size_t consumed = 0;

    while (consumed <= k_zero_based) {
        wide_f candidate_values(infinity);

        eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
            row_range,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                const auto values = eve::load[ignore](it);
                const auto valid = ignore.mask(eve::as<wide_f>()) && (values > wide_f(lower));
                candidate_values = eve::min(
                    candidate_values,
                    eve::if_else(valid, values, wide_f(infinity))
                );
            }
        );

        const float candidate = reduce_min_wide(candidate_values);
        if (candidate == infinity) {
            throw std::runtime_error("Could not find HDBSCAN core-distance order statistic");
        }

        wide_i equal_counts = wide_zero_i;
        eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
            row_range,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                const auto values = eve::load[ignore](it);
                const auto equal = ignore.mask(eve::as<wide_f>()) && (values == wide_f(candidate));
                equal_counts += eve::if_else(equal, wide_i(1), wide_zero_i);
            }
        );

        const std::size_t equal_count = static_cast<std::size_t>(reduce_sum_wide(equal_counts));
        if (equal_count == 0) {
            throw std::runtime_error("Internal HDBSCAN core-distance selection error");
        }

        if (consumed + equal_count > k_zero_based) {
            return candidate;
        }

        consumed += equal_count;
        lower = candidate;
    }

    throw std::runtime_error("Unreachable HDBSCAN core-distance selection state");
}

inline void core_distances(
    std::span<const float> distance_matrix,
    std::size_t N,
    std::size_t min_samples,
    std::vector<float, eve::aligned_allocator<float>>& core
) {
    if (distance_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN core stage expected a dense N x N distance matrix");
    }
    if (min_samples < 2 || min_samples > N) {
        throw std::runtime_error("HDBSCAN min_samples must be in [2, N]");
    }

    if (core.size() != N) {
        core.resize(N);
    }

    const std::size_t k_zero_based = min_samples - 1;
    for (std::size_t row = 0; row < N; ++row) {
        core[row] = kth_smallest_value_eve(
            distance_matrix.data() + row * N,
            N,
            k_zero_based
        );
    }
}
inline void mutual_reachability_matrix(
    std::span<const float> distance_matrix,
    std::size_t N,
    std::size_t min_samples,
    std::vector<float, eve::aligned_allocator<float>>& mutual_reachability
) {
    if (distance_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN mreach stage expected a dense N x N distance matrix");
    }

    std::vector<float, eve::aligned_allocator<float>> core;
    core_distances(distance_matrix, N, min_samples, core);

    if (mutual_reachability.size() != N * N) {
        mutual_reachability.resize(N * N);
    }

    for (std::size_t row = 0; row < N; ++row) {
        const float core_i = core[row];
        const float* row_in = distance_matrix.data() + row * N;
        float* row_out = mutual_reachability.data() + row * N;

        auto distance_range = eve::algo::as_range(row_in, row_in + N);
        auto core_range = eve::algo::as_range(core.data(), core.data() + N);
        auto out_range = eve::algo::as_range(row_out, row_out + N);
        auto zipped = eve::views::zip(distance_range, core_range, out_range);

        eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
            zipped,
            [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
                auto [distance_it, core_it, out_it] = it;
                const auto distance = eve::load[ignore](distance_it);
                const auto core_j = eve::load[ignore](core_it);
                const auto mreach = eve::max(eve::max(distance, wide_f(core_i)), core_j);
                eve::store[ignore](mreach, out_it);
            }
        );
    }
}
