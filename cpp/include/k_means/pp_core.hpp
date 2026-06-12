#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include <eve/memory/aligned_allocator.hpp>

namespace kmeans_pp {

using kmeans_pp_min_distance_vector = std::vector<float, eve::aligned_allocator<float>>;

// Shared greedy k-means++ seeding loop.
//
// The backend owns only representation-specific operations:
//   * centroid result storage
//   * min-distance update for one accepted centroid
//   * cost evaluation for a tile of local-trial candidates
template<class Backend>
auto greedy_kmeans_pp_core(
    Backend& backend,
    int K,
    int num_local_trials = 0,
    unsigned int seed = std::random_device{}()
) -> typename Backend::centroids_type {
    if (num_local_trials == 0) {
        num_local_trials = 2 + static_cast<int>(std::log(K));
    }

    auto centroids = backend.make_centroids(static_cast<std::size_t>(std::max(K, 0)));

    if (backend.N() == 0 || K <= 0) {
        return centroids;
    }

    const std::size_t N = backend.N();
    const std::size_t K_size = static_cast<std::size_t>(K);

    std::mt19937 gen(seed);

    // First centroid: Choose uniformly at random
    std::uniform_int_distribution<std::size_t> uni_dist(0, N - 1);
    backend.set_centroid_from_sample(
        centroids,
        0,
        uni_dist(gen)
    );

    if (K == 1) {
        return centroids;
    }

    kmeans_pp_min_distance_vector min_sq_dist(
        N,
        std::numeric_limits<float>::infinity()
    );

    backend.update_min_distances(
        std::span<float>{ min_sq_dist.data(), min_sq_dist.size() },
        centroids,
        0
    );

    std::vector<float> cdf(N);

    constexpr std::size_t local_trial_tile_size = Backend::local_trial_tile_size;
    static_assert(local_trial_tile_size > 0);

    std::array<std::size_t, local_trial_tile_size> candidate_indices{};
    std::array<float, local_trial_tile_size> trial_costs{};

    for (std::size_t k = 1; k < K_size; ++k) {
        std::partial_sum(min_sq_dist.begin(), min_sq_dist.end(), cdf.begin());
        const float total_weight = cdf.back();

        if (total_weight <= 0.0f) {
            // Edge case: All samples perfectly overlap with existing centroids. Pick uniformly.
            backend.set_centroid_from_sample(
                centroids,
                k,
                uni_dist(gen)
            );

            backend.update_min_distances(
                std::span<float>{ min_sq_dist.data(), min_sq_dist.size() },
                centroids,
                k
            );

            continue;
        }

        std::uniform_real_distribution<float> prob_dist(0.0f, total_weight);

        float best_cost = std::numeric_limits<float>::infinity();
        std::size_t best_candidate_n = 0;

        int trial = 0;

        while (trial < num_local_trials) {
            const std::size_t remaining = static_cast<std::size_t>(num_local_trials - trial);
            const std::size_t trial_count = std::min(local_trial_tile_size, remaining);

            for (std::size_t t = 0; t < trial_count; ++t) {
                const float r = prob_dist(gen);
                auto it = std::lower_bound(cdf.begin(), cdf.end(), r);

                candidate_indices[t] = std::min<std::size_t>(
                    static_cast<std::size_t>(it - cdf.begin()),
                    N - 1
                );
            }

            backend.evaluate_candidate_costs(
                std::span<const float>{ min_sq_dist.data(), min_sq_dist.size() },
                std::span<const std::size_t>{ candidate_indices.data(), trial_count },
                std::span<float>{ trial_costs.data(), trial_count }
            );

            for (std::size_t t = 0; t < trial_count; ++t) {
                if (trial_costs[t] < best_cost) {
                    best_cost = trial_costs[t];
                    best_candidate_n = candidate_indices[t];
                }
            }

            trial += static_cast<int>(trial_count);
        }

        backend.set_centroid_from_sample(
            centroids,
            k,
            best_candidate_n
        );

        backend.update_min_distances(
            std::span<float>{ min_sq_dist.data(), min_sq_dist.size() },
            centroids,
            k
        );
    }

    return centroids;
}

} // namespace kmeans_pp
