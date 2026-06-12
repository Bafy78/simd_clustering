#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../pp_core.hpp"

// Computes SIMD squared distance between a block of samples and a centroid.
constexpr auto compute_simd_dist_sq = [](const auto& sample, const auto& centroid) {
    auto dist_sq = wide_zero_f;

    kumi::for_each([&](auto p, auto c) {
        auto diff = p - c;
        dist_sq = eve::fma(diff, diff, dist_sq);
    }, sample, centroid);

    return dist_sq;
};

template <eve::product_type SampleType, typename MinDistView>
void update_min_distances(
    const eve::algo::soa_vector<SampleType>& samples,
    MinDistView min_dist_view,
    const SampleType& new_centroid
) {
    auto zipped = eve::views::zip(samples, min_dist_view);

    eve::algo::for_each[eve::algo::force_cardinal<kmeans_pp::cardinal{}()>]
    (zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
        auto [sample_it, dist_it] = it;
        auto sample = eve::load[ignore](sample_it);
        auto current_min_dist = eve::load[ignore](dist_it);

        auto dist_sq = compute_simd_dist_sq(sample, new_centroid);

        auto new_min = eve::min(current_min_dist, dist_sq);
        eve::store[ignore](new_min, dist_it);
    });
}

template <eve::product_type SampleType, typename MinDistView>
float evaluate_candidate_cost(
    const eve::algo::soa_vector<SampleType>& samples,
    MinDistView min_dist_view,
    const SampleType& candidate
) {
    auto cost_accumulator = wide_zero_f;
    auto zipped = eve::views::zip(samples, min_dist_view);

    eve::algo::for_each[eve::algo::force_cardinal<kmeans_pp::cardinal{}()>]
    (zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
        auto [sample_it, dist_it] = it;
        auto sample = eve::load[ignore](sample_it);
        auto current_min_dist = eve::load[ignore](dist_it);

        auto dist_sq = compute_simd_dist_sq(sample, candidate);
        auto trial_min = eve::min(current_min_dist, dist_sq);

        auto mask = ignore.mask(eve::as<wide_f>());
        cost_accumulator += eve::if_else(mask, trial_min, eve::zero);
    });

    return eve::reduce(cost_accumulator);
}

template <eve::product_type SampleType>
struct static_kmeans_pp_backend {
    using centroids_type = std::vector<SampleType>;

    static constexpr std::size_t local_trial_tile_size = 1;

    const eve::algo::soa_vector<SampleType>& samples;

    std::size_t N() const {
        return samples.size();
    }

    centroids_type make_centroids(std::size_t K) const {
        centroids_type centroids;
        centroids.reserve(K);
        return centroids;
    }

    void set_centroid_from_sample(
        centroids_type& centroids,
        std::size_t k,
        std::size_t n
    ) const {
        if (centroids.size() == k) {
            centroids.push_back(samples.get(n));
        } else {
            centroids[k] = samples.get(n);
        }
    }

    void update_min_distances(
        std::span<float> min_sq_dist,
        const centroids_type& centroids,
        std::size_t centroid_k
    ) const {
        auto min_dist_view = eve::algo::as_range(
            eve::as_aligned(min_sq_dist.data()),
            min_sq_dist.data() + min_sq_dist.size()
        );

        ::update_min_distances(
            samples,
            min_dist_view,
            centroids[centroid_k]
        );
    }

    void evaluate_candidate_costs(
        std::span<const float> min_sq_dist,
        std::span<const std::size_t> candidate_indices,
        std::span<float> trial_costs
    ) const {
        auto min_dist_view = eve::algo::as_range(
            eve::as_aligned(min_sq_dist.data()),
            min_sq_dist.data() + min_sq_dist.size()
        );

        for (std::size_t t = 0; t < candidate_indices.size(); ++t) {
            const SampleType candidate = samples.get(candidate_indices[t]);

            trial_costs[t] = ::evaluate_candidate_cost(
                samples,
                min_dist_view,
                candidate
            );
        }
    }
};

template <eve::product_type SampleType>
std::vector<SampleType> greedy_kmeans_pp_init(
    const eve::algo::soa_vector<SampleType>& samples,
    int K,
    int num_local_trials = 0,
    unsigned int seed = std::random_device{}()
) {
    static_kmeans_pp_backend<SampleType> backend{ samples };

    return kmeans_pp::greedy_kmeans_pp_core(
        backend,
        K,
        num_local_trials,
        seed
    );
}
