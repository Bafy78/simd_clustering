#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <limits>
#include <algorithm>

#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

// Computes SIMD squared distance between a block of samples and a centroid
constexpr auto compute_simd_dist_sq = [](const auto& sample, const auto& centroid) {
    auto dist_sq = eve::zero(eve::as<eve::wide<float>>());
    kumi::for_each([&](auto p, auto c) {
        auto diff = p - c;
        dist_sq = eve::fma(diff, diff, dist_sq);
    }, sample, centroid);
    return dist_sq;
};

// Permanently updates the minimum distance array with a new centroid
template <eve::product_type SampleType, typename MinDistView>
void update_min_distances(
    const eve::algo::soa_vector<SampleType>& samples,
    MinDistView min_dist_view,
    const SampleType& new_centroid
) {
    auto zipped = eve::views::zip(samples, min_dist_view);
    eve::algo::for_each(zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
        auto [sample_it, dist_it] = it;
        auto sample = eve::load[ignore](sample_it);
        auto current_min_dist = eve::load[ignore](dist_it);

        auto dist_sq = compute_simd_dist_sq(sample, new_centroid);

        auto new_min = eve::min(current_min_dist, dist_sq);
        eve::store[ignore](new_min, dist_it);
    });
}

// Evaluates the potential total inertia if a candidate were chosen
template <eve::product_type SampleType, typename MinDistView>
float evaluate_candidate_cost(
    const eve::algo::soa_vector<SampleType>& samples,
    MinDistView min_dist_view,
    const SampleType& candidate
) {
    auto cost_accumulator = eve::zero(eve::as<eve::wide<float>>());
    auto zipped = eve::views::zip(samples, min_dist_view);

    eve::algo::for_each(zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
        auto [sample_it, dist_it] = it;
        auto sample = eve::load[ignore](sample_it);
        auto current_min_dist = eve::load[ignore](dist_it);

        auto dist_sq = compute_simd_dist_sq(sample, candidate);
        auto trial_min = eve::min(current_min_dist, dist_sq);

        // Mask out inactive lanes to prevent adding garbage data to the reduction sum
        auto mask = ignore.mask(eve::as<eve::wide<float>>());
        cost_accumulator += eve::if_else(mask, trial_min, eve::zero);
    });

    // Reduce the wide vector to a single scalar cost
    return eve::reduce(cost_accumulator);
}

// Main Function
template <eve::product_type SampleType>
std::vector<SampleType> greedy_kmeans_pp_init(
    const eve::algo::soa_vector<SampleType>& samples,
    int K,
    int num_local_trials = 0,
    unsigned int seed = std::random_device{}()
) {
    if (num_local_trials == 0) num_local_trials = 2 + static_cast<int>(std::log(K));

    std::vector<SampleType> centroids;
    if (samples.empty() || K <= 0) return centroids;

    centroids.reserve(K);
    std::mt19937 gen(seed);

    // First centroid: Choose uniformly at random
    std::uniform_int_distribution<std::size_t> uni_dist(0, samples.size() - 1);
    centroids.push_back(samples.get(uni_dist(gen)));

    if (K == 1) return centroids;

    // Maintain the minimum squared distance (D^2) from each sample to the closest chosen centroid
    std::vector<float, eve::aligned_allocator<float>> min_sq_dist(
        samples.size(), std::numeric_limits<float>::infinity()
    );

    auto min_dist_view = eve::algo::as_range(
        eve::as_aligned(min_sq_dist.data()),
        min_sq_dist.data() + min_sq_dist.size()
    );

    // Update D^2 array for the very first randomly chosen centroid
    update_min_distances(samples, min_dist_view, centroids.back());

    std::vector<float> cdf(samples.size());

    // Iteratively select the remaining centroids
    for (int k = 1; k < K; ++k) {
        std::partial_sum(min_sq_dist.begin(), min_sq_dist.end(), cdf.begin());
        float total_weight = cdf.back();

        if (total_weight <= 0.0f) {
            // Edge case: All samples perfectly overlap with existing centroids. Pick uniformly.
            centroids.push_back(samples.get(uni_dist(gen)));
            update_min_distances(samples, min_dist_view, centroids.back());
            continue;
        }

        std::uniform_real_distribution<float> prob_dist(0.0f, total_weight);

        float best_cost = std::numeric_limits<float>::infinity();
        std::size_t best_candidate_n = 0;

        // The Greedy Step: Sample l candidates and evaluate them
        for (int trial = 0; trial < num_local_trials; ++trial) {
            // Probabilistic selection using binary search on the CDF
            float r = prob_dist(gen);
            auto it = std::ranges::lower_bound(cdf, r);
            std::size_t candidate_n = std::min<std::size_t>(it - cdf.begin(), samples.size() - 1);

            SampleType candidate = samples.get(candidate_n);

            float current_trial_cost = evaluate_candidate_cost(samples, min_dist_view, candidate);

            // Greedily track the candidate that yields the lowest overall inertia
            if (current_trial_cost < best_cost) {
                best_cost = current_trial_cost;
                best_candidate_n = candidate_n;
            }
        }

        // Finalize the best candidate and update the D^2 array permanently
        centroids.push_back(samples.get(best_candidate_n));
        update_min_distances(samples, min_dist_view, centroids.back());
    }

    return centroids;
}