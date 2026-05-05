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

#include "./simd_common.hpp"

// Permanently updates the minimum distance array with a new centroid
template <eve::product_type PointType, typename MinDistView>
void update_min_distances(
    const eve::algo::soa_vector<PointType>& points,
    MinDistView min_dist_view,
    const PointType& new_centroid
) {
    auto zipped = eve::views::zip(points, min_dist_view);
    eve::algo::for_each(zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
        auto [pt_it, dist_it] = it;
        auto pt = eve::load[ignore](pt_it);
        auto current_min_dist = eve::load[ignore](dist_it);
        
        auto dist_sq = compute_simd_dist_sq(pt, new_centroid);
        
        auto new_min = eve::min(current_min_dist, dist_sq);
        eve::store[ignore](new_min, dist_it);
    });
}

// Evaluates the potential total inertia if a candidate were chosen
template <eve::product_type PointType, typename MinDistView>
float evaluate_candidate_cost(
    const eve::algo::soa_vector<PointType>& points,
    MinDistView min_dist_view,
    const PointType& candidate
) {
    auto cost_accumulator = eve::zero(eve::as<eve::wide<float>>());
    auto zipped = eve::views::zip(points, min_dist_view);
    
    eve::algo::for_each(zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
        auto [pt_it, dist_it] = it;
        auto pt = eve::load[ignore](pt_it);
        auto current_min_dist = eve::load[ignore](dist_it);
        
        auto dist_sq = compute_simd_dist_sq(pt, candidate);                
        auto trial_min = eve::min(current_min_dist, dist_sq);
        
        // Mask out inactive lanes to prevent adding garbage data to the reduction sum
        auto mask = ignore.mask(eve::as<eve::wide<float>>());
        cost_accumulator += eve::if_else(mask, trial_min, eve::zero);
    });
    
    // Reduce the wide vector to a single scalar cost
    return eve::reduce(cost_accumulator);
}

// Main Function
template <eve::product_type PointType>
std::vector<PointType> greedy_kmeans_pp_init(
    const eve::algo::soa_vector<PointType>& points,
    int num_clusters,
    int num_local_trials = 0,
    unsigned int seed = std::random_device{}()
) {
    if (num_local_trials == 0) num_local_trials = 2 + static_cast<int>(std::log(num_clusters));

    std::vector<PointType> centroids;
    if (points.empty() || num_clusters <= 0) return centroids;
    
    centroids.reserve(num_clusters);
    std::mt19937 gen(seed);
    
    // First centroid: Choose uniformly at random
    std::uniform_int_distribution<std::size_t> uni_dist(0, points.size() - 1);
    centroids.push_back(points.get(uni_dist(gen)));

    if (num_clusters == 1) return centroids;

    // Maintain the minimum squared distance (D^2) from each point to the closest chosen centroid
    std::vector<float, eve::aligned_allocator<float>> min_sq_dist(
        points.size(), std::numeric_limits<float>::infinity()
    );
    
    auto min_dist_view = eve::algo::as_range(
        eve::as_aligned(min_sq_dist.data()), 
        min_sq_dist.data() + min_sq_dist.size()
    );

    // Update D^2 array for the very first randomly chosen centroid
    update_min_distances(points, min_dist_view, centroids.back());

    std::vector<float> cdf(points.size());

    // Iteratively select the remaining centroids
    for (int k = 1; k < num_clusters; ++k) {
        std::partial_sum(min_sq_dist.begin(), min_sq_dist.end(), cdf.begin());
        float total_weight = cdf.back();

        if (total_weight <= 0.0f) {
            // Edge case: All points perfectly overlap with existing centroids. Pick uniformly.
            centroids.push_back(points.get(uni_dist(gen)));
            update_min_distances(points, min_dist_view, centroids.back());
            continue;
        }

        std::uniform_real_distribution<float> prob_dist(0.0f, total_weight);
        
        float best_cost = std::numeric_limits<float>::infinity();
        std::size_t best_candidate_idx = 0;

        // The Greedy Step: Sample l candidates and evaluate them
        for (int trial = 0; trial < num_local_trials; ++trial) {
            // Probabilistic selection using binary search on the CDF
            float r = prob_dist(gen);
            auto it = std::ranges::lower_bound(cdf, r);
            std::size_t candidate_idx = std::min<std::size_t>(it - cdf.begin(), points.size() - 1);

            PointType candidate = points.get(candidate_idx);

            float current_trial_cost = evaluate_candidate_cost(points, min_dist_view, candidate);

            // Greedily track the candidate that yields the lowest overall inertia
            if (current_trial_cost < best_cost) {
                best_cost = current_trial_cost;
                best_candidate_idx = candidate_idx;
            }
        }

        // Finalize the best candidate and update the D^2 array permanently
        centroids.push_back(points.get(best_candidate_idx));
        update_min_distances(points, min_dist_view, centroids.back());
    }

    return centroids;
}