#include <iostream>
#include <iomanip>
#include <vector>
#include <span>
#include <ranges>
#include <random>
#include <numeric>
#include <algorithm>
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

struct Point3D : eve::struct_support<Point3D, float, float, float> {};

template <eve::product_type PointType>
void print_clustering_results(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const int> assignments,
    const std::vector<PointType>& centroids
) {
    constexpr std::size_t num_dims = kumi::size_v<PointType>;
    if (points.empty()) return;

    int coord_width = num_dims * 8; 

    std::cout << std::left 
              << std::setw(coord_width) << "Point" 
              << std::setw(12) << "Cluster ID" 
              << "Centroid\n";
    std::cout << std::string(coord_width + 12 + coord_width, '-') << "\n";

    auto print_point = [](const auto& point) {
        std::cout << "(";
        std::size_t dim_idx = 0;
        kumi::for_each([&](auto val) {
            std::cout << std::setw(5) << val 
                      << (++dim_idx == num_dims ? "" : ", ");
        }, point);
        
        std::cout << ")";
    };

    for (std::size_t i = 0; i < points.size(); ++i) {
        int k = assignments[i];

        std::cout << std::right;
        print_point(points.get(i)); 
        std::cout << " " << std::setw(8) << k << "      ";
        print_point(centroids[k]); 
        std::cout << "\n";
    }
}

// Helper lambda: Computes SIMD squared distance between a block of points and a centroid
constexpr auto compute_simd_dist_sq = [](const auto& pt, const auto& centroid) {
    auto dist_sq = eve::zero(eve::as<eve::wide<float>>());
    kumi::for_each([&](auto p, auto c) {
        auto diff = p - c;
        dist_sq = eve::fma(diff, diff, dist_sq); 
    }, pt, centroid);
    return dist_sq;
};

template <eve::algo::relaxed_range R, typename PointType>
void assign_points_to_centroids(
    R&& zipped_data, 
    const std::vector<PointType>& centroids
) {
    eve::algo::for_each(
        zipped_data, 
        [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
            
            auto [pt_it, assign_it] = it; 

            auto pt = eve::load[ignore](pt_it); 

            auto min_distances = eve::valmax(eve::as<eve::wide<float>>());
            auto closest_centroid_indices = eve::zero(eve::as<eve::wide<int>>());

            for (std::size_t k = 0; k < centroids.size(); ++k) {
                auto total_dist_sq = compute_simd_dist_sq(pt, centroids[k]);
                
                auto is_closer = total_dist_sq < min_distances;
                min_distances = eve::min(min_distances, total_dist_sq);
                closest_centroid_indices = eve::if_else(is_closer, eve::wide<int>(k), closest_centroid_indices);
            }

            eve::store[ignore](closest_centroid_indices, assign_it);
        }
    );
}

template <eve::product_type PointType>
void resolve_dead_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<int> assignments,
    const std::vector<PointType>& old_centroids,
    std::vector<PointType>& sum,
    std::vector<int>& counts
) {
    // Helper: Compute squared distance between two points using Kumi
    auto get_dist_sq = [](const auto& p1, const auto& p2) {
        auto sq_diffs = kumi::map([](auto a, auto b) { return (a - b) * (a - b); }, p1, p2);
        return kumi::sum(sq_diffs, 0.0f); 
    };

    // Helper: Move a point's value from one sum to another using Kumi
    auto transfer_sum = [](auto& from, auto& to, const auto& pt) {
        kumi::for_each([](auto& f, auto& t, auto p) {
            f -= p;
            t += p;
        }, from, to, pt);
    };

    for (std::size_t k = 0; k < old_centroids.size(); ++k) {
        if (counts[k] != 0) continue; 

        // Identify the biggest cluster
        auto max_it = std::ranges::max_element(counts);
        if (*max_it <= 1) break; 
        int biggest_cluster = std::ranges::distance(counts.begin(), max_it);

        // Find the farthest data point within that biggest cluster
        const auto& biggest_centroid = old_centroids[biggest_cluster];

        auto cluster_indices = std::views::iota(std::size_t{0}, points.size())
                             | std::views::filter([&](std::size_t i) { return assignments[i] == biggest_cluster; });

        auto farthest_it = std::ranges::max_element(cluster_indices, {}, [&](std::size_t i) {
            return get_dist_sq(points.get(i), biggest_centroid);
        });

        if (farthest_it == cluster_indices.end()) continue;
        
        int farthest_idx = *farthest_it;

        // Exclude this farthest point and reassign to the empty cluster
        assignments[farthest_idx] = k;
        counts[biggest_cluster]--;
        counts[k]++;

        // Update the sums in place
        transfer_sum(sum[biggest_cluster], sum[k], points.get(farthest_idx));
    }
}

template <eve::product_type PointType>
void update_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<int> assignments,
    std::vector<PointType>& centroids,
    std::vector<PointType>& sum,
    std::vector<int>& counts
) {
    eve::algo::fill(counts, 0);

    // Zero out the pre-allocated buffers
    for (auto& row : sum) {
        kumi::for_each([](auto& v) { v = 0.0f; }, row);
    }

    // Accumulate sums and counts
    for (std::size_t i = 0; i < points.size(); ++i) {
        int cluster_idx = assignments[i];
        counts[cluster_idx]++;

        auto pt = points.get(i);
        kumi::for_each([](auto& s, auto p) { s += p; }, sum[cluster_idx], pt);
    }

    resolve_dead_centroids(points, assignments, centroids, sum, counts);

    // Compute the new means and update the centroids
    for (std::size_t k = 0; k < centroids.size(); ++k) {
        if (counts[k] > 0) {
            kumi::for_each([count = counts[k]](auto& cent, auto s) { 
                cent = s / count; 
            }, centroids[k], sum[k]);
        }
    }
}

// Helper: Calculate squared Frobenius norm of the shift between old and new centroids
template <eve::product_type PointType>
float calculate_centroid_shift_sq(
    const std::vector<PointType>& old_centroids,
    const std::vector<PointType>& new_centroids
) {
    float shift_sq = 0.0f;
    
    for (std::size_t k = 0; k < new_centroids.size(); ++k) {
        kumi::for_each([&shift_sq](auto new_val, auto old_val) {
            float diff = new_val - old_val;
            shift_sq += diff * diff;
        }, new_centroids[k], old_centroids[k]);
    }
    return shift_sq;
}

template <eve::product_type PointType>
std::vector<int, eve::aligned_allocator<int>> k_means(
    const eve::algo::soa_vector<PointType>& points,
    std::vector<PointType>& centroids,
    int max_iterations = 100,
    float tol = 1e-4f
) {
    // Align these to zip them together in SIMD kernels
    std::vector<int, eve::aligned_allocator<int>> centroid_assignments(points.size(), -1);
    auto aligned_ptr = eve::as_aligned(centroid_assignments.data());
    auto unaligned_end = centroid_assignments.data() + points.size();
    auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
    auto zipped_data = eve::views::zip(points, assignments_range);

    // Buffers for the update step
    std::vector<PointType> sum(centroids.size());
    std::vector<int> counts(centroids.size(), 0);
    std::vector<PointType> previous_centroids(centroids.size());

    bool converged = false;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        assign_points_to_centroids(zipped_data, centroids);

        previous_centroids = centroids;

        update_centroids(points, centroid_assignments, centroids, sum, counts);
        
        float shift_sq = calculate_centroid_shift_sq(previous_centroids, centroids);
        if (shift_sq <= tol * tol) {
            converged = true;
        }

        iterations++;
    }

    // Post-loop step: Reassign labels to perfectly match the final centroid positions
    assign_points_to_centroids(zipped_data, centroids);

    return centroid_assignments;
}

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

    // Helper lambda: Updates the D^2 array when a new centroid is finalized
    auto update_distances = [&](const PointType& new_centroid) {
        auto zipped = eve::views::zip(points, min_dist_view);
        eve::algo::for_each(zipped, [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
            auto [pt_it, dist_it] = it;
            auto pt = eve::load[ignore](pt_it);
            auto current_min_dist = eve::load[ignore](dist_it);
            
            auto dist_sq = compute_simd_dist_sq(pt, new_centroid);
            
            auto new_min = eve::min(current_min_dist, dist_sq);
            eve::store[ignore](new_min, dist_it);
        });
    };

    // Update D^2 array for the very first randomly chosen centroid
    update_distances(centroids.back());

    std::vector<float> cdf(points.size());

    // Iteratively select the remaining centroids
    for (int k = 1; k < num_clusters; ++k) {
        std::partial_sum(min_sq_dist.begin(), min_sq_dist.end(), cdf.begin());
        float total_weight = cdf.back();

        if (total_weight <= 0.0f) {
            // Edge case: All points perfectly overlap with existing centroids. Pick uniformly.
            centroids.push_back(points.get(uni_dist(gen)));
            update_distances(centroids.back());
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
            std::size_t candidate_idx = it - cdf.begin();
            if (candidate_idx >= points.size()) candidate_idx = points.size() - 1;

            PointType candidate = points.get(candidate_idx);

            // Vectorized Evaluation: Calculate potential inertia if this candidate were chosen
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
            float current_trial_cost = eve::reduce(cost_accumulator);

            // Greedily track the candidate that yields the lowest overall inertia
            if (current_trial_cost < best_cost) {
                best_cost = current_trial_cost;
                best_candidate_idx = candidate_idx;
            }
        }

        // 4. Finalize the best candidate and update the D^2 array permanently
        centroids.push_back(points.get(best_candidate_idx));
        update_distances(centroids.back());
    }

    return centroids;
}

int main()
{
    eve::algo::soa_vector<Point3D> points {
        // Cluster 1 roughly around (1, 1, 1)
        Point3D{1.0f, 1.0f, 1.5f}, Point3D{1.2f, 1.1f, 1.4f}, Point3D{0.8f, 1.3f, 1.6f},
        Point3D{1.1f, 0.9f, 1.2f}, Point3D{1.5f, 1.2f, 1.7f}, Point3D{0.9f, 1.0f, 1.5f},
        Point3D{1.3f, 1.4f, 1.3f}, Point3D{1.0f, 1.2f, 1.1f}, Point3D{1.4f, 0.8f, 1.4f},
        Point3D{1.2f, 1.5f, 1.6f},
        
        // Cluster 2 roughly around (5, 5, 5)
        Point3D{5.0f, 5.0f, 5.0f}, Point3D{5.2f, 4.8f, 5.1f}, Point3D{4.9f, 5.3f, 4.8f},
        Point3D{5.1f, 5.1f, 5.2f}, Point3D{4.8f, 4.9f, 4.9f}, Point3D{5.3f, 5.0f, 5.3f},
        Point3D{5.0f, 5.2f, 4.7f}, Point3D{4.7f, 5.1f, 5.0f}, Point3D{5.2f, 5.3f, 5.1f},
        Point3D{4.9f, 4.7f, 5.2f},
        
        // Cluster 3 roughly around (8, 1, 8)
        Point3D{8.0f, 1.0f, 8.0f}, Point3D{8.1f, 1.2f, 7.8f}, Point3D{7.9f, 0.9f, 8.2f},
        Point3D{8.2f, 1.1f, 8.1f}, Point3D{7.8f, 1.0f, 7.9f}, Point3D{8.3f, 0.8f, 8.0f},
        Point3D{8.0f, 1.3f, 7.7f}, Point3D{7.7f, 1.1f, 8.3f}, Point3D{8.1f, 0.9f, 8.1f},
        Point3D{7.9f, 1.2f, 7.8f}
    };

    /*
    std::vector<Point3D> centroids {
        Point3D{2.5f, 3.0f, 1.0f}, 
        Point3D{6.0f, 2.0f, 4.5f},
        Point3D{7.5f, 4.0f, 7.0f}
    };
    */

    int num_clusters = 3;
    std::vector<Point3D> centroids = greedy_kmeans_pp_init(points, num_clusters);

    auto centroid_assignments = k_means(points, centroids);

    print_clustering_results(points, centroid_assignments, centroids);

    return 0;
}