#include <ranges>

#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "./simd_common.hpp"

template <typename PointType>
using AccumulatorType = kumi::result::fill_t<kumi::size_v<PointType>, double>;

template <eve::product_type PointType>
double compute_sklearn_tolerance(const eve::algo::soa_vector<PointType>& points, float tol) {
    if (tol == 0.0f) return 0.0;
    
    std::size_t n = points.size();
    if (n == 0) return 0.0;

    // Initialize accumulators for E[X] and E[X^2]
    AccumulatorType<PointType> sums;
    AccumulatorType<PointType> sq_sums;
    kumi::for_each([](auto& s, auto& sq) { s = 0.0; sq = 0.0; }, sums, sq_sums);

    // Accumulate sum and sum of squares for each feature
    for (std::size_t i = 0; i < n; ++i) {
        auto pt = points.get(i);
        kumi::for_each([](auto& s, auto& sq, auto p) { 
            double pd = static_cast<double>(p);
            s += pd; 
            sq += pd * pd;
        }, sums, sq_sums, pt);
    }

    double total_variance = 0.0;
    constexpr std::size_t num_features = kumi::size_v<PointType>;

    // Calculate variance for each feature: E[X^2] - (E[X])^2
    kumi::for_each([n, &total_variance](auto sum, auto sq_sum) {
        double mean = sum / n;
        double variance = (sq_sum / n) - (mean * mean);
        total_variance += variance;
    }, sums, sq_sums);

    double mean_variance = total_variance / num_features;

    // Scikit-learn returns mean(variances) * tol
    return mean_variance * static_cast<double>(tol);
}

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
    std::vector<AccumulatorType<PointType>>& sum,
    std::vector<int>& counts
) {
    // Helper: Compute squared distance between two points using Kumi
    auto get_dist_sq = [](const auto& p1, const auto& p2) {
    auto sq_diffs = kumi::map([](auto a, auto b) { 
        double diff = static_cast<double>(a) - static_cast<double>(b);
        return diff * diff; 
    }, p1, p2);
        return kumi::sum(sq_diffs, 0.0); 
    };

    // Helper: Move a point's value from one sum to another using Kumi
    auto transfer_sum = [](auto& from, auto& to, const auto& pt) {
        kumi::for_each([](auto& f, auto& t, auto p) {
            double pd = static_cast<double>(p);
            f -= pd;
            t += pd;
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
    std::vector<AccumulatorType<PointType>>& sum,
    std::vector<int>& counts
) {
    eve::algo::fill(counts, 0);

    // Zero out the pre-allocated buffers
    for (auto& row : sum) {
        kumi::for_each([](auto& v) { v = 0.0; }, row);
    }

    // Accumulate sums and counts
    for (std::size_t i = 0; i < points.size(); ++i) {
        int cluster_idx = assignments[i];
        counts[cluster_idx]++;

        auto pt = points.get(i);
        kumi::for_each([](auto& s, auto p) { s += static_cast<double>(p); }, sum[cluster_idx], pt);
    }

    resolve_dead_centroids(points, assignments, centroids, sum, counts);

    // Compute the new means and update the centroids
    for (std::size_t k = 0; k < centroids.size(); ++k) {
        if (counts[k] > 0) {
            kumi::for_each([count = counts[k]](auto& cent, auto s) { 
                cent = static_cast<float>(s / static_cast<double>(count));
            }, centroids[k], sum[k]);
        }
    }
}

// Helper: Calculate squared Frobenius norm of the shift between old and new centroids
template <eve::product_type PointType>
double calculate_centroid_shift_sq(
    const std::vector<PointType>& old_centroids,
    const std::vector<PointType>& new_centroids
) {
    double shift_sq = 0.0;
    
    for (std::size_t k = 0; k < new_centroids.size(); ++k) {
        kumi::for_each([&shift_sq](auto new_val, auto old_val) {
            double diff = static_cast<double>(new_val) - static_cast<double>(old_val);
            shift_sq += diff * diff;
        }, new_centroids[k], old_centroids[k]);
    }
    return shift_sq;
}

template <eve::product_type PointType>
std::vector<int, eve::aligned_allocator<int>> k_means(
    const eve::algo::soa_vector<PointType>& points,
    std::vector<PointType>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    // Align these to zip them together in SIMD kernels
    std::vector<int, eve::aligned_allocator<int>> centroid_assignments(points.size(), -1);
    auto aligned_ptr = eve::as_aligned(centroid_assignments.data());
    auto unaligned_end = centroid_assignments.data() + points.size();
    auto assignments_range = eve::algo::as_range(aligned_ptr, unaligned_end);
    auto zipped_data = eve::views::zip(points, assignments_range);

    // Buffers for the update step
    std::vector<AccumulatorType<PointType>> sum(centroids.size());
    std::vector<int> counts(centroids.size(), 0);
    std::vector<PointType> previous_centroids(centroids.size());

    double scaled_tol = compute_sklearn_tolerance(points, tol);
    bool converged = false;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        assign_points_to_centroids(zipped_data, centroids);

        previous_centroids = centroids;

        update_centroids(points, centroid_assignments, centroids, sum, counts);
        
        double shift_sq = calculate_centroid_shift_sq(previous_centroids, centroids);
        if (shift_sq <= scaled_tol) {
            converged = true;
        }

        iterations++;
    }

    out_iterations = iterations;

    // Post-loop step: Reassign labels to perfectly match the final centroid positions
    assign_points_to_centroids(zipped_data, centroids);

    return centroid_assignments;
}