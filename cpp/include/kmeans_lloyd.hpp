#include <ranges>
#include <algorithm>
#include <vector>
#include <numeric>

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
    // Squared distance between two Kumi product points.
    auto get_dist_sq = [](const auto& p1, const auto& p2) {
        auto sq_diffs = kumi::map([](auto a, auto b) {
            double diff = static_cast<double>(a) - static_cast<double>(b);
            return diff * diff;
        }, p1, p2);

        return kumi::sum(sq_diffs, 0.0);
    };

    // sum[cluster] -= pt
    auto subtract_from_sum = [](auto& dst, const auto& pt) {
        kumi::for_each([](auto& d, auto p) {
            d -= static_cast<double>(p);
        }, dst, pt);
    };

    // sum[cluster] = pt
    //
    // In sklearn, centers_new[new_cluster_id, k] = X[far_idx, k] * weight.
    // Here weight is implicitly 1.
    auto set_sum_to_point = [](auto& dst, const auto& pt) {
        kumi::for_each([](auto& d, auto p) {
            d = static_cast<double>(p);
        }, dst, pt);
    };

    const std::size_t n_samples = points.size();
    const std::size_t n_clusters = old_centroids.size();

    // Find empty clusters, in cluster-id order, matching np.where(counts == 0)[0].
    std::vector<std::size_t> empty_clusters;
    empty_clusters.reserve(n_clusters);

    for (std::size_t k = 0; k < n_clusters; ++k) {
        if (counts[k] == 0) {
            empty_clusters.push_back(k);
        }
    }

    const std::size_t n_empty = empty_clusters.size();

    if (n_empty == 0) return;

    // Compute distance of each point to its currently assigned old centroid:
    // sklearn: distances = ((X - centers_old[labels]) ** 2).sum(axis=1)
    std::vector<double> distances(n_samples);

    double max_distance = 0.0;

    for (std::size_t i = 0; i < n_samples; ++i) {
        const int label = assignments[i];

        const double dist = get_dist_sq(points.get(i), old_centroids[label]);

        distances[i] = dist;
        max_distance = std::max(max_distance, dist);
    }

    // sklearn returns early when all distances are exactly zero.
    // This happens when there are more clusters than distinct non-duplicate samples.
    if (max_distance == 0.0) return;

    // Select the n_empty farthest points globally.
    // sklearn uses np.argpartition(...)[-n_empty:][::-1].
    std::vector<std::size_t> farthest_indices(n_samples);
    std::iota(farthest_indices.begin(), farthest_indices.end(), std::size_t{0});

    auto farther = [&](std::size_t a, std::size_t b) {
        if (distances[a] != distances[b]) {
            return distances[a] > distances[b];
        }

        return a < b;
    };

    std::partial_sort(
        farthest_indices.begin(),
        farthest_indices.begin() + static_cast<std::ptrdiff_t>(n_empty),
        farthest_indices.end(),
        farther
    );

    // Relocate each empty cluster using one farthest point.
    // Important sklearn detail:
    // labels / assignments are NOT changed here. Only the sums and counts are changed.
    for (std::size_t idx = 0; idx < n_empty; ++idx) {
        const std::size_t new_cluster_id = empty_clusters[idx];
        const std::size_t far_idx = farthest_indices[idx];

        const int old_cluster_id = assignments[far_idx];
        const auto pt = points.get(far_idx);

        subtract_from_sum(sum[old_cluster_id], pt);
        set_sum_to_point(sum[new_cluster_id], pt);

        counts[old_cluster_id] -= 1;
        counts[new_cluster_id] = 1;
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

    std::vector<int, eve::aligned_allocator<int>> previous_assignments(points.size(), -1);

    double scaled_tol = compute_sklearn_tolerance(points, tol);
    bool converged = false;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        previous_assignments = centroid_assignments;

        assign_points_to_centroids(zipped_data, centroids);

        previous_centroids = centroids;

        update_centroids(points, centroid_assignments, centroids, sum, counts);
        
        if (std::ranges::equal(centroid_assignments, previous_assignments)) {
            converged = true;
        } else {
            double shift_sq = calculate_centroid_shift_sq(previous_centroids, centroids);

            if (shift_sq <= scaled_tol) {
                converged = true;
            }
        }

        iterations++;
    }

    out_iterations = iterations;

    // Post-loop step: Reassign labels to perfectly match the final centroid positions
    assign_points_to_centroids(zipped_data, centroids);

    return centroid_assignments;
}