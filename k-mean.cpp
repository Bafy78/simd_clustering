#include <iostream>
#include <iomanip>
#include <vector>
#include <span>
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

struct Point2D : eve::struct_support<Point2D, float, float> {};
struct Point3D : eve::struct_support<Point3D, float, float, float> {};

template <eve::product_type PointType>
void print_clustering_results(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const int> assignments,
    const std::vector<PointType>& centroids
) {
    constexpr std::size_t num_dims = kumi::size_v<PointType>;
    if (points.size() == 0) return;

    int coord_width = num_dims * 8; 

    std::cout << std::left 
              << std::setw(coord_width) << "Point" 
              << std::setw(12) << "Cluster ID" 
              << "Centroid\n";
    std::cout << std::string(coord_width + 12 + coord_width, '-') << "\n";

    for (std::size_t i = 0; i < points.size(); ++i) {
        int k = assignments[i];
        auto pt = points.get(i);

        std::cout<< std::right << "(";
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((std::cout << std::setw(5) << get<I>(pt) << (I == num_dims - 1 ? "" : ", ")), ...);
        }(std::make_index_sequence<num_dims>{});
        std::cout << ") ";

        std::cout << std::setw(8) << k << "      ";

        std::cout << "(";
        auto c = centroids[k];
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((std::cout << std::setw(5) << get<I>(c) << (I == num_dims - 1 ? "" : ", ")), ...);
        }(std::make_index_sequence<num_dims>{});
        std::cout << ")\n";
    }
}

template <eve::product_type PointType>
void assign_points_to_centroids(
    const eve::algo::soa_vector<PointType>& points,
    const std::vector<PointType>& centroids, 
    std::span<int> assignments
) {
    auto zipped_data = eve::views::zip(points, assignments);

    eve::algo::for_each(
        zipped_data, 
        [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
            
            auto [pt_it, assign_it] = it; 

            auto pt = eve::load[ignore](pt_it); 

            auto min_distances = eve::valmax(eve::as<eve::wide<float>>());
            auto closest_centroid_indices = eve::zero(eve::as<eve::wide<int>>());

            for (std::size_t k = 0; k < centroids.size(); ++k) {
                auto total_dist_sq = eve::zero(eve::as<eve::wide<float>>());
                auto centroid = centroids[k]; 

                // Compile-time loop over the dimensions using fold expressions
                [&]<std::size_t... I>(std::index_sequence<I...>) {
                    (..., (
                        [&] {
                            // Extract the i-th dimension from the SIMD point and scalar centroid
                            auto diff = get<I>(pt) - get<I>(centroid);
                            total_dist_sq = eve::fma(diff, diff, total_dist_sq); 
                        }()
                    ));
                }(std::make_index_sequence<kumi::size_v<PointType>>{});
                
                auto is_closer = total_dist_sq < min_distances;
                min_distances = eve::min(min_distances, total_dist_sq);
                closest_centroid_indices = eve::if_else(is_closer, eve::wide<int>(k), closest_centroid_indices);
            }

            eve::store[ignore](closest_centroid_indices, assign_it);
        }
    );
}

template <eve::product_type PointType>
void update_centroids(
    const eve::algo::soa_vector<PointType>& points,
    std::span<const int> assignments,
    std::vector<PointType>& centroids,
    std::vector<PointType>& sum,
    std::vector<int>& counts
) {
    constexpr std::size_t num_dims = kumi::size_v<PointType>;

    std::fill(counts.begin(), counts.end(), 0);
    
    // Zero out the pre-allocated buffers using fold expressions
    for (auto& row : sum) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((get<I>(row) = 0.0f), ...);
        }(std::make_index_sequence<num_dims>{});
    }

    // Accumulate sums and counts
    for (std::size_t i = 0; i < points.size(); ++i) {
        int cluster_idx = assignments[i];
        counts[cluster_idx]++;
        
        auto pt = points.get(i);
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((get<I>(sum[cluster_idx]) += get<I>(pt)), ...);
        }(std::make_index_sequence<num_dims>{});
    }

    // Compute the new means and update the centroids
    for (std::size_t k = 0; k < centroids.size(); ++k) {
        if (counts[k] > 0) {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((get<I>(centroids[k]) = get<I>(sum[k]) / counts[k]), ...);
            }(std::make_index_sequence<num_dims>{});
        }
    }
}

// Helper: Calculate squared Frobenius norm of the shift between old and new centroids
template <eve::product_type PointType>
float calculate_centroid_shift_sq(
    const std::vector<PointType>& old_centroids,
    const std::vector<PointType>& new_centroids
) {
    constexpr std::size_t num_dims = kumi::size_v<PointType>;
    float shift_sq = 0.0f;
    
    for (std::size_t k = 0; k < new_centroids.size(); ++k) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (..., (
                [&] {
                    float diff = get<I>(new_centroids[k]) - get<I>(old_centroids[k]);
                    shift_sq += diff * diff;
                }()
            ));
        }(std::make_index_sequence<num_dims>{});
    }
    return shift_sq;
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

    std::vector<Point3D> centroids {
        Point3D{2.5f, 3.0f, 1.0f}, 
        Point3D{6.0f, 2.0f, 4.5f},
        Point3D{7.5f, 4.0f, 7.0f}
    };

    // Align these as we'll zip them together later
    std::vector<int, eve::aligned_allocator<int>> centroid_assignments(points.size(), -1);

    // Buffers for the update step
    std::vector<Point3D> sum(centroids.size());
    std::vector<int> counts(centroids.size(), 0);

    std::vector<Point3D> previous_centroids(centroids.size());

    bool converged = false;
    int max_iterations = 100;
    int iterations = 0;
    float tol = 1e-4f;

    while (!converged && iterations < max_iterations) {
        assign_points_to_centroids(points, centroids, centroid_assignments);

        previous_centroids = centroids;

        update_centroids(points, centroid_assignments, centroids, sum, counts);
        
        float shift_sq = calculate_centroid_shift_sq(previous_centroids, centroids);
        if (shift_sq <= tol * tol) {
            converged = true;
        }

        iterations++;
    }

    // Post-loop step: Reassign labels to perfectly match the final centroid positions
    assign_points_to_centroids(points, centroids, centroid_assignments);

    print_clustering_results(points, centroid_assignments, centroids);

    return 0;
}