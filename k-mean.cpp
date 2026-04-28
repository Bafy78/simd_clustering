#include <iostream>
#include <iomanip>
#include <eve/wide.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/module/algo.hpp>
#include <vector>

struct point2d : eve::struct_support<point2d, float, float> {};

void print_clustering_results(
    eve::algo::soa_vector<point2d> const& points,
    std::span<const int> assignments,
    eve::algo::soa_vector<point2d> const& centroids
) {
    std::cout << std::left 
              << std::setw(18) << "Point (x, y)" 
              << std::setw(12) << "Cluster ID" 
              << "Centroid (cx, cy)\n";
    std::cout << std::string(50, '-') << "\n";

    for (size_t i = 0; i < points.size(); ++i) {
        int k = assignments[i];

        auto [px, py] = points.get(i);
        auto [cx, cy] = centroids.get(k);

        std::cout << "(" << std::setw(4) << px << ", " 
                  << std::setw(4) << py << ")"
                  << std::right << std::setw(8) << k << "      "
                  << "(" << cx << ", " << cy << ")\n";
    }
}

void assign_points_to_centroids(
    eve::algo::soa_vector<point2d> const& points,
    eve::algo::soa_vector<point2d> const& centroids,
    std::span<int> assignments
) {
    eve::algo::transform_to[eve::algo::expensive_callable](points, assignments, 
        [&](auto batch) {
            auto [points_batch_x, points_batch_y] = batch;

            auto min_distances = eve::valmax(eve::as(points_batch_x));
            auto closest_centroid_indices = eve::convert(eve::zero(eve::as(points_batch_x)), eve::as<int>{});

            for (int k = 0; k < centroids.size(); ++k) {
                auto [cx, cy] = centroids.get(k);

                auto dist_sq_x = eve::sqr(points_batch_x - cx);
                auto y_diff = points_batch_y - cy;
                
                auto total_dist_sq = eve::fma(y_diff, y_diff, dist_sq_x);
                
                auto is_closer = total_dist_sq < min_distances;
                min_distances = eve::min(min_distances, total_dist_sq);
                
                closest_centroid_indices = eve::if_else(is_closer, k, closest_centroid_indices);
            }

            return closest_centroid_indices; 
        }
    );
}

void update_centroids(
    eve::algo::soa_vector<point2d> const& points,
    std::span<int> assignments,
    eve::algo::soa_vector<point2d>& centroids
) {
    std::size_t k = centroids.size();
    std::vector<float> sum_x(k, 0.0f);
    std::vector<float> sum_y(k, 0.0f);
    std::vector<int> counts(k, 0);

    // Accumulate sums and counts for each centroid
    for (std::size_t i = 0; i < points.size(); ++i) {
        int cluster_idx = assignments[i];
        auto [px, py] = points.get(i);
        
        sum_x[cluster_idx] += px;
        sum_y[cluster_idx] += py;
        counts[cluster_idx]++;
    }

    // Compute the new means and update the centroids
    for (std::size_t i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            centroids.set(i, point2d{sum_x[i] / counts[i], sum_y[i] / counts[i]});
        }
    }
}

int main()
{
    eve::algo::soa_vector<point2d> points {
        point2d{1.0f, 1.0f}, point2d{2.0f, 1.5f}, point2d{3.0f, 2.0f}, 
        point2d{4.0f, 2.5f}, point2d{5.0f, 3.0f}, point2d{6.0f, 3.5f}, 
        point2d{7.0f, 4.0f}, point2d{8.0f, 4.5f}
    };

    eve::algo::soa_vector<point2d> centroids {
        point2d{2.5f, 3.0f}, point2d{6.0f, 2.0f}
    };

    std::vector<int> centroid_assignments(points.size());
    std::vector<int> previous_assignments(points.size(), -1);

    bool converged = false;
    int max_iterations = 100;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        assign_points_to_centroids(points, centroids, centroid_assignments);

        converged = (centroid_assignments == previous_assignments);
        if (converged) break;
        previous_assignments = centroid_assignments;

        update_centroids(points, centroid_assignments, centroids);
        
        iterations++;
    }

    print_clustering_results(points, centroid_assignments, centroids);

  return 0;
}