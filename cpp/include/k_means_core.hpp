#pragma once

#include <ranges>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <array>
#include <vector>

#ifndef KMEANS_MAX_CLUSTERS
#define KMEANS_MAX_CLUSTERS 255
#endif

namespace kmeans {

template<std::size_t MaxClusters>
using label_for_max_clusters_t =
std::conditional_t<
    MaxClusters <= 255,
    std::uint8_t,
    std::conditional_t<
    MaxClusters <= 65535,
    std::uint16_t,
    std::uint32_t
    >
>;

using default_label_t = label_for_max_clusters_t<KMEANS_MAX_CLUSTERS>;

template<class Label>
inline constexpr Label invalid_label_v = std::numeric_limits<Label>::max();

template<class Label>
inline constexpr std::size_t max_clusters_for_label_v =
static_cast<std::size_t>(std::numeric_limits<Label>::max());

template<class Label>
inline void check_cluster_count_fits(std::size_t n_clusters) {
    static_assert(std::is_integral_v<Label>);

    if (n_clusters > max_clusters_for_label_v<Label>) {
        throw std::invalid_argument("n_clusters exceeds label_t capacity");
    }
}

template<std::size_t D, typename Function>
inline constexpr void for_each_feature(Function&& f) {
    auto&& fn = f;
    kumi::for_each_index(
        [&](auto index, auto) { fn(index); },
        kumi::fill<D>(0)
    );
}

template<std::size_t D, typename SampleValue>
inline float compute_sklearn_tolerance_common(
    std::size_t n_samples,
    SampleValue&& sample_value,
    float tol
) {
    if (tol == 0.0f) return 0.0f;
    if (n_samples == 0) return 0.0f;

    std::array<float, D> means{};

    for_each_feature<D>([&](auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;

        float sum = 0.0f;

        for (std::size_t i = 0; i < n_samples; ++i) {
            sum += sample_value(feature_index, i);
        }

        means[d] = sum / static_cast<float>(n_samples);
    });

    float total_variance = 0.0f;

    for_each_feature<D>([&](auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;

        const float mean = means[d];
        float variance = 0.0f;

        for (std::size_t i = 0; i < n_samples; ++i) {
            const float diff = sample_value(feature_index, i) - mean;
            variance += diff * diff;
        }

        total_variance += variance / static_cast<float>(n_samples);
    });

    const float mean_variance = total_variance / static_cast<float>(D);

    return mean_variance * tol;
}

template<std::size_t D, typename OldCentroidValue, typename NewCentroidValue>
inline float calculate_centroid_shift_sq_common(
    std::size_t n_clusters,
    OldCentroidValue&& old_centroid_value,
    NewCentroidValue&& new_centroid_value
) {
    float shift_sq = 0.0f;

    for (std::size_t k = 0; k < n_clusters; ++k) {
        for_each_feature<D>([&](auto feature_index) {
            const float diff =
                new_centroid_value(k, feature_index)
                - old_centroid_value(k, feature_index);

            shift_sq += diff * diff;
        });
    }

    return shift_sq;
}

template<class Ops, class Label>
void resolve_dead_centroids_common(
    Ops& ops,
    std::span<const Label> assignments,
    std::span<int> counts
) {
    const std::size_t n_samples = ops.n_samples();
    const std::size_t n_clusters = counts.size();

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
    std::vector<float> distances(n_samples);

    float max_distance = 0.0f;

    for (std::size_t i = 0; i < n_samples; ++i) {
        const std::size_t label = static_cast<std::size_t>(assignments[i]);

        const float dist = ops.distance_to_old_centroid(i, label);

        distances[i] = dist;
        max_distance = std::max(max_distance, dist);
    }

    // sklearn returns early when all distances are exactly zero.
    // This happens when there are more clusters than distinct non-duplicate samples.
    if (max_distance == 0.0f) return;

    // Select the n_empty farthest points globally.
    // sklearn uses np.argpartition(...)[-n_empty:][::-1].
    std::vector<std::size_t> farthest_indices(n_samples);
    std::iota(
        farthest_indices.begin(),
        farthest_indices.end(),
        std::size_t{ 0 }
    );

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
    //
    // Important sklearn detail:
    // labels / assignments are NOT changed here. Only the sums and counts are changed.
    for (std::size_t idx = 0; idx < n_empty; ++idx) {
        const std::size_t new_cluster_id = empty_clusters[idx];
        const std::size_t far_idx = farthest_indices[idx];

        const std::size_t old_cluster_id = static_cast<std::size_t>(assignments[far_idx]);

        ops.relocate_empty_cluster(
            old_cluster_id,
            new_cluster_id,
            far_idx
        );

        counts[old_cluster_id] -= 1;
        counts[new_cluster_id] = 1;
    }
}

template<class Ops, class Label>
void update_centroids_common(
    Ops& ops,
    std::span<const Label> assignments,
    std::span<int> counts
) {
    ops.reset_sums();
    std::fill(counts.begin(), counts.end(), 0);

    for (std::size_t i = 0; i < ops.n_samples(); ++i) {
        const std::size_t cluster_idx = static_cast<std::size_t>(assignments[i]);

        counts[cluster_idx]++;
        ops.add_point_to_sum(cluster_idx, i);
    }

    ops.resolve_dead_centroids(assignments, counts);

    for (std::size_t k = 0; k < ops.n_clusters(); ++k) {
        if (counts[k] > 0) {
            ops.write_centroid_from_sum(k, counts[k]);
        }
    }

    ops.after_centroids_updated();
}

template<class Backend>
auto k_means_core(
    Backend& backend,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) -> typename Backend::assignment_vector {
    using label_type = typename Backend::label_type;

    if constexpr (requires(const Backend & b) { b.check_cluster_count(); }) {
        backend.check_cluster_count();
    }

    auto assignments = backend.make_assignment_vector(invalid_label_v<label_type>);
    auto counts = backend.make_counts_vector();
    auto previous_centroids = backend.make_centroid_snapshot();

    const float scaled_tol = backend.compute_tolerance(tol);

    bool converged = false;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        const bool labels_changed = backend.assign_and_check_changed(assignments);

        backend.save_centroids(previous_centroids);

        backend.update_centroids(assignments, counts);

        if (!labels_changed) {
            converged = true;
        } else {
            const float shift_sq = backend.centroid_shift_sq(previous_centroids);

            if (shift_sq <= scaled_tol) {
                converged = true;
            }
        }

        ++iterations;
    }

    out_iterations = iterations;

    // Final label refresh against final centroid positions.
    backend.assign(assignments);

    return assignments;
}

}
