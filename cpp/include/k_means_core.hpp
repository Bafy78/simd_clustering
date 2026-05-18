#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <array>
#include <vector>
#include <utility>

namespace kmeans {


inline void check_cluster_count(std::size_t n_clusters, std::size_t n_samples) {
    if (n_clusters == 0) {
        throw std::invalid_argument("n_clusters must be greater than zero");
    }

    if (n_clusters > n_samples) {
        throw std::invalid_argument("n_clusters must be <= n_samples");
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
inline std::pair<std::array<float, D>, float>
compute_feature_mean_and_sklearn_tolerance_common(
    std::size_t n_samples,
    SampleValue&& sample_value,
    float tol
) {
    std::array<float, D> means{};

    if (n_samples == 0) return { means, 0.0f };

    for_each_feature<D>([&](auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;

        float sum = 0.0f;

        for (std::size_t i = 0; i < n_samples; ++i) {
            sum += sample_value(feature_index, i);
        }

        means[d] = sum / static_cast<float>(n_samples);
    });

    if (tol == 0.0f) return { means, 0.0f };

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

    return { means, mean_variance * tol };
}

template<std::size_t D, typename SampleValue>
inline float compute_sklearn_tolerance_common(
    std::size_t n_samples,
    SampleValue&& sample_value,
    float tol
) {
    return compute_feature_mean_and_sklearn_tolerance_common<D>(
        n_samples,
        std::forward<SampleValue>(sample_value),
        tol
    ).second;
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

template<class Ops>
void resolve_dead_centroids_common(
    Ops& ops,
    std::span<const int> assignments,
    std::span<int> counts
) {
    const std::size_t n_samples = ops.n_samples();
    const std::size_t n_clusters = counts.size();

    // Find empty clusters, in cluster-id order, matching np.where(counts == 0)[0].
    std::vector<int> empty_clusters;
    empty_clusters.reserve(n_clusters);

    for (std::size_t k = 0; k < n_clusters; ++k) {
        if (counts[k] == 0) {
            empty_clusters.push_back(static_cast<int>(k));
        }
    }

    const std::size_t n_empty = empty_clusters.size();

    if (n_empty == 0) return;

    // Dead-cluster repair keeps sample ids in uint32_t. 
    // This path assumes n_samples <= UINT32_MAX
    if (n_samples > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("dead-centroid repair requires n_samples <= UINT32_MAX");
    }

    struct candidate_t {
        float distance;
        std::uint32_t index;
    };

    // Compute distance of each point to its currently assigned old centroid:
    // sklearn: distances = ((X - centers_old[labels]) ** 2).sum(axis=1)
    std::vector<candidate_t> farthest_candidates;

    for (std::size_t i = 0; i < n_samples; ++i) {
        const std::size_t assigned_cluster = static_cast<std::size_t>(assignments[i]);
        const float dist = ops.distance_to_old_centroid(i, assigned_cluster);

        if (farthest_candidates.empty()) {
            if (dist == 0.0f) continue;

            farthest_candidates.reserve(n_samples);

            for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(i); ++j) {
                farthest_candidates.push_back({ 0.0f, j });
            }
        }

        farthest_candidates.push_back({ dist, static_cast<std::uint32_t>(i) });
    }

    // sklearn returns early when all distances are exactly zero.
    // This happens when there are more clusters than distinct non-duplicate samples.
    if (farthest_candidates.empty()) return;

    // Select the n_empty farthest points globally.
    // sklearn uses np.argpartition(...)[-n_empty:][::-1].
    auto farther = [](const candidate_t& a, const candidate_t& b) {
        if (a.distance != b.distance) {
            return a.distance > b.distance;
        }

        return a.index < b.index;
    };

    std::partial_sort(
        farthest_candidates.begin(),
        farthest_candidates.begin() + static_cast<std::ptrdiff_t>(n_empty),
        farthest_candidates.end(),
        farther
    );

    // Relocate each empty cluster using one farthest point.
    //
    // Important sklearn detail:
    // labels / assignments are NOT changed here. Only the sums and counts are changed.
    for (std::size_t idx = 0; idx < n_empty; ++idx) {
        const std::size_t new_cluster_id = static_cast<std::size_t>(empty_clusters[idx]);
        const std::size_t far_idx = static_cast<std::size_t>(farthest_candidates[idx].index);

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

template<class Ops>
void update_centroids_common(
    Ops& ops,
    std::span<const int> assignments,
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
    backend.check_cluster_count();

    auto assignments = backend.make_assignment_vector(-1);
    auto counts = backend.make_counts_vector();
    auto previous_centroids = backend.make_centroid_snapshot();

    const float scaled_tol = backend.prepare_data_for_fit(tol);

    bool converged = false;
    bool strict_convergence = false;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        const bool labels_changed = backend.assign_and_check_changed(assignments);

        backend.save_centroids(previous_centroids);

        backend.update_centroids(assignments, counts);

        if (!labels_changed) {
            strict_convergence = true;
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

    // sklearn-like final label refresh:
    // If Lloyd stopped by strict label convergence, labels already match the final centers.
    // If it stopped by tolerance or max_iterations, rerun the E-step while still centered.
    if (!strict_convergence) {
        backend.assign(assignments);
    }

    backend.finish_fit_after_final_assignment();

    return assignments;
}

}
