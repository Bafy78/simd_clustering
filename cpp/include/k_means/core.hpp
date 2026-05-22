#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>

#include <eve/wide.hpp>

namespace kmeans {

using wide_f = eve::wide<float>;
using cardinal = typename wide_f::cardinal_type;
using wide_i = eve::wide<int, cardinal>;

inline void check_cluster_count(std::size_t n_clusters, std::size_t n_samples) {
    if (n_clusters == 0) {
        throw std::invalid_argument("n_clusters must be greater than zero");
    }

    if (n_clusters > n_samples) {
        throw std::invalid_argument("n_clusters must be <= n_samples");
    }
}

template<class Ops>
bool resolve_dead_centroids_common(
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

    if (n_empty == 0) return false;

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
    if (farthest_candidates.empty()) return false;

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

    return true;
}

template<class Ops>
bool update_centroids_common(
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

    const bool dead_repair_happened = ops.resolve_dead_centroids(assignments, counts);

    for (std::size_t k = 0; k < ops.n_clusters(); ++k) {
        if (counts[k] > 0) {
            ops.write_centroid_from_sum(k, counts[k]);
        }
    }

    return dead_repair_happened;
}


template<class IntVector>
std::span<const int> const_int_span(const IntVector& values) {
    return std::span<const int>{ values.data(), values.size() };
}

template<class IntVector>
std::span<int> mutable_int_span(IntVector& values) {
    return std::span<int>{ values.data(), values.size() };
}

template<class AssignmentVector>
struct incremental_centroid_update_state {
    AssignmentVector previous_assignments;
    std::vector<int> dirty_clusters;
    std::vector<unsigned char> dirty_marker;

    // True means sums/counts are exactly derivable from the current assignments.
    // Dead-centroid repair makes this false because sklearn-style repair changes
    // sums/counts but intentionally leaves labels unchanged.
    bool sums_match_assignments = false;

    explicit incremental_centroid_update_state(std::size_t n_clusters = 0)
        : dirty_marker(n_clusters, 0) {}

    void snapshot_before_assignment(std::span<const int> assignments) {
        if (!sums_match_assignments) return;

        previous_assignments.assign(assignments.begin(), assignments.end());
    }

    std::span<const int> previous_assignments_span() const {
        return const_int_span(previous_assignments);
    }

    std::span<const int> dirty_clusters_span() const {
        return const_int_span(dirty_clusters);
    }

    void clear_dirty_clusters() {
        for (int cluster_idx : dirty_clusters) {
            dirty_marker[static_cast<std::size_t>(cluster_idx)] = 0;
        }

        dirty_clusters.clear();
    }

    void mark_dirty_cluster(std::size_t cluster_idx) {
        if (dirty_marker[cluster_idx] != 0) return;

        dirty_marker[cluster_idx] = 1;
        dirty_clusters.push_back(static_cast<int>(cluster_idx));
    }
};


template<class Ops>
std::size_t count_assignment_changes_until_common(
    Ops& ops,
    std::span<const int> previous_assignments,
    std::span<const int> assignments,
    std::size_t max_changes
) {
    std::size_t changes = 0;

    for (std::size_t i = 0; i < ops.n_samples(); ++i) {
        if (previous_assignments[i] == assignments[i]) continue;

        ++changes;
        if (changes > max_changes) return changes;
    }

    return changes;
}

template<class Ops>
void apply_assignment_deltas_common(
    Ops& ops,
    std::span<const int> previous_assignments,
    std::span<const int> assignments,
    std::span<int> counts
) {
    for (std::size_t i = 0; i < ops.n_samples(); ++i) {
        const int old_label_int = previous_assignments[i];
        const int new_label_int = assignments[i];

        if (old_label_int == new_label_int) continue;

        if (old_label_int >= 0) {
            const std::size_t old_label = static_cast<std::size_t>(old_label_int);

            counts[old_label] -= 1;
            ops.subtract_point_from_sum(old_label, i);
            ops.mark_dirty_cluster(old_label);
        }

        const std::size_t new_label = static_cast<std::size_t>(new_label_int);

        counts[new_label] += 1;
        ops.add_point_to_sum(new_label, i);
        ops.mark_dirty_cluster(new_label);
    }
}

template<class Ops>
bool try_update_centroids_incremental_common(
    Ops& ops,
    std::span<const int> previous_assignments,
    std::span<const int> assignments,
    std::span<int> counts,
    std::size_t max_incremental_changes,
    bool& dead_repair_happened
) {
    const std::size_t changed_labels = count_assignment_changes_until_common(
        ops,
        previous_assignments,
        assignments,
        max_incremental_changes
    );

    if (changed_labels > max_incremental_changes) {
        return false;
    }

    ops.clear_dirty_clusters();

    apply_assignment_deltas_common(
        ops,
        previous_assignments,
        assignments,
        counts
    );

    dead_repair_happened = ops.resolve_dead_centroids_and_mark_dirty(
        assignments,
        counts
    );

    ops.write_dirty_centroids(counts);
    ops.refresh_dirty_centroid_data();
    ops.clear_dirty_clusters();

    return true;
}



template<class Ops, class State, class AssignmentVector, class CountsVector>
bool update_centroids_incremental_or_full_common(
    Ops& ops,
    State& state,
    const AssignmentVector& assignments,
    CountsVector& counts,
    std::size_t max_incremental_changes
) {
    bool dead_repair_happened = false;
    bool used_incremental_update = false;

    if (state.sums_match_assignments) {
        used_incremental_update = try_update_centroids_incremental_common(
            ops,
            state.previous_assignments_span(),
            const_int_span(assignments),
            mutable_int_span(counts),
            max_incremental_changes,
            dead_repair_happened
        );
    }

    if (!used_incremental_update) {
        dead_repair_happened = ops.update_centroids_full(assignments, counts);
        ops.refresh_all_centroid_data();
    }

    state.sums_match_assignments = !dead_repair_happened;

    return dead_repair_happened;
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

        const bool dead_repair_happened = backend.update_centroids(assignments, counts);

        if (!labels_changed && !dead_repair_happened) {
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
