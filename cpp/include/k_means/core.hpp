#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>

#include "../simd.hpp"

namespace kmeans {

using ::wide_f;
using ::wide_i;
using ::cardinal;

inline void check_cluster_count(std::size_t K, std::size_t N) {
    if (K == 0) {
        throw std::invalid_argument("K must be greater than zero");
    }

    if (K > N) {
        throw std::invalid_argument("K must be <= N");
    }
}

template<class Ops>
bool resolve_dead_centroids_common(
    Ops& ops,
    std::span<const int> assignments,
    std::span<int> counts
) {
    const std::size_t N = ops.N();
    const std::size_t K = counts.size();

    // Find empty clusters, in cluster-id order, matching np.where(counts == 0)[0].
    std::vector<int> empty_clusters;
    empty_clusters.reserve(K);

    for (std::size_t k = 0; k < K; ++k) {
        if (counts[k] == 0) {
            empty_clusters.push_back(static_cast<int>(k));
        }
    }

    const std::size_t empty_cluster_count = empty_clusters.size();

    if (empty_cluster_count == 0) return false;

    // Dead-cluster repair keeps sample ids in uint32_t.
    // This path assumes N <= UINT32_MAX
    if (N > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("dead-centroid repair requires N <= UINT32_MAX");
    }

    struct candidate_t {
        float distance;
        std::uint32_t n;
    };

    // Compute distance of each sample to its currently assigned old centroid:
    // sklearn: distances = ((X - centers_old[labels]) ** 2).sum(axis=1)
    std::vector<candidate_t> farthest_candidates;

    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t assigned_k = static_cast<std::size_t>(assignments[n]);
        const float dist = ops.distance_to_old_centroid(n, assigned_k);

        if (farthest_candidates.empty()) {
            if (dist == 0.0f) continue;

            farthest_candidates.reserve(N);

            for (std::uint32_t candidate_n = 0; candidate_n < static_cast<std::uint32_t>(n); ++candidate_n) {
                farthest_candidates.push_back({ 0.0f, candidate_n });
            }
        }

        farthest_candidates.push_back({ dist, static_cast<std::uint32_t>(n) });
    }

    // sklearn returns early when all distances are exactly zero.
    // This happens when there are more clusters than distinct non-duplicate samples.
    if (farthest_candidates.empty()) return false;

    // Select the empty_cluster_count farthest samples globally.
    // sklearn uses np.argpartition(...)[-empty_cluster_count:][::-1].
    auto farther = [](const candidate_t& a, const candidate_t& b) {
        if (a.distance != b.distance) {
            return a.distance > b.distance;
        }

        return a.n < b.n;
    };

    std::partial_sort(
        farthest_candidates.begin(),
        farthest_candidates.begin() + static_cast<std::ptrdiff_t>(empty_cluster_count),
        farthest_candidates.end(),
        farther
    );

    // Relocate each empty cluster using one farthest sample.
    //
    // Important sklearn detail:
    // labels / assignments are NOT changed here. Only the sums and counts are changed.
    for (std::size_t k = 0; k < empty_cluster_count; ++k) {
        const std::size_t new_k = static_cast<std::size_t>(empty_clusters[k]);
        const std::size_t far_n = static_cast<std::size_t>(farthest_candidates[k].n);

        const std::size_t old_k = static_cast<std::size_t>(assignments[far_n]);

        ops.relocate_empty_cluster(
            old_k,
            new_k,
            far_n
        );

        counts[old_k] -= 1;
        counts[new_k] = 1;
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

    for (std::size_t n = 0; n < ops.N(); ++n) {
        const std::size_t k = static_cast<std::size_t>(assignments[n]);

        counts[k]++;
        ops.add_sample_to_sum(k, n);
    }

    const bool dead_repair_happened = ops.resolve_dead_centroids(assignments, counts);

    for (std::size_t k = 0; k < ops.K(); ++k) {
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

    explicit incremental_centroid_update_state(std::size_t K = 0)
        : dirty_marker(K, 0) {}

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
        for (int k : dirty_clusters) {
            dirty_marker[static_cast<std::size_t>(k)] = 0;
        }

        dirty_clusters.clear();
    }

    void mark_dirty_cluster(std::size_t k) {
        if (dirty_marker[k] != 0) return;

        dirty_marker[k] = 1;
        dirty_clusters.push_back(static_cast<int>(k));
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

    for (std::size_t n = 0; n < ops.N(); ++n) {
        if (previous_assignments[n] == assignments[n]) continue;

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
    for (std::size_t n = 0; n < ops.N(); ++n) {
        const int old_k_int = previous_assignments[n];
        const int new_k_int = assignments[n];

        if (old_k_int == new_k_int) continue;

        if (old_k_int >= 0) {
            const std::size_t old_k = static_cast<std::size_t>(old_k_int);

            counts[old_k] -= 1;
            ops.subtract_sample_from_sum(old_k, n);
            ops.mark_dirty_cluster(old_k);
        }

        const std::size_t new_k = static_cast<std::size_t>(new_k_int);

        counts[new_k] += 1;
        ops.add_sample_to_sum(new_k, n);
        ops.mark_dirty_cluster(new_k);
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
