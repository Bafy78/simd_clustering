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

struct centroid_update_result {
    bool dead_repair_happened = false;
    float shift_sq = 0.0f;
};

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

    // Select the empty_cluster_count farthest samples globally.
    // sklearn uses np.argpartition(...)[-empty_cluster_count:][::-1].
    // We keep the same top-E rule, but maintain only the current top E candidates
    // instead of materializing all N distances/candidates.
    auto farther = [](const candidate_t& a, const candidate_t& b) {
        if (a.distance != b.distance) {
            return a.distance > b.distance;
        }

        return a.n < b.n;
    };

    std::vector<candidate_t> farthest_candidates;
    farthest_candidates.reserve(empty_cluster_count);

    float max_distance = 0.0f;

    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t assigned_k = static_cast<std::size_t>(assignments[n]);
        const float dist = ops.distance_to_old_centroid(n, assigned_k);

        max_distance = std::max(max_distance, dist);

        const candidate_t candidate{
            dist,
            static_cast<std::uint32_t>(n)
        };

        if (farthest_candidates.size() < empty_cluster_count) {
            farthest_candidates.push_back(candidate);
            std::push_heap(
                farthest_candidates.begin(),
                farthest_candidates.end(),
                farther
            );
        } else if (farther(candidate, farthest_candidates.front())) {
            std::pop_heap(
                farthest_candidates.begin(),
                farthest_candidates.end(),
                farther
            );

            farthest_candidates.back() = candidate;

            std::push_heap(
                farthest_candidates.begin(),
                farthest_candidates.end(),
                farther
            );
        }
    }

    // sklearn returns early when all distances are exactly zero.
    // This happens when there are more clusters than distinct non-duplicate samples.
    if (max_distance == 0.0f) return false;

    // Relocate in deterministic farthest-first order. This preserves the intended
    // sklearn top-E selection rule, though not NumPy argpartition's internal order.
    std::sort(
        farthest_candidates.begin(),
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
centroid_update_result update_centroids_common(
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

    centroid_update_result result;
    result.dead_repair_happened = ops.resolve_dead_centroids(assignments, counts);

    for (std::size_t k = 0; k < ops.K(); ++k) {
        if (counts[k] > 0) {
            result.shift_sq += ops.write_centroid_from_sum(k, counts[k]);
        }
    }

    return result;
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
    centroid_update_result& result
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

    result.dead_repair_happened = ops.resolve_dead_centroids_and_mark_dirty(
        assignments,
        counts
    );

    result.shift_sq = ops.write_dirty_centroids(counts);
    ops.refresh_dirty_centroid_data();
    ops.clear_dirty_clusters();

    return true;
}



template<class Ops, class State, class AssignmentVector, class CountsVector>
centroid_update_result update_centroids_incremental_or_full_common(
    Ops& ops,
    State& state,
    const AssignmentVector& assignments,
    CountsVector& counts,
    std::size_t max_incremental_changes
) {
    centroid_update_result result;
    bool used_incremental_update = false;

    if (state.sums_match_assignments) {
        used_incremental_update = try_update_centroids_incremental_common(
            ops,
            state.previous_assignments_span(),
            const_int_span(assignments),
            mutable_int_span(counts),
            max_incremental_changes,
            result
        );
    }

    if (!used_incremental_update) {
        result = ops.update_centroids_full(assignments, counts);
        ops.refresh_all_centroid_data();
    }

    state.sums_match_assignments = !result.dead_repair_happened;

    return result;
}

template<class Backend>
auto k_means_core(
    Backend& backend,
    int& out_algorithm_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) -> typename Backend::assignment_vector {
    backend.check_cluster_count();

    auto assignments = backend.make_assignment_vector(-1);
    auto counts = backend.make_counts_vector();

    const float scaled_tol = backend.prepare_data_for_fit(tol);

    bool converged = false;
    bool strict_convergence = false;
    int algorithm_iterations = 0;

    while (!converged && algorithm_iterations < max_iterations) {
        const bool labels_changed = backend.assign_and_check_changed(assignments);
        const centroid_update_result update = backend.update_centroids(assignments, counts);

        if (!labels_changed && !update.dead_repair_happened) {
            strict_convergence = true;
            converged = true;
        } else if (update.shift_sq <= scaled_tol) {
            converged = true;
        }

        ++algorithm_iterations;
    }

    out_algorithm_iterations = algorithm_iterations;

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
