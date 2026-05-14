#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>
#include <vector>
#include <numeric>
#include <ranges>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>


using wide_f = eve::wide<float>;
using wide_i = eve::wide<int, typename wide_f::cardinal_type>;
using aligned_float_vector = std::vector<float, eve::aligned_allocator<float>>;
using aligned_int_vector = std::vector<int, eve::aligned_allocator<int>>;

inline constexpr std::size_t simd_cardinal() {
    return static_cast<std::size_t>(wide_f::size());
}

inline std::size_t round_up_to_multiple(std::size_t n, std::size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

// Dynamic feature-major point storage: points[d][i]
// The stride is padded to SIMD cardinality so every feature column can be
// loaded safely in SIMD chunks.
struct points_soa_view {
    const float* data = nullptr;

    std::size_t n_samples = 0;
    std::size_t n_features = 0;
    std::size_t stride = 0;

    const float* feature(std::size_t d) const {
        return data + d * stride;
    }
};

struct points_soa_storage {
    aligned_float_vector data;

    std::size_t n_samples = 0;
    std::size_t n_features = 0;
    std::size_t stride = 0;

    points_soa_storage() = default;

    points_soa_storage(std::size_t samples, std::size_t features) {
        resize(samples, features);
    }

    void resize(std::size_t samples, std::size_t features) {
        n_samples = samples;
        n_features = features;
        stride = round_up_to_multiple(samples, simd_cardinal());

        data.assign(n_features * stride, 0.0f);
    }

    float& operator()(std::size_t sample, std::size_t feature) {
        return data[feature * stride + sample];
    }

    float operator()(std::size_t sample, std::size_t feature) const {
        return data[feature * stride + sample];
    }

    const float* feature(std::size_t d) const {
        return data.data() + d * stride;
    }

    points_soa_view view() const {
        return points_soa_view{
            .data = data.data(),
            .n_samples = n_samples,
            .n_features = n_features,
            .stride = stride
        };
    }
};

// Dual centroid storage
// row_major: centroids[k][d]
//     Useful for update/dead-centroid/scalar logic.
//
// feature_major: centroids_T[d][k]
//     Useful for the tiled assignment kernel.
struct centroids_storage {
    aligned_float_vector row_major;
    aligned_float_vector feature_major;

    std::size_t n_clusters = 0;
    std::size_t n_features = 0;
    std::size_t feature_major_stride = 0;

    centroids_storage() = default;

    template<std::size_t K_TILE>
    void resize_for_tile(std::size_t clusters, std::size_t features) {
        n_clusters = clusters;
        n_features = features;

        feature_major_stride = round_up_to_multiple(n_clusters, K_TILE);

        row_major.assign(n_clusters * n_features, 0.0f);
        feature_major.assign(n_features * feature_major_stride, 0.0f);
    }

    float& row(std::size_t k, std::size_t d) {
        return row_major[k * n_features + d];
    }

    float row(std::size_t k, std::size_t d) const {
        return row_major[k * n_features + d];
    }

    const float* feature_centroids(std::size_t d) const {
        return feature_major.data() + d * feature_major_stride;
    }

    // Call this once after updating row_major centroids.
    void sync_feature_major_from_row_major() {
        std::fill(feature_major.begin(), feature_major.end(), 0.0f);

        for (std::size_t d = 0; d < n_features; ++d) {
            float* dst = feature_major.data() + d * feature_major_stride;

            for (std::size_t k = 0; k < n_clusters; ++k) {
                dst[k] = row_major[k * n_features + d];
            }
        }
    }
};

// Process one full compile-time centroid tile
// This is the high-D replacement for: `compute_simd_dist_sq(pt, centroid[k])`
// except now dimensions are streamed one at a time, and one point feature
// load is reused across K_TILE centroid accumulators.
template<std::size_t K_TILE, typename Ignore>
inline void process_centroid_tile(
    points_soa_view points,
    const centroids_storage& centroids,
    std::size_t sample_i,
    std::size_t k0,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    auto distances = kumi::fill<K_TILE>(eve::zero(eve::as<wide_f>()));

    for (std::size_t d = 0; d < points.n_features; ++d) {
        // points are feature-major:
        //     points[d][sample_i : sample_i + SIMD_WIDTH]
        //
        // The storage pads each feature row, so this load is safe for the
        // tail as long as the same ignore is used consistently.
        auto x = eve::load[ignore](eve::as_aligned(points.feature(d) + sample_i));

        const float* centroid_feature = centroids.feature_centroids(d) + k0;

        kumi::for_each_index([&](auto index, auto& dist) {
            constexpr std::size_t t = decltype(index)::value;
            auto c = wide_f(centroid_feature[t]);
            auto diff = x - c;
            dist = eve::fma(diff, diff, dist);
            },
            distances
        );
    }

    kumi::for_each_index([&](auto index, auto dist) {
        constexpr std::size_t t = decltype(index)::value;
        const auto candidate_k = static_cast<int>(k0 + t);
        auto closer = dist < best_dist;

        best_dist = eve::min(best_dist, dist);
        best_k = eve::if_else(
            closer,
            wide_i(candidate_k),
            best_k
        );
        },
        distances
    );
}

// Compile-time recursive tail decomposition.
template<std::size_t TILE, typename Ignore>
inline void process_centroid_tail(
    points_soa_view points,
    const centroids_storage& centroids,
    std::size_t sample_i,
    std::size_t& k0,
    std::size_t& remaining,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    static_assert(TILE > 0);
    static_assert(std::has_single_bit(TILE));

    if (remaining >= TILE) {
        process_centroid_tile<TILE>(
            points,
            centroids,
            sample_i,
            k0,
            ignore,
            best_dist,
            best_k
        );

        k0 += TILE;
        remaining -= TILE;
    }

    if constexpr (TILE > 1) {
        process_centroid_tail<TILE / 2>(
            points,
            centroids,
            sample_i,
            k0,
            remaining,
            ignore,
            best_dist,
            best_k
        );
    }
}

// Dynamic high-D assignment kernel.
// This replaces assign_points_to_centroids for the high-dimensional path.
// Register pressure depends on K_TILE, not on D.
template<std::size_t K_TILE>
void assign_points_to_centroids_tiled(
    points_soa_view points,
    const centroids_storage& centroids,
    std::span<int> assignments
) {
    constexpr std::size_t card = simd_cardinal();
    const std::size_t n = points.n_samples;
    const std::size_t K = centroids.n_clusters;

    for (std::size_t i = 0; i < n; i += card) {
        const std::size_t valid = std::min(card, n - i);
        const std::size_t ignored_lanes = card - valid;
        auto ignore = eve::ignore_last(ignored_lanes);

        auto best_dist = eve::valmax(eve::as<wide_f>());
        auto best_k = eve::zero(eve::as<wide_i>());

        std::size_t k0 = 0;

        for (; k0 + K_TILE <= K; k0 += K_TILE)
        {
            process_centroid_tile<K_TILE>(
                points,
                centroids,
                i,
                k0,
                ignore,
                best_dist,
                best_k
            );
        }

        if constexpr (K_TILE > 1) {
            std::size_t remaining = K - k0;

            process_centroid_tail<K_TILE / 2>(
                points,
                centroids,
                i,
                k0,
                remaining,
                ignore,
                best_dist,
                best_k
            );
        }

        eve::store[ignore](best_k, eve::as_aligned(assignments.data() + i));
    }
}

inline float compute_sklearn_tolerance(points_soa_view points, float tol) {
    if (tol == 0.0f) return 0.0f;

    const std::size_t n = points.n_samples;
    const std::size_t D = points.n_features;

    if (n == 0 || D == 0) return 0.0f;

    aligned_float_vector means(D, 0.0f);

    // Accumulate mean for each feature.
    // points are feature-major, so this is cache-friendly.
    for (std::size_t d = 0; d < D; ++d) {
        const float* x = points.feature(d);

        float sum = 0.0f;

        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i];
        }

        means[d] = sum / static_cast<float>(n);
    }

    float total_variance = 0.0f;

    // Accumulate variance for each feature.
    for (std::size_t d = 0; d < D; ++d) {
        const float* x = points.feature(d);
        const float mean = means[d];

        float variance = 0.0f;

        for (std::size_t i = 0; i < n; ++i) {
            const float diff = x[i] - mean;
            variance += diff * diff;
        }

        total_variance += variance / static_cast<float>(n);
    }

    const float mean_variance =
        total_variance / static_cast<float>(D);

    // Scikit-learn returns mean(variances) * tol.
    return mean_variance * tol;
}

inline float point_to_centroid_dist_sq(
    points_soa_view points,
    std::size_t sample_i,
    std::span<const float> centroids_row_major,
    std::size_t centroid_k
) {
    const std::size_t D = points.n_features;
    const float* centroid = centroids_row_major.data() + centroid_k * D;

    float dist = 0.0f;

    for (std::size_t d = 0; d < D; ++d) {
        const float diff = points.feature(d)[sample_i] - centroid[d];
        dist += diff * diff;
    }

    return dist;
}

inline void resolve_dead_centroids(
    points_soa_view points,
    std::span<int> assignments,
    std::span<const float> old_centroids_row_major,
    std::span<float> sums_row_major,
    std::span<int> counts
) {
    const std::size_t n_samples = points.n_samples;
    const std::size_t n_features = points.n_features;
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
        const int label = assignments[i];

        const float dist = point_to_centroid_dist_sq(
            points,
            i,
            old_centroids_row_major,
            static_cast<std::size_t>(label)
        );

        distances[i] = dist;
        max_distance = std::max(max_distance, dist);
    }

    // sklearn returns early when all distances are exactly zero.
    // This happens when there are more clusters than distinct non-duplicate samples.
    if (max_distance == 0.0f) return;

    // Select the n_empty farthest points globally.
    // sklearn uses np.argpartition(...)[-n_empty:][::-1].
    std::vector<std::size_t> farthest_indices(n_samples);
    std::iota(farthest_indices.begin(), farthest_indices.end(), std::size_t{ 0 });

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

        const int old_cluster_id_int = assignments[far_idx];

        const std::size_t old_cluster_id = static_cast<std::size_t>(old_cluster_id_int);

        float* old_sum = sums_row_major.data() + old_cluster_id * n_features;
        float* new_sum = sums_row_major.data() + new_cluster_id * n_features;

        for (std::size_t d = 0; d < n_features; ++d) {
            const float x = points.feature(d)[far_idx];

            // sum[old_cluster_id] -= X[far_idx]
            old_sum[d] -= x;
            new_sum[d] = x;
        }

        counts[old_cluster_id] -= 1;
        counts[new_cluster_id] = 1;
    }
}

inline void update_centroids(
    points_soa_view points,
    std::span<int> assignments,
    centroids_storage& centroids,
    aligned_float_vector& sums_row_major,
    aligned_int_vector& counts
) {
    const std::size_t n_samples = points.n_samples;
    const std::size_t n_features = points.n_features;
    const std::size_t n_clusters = centroids.n_clusters;

    if (sums_row_major.size() != n_clusters * n_features) {
        sums_row_major.assign(n_clusters * n_features, 0.0f);
    }
    else {
        std::fill(sums_row_major.begin(), sums_row_major.end(), 0.0f);
    }

    if (counts.size() != n_clusters) {
        counts.assign(n_clusters, 0);
    }
    else {
        std::fill(counts.begin(), counts.end(), 0);
    }

    // Accumulate sums and counts.
    for (std::size_t i = 0; i < n_samples; ++i) {
        const int cluster_idx_int = assignments[i];

        const std::size_t cluster_idx =
            static_cast<std::size_t>(cluster_idx_int);

        counts[cluster_idx]++;

        float* dst = sums_row_major.data() + cluster_idx * n_features;

        for (std::size_t d = 0; d < n_features; ++d) {
            dst[d] += points.feature(d)[i];
        }
    }

    resolve_dead_centroids(
        points,
        assignments,
        std::span<const float>(centroids.row_major.data(), centroids.row_major.size()),
        std::span<float>(sums_row_major.data(), sums_row_major.size()),
        std::span<int>(counts.data(), counts.size())
    );

    // Compute the new means and update row-major centroids.
    for (std::size_t k = 0; k < n_clusters; ++k) {
        if (counts[k] > 0) {
            const float inv_count = 1.0f / static_cast<float>(counts[k]);

            const float* src = sums_row_major.data() + k * n_features;
            float* dst = centroids.row_major.data() + k * n_features;

            for (std::size_t d = 0; d < n_features; ++d) {
                dst[d] = src[d] * inv_count;
            }
        }
    }

    // Keep assignment-friendly centroid storage in sync.
    centroids.sync_feature_major_from_row_major();
}

inline float calculate_centroid_shift_sq(
    std::span<const float> old_centroids_row_major,
    std::span<const float> new_centroids_row_major
) {
    float shift_sq = 0.0f;

    for (std::size_t i = 0; i < new_centroids_row_major.size(); ++i) {
        const float diff =
            new_centroids_row_major[i] - old_centroids_row_major[i];

        shift_sq += diff * diff;
    }

    return shift_sq;
}

template<std::size_t K_TILE>
aligned_int_vector k_means_tiled(
    points_soa_view points,
    centroids_storage& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(K_TILE > 0);
    static_assert(std::has_single_bit(K_TILE));

    // Ensure assignment-friendly storage matches the initial row-major centroids.
    centroids.sync_feature_major_from_row_major();

    aligned_int_vector centroid_assignments(points.n_samples, -1);
    aligned_int_vector previous_assignments(points.n_samples, -1);

    aligned_float_vector sums_row_major(
        centroids.n_clusters * centroids.n_features,
        0.0f
    );

    aligned_int_vector counts(centroids.n_clusters, 0);

    aligned_float_vector previous_centroids_row_major(
        centroids.row_major.size(),
        0.0f
    );

    const float scaled_tol =
        compute_sklearn_tolerance(points, tol);

    bool converged = false;
    int iterations = 0;

    while (!converged && iterations < max_iterations) {
        previous_assignments = centroid_assignments;

        assign_points_to_centroids_tiled<K_TILE>(
            points,
            centroids,
            std::span<int>(centroid_assignments.data(), centroid_assignments.size())
        );

        previous_centroids_row_major = centroids.row_major;

        update_centroids(
            points,
            std::span<int>(centroid_assignments.data(), centroid_assignments.size()),
            centroids,
            sums_row_major,
            counts
        );

        if (std::ranges::equal(centroid_assignments, previous_assignments)) {
            converged = true;
        }
        else {
            const float shift_sq = calculate_centroid_shift_sq(
                std::span<const float>(
                    previous_centroids_row_major.data(),
                    previous_centroids_row_major.size()
                ),
                std::span<const float>(
                    centroids.row_major.data(),
                    centroids.row_major.size()
                )
            );

            if (shift_sq <= scaled_tol) {
                converged = true;
            }
        }

        iterations++;
    }

    out_iterations = iterations;

    // Post-loop step: reassign labels to perfectly match final centroid positions.
    assign_points_to_centroids_tiled<K_TILE>(
        points,
        centroids,
        std::span<int>(centroid_assignments.data(), centroid_assignments.size())
    );

    return centroid_assignments;
}