#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>
#include <array>

#include <eve/arch.hpp>
#include <eve/module/core.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../core.hpp"

using wide_f = kmeans::wide_f;
using wide_i = kmeans::wide_i;

using aligned_float_vector = std::vector<float, eve::aligned_allocator<float>>;
using aligned_int_vector = std::vector<int, eve::aligned_allocator<int>>;

inline constexpr std::size_t simd_cardinal() {
    return static_cast<std::size_t>(wide_f::size());
}

inline std::size_t round_up_to_multiple(std::size_t n, std::size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

// Static-D feature-major point view:
//     points[d][i]
//
// D is compile-time.
// stride is runtime because n_samples is runtime.
// stride is padded to SIMD cardinality so each feature column can be loaded safely in SIMD chunks
template<std::size_t D>
struct points_soa_view {
    static constexpr std::size_t n_features = D;

    float* data = nullptr;

    std::size_t n_samples = 0;
    std::size_t stride = 0;

    float* feature(std::size_t d) const {
        return data + d * stride;
    }
};

template<std::size_t D>
struct points_soa_storage {
    static constexpr std::size_t n_features = D;

    aligned_float_vector data;

    std::size_t n_samples = 0;
    std::size_t stride = 0;

    points_soa_storage() = default;

    explicit points_soa_storage(std::size_t samples) {
        resize(samples);
    }

    void resize(std::size_t samples) {
        n_samples = samples;
        stride = round_up_to_multiple(samples, simd_cardinal());

        data.resize(D * stride);
        std::fill(data.begin(), data.end(), 0.0f);
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

    points_soa_view<D> view() {
        return points_soa_view<D>{
            .data = data.data(),
            .n_samples = n_samples,
            .stride = stride,
        };
    }

    points_soa_view<D> view() const {
        return points_soa_view<D>{
            .data = const_cast<float*>(data.data()),
            .n_samples = n_samples,
            .stride = stride,
        };
    }
};

// Static-D centroid storage.
//
// row_major is the single source of truth:
//     centroids[k][d]
//
// Assignment-specific layouts are owned and refreshed by the assignment
// backends when on_centroids_changed() is called.
template<std::size_t D>
struct centroids_storage {
    static constexpr std::size_t n_features = D;

    aligned_float_vector row_major;

    std::size_t n_clusters = 0;

    centroids_storage() = default;

    void resize(std::size_t clusters) {
        n_clusters = clusters;
        row_major.resize(n_clusters * D);
        std::fill(row_major.begin(), row_major.end(), 0.0f);
    }

    float& row(std::size_t k, std::size_t d) {
        return row_major[k * D + d];
    }

    float row(std::size_t k, std::size_t d) const {
        return row_major[k * D + d];
    }

    template<std::size_t Feature>
    float& row(std::size_t k) {
        static_assert(Feature < D);
        return row_major[k * D + Feature];
    }

    template<std::size_t Feature>
    float row(std::size_t k) const {
        static_assert(Feature < D);
        return row_major[k * D + Feature];
    }
};

template<std::size_t D>
inline void add_point_to_sum_row_major(
    points_soa_view<D> points,
    aligned_float_vector& sums_row_major,
    std::size_t cluster_idx,
    std::size_t sample_i
) {
    float* dst = sums_row_major.data() + cluster_idx * D;

    for (std::size_t d = 0; d < D; ++d)
        dst[d] += points.feature(d)[sample_i];
}

template<std::size_t D>
inline void subtract_point_from_sum_row_major(
    points_soa_view<D> points,
    aligned_float_vector& sums_row_major,
    std::size_t cluster_idx,
    std::size_t sample_i
) {
    float* dst = sums_row_major.data() + cluster_idx * D;

    for (std::size_t d = 0; d < D; ++d)
        dst[d] -= points.feature(d)[sample_i];
}

#include "./assignment_centroid_tiled.hpp"
#include "./assignment_micro_gemm.hpp"

template<std::size_t D>
inline float point_to_centroid_dist_sq(
    points_soa_view<D> points,
    std::size_t sample_i,
    std::span<const float> centroids_row_major,
    std::size_t centroid_k
) {
    const float* centroid = centroids_row_major.data() + centroid_k * D;

    float dist = 0.0f;

    for (std::size_t d = 0; d < D; ++d) {
        const float diff = points.feature(d)[sample_i] - centroid[d];
        dist += diff * diff;
    }

    return dist;
}

template<std::size_t D>
inline bool resolve_dead_centroids(
    points_soa_view<D> points,
    std::span<const int> assignments,
    std::span<const float> old_centroids_row_major,
    std::span<float> sums_row_major,
    std::span<int> counts
) {
    struct ops_t {
        points_soa_view<D> points;
        std::span<const float> old_centroids_row_major;
        std::span<float> sums_row_major;

        std::size_t n_samples() const { return points.n_samples; }

        float distance_to_old_centroid(std::size_t sample_i, std::size_t old_label) const {
            return point_to_centroid_dist_sq<D>(
                points,
                sample_i,
                old_centroids_row_major,
                old_label
            );
        }

        void relocate_empty_cluster(
            std::size_t old_cluster_id,
            std::size_t new_cluster_id,
            std::size_t sample_i
        ) {
            float* old_sum = sums_row_major.data() + old_cluster_id * D;
            float* new_sum = sums_row_major.data() + new_cluster_id * D;

            for (std::size_t d = 0; d < D; ++d) {
                const float x = points.feature(d)[sample_i];
                old_sum[d] -= x;
                new_sum[d] = x;
            }
        }
    };

    ops_t ops{ points, old_centroids_row_major, sums_row_major };

    return kmeans::resolve_dead_centroids_common(
        ops,
        assignments,
        counts
    );
}

template<std::size_t D>
inline bool update_centroids(
    points_soa_view<D> points,
    std::span<const int> assignments,
    centroids_storage<D>& centroids,
    aligned_float_vector& sums_row_major,
    aligned_int_vector& counts
) {
    struct ops_t {
        points_soa_view<D> points;
        centroids_storage<D>& centroids;
        aligned_float_vector& sums_row_major;

        std::size_t n_samples() const { return points.n_samples; }
        std::size_t n_clusters() const { return centroids.n_clusters; }

        void reset_sums() {
            std::fill(sums_row_major.begin(), sums_row_major.end(), 0.0f);
        }

        void add_point_to_sum(std::size_t cluster_idx, std::size_t sample_i) {
            add_point_to_sum_row_major<D>(points, sums_row_major, cluster_idx, sample_i);
        }

        bool resolve_dead_centroids(std::span<const int> assignments, std::span<int> counts) {
            return ::resolve_dead_centroids<D>(
                points,
                assignments,
                std::span<const float>{centroids.row_major},
                std::span<float>{sums_row_major},
                counts
            );
        }

        void write_centroid_from_sum(std::size_t cluster_idx, int count) {
            const float inv_count = 1.0f / static_cast<float>(count);

            const float* src = sums_row_major.data() + cluster_idx * D;
            float* dst = centroids.row_major.data() + cluster_idx * D;

            for (std::size_t d = 0; d < D; ++d) {
                dst[d] = src[d] * inv_count;
            }
        }
    };

    ops_t ops{ points, centroids, sums_row_major };

    return kmeans::update_centroids_common(
        ops,
        assignments,
        std::span<int>{counts}
    );
}

template<std::size_t D>
inline float calculate_centroid_shift_sq(
    std::span<const float> old_centroids_row_major,
    std::span<const float> new_centroids_row_major
) {
    const std::size_t n_clusters = new_centroids_row_major.size() / D;

    float shift_sq = 0.0f;

    for (std::size_t k = 0; k < n_clusters; ++k) {
        const std::size_t base = k * D;

        for (std::size_t j = 0; j < D; ++j) {
            const float diff =
                new_centroids_row_major[base + j]
                - old_centroids_row_major[base + j];

            shift_sq += diff * diff;
        }
    }

    return shift_sq;
}

template<std::size_t D, class AssignmentBackend>
struct dynamic_kmeans_backend {
    using assignment_vector = aligned_int_vector;
    using counts_vector = aligned_int_vector;
    using centroid_snapshot = aligned_float_vector;

    points_soa_view<D> original_points;

    points_soa_storage<D> centered_points_storage;
    points_soa_view<D> points;
    centroids_storage<D>& centroids;

    aligned_float_vector sums_row_major;
    kmeans::incremental_centroid_update_state<assignment_vector> incremental_update;
    std::array<float, D> feature_mean;

    AssignmentBackend assignment_backend;

    dynamic_kmeans_backend(
        points_soa_view<D> points_,
        centroids_storage<D>& centroids_,
        AssignmentBackend assignment_backend_
    )
     : original_points(points_),
      centered_points_storage(points_.n_samples),
      points(centered_points_storage.view()),
      centroids(centroids_),
      sums_row_major(centroids_.n_clusters * D, 0.0f),
      incremental_update(centroids_.n_clusters),
      assignment_backend(std::move(assignment_backend_)) {
    }

    void compute_feature_mean_from_original() {
        constexpr std::size_t card = simd_cardinal();

        const std::size_t n = original_points.n_samples;

        if (n == 0) {
            feature_mean.fill(0.0f);
            return;
        }

        const float inv_n = 1.0f / static_cast<float>(n);

        for (std::size_t d = 0; d < D; ++d) {
            const float* src = original_points.feature(d);

            wide_f sum_v = eve::zero(eve::as<wide_f>());

            std::size_t i = 0;

            for (; i + card <= n; i += card) {
                const auto x = eve::load(eve::as_aligned(src + i));
                sum_v += x;
            }

            float sum = eve::reduce(sum_v);

            for (; i < n; ++i) {
                sum += src[i];
            }

            feature_mean[d] = sum * inv_n;
        }
    }

    float copy_centered_points_from_original_and_compute_scaled_tolerance(float tol) {
        constexpr std::size_t card = simd_cardinal();
        const std::size_t n = points.n_samples;

        float total_variance = 0.0f;

        for (std::size_t d = 0; d < D; ++d) {
            const float* src = original_points.feature(d);
            float* dst = points.feature(d);
            const float mean = feature_mean[d];
            const wide_f mean_v(mean);

            std::size_t i = 0;

            if (tol == 0.0f) {
                for (; i + card <= n; i += card) {
                    const auto x = eve::load(eve::as_aligned(src + i));
                    const auto centered = x - mean_v;
                    eve::store(centered, eve::as_aligned(dst + i));
                }

                for (; i < n; ++i)
                    dst[i] = src[i] - mean;

            } else {
                wide_f variance_v = eve::zero(eve::as<wide_f>());

                for (; i + card <= n; i += card) {
                    const auto x = eve::load(eve::as_aligned(src + i));
                    const auto centered = x - mean_v;

                    eve::store(centered, eve::as_aligned(dst + i));
                    variance_v = eve::fma(centered, centered, variance_v);
                }

                float variance = eve::reduce(variance_v);

                for (; i < n; ++i) {
                    const float centered = src[i] - mean;
                    dst[i] = centered;
                    variance += centered * centered;
                }

                total_variance += variance / static_cast<float>(n);
            }
        }

        if (tol == 0.0f) return 0.0f;

        return tol * (total_variance / static_cast<float>(D));
    }

    float prepare_data_for_fit(float tol) {
        compute_feature_mean_from_original();
        const float scaled_tol =
            copy_centered_points_from_original_and_compute_scaled_tolerance(tol);

        subtract_feature_mean_from_centroids();
        assignment_backend.on_centroids_changed(centroids);

        return scaled_tol;
    }

    void finish_fit_after_final_assignment() {
        add_feature_mean_to_centroids();
    }

    void subtract_feature_mean_from_centroids() {
        for (std::size_t k = 0; k < centroids.n_clusters; ++k) {
            for (std::size_t d = 0; d < D; ++d) 
                centroids.row(k, d) -= feature_mean[d];
        }
    }

    void add_feature_mean_to_centroids() {
        for (std::size_t k = 0; k < centroids.n_clusters; ++k) {
            for (std::size_t d = 0; d < D; ++d)
                centroids.row(k, d) += feature_mean[d];
        }
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count(centroids.n_clusters, points.n_samples);
    }

    std::size_t n_samples() const {
        return points.n_samples;
    }

    assignment_vector make_assignment_vector(int initial_value) const {
        return assignment_vector(points.n_samples, initial_value);
    }

    counts_vector make_counts_vector() const {
        return counts_vector(centroids.n_clusters, 0);
    }

    centroid_snapshot make_centroid_snapshot() const {
        return centroid_snapshot(centroids.row_major.size(), 0.0f);
    }

    void save_centroids(centroid_snapshot& snapshot) const {
        snapshot = centroids.row_major;
    }

    void assign(assignment_vector& assignments) const {
        assignment_backend.assign(points, assignments);
    }

    bool assign_and_check_changed(assignment_vector& assignments) {
        incremental_update.snapshot_before_assignment(
            std::span<const int>{assignments}
        );

        return assignment_backend.assign_and_check_changed(points, assignments);
    }

    void clear_dirty_clusters() {
        incremental_update.clear_dirty_clusters();
    }

    void mark_dirty_cluster(std::size_t cluster_idx) {
        incremental_update.mark_dirty_cluster(cluster_idx);
    }

    void add_point_to_sum(std::size_t cluster_idx, std::size_t sample_i) {
        add_point_to_sum_row_major<D>(points, sums_row_major, cluster_idx, sample_i);
    }

    void subtract_point_from_sum(std::size_t cluster_idx, std::size_t sample_i) {
        subtract_point_from_sum_row_major<D>(points, sums_row_major, cluster_idx, sample_i);
    }

    bool resolve_dead_centroids_and_mark_dirty(
        std::span<const int> assignments,
        std::span<int> counts
    ) {
        struct ops_t {
            dynamic_kmeans_backend& backend;

            std::size_t n_samples() const { return backend.points.n_samples; }

            float distance_to_old_centroid(std::size_t sample_i, std::size_t old_label) const {
                return point_to_centroid_dist_sq<D>(
                    backend.points,
                    sample_i,
                    std::span<const float>{backend.centroids.row_major},
                    old_label
                );
            }

            void relocate_empty_cluster(
                std::size_t old_cluster_id,
                std::size_t new_cluster_id,
                std::size_t sample_i
            ) {
                subtract_point_from_sum_row_major<D>(
                    backend.points,
                    backend.sums_row_major,
                    old_cluster_id,
                    sample_i
                );

                float* new_sum = backend.sums_row_major.data() + new_cluster_id * D;
                for (std::size_t d = 0; d < D; ++d) {
                    new_sum[d] = backend.points.feature(d)[sample_i];
                }

                backend.mark_dirty_cluster(old_cluster_id);
                backend.mark_dirty_cluster(new_cluster_id);
            }
        };

        ops_t ops{ *this };

        return kmeans::resolve_dead_centroids_common(
            ops,
            assignments,
            counts
        );
    }

    void write_dirty_centroids(std::span<const int> counts) {
        for (int cluster_idx : incremental_update.dirty_clusters_span()) {
            const std::size_t k = static_cast<std::size_t>(cluster_idx);

            if (counts[k] <= 0) continue;

            const float inv_count = 1.0f / static_cast<float>(counts[k]);
            const float* src = sums_row_major.data() + k * D;
            float* dst = centroids.row_major.data() + k * D;

            for (std::size_t d = 0; d < D; ++d) {
                dst[d] = src[d] * inv_count;
            }
        }
    }

    void refresh_dirty_centroid_data() {
        assignment_backend.on_centroids_changed_for_clusters(
            centroids,
            incremental_update.dirty_clusters_span()
        );
    }

    bool update_centroids_full(
        const assignment_vector& assignments,
        counts_vector& counts
    ) {
        return ::update_centroids<D>(
            points,
            std::span<const int>{assignments},
            centroids,
            sums_row_major,
            counts
        );
    }

    void refresh_all_centroid_data() {
        assignment_backend.on_centroids_changed(centroids);
    }

    bool update_centroids(assignment_vector& assignments, counts_vector& counts) {
        return kmeans::update_centroids_incremental_or_full_common(
            *this,
            incremental_update,
            assignments,
            counts,
            points.n_samples / 4
        );
    }

    float centroid_shift_sq(const centroid_snapshot& previous_centroids) const {
        return ::calculate_centroid_shift_sq<D>(
            std::span<const float>{previous_centroids},
            std::span<const float>{centroids.row_major}
        );
    }
};

template<std::size_t D, std::size_t K_TILE>
using centroid_tiled_kmeans_backend = dynamic_kmeans_backend<
    D,
    centroid_tiled_assignment_backend<D, K_TILE>
>;

template<std::size_t D, std::size_t K_TILE>
aligned_int_vector k_means_centroid_tiled(
    points_soa_view<D> points,
    centroids_storage<D>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(D > 0);
    static_assert(K_TILE > 0);
    centroid_tiled_kmeans_backend<D, K_TILE> backend{
        points,
        centroids,
        centroid_tiled_assignment_backend<D, K_TILE>{}
    };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}

template<std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
using micro_gemm_kmeans_backend = dynamic_kmeans_backend<
    D,
    micro_gemm_assignment_backend<D, K_TILE, M_VECTORS>
>;

template<std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
aligned_int_vector k_means_micro_gemm(
    points_soa_view<D> points,
    centroids_storage<D>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(D > 0);
    static_assert(K_TILE > 0);
    static_assert(M_VECTORS > 0);

    micro_gemm_kmeans_backend<D, K_TILE, M_VECTORS> backend{
        points,
        centroids,
        micro_gemm_assignment_backend<D, K_TILE, M_VECTORS>{}
    };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}