#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>
#include <array>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>

#include "../core.hpp"
#include "layout.hpp"

template<std::size_t D>
inline void add_sample_to_sum_row_major(
    samples_soa_view<D> samples,
    aligned_float_vector& sums_row_major,
    std::size_t k,
    std::size_t n
) {
    float* dst = sums_row_major.data() + k * D;

    for (std::size_t d = 0; d < D; ++d)
        dst[d] += samples.dimension(d)[n];
}

template<std::size_t D>
inline void subtract_sample_from_sum_row_major(
    samples_soa_view<D> samples,
    aligned_float_vector& sums_row_major,
    std::size_t k,
    std::size_t n
) {
    float* dst = sums_row_major.data() + k * D;

    for (std::size_t d = 0; d < D; ++d)
        dst[d] -= samples.dimension(d)[n];
}

#include "./assignment_centroid_tiled.hpp"
#include "./assignment_micro_gemm.hpp"

template<std::size_t D>
inline float sample_to_centroid_dist_sq(
    samples_soa_view<D> samples,
    std::size_t n,
    std::span<const float> centroids_row_major,
    std::size_t centroid_k
) {
    const float* centroid = centroids_row_major.data() + centroid_k * D;

    float dist = 0.0f;

    for (std::size_t d = 0; d < D; ++d) {
        const float diff = samples.dimension(d)[n] - centroid[d];
        dist += diff * diff;
    }

    return dist;
}

template<std::size_t D>
inline bool resolve_dead_centroids(
    samples_soa_view<D> samples,
    std::span<const int> assignments,
    std::span<const float> old_centroids_row_major,
    std::span<float> sums_row_major,
    std::span<int> counts
) {
    struct ops_t {
        samples_soa_view<D> samples;
        std::span<const float> old_centroids_row_major;
        std::span<float> sums_row_major;

        std::size_t N() const { return samples.N; }

        float distance_to_old_centroid(std::size_t n, std::size_t old_k) const {
            return sample_to_centroid_dist_sq<D>(
                samples,
                n,
                old_centroids_row_major,
                old_k
            );
        }

        void relocate_empty_cluster(
            std::size_t old_k,
            std::size_t new_k,
            std::size_t n
        ) {
            float* old_sum = sums_row_major.data() + old_k * D;
            float* new_sum = sums_row_major.data() + new_k * D;

            for (std::size_t d = 0; d < D; ++d) {
                const float x = samples.dimension(d)[n];
                old_sum[d] -= x;
                new_sum[d] = x;
            }
        }
    };

    ops_t ops{ samples, old_centroids_row_major, sums_row_major };

    return kmeans::resolve_dead_centroids_common(
        ops,
        assignments,
        counts
    );
}

template<std::size_t D>
inline bool update_centroids(
    samples_soa_view<D> samples,
    std::span<const int> assignments,
    centroids_storage<D>& centroids,
    aligned_float_vector& sums_row_major,
    aligned_int_vector& counts
) {
    struct ops_t {
        samples_soa_view<D> samples;
        centroids_storage<D>& centroids;
        aligned_float_vector& sums_row_major;

        std::size_t N() const { return samples.N; }
        std::size_t K() const { return centroids.K; }

        void reset_sums() {
            std::fill(sums_row_major.begin(), sums_row_major.end(), 0.0f);
        }

        void add_sample_to_sum(std::size_t k, std::size_t n) {
            add_sample_to_sum_row_major<D>(samples, sums_row_major, k, n);
        }

        bool resolve_dead_centroids(std::span<const int> assignments, std::span<int> counts) {
            return ::resolve_dead_centroids<D>(
                samples,
                assignments,
                std::span<const float>{centroids.row_major},
                std::span<float>{sums_row_major},
                counts
            );
        }

        void write_centroid_from_sum(std::size_t k, int count) {
            const float inv_count = 1.0f / static_cast<float>(count);

            const float* src = sums_row_major.data() + k * D;
            float* dst = centroids.row_major.data() + k * D;

            for (std::size_t d = 0; d < D; ++d) {
                dst[d] = src[d] * inv_count;
            }
        }
    };

    ops_t ops{ samples, centroids, sums_row_major };

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
    const std::size_t K = new_centroids_row_major.size() / D;

    float shift_sq = 0.0f;

    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t base = k * D;

        for (std::size_t d = 0; d < D; ++d) {
            const float diff =
                new_centroids_row_major[base + d]
                - old_centroids_row_major[base + d];

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

    samples_soa_view<D> original_samples;

    samples_soa_storage<D> centered_samples_storage;
    samples_soa_view<D> samples;
    centroids_storage<D>& centroids;

    aligned_float_vector sums_row_major;
    kmeans::incremental_centroid_update_state<assignment_vector> incremental_update;
    std::array<float, D> dimension_mean;

    AssignmentBackend assignment_backend;

    dynamic_kmeans_backend(
        samples_soa_view<D> samples_,
        centroids_storage<D>& centroids_,
        AssignmentBackend assignment_backend_
    )
     : original_samples(samples_),
      centered_samples_storage(samples_.N),
      samples(centered_samples_storage.view()),
      centroids(centroids_),
      sums_row_major(centroids_.K * D, 0.0f),
      incremental_update(centroids_.K),
      assignment_backend(std::move(assignment_backend_)) {
    }

    void compute_dimension_mean_from_original() {
        constexpr std::size_t card = simd_cardinal();

        const std::size_t N = original_samples.N;

        if (N == 0) {
            dimension_mean.fill(0.0f);
            return;
        }

        const float inv_N = 1.0f / static_cast<float>(N);

        for (std::size_t d = 0; d < D; ++d) {
            const float* src = original_samples.dimension(d);

            wide_f sum_v = eve::zero(eve::as<wide_f>());

            std::size_t n = 0;

            for (; n + card <= N; n += card) {
                const auto x = eve::load(eve::as_aligned(src + n));
                sum_v += x;
            }

            float sum = eve::reduce(sum_v);

            for (; n < N; ++n) {
                sum += src[n];
            }

            dimension_mean[d] = sum * inv_N;
        }
    }

    float copy_centered_samples_from_original_and_compute_scaled_tolerance(float tol) {
        constexpr std::size_t card = simd_cardinal();
        const std::size_t N = samples.N;

        float total_variance = 0.0f;

        for (std::size_t d = 0; d < D; ++d) {
            const float* src = original_samples.dimension(d);
            float* dst = samples.dimension(d);
            const float mean = dimension_mean[d];
            const wide_f mean_v(mean);

            std::size_t n = 0;

            if (tol == 0.0f) {
                for (; n + card <= N; n += card) {
                    const auto x = eve::load(eve::as_aligned(src + n));
                    const auto centered = x - mean_v;
                    eve::store(centered, eve::as_aligned(dst + n));
                }

                for (; n < N; ++n)
                    dst[n] = src[n] - mean;

            } else {
                wide_f variance_v = eve::zero(eve::as<wide_f>());

                for (; n + card <= N; n += card) {
                    const auto x = eve::load(eve::as_aligned(src + n));
                    const auto centered = x - mean_v;

                    eve::store(centered, eve::as_aligned(dst + n));
                    variance_v = eve::fma(centered, centered, variance_v);
                }

                float variance = eve::reduce(variance_v);

                for (; n < N; ++n) {
                    const float centered = src[n] - mean;
                    dst[n] = centered;
                    variance += centered * centered;
                }

                total_variance += variance / static_cast<float>(N);
            }
        }

        if (tol == 0.0f) return 0.0f;

        return tol * (total_variance / static_cast<float>(D));
    }

    float prepare_data_for_fit(float tol) {
        compute_dimension_mean_from_original();
        const float scaled_tol =
            copy_centered_samples_from_original_and_compute_scaled_tolerance(tol);

        subtract_dimension_mean_from_centroids();
        assignment_backend.on_centroids_changed(centroids);

        return scaled_tol;
    }

    void finish_fit_after_final_assignment() {
        add_dimension_mean_to_centroids();
    }

    void subtract_dimension_mean_from_centroids() {
        for (std::size_t k = 0; k < centroids.K; ++k) {
            for (std::size_t d = 0; d < D; ++d) 
                centroids.row(k, d) -= dimension_mean[d];
        }
    }

    void add_dimension_mean_to_centroids() {
        for (std::size_t k = 0; k < centroids.K; ++k) {
            for (std::size_t d = 0; d < D; ++d)
                centroids.row(k, d) += dimension_mean[d];
        }
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count(centroids.K, samples.N);
    }

    std::size_t N() const {
        return samples.N;
    }

    assignment_vector make_assignment_vector(int initial_value) const {
        return assignment_vector(samples.N, initial_value);
    }

    counts_vector make_counts_vector() const {
        return counts_vector(centroids.K, 0);
    }

    centroid_snapshot make_centroid_snapshot() const {
        return centroid_snapshot(centroids.row_major.size(), 0.0f);
    }

    void save_centroids(centroid_snapshot& snapshot) const {
        snapshot = centroids.row_major;
    }

    void assign(assignment_vector& assignments) const {
        assignment_backend.assign(samples, assignments);
    }

    bool assign_and_check_changed(assignment_vector& assignments) {
        incremental_update.snapshot_before_assignment(
            std::span<const int>{assignments}
        );

        return assignment_backend.assign_and_check_changed(samples, assignments);
    }

    void clear_dirty_clusters() {
        incremental_update.clear_dirty_clusters();
    }

    void mark_dirty_cluster(std::size_t k) {
        incremental_update.mark_dirty_cluster(k);
    }

    void add_sample_to_sum(std::size_t k, std::size_t n) {
        add_sample_to_sum_row_major<D>(samples, sums_row_major, k, n);
    }

    void subtract_sample_from_sum(std::size_t k, std::size_t n) {
        subtract_sample_from_sum_row_major<D>(samples, sums_row_major, k, n);
    }

    bool resolve_dead_centroids_and_mark_dirty(
        std::span<const int> assignments,
        std::span<int> counts
    ) {
        struct ops_t {
            dynamic_kmeans_backend& backend;

            std::size_t N() const { return backend.samples.N; }

            float distance_to_old_centroid(std::size_t n, std::size_t old_k) const {
                return sample_to_centroid_dist_sq<D>(
                    backend.samples,
                    n,
                    std::span<const float>{backend.centroids.row_major},
                    old_k
                );
            }

            void relocate_empty_cluster(
                std::size_t old_k,
                std::size_t new_k,
                std::size_t n
            ) {
                subtract_sample_from_sum_row_major<D>(
                    backend.samples,
                    backend.sums_row_major,
                    old_k,
                    n
                );

                float* new_sum = backend.sums_row_major.data() + new_k * D;
                for (std::size_t d = 0; d < D; ++d) {
                    new_sum[d] = backend.samples.dimension(d)[n];
                }

                backend.mark_dirty_cluster(old_k);
                backend.mark_dirty_cluster(new_k);
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
        for (int dirty_k : incremental_update.dirty_clusters_span()) {
            const std::size_t k = static_cast<std::size_t>(dirty_k);

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
            samples,
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
            samples.N / 4
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
    samples_soa_view<D> samples,
    centroids_storage<D>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(D > 0);
    static_assert(K_TILE > 0);
    centroid_tiled_kmeans_backend<D, K_TILE> backend{
        samples,
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

template<std::size_t D, std::size_t N_VECTORS, std::size_t K_TILE>
using micro_gemm_kmeans_backend = dynamic_kmeans_backend<
    D,
    micro_gemm_assignment_backend<D, N_VECTORS, K_TILE>
>;

template<std::size_t D, std::size_t N_VECTORS, std::size_t K_TILE>
aligned_int_vector k_means_micro_gemm(
    samples_soa_view<D> samples,
    centroids_storage<D>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(D > 0);
    static_assert(K_TILE > 0);
    static_assert(N_VECTORS > 0);

    micro_gemm_kmeans_backend<D, N_VECTORS, K_TILE> backend{
        samples,
        centroids,
        micro_gemm_assignment_backend<D, N_VECTORS, K_TILE>{}
    };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}