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

template<std::size_t K_TILE>
inline constexpr std::size_t centroid_tiles_at_once() {
    constexpr std::size_t register_count = static_cast<std::size_t>(eve::register_count::simd);
    constexpr std::size_t kernel_register_overhead = 4;

    constexpr std::size_t accumulator_registers =
        register_count > kernel_register_overhead
        ? register_count - kernel_register_overhead
        : K_TILE;

    constexpr std::size_t tiles = accumulator_registers / K_TILE;

    if constexpr (tiles == 0) {
        return 1;
    } else {
        return tiles;
    }
}

// Static-D feature-major point view:
//
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

    template<std::size_t Feature>
    float* feature() const {
        static_assert(Feature < D);
        return data + Feature * stride;
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

        data.assign(D * stride, 0.0f);
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

    template<std::size_t Feature>
    const float* feature() const {
        static_assert(Feature < D);
        return data.data() + Feature * stride;
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

// Static-D dual centroid storage.
//
// row_major:
//     centroids[k][d]
//     Useful for update/dead-centroid/scalar logic.
//
// feature_major:
//     centroids_T[d][k]
//     Useful for the tiled assignment kernel.
template<std::size_t D>
struct centroids_storage {
    static constexpr std::size_t n_features = D;

    aligned_float_vector row_major;
    aligned_float_vector feature_major;
    aligned_float_vector centroid_norm_sq;

    std::size_t n_clusters = 0;
    std::size_t feature_major_stride = 0;

    centroids_storage() = default;

    template<std::size_t K_TILE>
    void resize_for_tile(std::size_t clusters) {
        static_assert(K_TILE > 0);
        n_clusters = clusters;
        feature_major_stride = round_up_to_multiple(n_clusters, K_TILE);

        row_major.assign(n_clusters * D, 0.0f);
        feature_major.assign(D * feature_major_stride, 0.0f);
        centroid_norm_sq.assign(n_clusters, 0.0f);
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

    const float* feature_centroids(std::size_t d) const {
        return feature_major.data() + d * feature_major_stride;
    }

    template<std::size_t Feature>
    const float* feature_centroids() const {
        static_assert(Feature < D);
        return feature_major.data() + Feature * feature_major_stride;
    }

    void recompute_centroid_norms_from_row_major() {
        centroid_norm_sq.resize(n_clusters);

        for (std::size_t k = 0; k < n_clusters; ++k) {
            const float* centroid = row_major.data() + k * D;

            float norm = 0.0f;
            kmeans::for_each_feature<D>([&](auto feature_index) {
                constexpr std::size_t d = decltype(feature_index)::value;
                const float c = centroid[d];
                norm += c * c;
            });

            centroid_norm_sq[k] = norm;
        }
    }

    void sync_fused_assignment_layout_from_row_major() {
        std::fill(feature_major.begin(), feature_major.end(), 0.0f);

        if (centroid_norm_sq.size() != n_clusters) {
            centroid_norm_sq.assign(n_clusters, 0.0f);
        } else {
            std::fill(centroid_norm_sq.begin(), centroid_norm_sq.end(), 0.0f);
        }

        kmeans::for_each_feature<D>([&](auto feature_index) {
            constexpr std::size_t d = decltype(feature_index)::value;

            float* dst = feature_major.data() + d * feature_major_stride;

            for (std::size_t k = 0; k < n_clusters; ++k) {
                const float c = row_major[k * D + d];
                dst[k] = -2.0f * c;
                centroid_norm_sq[k] += c * c;
            }
        });
    }
};

#include "./assignment_fused.hpp"

template<std::size_t D>
inline float compute_sklearn_tolerance(
    points_soa_view<D> points,
    float tol
) {
    auto sample_value = [&](auto feature_index, std::size_t sample_i) {
        constexpr std::size_t d = decltype(feature_index)::value;
        return points.template feature<d>()[sample_i];
    };

    return kmeans::compute_sklearn_tolerance_common<D>(
        points.n_samples,
        sample_value,
        tol
    );
}

template<std::size_t D>
inline float point_to_centroid_dist_sq(
    points_soa_view<D> points,
    std::size_t sample_i,
    std::span<const float> centroids_row_major,
    std::size_t centroid_k
) {
    const float* centroid = centroids_row_major.data() + centroid_k * D;

    float dist = 0.0f;

    kmeans::for_each_feature<D>([&](auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;
        const float diff = points.template feature<d>()[sample_i] - centroid[d];
        dist += diff * diff;
    });

    return dist;
}

template<std::size_t D>
inline void resolve_dead_centroids(
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

            kmeans::for_each_feature<D>([&](auto feature_index) {
                constexpr std::size_t d = decltype(feature_index)::value;
                const float x = points.template feature<d>()[sample_i];
                old_sum[d] -= x;
                new_sum[d] = x;
            });
        }
    };

    ops_t ops{ points, old_centroids_row_major, sums_row_major };

    kmeans::resolve_dead_centroids_common(
        ops,
        assignments,
        counts
    );
}

template<std::size_t D>
inline void update_centroids(
    points_soa_view<D> points,
    std::span<const int> assignments,
    centroids_storage<D>& centroids,
    aligned_float_vector& sums_row_major,
    aligned_int_vector& counts
) {
    const std::size_t n_clusters = centroids.n_clusters;

    if (sums_row_major.size() != n_clusters * D) {
        sums_row_major.resize(n_clusters * D);
    }

    if (counts.size() != n_clusters) {
        counts.resize(n_clusters);
    }

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
            float* dst = sums_row_major.data() + cluster_idx * D;

            kmeans::for_each_feature<D>([&](auto feature_index) {
                constexpr std::size_t d = decltype(feature_index)::value;
                dst[d] += points.template feature<d>()[sample_i];
            });
        }

        void resolve_dead_centroids(std::span<const int> assignments, std::span<int> counts) {
            ::resolve_dead_centroids<D>(
                points,
                assignments,
                std::span<const float>(centroids.row_major.data(), centroids.row_major.size()),
                std::span<float>(sums_row_major.data(), sums_row_major.size()),
                counts
            );
        }

        void write_centroid_from_sum(std::size_t cluster_idx, int count) {
            const float inv_count = 1.0f / static_cast<float>(count);

            const float* src = sums_row_major.data() + cluster_idx * D;
            float* dst = centroids.row_major.data() + cluster_idx * D;

            kmeans::for_each_feature<D>([&](auto feature_index) {
                constexpr std::size_t d = decltype(feature_index)::value;
                dst[d] = src[d] * inv_count;
            });
        }
    };

    ops_t ops{ points, centroids, sums_row_major };

    kmeans::update_centroids_common(
        ops,
        assignments,
        std::span<int>(counts.data(), counts.size())
    );
}

template<std::size_t D>
inline float calculate_centroid_shift_sq(
    std::span<const float> old_centroids_row_major,
    std::span<const float> new_centroids_row_major
) {
    const std::size_t n_clusters = new_centroids_row_major.size() / D;

    auto old_value = [&](std::size_t k, auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;
        return old_centroids_row_major[k * D + d];
    };

    auto new_value = [&](std::size_t k, auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;
        return new_centroids_row_major[k * D + d];
    };

    return kmeans::calculate_centroid_shift_sq_common<D>(
        n_clusters,
        old_value,
        new_value
    );
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
    std::array<float, D> feature_mean{};

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
      assignment_backend(std::move(assignment_backend_)) {}

    void compute_feature_mean_from_original() {
        if (original_points.n_samples == 0) {
            feature_mean.fill(0.0f);
            return;
        }

        kmeans::for_each_feature<D>([&](auto feature_index) {
            constexpr std::size_t d = decltype(feature_index)::value;

            const float* src = original_points.template feature<d>();

            float sum = 0.0f;
            for (std::size_t i = 0; i < original_points.n_samples; ++i) {
                sum += src[i];
            }

            feature_mean[d] = sum / static_cast<float>(original_points.n_samples);
        });
    }

    float copy_centered_points_from_original_and_compute_scaled_tolerance(float tol) {
        if (points.n_samples == 0) {
            return 0.0f;
        }

        constexpr std::size_t card = simd_cardinal();
        const std::size_t n = points.n_samples;

        float total_variance = 0.0f;

        kmeans::for_each_feature<D>([&](auto feature_index) {
            constexpr std::size_t d = decltype(feature_index)::value;

            const float* src = original_points.template feature<d>();
            float* dst = points.template feature<d>();
            const float mean = feature_mean[d];
            const wide_f mean_v(mean);

            std::size_t i = 0;

            if (tol == 0.0f) {
                for (; i + card <= n; i += card) {
                    const auto x = eve::load(eve::as_aligned(src + i));
                    const auto centered = x - mean_v;
                    eve::store(centered, eve::as_aligned(dst + i));
                }

                for (; i < n; ++i) {
                    dst[i] = src[i] - mean;
                }
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
        });

        if (tol == 0.0f) {
            return 0.0f;
        }

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
            kmeans::for_each_feature<D>([&](auto feature_index) {
                constexpr std::size_t d = decltype(feature_index)::value;
                centroids.row(k, d) -= feature_mean[d];
            });
        }
    }

    void add_feature_mean_to_centroids() {
        for (std::size_t k = 0; k < centroids.n_clusters; ++k) {
            kmeans::for_each_feature<D>([&](auto feature_index) {
                constexpr std::size_t d = decltype(feature_index)::value;
                centroids.row(k, d) += feature_mean[d];
            });
        }
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count(centroids.n_clusters, points.n_samples);
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
        assignment_backend.assign(
            points,
            centroids,
            std::span<int>(assignments.data(), assignments.size())
        );
    }

    bool assign_and_check_changed(assignment_vector& assignments) const {
        return assignment_backend.assign_and_check_changed(
            points,
            centroids,
            std::span<int>(assignments.data(), assignments.size())
        );
    }

    void update_centroids(
        assignment_vector& assignments,
        counts_vector& counts
    ) {
        ::update_centroids<D>(
            points,
            std::span<const int>(assignments.data(), assignments.size()),
            centroids,
            sums_row_major,
            counts
        );

        assignment_backend.on_centroids_changed(centroids);
    }

    float centroid_shift_sq(
        const centroid_snapshot& previous_centroids
    ) const {
        return ::calculate_centroid_shift_sq<D>(
            std::span<const float>(previous_centroids.data(), previous_centroids.size()),
            std::span<const float>(centroids.row_major.data(), centroids.row_major.size())
        );
    }
};

template<std::size_t D, std::size_t K_TILE>
using fused_kmeans_backend = dynamic_kmeans_backend<
    D,
    fused_assignment_backend<D, K_TILE>
>;

template<std::size_t D, std::size_t K_TILE>
aligned_int_vector k_means_tiled(
    points_soa_view<D> points,
    centroids_storage<D>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(D > 0);
    static_assert(K_TILE > 0);
    fused_kmeans_backend<D, K_TILE> backend{
        points,
        centroids,
        fused_assignment_backend<D, K_TILE>{}
    };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}
