#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>
#include <vector>
#include <numeric>
#include <ranges>
#include <array>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "./k_means_core.hpp"

using wide_f = eve::wide<float>;
using wide_i = eve::wide<int, typename wide_f::cardinal_type>;

using aligned_float_vector = std::vector<float, eve::aligned_allocator<float>>;
using aligned_int_vector = std::vector<int, eve::aligned_allocator<int>>;

template<class Label>
using aligned_label_vector = std::vector<Label, eve::aligned_allocator<Label>>;

inline constexpr std::size_t simd_cardinal() {
    return static_cast<std::size_t>(wide_f::size());
}

inline std::size_t round_up_to_multiple(std::size_t n, std::size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
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

    const float* data = nullptr;

    std::size_t n_samples = 0;
    std::size_t stride = 0;

    const float* feature(std::size_t d) const {
        return data + d * stride;
    }

    template<std::size_t Feature>
    const float* feature() const {
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

    points_soa_view<D> view() const {
        return points_soa_view<D>{
            .data = data.data(),
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

    std::size_t n_clusters = 0;
    std::size_t feature_major_stride = 0;

    centroids_storage() = default;

    template<std::size_t K_TILE>
    void resize_for_tile(std::size_t clusters) {
        static_assert(K_TILE > 0);
        static_assert(std::has_single_bit(K_TILE));

        n_clusters = clusters;
        feature_major_stride = round_up_to_multiple(n_clusters, K_TILE);

        row_major.assign(n_clusters * D, 0.0f);
        feature_major.assign(D * feature_major_stride, 0.0f);
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

    // Call this once after updating row_major centroids.
    void sync_feature_major_from_row_major() {
        std::fill(feature_major.begin(), feature_major.end(), 0.0f);

        kmeans::for_each_feature<D>([&](auto feature_index) {
            constexpr std::size_t d = decltype(feature_index)::value;

            float* dst = feature_major.data() + d * feature_major_stride;

            for (std::size_t k = 0; k < n_clusters; ++k) {
                dst[k] = row_major[k * D + d];
            }
        });
    }
};

// Process one full compile-time centroid tile
// This is the high-D replacement for: `compute_simd_dist_sq(pt, centroid[k])`
// D is compile-time, but dimensions are still streamed one at a time.
// We do NOT materialize a kumi::tuple<wide_f, ..., wide_f> point.
// Each feature vector is loaded, used for all K_TILE accumulators, then discarded.
template<std::size_t D, std::size_t K_TILE, typename Ignore>
inline void process_centroid_tile(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    std::size_t k0,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    auto distances = kumi::fill<K_TILE>(eve::zero(eve::as<wide_f>()));

    kmeans::for_each_feature<D>([&](auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;

        // points are feature-major:
        //     points[d][sample_i : sample_i + SIMD_WIDTH]
        //
        // D is compile-time, but the feature column is streamed.
        auto x = eve::load[ignore](
            eve::as_aligned(points.template feature<d>() + sample_i)
            );

        const float* centroid_feature = centroids.template feature_centroids<d>() + k0;

        kumi::for_each_index(
            [&](auto index, auto& dist) {
            constexpr std::size_t t = decltype(index)::value;

            auto c = wide_f(centroid_feature[t]);
            auto diff = x - c;

            dist = eve::fma(diff, diff, dist);
        },
            distances
        );
    });

    kumi::for_each_index(
        [&](auto index, auto dist) {
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
template<std::size_t D, std::size_t TILE, typename Ignore>
inline void process_centroid_tail(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    std::size_t& k0,
    std::size_t& remaining,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    if (remaining >= TILE) {
        process_centroid_tile<D, TILE>(
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
        process_centroid_tail<D, TILE / 2>(
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

template<std::size_t D, std::size_t K_TILE, bool TrackChanges, class Label, typename Ignore>
inline bool assign_one_sample_block_tiled(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    Ignore ignore,
    Label* assignment_ptr
) {
    const std::size_t K = centroids.n_clusters;

    using wide_label = eve::wide<Label, typename wide_f::cardinal_type>;

    auto assignment_aligned_ptr = eve::as_aligned(
        assignment_ptr, typename wide_label::cardinal_type{}
    );

    auto best_dist = eve::valmax(eve::as<wide_f>());
    auto best_k = eve::zero(eve::as<wide_i>());

    std::size_t k0 = 0;

    for (; k0 + K_TILE <= K; k0 += K_TILE) {
        process_centroid_tile<D, K_TILE>(
            points,
            centroids,
            sample_i,
            k0,
            ignore,
            best_dist,
            best_k
        );
    }

    if constexpr (K_TILE > 1) {
        std::size_t remaining = K - k0;

        process_centroid_tail<D, K_TILE / 2>(
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

    bool changed = false;

    const wide_label best_label = eve::convert(best_k, eve::as<Label>());

    if constexpr (TrackChanges) {
        const wide_label previous_label =
            eve::load[ignore](assignment_aligned_ptr, eve::as<wide_label>{});

        changed = eve::any[ignore](best_label != previous_label);
    }

    eve::store[ignore](best_label, assignment_aligned_ptr);

    return changed;
}

template<std::size_t D, std::size_t K_TILE, bool TrackChanges, class Label>
bool assign_points_to_centroids_tiled_impl(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::span<Label> assignments
) {
    constexpr std::size_t card = simd_cardinal();
    const std::size_t n = points.n_samples;

    bool any_changed = false;
    std::size_t i = 0;

    auto assign_block = [&](std::size_t sample_i, auto ignore) {
        if constexpr (TrackChanges) {
            if (!any_changed) {
                any_changed =
                    assign_one_sample_block_tiled<D, K_TILE, true>(
                        points,
                        centroids,
                        sample_i,
                        ignore,
                        assignments.data() + sample_i
                    );
            } else {
                (void)assign_one_sample_block_tiled<D, K_TILE, false>(
                    points,
                    centroids,
                    sample_i,
                    ignore,
                    assignments.data() + sample_i
                );
            }
        } else {
            (void)assign_one_sample_block_tiled<D, K_TILE, false>(
                points,
                centroids,
                sample_i,
                ignore,
                assignments.data() + sample_i
            );
        }
    };

    // Fast unmasked main loop.
    for (; i + card <= n; i += card) {
        assign_block(i, eve::ignore_none);
    }

    // Masked tail only.
    if (i < n) {
        const std::size_t valid = n - i;
        const std::size_t ignored_lanes = card - valid;

        assign_block(i, eve::ignore_last(ignored_lanes));
    }

    return any_changed;
}

// Static-D high-D assignment kernel.
// Register pressure depends on K_TILE, not on D.
// The main loop is unmasked; only the final tail block uses ignore_last.
// assignments.data() must be SIMD-aligned
template<std::size_t D, std::size_t K_TILE, class Label>
void assign_points_to_centroids_tiled(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::span<Label> assignments
) {
    (void)assign_points_to_centroids_tiled_impl<D, K_TILE, false>(
        points,
        centroids,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE, class Label>
bool assign_points_to_centroids_tiled_and_check_changed(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::span<Label> assignments
) {
    return assign_points_to_centroids_tiled_impl<D, K_TILE, true>(
        points,
        centroids,
        assignments
    );
}

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

template<std::size_t D, class Label>
inline void resolve_dead_centroids(
    points_soa_view<D> points,
    std::span<const Label> assignments,
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

template<std::size_t D, class Label>
inline void update_centroids(
    points_soa_view<D> points,
    std::span<const Label> assignments,
    centroids_storage<D>& centroids,
    aligned_float_vector& sums_row_major,
    aligned_int_vector& counts
) {
    const std::size_t n_clusters = centroids.n_clusters;

    if (sums_row_major.size() != n_clusters * D) {
        sums_row_major.assign(n_clusters * D, 0.0f);
    }

    if (counts.size() != n_clusters) {
        counts.assign(n_clusters, 0);
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

        void resolve_dead_centroids(std::span<const Label> assignments, std::span<int> counts) {
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

        void after_centroids_updated() {
            centroids.sync_feature_major_from_row_major();
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

template<std::size_t D, std::size_t K_TILE, class Label = kmeans::default_label_t>
struct tiled_kmeans_backend {
    using label_type = Label;
    using assignment_vector = aligned_label_vector<label_type>;
    using counts_vector = aligned_int_vector;
    using centroid_snapshot = aligned_float_vector;

    static constexpr label_type invalid_label = kmeans::invalid_label_v<label_type>;

    points_soa_view<D> points;
    centroids_storage<D>& centroids;

    aligned_float_vector sums_row_major;

    tiled_kmeans_backend(
        points_soa_view<D> points_,
        centroids_storage<D>& centroids_
    ) : points(points_), centroids(centroids_), sums_row_major(centroids_.n_clusters* D, 0.0f) {
        centroids.sync_feature_major_from_row_major();
    }

    void check_cluster_count() const {
        kmeans::check_cluster_count_fits<label_type>(centroids.n_clusters);
        kmeans::check_cluster_count(centroids.n_clusters, points.n_samples);
    }

    assignment_vector make_assignment_vector(label_type initial_value) const {
        return assignment_vector(points.n_samples, initial_value);
    }

    counts_vector make_counts_vector() const {
        return counts_vector(centroids.n_clusters, 0);
    }

    float compute_tolerance(float tol) const {
        return ::compute_sklearn_tolerance<D>(points, tol);
    }

    centroid_snapshot make_centroid_snapshot() const {
        return centroid_snapshot(centroids.row_major.size(), 0.0f);
    }

    void save_centroids(centroid_snapshot& snapshot) const {
        snapshot = centroids.row_major;
    }

    void assign(assignment_vector& assignments) const {
        ::assign_points_to_centroids_tiled<D, K_TILE>(
            points,
            centroids,
            std::span<label_type>(assignments.data(), assignments.size())
        );
    }
    bool assign_and_check_changed(assignment_vector& assignments) const {
        return ::assign_points_to_centroids_tiled_and_check_changed<D, K_TILE>(
            points,
            centroids,
            std::span<label_type>(assignments.data(), assignments.size())
        );
    }

    void update_centroids(
        assignment_vector& assignments,
        counts_vector& counts
    ) {
        ::update_centroids<D>(
            points,
            std::span<const label_type>(assignments.data(), assignments.size()),
            centroids,
            sums_row_major,
            counts
        );
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

template<std::size_t D, std::size_t K_TILE, class Label = kmeans::default_label_t>
aligned_label_vector<Label> k_means_tiled(
    points_soa_view<D> points,
    centroids_storage<D>& centroids,
    int& out_iterations,
    int max_iterations = 300,
    float tol = 1e-4f
) {
    static_assert(D > 0);
    static_assert(K_TILE > 0);
    static_assert(std::has_single_bit(K_TILE));

    tiled_kmeans_backend<D, K_TILE, Label> backend{ points, centroids };

    return kmeans::k_means_core(
        backend,
        out_iterations,
        max_iterations,
        tol
    );
}
