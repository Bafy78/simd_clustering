#pragma once

#include "layout.hpp"

template<std::size_t D, std::size_t K_TILE>
struct centroid_tiled_assignment_layout {
    aligned_float_vector dimension_major;
    aligned_float_vector centroid_norm_sq;

    std::size_t K = 0;
    std::size_t dimension_major_stride = 0;

    void sync_from_row_major(const centroids_storage<D>& centroids) {
        K = centroids.K;
        dimension_major_stride = round_up_to_multiple(K, K_TILE);

        dimension_major.resize(D * dimension_major_stride);
        centroid_norm_sq.resize(K);

        for (std::size_t k = 0; k < K; ++k) {
            const float c = centroids.row_major[k * D];
            dimension_major.data()[k] = -2.0f * c;
            centroid_norm_sq[k] = c * c;
        }

        for (std::size_t d = 1; d < D; ++d) {
            float* dst = dimension_major.data() + d * dimension_major_stride;

            for (std::size_t k = 0; k < K; ++k) {
                const float c = centroids.row_major[k * D + d];
                dst[k] = -2.0f * c;
                centroid_norm_sq[k] += c * c;
            }
        }
    }

    void update_cluster_from_row_major(
        const centroids_storage<D>& centroids,
        std::size_t k
    ) {
        float norm = 0.0f;

        for (std::size_t d = 0; d < D; ++d) {
            const float c = centroids.row_major[k * D + d];
            dimension_major[d * dimension_major_stride + k] = -2.0f * c;
            norm += c * c;
        }

        centroid_norm_sq[k] = norm;
    }

    const float* dimension_centroids(std::size_t d) const {
        return dimension_major.data() + d * dimension_major_stride;
    }

    template<std::size_t Dimension>
    const float* dimension_centroids() const {
        static_assert(Dimension < D);
        return dimension_major.data() + Dimension * dimension_major_stride;
    }
};

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

inline void update_best_centroid(
    wide_f dist,
    std::size_t candidate_k,
    wide_f& best_dist,
    wide_i& best_k
) {
    auto closer = eve::is_less(dist, best_dist);

    best_dist = eve::min(best_dist, dist);

    best_k = eve::if_else(
        closer,
        wide_i(static_cast<int>(candidate_k)),
        best_k
    );
}

// Process K_TILE_GROUPS consecutive full compile-time centroid tiles while reusing each
// loaded sample dimension across all tile groups.
template<
    std::size_t K_COUNT,
    std::size_t K_TILE_GROUPS,
    std::size_t D,
    std::size_t K_TILE,
    typename Ignore
>
inline void process_centroid_tiles(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t k0,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    constexpr std::size_t TOTAL_TILE = K_COUNT * K_TILE_GROUPS;
    auto scores = kumi::generate<TOTAL_TILE>([&](auto index) {
        return wide_f(layout.centroid_norm_sq[k0 + index]);
    });

    for (std::size_t d = 0; d < D; ++d) {
        // samples are dimension-major:
        //     samples[d][n : n + SIMD_WIDTH]
        //
        // dimension_major stores the assignment coefficient -2*c:
        //     score = ||c||^2 + sum_d x[d] * (-2*c[d])
        auto x = eve::load[ignore](
            eve::as_aligned(samples.dimension(d) + n)
        );

        const float* centroid_dimension =
            layout.dimension_centroids(d) + k0;

        kumi::for_each_index([&](auto index, auto& score) {
            auto neg_two_c = wide_f(centroid_dimension[index]);

            score = eve::fma(x, neg_two_c, score);
        }, scores);
    }

    kumi::for_each_index([&](auto index, auto score) {
        const std::size_t k = k0 + index;

        update_best_centroid(
            score,
            k,
            best_dist,
            best_k
        );
    }, scores);
}

template<std::size_t K_TILE_GROUPS, std::size_t D, std::size_t K_TILE, typename Ignore>
inline void process_full_centroid_tile_groups(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t& k0,
    std::size_t K,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    constexpr std::size_t K_GROUP_TILE = K_TILE_GROUPS * K_TILE;

    for (; k0 + K_GROUP_TILE <= K; k0 += K_GROUP_TILE) {
        process_centroid_tiles<K_TILE, K_TILE_GROUPS>(
            samples,
            layout,
            n,
            k0,
            ignore,
            best_dist,
            best_k
        );
    }

    if constexpr (K_TILE_GROUPS > 1) {
        process_full_centroid_tile_groups<K_TILE_GROUPS - 1>(
            samples,
            layout,
            n,
            k0,
            K,
            ignore,
            best_dist,
            best_k
        );
    }
}

// Compile-time exact tail dispatch.
template<std::size_t TAIL, std::size_t D, std::size_t K_TILE, typename Ignore>
inline void process_centroid_tail_exact(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t k0,
    std::size_t remaining,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    if (remaining == TAIL) {
        process_centroid_tiles<TAIL, 1>(
            samples,
            layout,
            n,
            k0,
            ignore,
            best_dist,
            best_k
        );
    } else if constexpr (TAIL > 1) {
        process_centroid_tail_exact<TAIL - 1>(
            samples,
            layout,
            n,
            k0,
            remaining,
            ignore,
            best_dist,
            best_k
        );
    }
}

template<bool TrackChanges, std::size_t D, std::size_t K_TILE, typename Ignore>
inline bool assign_one_sample_block_tiled(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    Ignore ignore,
    int* assignment_ptr
) {
    const std::size_t K = layout.K;

    auto assignment_aligned_ptr = eve::as_aligned(
        assignment_ptr, typename wide_i::cardinal_type{}
    );

    auto best_dist = eve::valmax(eve::as<wide_f>());
    auto best_k = wide_zero_i;

    std::size_t k0 = 0;

    constexpr std::size_t K_TILE_GROUPS_AT_ONCE = centroid_tiles_at_once<K_TILE>();

    process_full_centroid_tile_groups<K_TILE_GROUPS_AT_ONCE>(
        samples,
        layout,
        n,
        k0,
        K,
        ignore,
        best_dist,
        best_k
    );

    if constexpr (K_TILE > 1) {
        const std::size_t remaining = K - k0;

        if (remaining != 0) {
            process_centroid_tail_exact<K_TILE - 1>(
                samples,
                layout,
                n,
                k0,
                remaining,
                ignore,
                best_dist,
                best_k
            );
        }
    }

    bool changed = false;

    if constexpr (TrackChanges) {
        const wide_i previous_label =
            eve::load[ignore](assignment_aligned_ptr, eve::as<wide_i>{});

        changed = eve::any[ignore](best_k != previous_label);
    }

    eve::store[ignore](best_k, assignment_aligned_ptr);

    return changed;
}

template<bool TrackChanges, std::size_t D, std::size_t K_TILE>
bool assign_samples_to_centroids_tiled_impl(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    constexpr std::size_t card = simd_cardinal();
    const std::size_t N = samples.N;

    bool any_changed = false;
    std::size_t n = 0;

    auto assign_block = [&](std::size_t n, auto ignore) {
        if constexpr (TrackChanges) {
            if (!any_changed) {
                any_changed =
                    assign_one_sample_block_tiled<true>(
                        samples,
                        layout,
                        n,
                        ignore,
                        assignments.data() + n
                    );
            } else {
                (void)assign_one_sample_block_tiled<false>(
                    samples,
                    layout,
                    n,
                    ignore,
                    assignments.data() + n
                );
            }
        } else {
            (void)assign_one_sample_block_tiled<false>(
                samples,
                layout,
                n,
                ignore,
                assignments.data() + n
            );
        }
    };

    // Fast unmasked main loop.
    for (; n + card <= N; n += card) {
        assign_block(n, eve::ignore_none);
    }

    // Masked tail only.
    if (n < N) {
        const std::size_t valid = N - n;
        const std::size_t ignored_lanes = card - valid;

        assign_block(n, eve::ignore_last(ignored_lanes));
    }

    return any_changed;
}

// Static-D high-D assignment kernel.
// The main grouped loop keeps centroid_tiles_at_once<K_TILE>() * K_TILE dot accumulators live; D is streamed.
// The main loop is unmasked; only the final tail block uses ignore_last.
// assignments.data() must be SIMD-aligned
template<std::size_t D, std::size_t K_TILE>
void assign_samples_to_centroids_tiled(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    (void)assign_samples_to_centroids_tiled_impl<false>(
        samples,
        layout,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE>
bool assign_samples_to_centroids_tiled_and_check_changed(
    samples_soa_view<D> samples,
    const centroid_tiled_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    return assign_samples_to_centroids_tiled_impl<true>(
        samples,
        layout,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE>
struct centroid_tiled_assignment_backend {
    centroid_tiled_assignment_layout<D, K_TILE> layout;

    void on_centroids_changed(const centroids_storage<D>& centroids) {
        layout.sync_from_row_major(centroids);
    }

    void on_centroids_changed_for_clusters(
        const centroids_storage<D>& centroids,
        std::span<const int> dirty_clusters
    ) {
        if (dirty_clusters.size() * 2 > centroids.K) {
            on_centroids_changed(centroids);
            return;
        }

        for (int k : dirty_clusters) {
            layout.update_cluster_from_row_major(
                centroids,
                static_cast<std::size_t>(k)
            );
        }
    }

    void assign(
        samples_soa_view<D> samples,
        std::span<int> assignments
    ) const {
        assign_samples_to_centroids_tiled<D, K_TILE>(
            samples,
            layout,
            assignments
        );
    }

    bool assign_and_check_changed(
        samples_soa_view<D> samples,
        std::span<int> assignments
    ) const {
        return assign_samples_to_centroids_tiled_and_check_changed<D, K_TILE>(
            samples,
            layout,
            assignments
        );
    }
};
