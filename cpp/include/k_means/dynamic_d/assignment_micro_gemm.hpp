#pragma once

#include "layout.hpp"

template<std::size_t D, std::size_t K_TILE>
struct micro_gemm_assignment_layout {
    // Tile-major centroid coefficient pack:
    //     packed[panel][d][k_in_panel]
    // where packed values are -2 * row_major_centroid_value.
    aligned_float_vector packed;
    aligned_float_vector centroid_norm_sq;

    std::size_t K = 0;
    std::size_t K_panel_count = 0;

    void sync_from_row_major(const centroids_storage<D>& centroids) {
        K = centroids.K;
        K_panel_count = (K + K_TILE - 1) / K_TILE;

        packed.resize(K_panel_count * D * K_TILE);
        centroid_norm_sq.resize(K);

        for (std::size_t panel = 0; panel < K_panel_count; ++panel) {
            const std::size_t k_base = panel * K_TILE;

            for (std::size_t t = 0; t < K_TILE; ++t) {
                const std::size_t k = k_base + t;
                if (k < K) {
                    const float c = centroids.row_major[k * D];
                    panel_dimension(panel, 0)[t] = -2.0f * c;
                    centroid_norm_sq[k] = c * c;
                }
            }

            for (std::size_t d = 1; d < D; ++d) {
                for (std::size_t t = 0; t < K_TILE; ++t) {
                    const std::size_t k = k_base + t;
                    if (k < K) {
                        const float c = centroids.row_major[k * D + d];
                        panel_dimension(panel, d)[t] = -2.0f * c;
                        centroid_norm_sq[k] += c * c;
                    }
                }
            }
        }
    }

    void update_cluster_from_row_major(
        const centroids_storage<D>& centroids,
        std::size_t k
    ) {
        const std::size_t panel = k / K_TILE;
        const std::size_t t = k % K_TILE;

        float norm = 0.0f;

        for (std::size_t d = 0; d < D; ++d) {
            const float c = centroids.row_major[k * D + d];
            panel_dimension(panel, d)[t] = -2.0f * c;
            norm += c * c;
        }

        centroid_norm_sq[k] = norm;
    }

    float* panel_dimension(std::size_t panel, std::size_t d) {
        return packed.data() + (panel * D + d) * K_TILE;
    }

    const float* panel_dimension(std::size_t panel, std::size_t d) const {
        return packed.data() + (panel * D + d) * K_TILE;
    }

    const float* panel_dimension_for_k0(std::size_t k0, std::size_t d) const {
        return panel_dimension(k0 / K_TILE, d);
    }
};

inline void micro_gemm_update_best_centroid(
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

template<std::size_t K_COUNT, std::size_t D, std::size_t K_TILE, std::size_t N_VECTORS>
inline void process_micro_gemm_centroid_tile_full_vectors(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t k0,
    std::array<wide_f, N_VECTORS>& best_dist,
    std::array<wide_i, N_VECTORS>& best_k
) {
    constexpr std::size_t card = simd_cardinal();

    std::array<std::array<wide_f, K_COUNT>, N_VECTORS> scores;

    for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
        for (std::size_t t = 0; t < K_COUNT; ++t) {
            scores[sample_vector][t] = wide_f(layout.centroid_norm_sq[k0 + t]);
        }
    }

    // Micro-GEMM shape:
    //   N_VECTORS SIMD sample vectors
    //   K_COUNT   centroid columns
    // No B x K score block is materialized.
    for (std::size_t d = 0; d < D; ++d) {
        const float* sample_dimension = samples.dimension(d);
        const float* centroid_panel_dimension = layout.panel_dimension_for_k0(k0, d);

        std::array<wide_f, N_VECTORS> x;

        for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
            x[sample_vector] = eve::load(
                eve::as_aligned(sample_dimension + n + sample_vector * card)
            );
        }

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f neg_two_c(centroid_panel_dimension[t]);

            for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
                scores[sample_vector][t] = eve::fma(
                    x[sample_vector],
                    neg_two_c,
                    scores[sample_vector][t]
                );
            }
        }
    }

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        const std::size_t k = k0 + t;

        for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
            micro_gemm_update_best_centroid(
                scores[sample_vector][t],
                k,
                best_dist[sample_vector],
                best_k[sample_vector]
            );
        }
    }
}

template<std::size_t TAIL, std::size_t D, std::size_t K_TILE, std::size_t N_VECTORS>
inline void process_micro_gemm_centroid_tail_exact_full_vectors(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t k0,
    std::size_t remaining,
    std::array<wide_f, N_VECTORS>& best_dist,
    std::array<wide_i, N_VECTORS>& best_k
) {
    if (remaining == TAIL) {
        process_micro_gemm_centroid_tile_full_vectors<TAIL>(
            samples,
            layout,
            n,
            k0,
            best_dist,
            best_k
        );
    } else if constexpr (TAIL > 1) {
        process_micro_gemm_centroid_tail_exact_full_vectors<TAIL - 1>(
            samples,
            layout,
            n,
            k0,
            remaining,
            best_dist,
            best_k
        );
    }
}

template<std::size_t K_COUNT, std::size_t D, std::size_t K_TILE, typename Ignore>
inline void process_micro_gemm_centroid_tile_one_vector(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t k0,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    std::array<wide_f, K_COUNT> scores;

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        scores[t] = wide_f(layout.centroid_norm_sq[k0 + t]);
    }

    for (std::size_t d = 0; d < D; ++d) {
        const auto x = eve::load[ignore](
            eve::as_aligned(samples.dimension(d) + n)
        );

        const float* centroid_panel_dimension = layout.panel_dimension_for_k0(k0, d);

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f neg_two_c(centroid_panel_dimension[t]);

            scores[t] = eve::fma(
                x,
                neg_two_c,
                scores[t]
            );
        }
    }

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        micro_gemm_update_best_centroid(
            scores[t],
            k0 + t,
            best_dist,
            best_k
        );
    }
}

template<std::size_t TAIL, std::size_t D, std::size_t K_TILE, typename Ignore>
inline void process_micro_gemm_centroid_tail_exact_one_vector(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    std::size_t k0,
    std::size_t remaining,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    if (remaining == TAIL) {
        process_micro_gemm_centroid_tile_one_vector<TAIL>(
            samples,
            layout,
            n,
            k0,
            ignore,
            best_dist,
            best_k
        );
    } else if constexpr (TAIL > 1) {
        process_micro_gemm_centroid_tail_exact_one_vector<TAIL - 1>(
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

template<std::size_t N_VECTORS, bool TrackChanges, std::size_t D, std::size_t K_TILE>
inline bool assign_full_micro_gemm_sample_group(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    int* assignments
) {
    constexpr std::size_t card = simd_cardinal();

    const std::size_t K = layout.K;

    std::array<wide_f, N_VECTORS> best_dist;
    std::array<wide_i, N_VECTORS> best_k;

    for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
        best_dist[sample_vector] = eve::valmax(eve::as<wide_f>());
        best_k[sample_vector] = eve::zero(eve::as<wide_i>());
    }

    std::size_t k0 = 0;

    for (; k0 + K_TILE <= K; k0 += K_TILE) {
        process_micro_gemm_centroid_tile_full_vectors<K_TILE>(
            samples,
            layout,
            n,
            k0,
            best_dist,
            best_k
        );
    }

    if constexpr (K_TILE > 1) {
        const std::size_t remaining = K - k0;

        if (remaining != 0) {
            process_micro_gemm_centroid_tail_exact_full_vectors<K_TILE - 1>(
                samples,
                layout,
                n,
                k0,
                remaining,
                best_dist,
                best_k
            );
        }
    }

    bool changed = false;

    for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
        auto assignment_ptr = eve::as_aligned(
            assignments + n + sample_vector * card,
            typename wide_i::cardinal_type{}
        );

        if constexpr (TrackChanges) {
            const wide_i previous_label =
                eve::load[eve::ignore_none](
                    assignment_ptr,
                    eve::as<wide_i>{}
                );

            changed = changed || eve::any(best_k[sample_vector] != previous_label);
        }

        eve::store[eve::ignore_none](best_k[sample_vector], assignment_ptr);
    }

    return changed;
}

template<bool TrackChanges, std::size_t D, std::size_t K_TILE, typename Ignore>
inline bool assign_one_micro_gemm_sample_vector(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t n,
    Ignore ignore,
    int* assignments
) {
    const std::size_t K = layout.K;

    auto assignment_ptr = eve::as_aligned(
        assignments + n,
        typename wide_i::cardinal_type{}
    );

    wide_f best_dist = eve::valmax(eve::as<wide_f>());
    wide_i best_k = eve::zero(eve::as<wide_i>());

    std::size_t k0 = 0;

    for (; k0 + K_TILE <= K; k0 += K_TILE) {
        process_micro_gemm_centroid_tile_one_vector<K_TILE>(
            samples,
            layout,
            n,
            k0,
            ignore,
            best_dist,
            best_k
        );
    }

    if constexpr (K_TILE > 1) {
        const std::size_t remaining = K - k0;

        if (remaining != 0) {
            process_micro_gemm_centroid_tail_exact_one_vector<K_TILE - 1>(
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
            eve::load[ignore](assignment_ptr, eve::as<wide_i>{});

        changed = eve::any[ignore](best_k != previous_label);
    }

    eve::store[ignore](best_k, assignment_ptr);

    return changed;
}

template<
    std::size_t N_VECTORS,
    bool TrackChanges,
    std::size_t D,
    std::size_t K_TILE
>
bool assign_samples_to_centroids_micro_gemm_impl(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    constexpr std::size_t card = simd_cardinal();
    constexpr std::size_t group_samples = N_VECTORS * card;

    const std::size_t N = samples.N;

    bool any_changed = false;
    std::size_t n = 0;

    auto assign_full_group = [&](std::size_t n) {
        if constexpr (TrackChanges) {
            if (!any_changed) {
                any_changed =
                    assign_full_micro_gemm_sample_group<N_VECTORS, true>(
                        samples,
                        layout,
                        n,
                        assignments.data()
                    );
            } else {
                (void)assign_full_micro_gemm_sample_group<N_VECTORS, false>(
                    samples,
                    layout,
                    n,
                    assignments.data()
                );
            }
        } else {
            (void)assign_full_micro_gemm_sample_group<N_VECTORS, false>(
                samples,
                layout,
                n,
                assignments.data()
            );
        }
    };

    auto assign_one_vector = [&](std::size_t n, auto ignore) {
        if constexpr (TrackChanges) {
            if (!any_changed) {
                any_changed =
                    assign_one_micro_gemm_sample_vector<true>(
                        samples,
                        layout,
                        n,
                        ignore,
                        assignments.data()
                    );
            } else {
                (void)assign_one_micro_gemm_sample_vector<false>(
                    samples,
                    layout,
                    n,
                    ignore,
                    assignments.data()
                );
            }
        } else {
            (void)assign_one_micro_gemm_sample_vector<false>(
                samples,
                layout,
                n,
                ignore,
                assignments.data()
            );
        }
    };

    // Generic N_VECTORS main loop.
    // This is intentionally unmasked.
    for (; n + group_samples <= N; n += group_samples) {
        assign_full_group(n);
    }

    // Tail path: simpler one-SIMD-vector blocks.
    //
    // This avoids trying to store heterogeneous EVE ignore decorators
    // for N_VECTORS sample vectors. The hot path remains generic.
    for (; n + card <= N; n += card) {
        assign_one_vector(n, eve::ignore_none);
    }

    if (n < N) {
        const std::size_t valid = N - n;
        const std::size_t ignored_lanes = card - valid;

        assign_one_vector(n, eve::ignore_last(ignored_lanes));
    }

    return any_changed;
}

template<std::size_t D, std::size_t N_VECTORS, std::size_t K_TILE>
void assign_samples_to_centroids_micro_gemm(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    (void)assign_samples_to_centroids_micro_gemm_impl<N_VECTORS, false>(
        samples,
        layout,
        assignments
    );
}

template<std::size_t D, std::size_t N_VECTORS, std::size_t K_TILE>
bool assign_samples_to_centroids_micro_gemm_and_check_changed(
    samples_soa_view<D> samples,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    return assign_samples_to_centroids_micro_gemm_impl<N_VECTORS, true>(
        samples,
        layout,
        assignments
    );
}

template<std::size_t D, std::size_t N_VECTORS, std::size_t K_TILE>
struct micro_gemm_assignment_backend {
    micro_gemm_assignment_layout<D, K_TILE> layout;

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
        assign_samples_to_centroids_micro_gemm<D, N_VECTORS, K_TILE>(
            samples,
            layout,
            assignments
        );
    }

    bool assign_and_check_changed(
        samples_soa_view<D> samples,
        std::span<int> assignments
    ) const {
        return assign_samples_to_centroids_micro_gemm_and_check_changed<D, N_VECTORS, K_TILE>(
            samples,
            layout,
            assignments
        );
    }
};
