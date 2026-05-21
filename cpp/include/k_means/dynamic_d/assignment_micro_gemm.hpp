template<std::size_t D, std::size_t K_TILE>
struct micro_gemm_assignment_layout {
    // Tile-major centroid coefficient pack:
    //     packed[panel][d][k_in_panel]
    // where packed values are -2 * row_major_centroid_value.
    aligned_float_vector packed;
    aligned_float_vector centroid_norm_sq;

    std::size_t n_clusters = 0;
    std::size_t n_panels = 0;

    void sync_from_row_major(const centroids_storage<D>& centroids) {
        n_clusters = centroids.n_clusters;
        n_panels = (n_clusters + K_TILE - 1) / K_TILE;

        packed.resize(n_panels * D * K_TILE);
        centroid_norm_sq.resize(n_clusters);

        for (std::size_t panel = 0; panel < n_panels; ++panel) {
            const std::size_t k_base = panel * K_TILE;

            for (std::size_t t = 0; t < K_TILE; ++t) {
                const std::size_t k = k_base + t;
                if (k < n_clusters) {
                    const float c = centroids.row_major[k * D];
                    panel_feature(panel, 0)[t] = -2.0f * c;
                    centroid_norm_sq[k] = c * c;
                }
            }

            for (std::size_t d = 1; d < D; ++d) {
                for (std::size_t t = 0; t < K_TILE; ++t) {
                    const std::size_t k = k_base + t;
                    if (k < n_clusters) {
                        const float c = centroids.row_major[k * D + d];
                        panel_feature(panel, d)[t] = -2.0f * c;
                        centroid_norm_sq[k] += c * c;
                    }
                }
            }
        }
    }

    float* panel_feature(std::size_t panel, std::size_t d) {
        return packed.data() + (panel * D + d) * K_TILE;
    }

    const float* panel_feature(std::size_t panel, std::size_t d) const {
        return packed.data() + (panel * D + d) * K_TILE;
    }

    const float* panel_feature_for_k0(std::size_t k0, std::size_t d) const {
        return panel_feature(k0 / K_TILE, d);
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

template<std::size_t K_COUNT, std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
inline void process_micro_gemm_centroid_tile_full_vectors(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t sample_i,
    std::size_t k0,
    std::array<wide_f, M_VECTORS>& best_dist,
    std::array<wide_i, M_VECTORS>& best_k
) {
    constexpr std::size_t card = simd_cardinal();

    std::array<std::array<wide_f, K_COUNT>, M_VECTORS> scores;

    for (std::size_t m = 0; m < M_VECTORS; ++m) {
        for (std::size_t t = 0; t < K_COUNT; ++t) {
            scores[m][t] = wide_f(layout.centroid_norm_sq[k0 + t]);
        }
    }

    // Micro-GEMM shape:
    //   M_VECTORS SIMD sample vectors
    //   K_COUNT   centroid columns
    // No B x K score block is materialized.
    for (std::size_t d = 0; d < D; ++d) {
        const float* point_feature = points.feature(d);
        const float* centroid_panel_feature = layout.panel_feature_for_k0(k0, d);

        std::array<wide_f, M_VECTORS> x;

        for (std::size_t m = 0; m < M_VECTORS; ++m) {
            x[m] = eve::load(
                eve::as_aligned(point_feature + sample_i + m * card)
            );
        }

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f neg_two_c(centroid_panel_feature[t]);

            for (std::size_t m = 0; m < M_VECTORS; ++m) {
                scores[m][t] = eve::fma(
                    x[m],
                    neg_two_c,
                    scores[m][t]
                );
            }
        }
    }

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        const std::size_t k = k0 + t;

        for (std::size_t m = 0; m < M_VECTORS; ++m) {
            micro_gemm_update_best_centroid(
                scores[m][t],
                k,
                best_dist[m],
                best_k[m]
            );
        }
    }
}

template<std::size_t TAIL, std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
inline void process_micro_gemm_centroid_tail_exact_full_vectors(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t sample_i,
    std::size_t k0,
    std::size_t remaining,
    std::array<wide_f, M_VECTORS>& best_dist,
    std::array<wide_i, M_VECTORS>& best_k
) {
    if (remaining == TAIL) {
        process_micro_gemm_centroid_tile_full_vectors<TAIL>(
            points,
            layout,
            sample_i,
            k0,
            best_dist,
            best_k
        );
    } else if constexpr (TAIL > 1) {
        process_micro_gemm_centroid_tail_exact_full_vectors<TAIL - 1>(
            points,
            layout,
            sample_i,
            k0,
            remaining,
            best_dist,
            best_k
        );
    }
}

template<std::size_t K_COUNT, std::size_t D, std::size_t K_TILE, typename Ignore>
inline void process_micro_gemm_centroid_tile_one_vector(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t sample_i,
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
            eve::as_aligned(points.feature(d) + sample_i)
        );

        const float* centroid_panel_feature = layout.panel_feature_for_k0(k0, d);

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f neg_two_c(centroid_panel_feature[t]);

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
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t sample_i,
    std::size_t k0,
    std::size_t remaining,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    if (remaining == TAIL) {
        process_micro_gemm_centroid_tile_one_vector<TAIL>(
            points,
            layout,
            sample_i,
            k0,
            ignore,
            best_dist,
            best_k
        );
    } else if constexpr (TAIL > 1) {
        process_micro_gemm_centroid_tail_exact_one_vector<TAIL - 1>(
            points,
            layout,
            sample_i,
            k0,
            remaining,
            ignore,
            best_dist,
            best_k
        );
    }
}

template<std::size_t M_VECTORS, bool TrackChanges, std::size_t D, std::size_t K_TILE>
inline bool assign_full_micro_gemm_sample_group(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t sample_i,
    int* assignments
) {
    constexpr std::size_t card = simd_cardinal();

    const std::size_t K = layout.n_clusters;

    std::array<wide_f, M_VECTORS> best_dist;
    std::array<wide_i, M_VECTORS> best_k;

    for (std::size_t m = 0; m < M_VECTORS; ++m) {
        best_dist[m] = eve::valmax(eve::as<wide_f>());
        best_k[m] = eve::zero(eve::as<wide_i>());
    }

    std::size_t k0 = 0;

    for (; k0 + K_TILE <= K; k0 += K_TILE) {
        process_micro_gemm_centroid_tile_full_vectors<K_TILE>(
            points,
            layout,
            sample_i,
            k0,
            best_dist,
            best_k
        );
    }

    if constexpr (K_TILE > 1) {
        const std::size_t remaining = K - k0;

        if (remaining != 0) {
            process_micro_gemm_centroid_tail_exact_full_vectors<K_TILE - 1>(
                points,
                layout,
                sample_i,
                k0,
                remaining,
                best_dist,
                best_k
            );
        }
    }

    bool changed = false;

    for (std::size_t m = 0; m < M_VECTORS; ++m) {
        auto assignment_ptr = eve::as_aligned(
            assignments + sample_i + m * card,
            typename wide_i::cardinal_type{}
        );

        if constexpr (TrackChanges) {
            const wide_i previous_label =
                eve::load[eve::ignore_none](
                    assignment_ptr,
                    eve::as<wide_i>{}
                );

            changed = changed || eve::any(best_k[m] != previous_label);
        }

        eve::store[eve::ignore_none](best_k[m], assignment_ptr);
    }

    return changed;
}

template<bool TrackChanges, std::size_t D, std::size_t K_TILE, typename Ignore>
inline bool assign_one_micro_gemm_sample_vector(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::size_t sample_i,
    Ignore ignore,
    int* assignments
) {
    const std::size_t K = layout.n_clusters;

    auto assignment_ptr = eve::as_aligned(
        assignments + sample_i,
        typename wide_i::cardinal_type{}
    );

    wide_f best_dist = eve::valmax(eve::as<wide_f>());
    wide_i best_k = eve::zero(eve::as<wide_i>());

    std::size_t k0 = 0;

    for (; k0 + K_TILE <= K; k0 += K_TILE) {
        process_micro_gemm_centroid_tile_one_vector<K_TILE>(
            points,
            layout,
            sample_i,
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
                points,
                layout,
                sample_i,
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
    std::size_t M_VECTORS,
    bool TrackChanges,
    std::size_t D,
    std::size_t K_TILE
>
bool assign_points_to_centroids_micro_gemm_impl(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    constexpr std::size_t card = simd_cardinal();
    constexpr std::size_t group_samples = M_VECTORS * card;

    const std::size_t n = points.n_samples;

    bool any_changed = false;
    std::size_t i = 0;

    auto assign_full_group = [&](std::size_t sample_i) {
        if constexpr (TrackChanges) {
            if (!any_changed) {
                any_changed =
                    assign_full_micro_gemm_sample_group<M_VECTORS, true>(
                        points,
                        layout,
                        sample_i,
                        assignments.data()
                    );
            } else {
                (void)assign_full_micro_gemm_sample_group<M_VECTORS, false>(
                    points,
                    layout,
                    sample_i,
                    assignments.data()
                );
            }
        } else {
            (void)assign_full_micro_gemm_sample_group<M_VECTORS, false>(
                points,
                layout,
                sample_i,
                assignments.data()
            );
        }
    };

    auto assign_one_vector = [&](std::size_t sample_i, auto ignore) {
        if constexpr (TrackChanges) {
            if (!any_changed) {
                any_changed =
                    assign_one_micro_gemm_sample_vector<true>(
                        points,
                        layout,
                        sample_i,
                        ignore,
                        assignments.data()
                    );
            } else {
                (void)assign_one_micro_gemm_sample_vector<false>(
                    points,
                    layout,
                    sample_i,
                    ignore,
                    assignments.data()
                );
            }
        } else {
            (void)assign_one_micro_gemm_sample_vector<false>(
                points,
                layout,
                sample_i,
                ignore,
                assignments.data()
            );
        }
    };

    // Generic M_VECTORS main loop.
    // This is intentionally unmasked.
    for (; i + group_samples <= n; i += group_samples) {
        assign_full_group(i);
    }

    // Tail path: simpler one-SIMD-vector blocks.
    //
    // This avoids trying to store heterogeneous EVE ignore decorators
    // for M_VECTORS lanes. The hot path remains generic.
    for (; i + card <= n; i += card) {
        assign_one_vector(i, eve::ignore_none);
    }

    if (i < n) {
        const std::size_t valid = n - i;
        const std::size_t ignored_lanes = card - valid;

        assign_one_vector(i, eve::ignore_last(ignored_lanes));
    }

    return any_changed;
}

template<std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
void assign_points_to_centroids_micro_gemm(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    (void)assign_points_to_centroids_micro_gemm_impl<M_VECTORS, false>(
        points,
        layout,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
bool assign_points_to_centroids_micro_gemm_and_check_changed(
    points_soa_view<D> points,
    const micro_gemm_assignment_layout<D, K_TILE>& layout,
    std::span<int> assignments
) {
    return assign_points_to_centroids_micro_gemm_impl<M_VECTORS, true>(
        points,
        layout,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE, std::size_t M_VECTORS>
struct micro_gemm_assignment_backend {
    micro_gemm_assignment_layout<D, K_TILE> layout;

    void on_centroids_changed(const centroids_storage<D>& centroids) {
        layout.sync_from_row_major(centroids);
    }

    void assign(
        points_soa_view<D> points,
        std::span<int> assignments
    ) const {
        assign_points_to_centroids_micro_gemm<D, K_TILE, M_VECTORS>(
            points,
            layout,
            assignments
        );
    }

    bool assign_and_check_changed(
        points_soa_view<D> points,
        std::span<int> assignments
    ) const {
        return assign_points_to_centroids_micro_gemm_and_check_changed<D, K_TILE, M_VECTORS>(
            points,
            layout,
            assignments
        );
    }
};
