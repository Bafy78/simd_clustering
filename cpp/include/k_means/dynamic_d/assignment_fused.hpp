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

// Process N_TILES consecutive full compile-time centroid tiles while reusing each
// loaded point feature across all tile groups.
template<std::size_t D, std::size_t K_TILE, std::size_t N_TILES, typename Ignore>
inline void process_centroid_tiles(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    std::size_t k0,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    constexpr std::size_t TOTAL_TILE = K_TILE * N_TILES;
    auto scores = kumi::generate<TOTAL_TILE>([&](auto index) {
        constexpr std::size_t t = decltype(index)::value;
        return wide_f(centroids.centroid_norm_sq[k0 + t]);
    });

    kmeans::for_each_feature<D>([&](auto feature_index) {
        constexpr std::size_t d = decltype(feature_index)::value;

        // points are feature-major:
        //     points[d][sample_i : sample_i + SIMD_WIDTH]
        //
        // feature_major stores the assignment coefficient -2*c:
        //     score = ||c||^2 + sum_d x[d] * (-2*c[d])
        auto x = eve::load[ignore](
            eve::as_aligned(points.template feature<d>() + sample_i)
        );

        const float* centroid_feature = centroids.template feature_centroids<d>() + k0;

        kumi::for_each_index([&](auto index, auto& score) {
            constexpr std::size_t t = decltype(index)::value;

            auto neg_two_c = wide_f(centroid_feature[t]);

            score = eve::fma(x, neg_two_c, score);
        }, scores);
    });

    kumi::for_each_index([&](auto index, auto score) {
        constexpr std::size_t t = decltype(index)::value;
        const std::size_t k = k0 + t;

        update_best_centroid(
            score,
            k,
            best_dist,
            best_k
        );
    }, scores);
}

template<std::size_t D, std::size_t K_TILE, std::size_t N_TILES, typename Ignore>
inline void process_full_centroid_tile_groups(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    std::size_t& k0,
    std::size_t K,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    constexpr std::size_t K_GROUP_TILE = N_TILES * K_TILE;

    for (; k0 + K_GROUP_TILE <= K; k0 += K_GROUP_TILE) {
        process_centroid_tiles<D, K_TILE, N_TILES>(
            points,
            centroids,
            sample_i,
            k0,
            ignore,
            best_dist,
            best_k
        );
    }

    if constexpr (N_TILES > 1) {
        process_full_centroid_tile_groups<D, K_TILE, N_TILES - 1>(
            points,
            centroids,
            sample_i,
            k0,
            K,
            ignore,
            best_dist,
            best_k
        );
    }
}

// Compile-time exact tail dispatch
template<std::size_t D, std::size_t TAIL, typename Ignore>
inline void process_centroid_tail_exact(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    std::size_t k0,
    std::size_t remaining,
    Ignore ignore,
    wide_f& best_dist,
    wide_i& best_k
) {
    static_assert(TAIL > 0);

    if (remaining == TAIL) {
        process_centroid_tiles<D, TAIL, 1>(
            points,
            centroids,
            sample_i,
            k0,
            ignore,
            best_dist,
            best_k
        );
    } else if constexpr (TAIL > 1) {
        process_centroid_tail_exact<D, TAIL - 1>(
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

template<std::size_t D, std::size_t K_TILE, bool TrackChanges, typename Ignore>
inline bool assign_one_sample_block_tiled(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::size_t sample_i,
    Ignore ignore,
    int* assignment_ptr
) {
    const std::size_t K = centroids.n_clusters;

    auto assignment_aligned_ptr = eve::as_aligned(
        assignment_ptr, typename wide_i::cardinal_type{}
    );

    auto best_dist = eve::valmax(eve::as<wide_f>());
    auto best_k = eve::zero(eve::as<wide_i>());

    std::size_t k0 = 0;

    constexpr std::size_t N_TILES_AT_ONCE = centroid_tiles_at_once<K_TILE>();

    process_full_centroid_tile_groups<D, K_TILE, N_TILES_AT_ONCE>(
        points,
        centroids,
        sample_i,
        k0,
        K,
        ignore,
        best_dist,
        best_k
    );

    if constexpr (K_TILE > 1) {
        const std::size_t remaining = K - k0;

        if (remaining != 0) {
            process_centroid_tail_exact<D, K_TILE - 1>(
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

    bool changed = false;

    if constexpr (TrackChanges) {
        const wide_i previous_label =
            eve::load[ignore](assignment_aligned_ptr, eve::as<wide_i>{});

        changed = eve::any[ignore](best_k != previous_label);
    }

    eve::store[ignore](best_k, assignment_aligned_ptr);

    return changed;
}

template<std::size_t D, std::size_t K_TILE, bool TrackChanges>
bool assign_points_to_centroids_tiled_impl(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::span<int> assignments
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
// The main grouped loop keeps centroid_tiles_at_once<K_TILE>() * K_TILE dot accumulators live; D is streamed.
// The main loop is unmasked; only the final tail block uses ignore_last.
// assignments.data() must be SIMD-aligned
template<std::size_t D, std::size_t K_TILE>
void assign_points_to_centroids_tiled(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::span<int> assignments
) {
    (void)assign_points_to_centroids_tiled_impl<D, K_TILE, false>(
        points,
        centroids,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE>
bool assign_points_to_centroids_tiled_and_check_changed(
    points_soa_view<D> points,
    const centroids_storage<D>& centroids,
    std::span<int> assignments
) {
    return assign_points_to_centroids_tiled_impl<D, K_TILE, true>(
        points,
        centroids,
        assignments
    );
}

template<std::size_t D, std::size_t K_TILE>
struct fused_assignment_backend {
    void on_centroids_changed(centroids_storage<D>& centroids) const {
        centroids.sync_fused_assignment_layout_from_row_major();
    }

    void assign(
        points_soa_view<D> points,
        const centroids_storage<D>& centroids,
        std::span<int> assignments
    ) const {
        assign_points_to_centroids_tiled<D, K_TILE>(
            points,
            centroids,
            assignments
        );
    }

    bool assign_and_check_changed(
        points_soa_view<D> points,
        const centroids_storage<D>& centroids,
        std::span<int> assignments
    ) const {
        return assign_points_to_centroids_tiled_and_check_changed<D, K_TILE>(
            points,
            centroids,
            assignments
        );
    }
};
