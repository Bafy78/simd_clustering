#pragma once

#include <array>
#include <cstddef>
#include <span>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>

#include "layout.hpp"
#include "../pp_core.hpp"

inline constexpr std::size_t dynamic_kmeans_pp_default_n_vectors = 2;
inline constexpr std::size_t dynamic_kmeans_pp_default_local_trial_tile = 5;

template<std::size_t D>
inline void write_dynamic_centroid_from_sample(
    samples_soa_view<D> samples,
    std::size_t sample_n,
    centroids_storage<D>& centroids,
    std::size_t centroid_k
) {
    for (std::size_t d = 0; d < D; ++d) {
        centroids.row(centroid_k, d) = samples.dimension(d)[sample_n];
    }
}

template<std::size_t D, std::size_t LOCAL_TRIAL_TILE>
inline void pack_dynamic_kmeans_pp_candidate_tile(
    samples_soa_view<D> samples,
    std::span<const std::size_t> candidate_indices,
    std::array<float, D * LOCAL_TRIAL_TILE>& candidate_pack
) {
    for (std::size_t d = 0; d < D; ++d) {
        const float* sample_dimension = samples.dimension(d);

        for (std::size_t t = 0; t < candidate_indices.size(); ++t) {
            candidate_pack[d * LOCAL_TRIAL_TILE + t] = sample_dimension[candidate_indices[t]];
        }
    }
}

template<std::size_t K_COUNT, std::size_t D, std::size_t LOCAL_TRIAL_TILE, std::size_t N_VECTORS>
inline void evaluate_dynamic_kmeans_pp_candidate_tile_full_vectors(
    samples_soa_view<D> samples,
    std::span<const float> min_sq_dist,
    const std::array<float, D * LOCAL_TRIAL_TILE>& candidate_pack,
    std::size_t n,
    std::array<wide_f, LOCAL_TRIAL_TILE>& cost_accumulators
) {
    constexpr std::size_t card = simd_cardinal();

    std::array<std::array<wide_f, K_COUNT>, N_VECTORS> dist_sq;

    for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
        for (std::size_t t = 0; t < K_COUNT; ++t) {
            dist_sq[sample_vector][t] = wide_zero_f;
        }
    }

    for (std::size_t d = 0; d < D; ++d) {
        const float* sample_dimension = samples.dimension(d);

        std::array<wide_f, N_VECTORS> x;

        for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
            x[sample_vector] = eve::load(
                eve::as_aligned(sample_dimension + n + sample_vector * card)
            );
        }

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f c(candidate_pack[d * LOCAL_TRIAL_TILE + t]);

            for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
                const auto diff = x[sample_vector] - c;

                dist_sq[sample_vector][t] = eve::fma(
                    diff,
                    diff,
                    dist_sq[sample_vector][t]
                );
            }
        }
    }

    std::array<wide_f, N_VECTORS> current_min_dist;

    for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
        current_min_dist[sample_vector] = eve::load(
            eve::as_aligned(min_sq_dist.data() + n + sample_vector * card)
        );
    }

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        for (std::size_t sample_vector = 0; sample_vector < N_VECTORS; ++sample_vector) {
            cost_accumulators[t] += eve::min(
                current_min_dist[sample_vector],
                dist_sq[sample_vector][t]
            );
        }
    }
}

template<std::size_t K_COUNT, std::size_t D, std::size_t LOCAL_TRIAL_TILE>
inline void evaluate_dynamic_kmeans_pp_candidate_tile_one_vector_full(
    samples_soa_view<D> samples,
    std::span<const float> min_sq_dist,
    const std::array<float, D * LOCAL_TRIAL_TILE>& candidate_pack,
    std::size_t n,
    std::array<wide_f, LOCAL_TRIAL_TILE>& cost_accumulators
) {
    std::array<wide_f, K_COUNT> dist_sq;

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        dist_sq[t] = wide_zero_f;
    }

    for (std::size_t d = 0; d < D; ++d) {
        const auto x = eve::load(
            eve::as_aligned(samples.dimension(d) + n)
        );

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f c(candidate_pack[d * LOCAL_TRIAL_TILE + t]);
            const auto diff = x - c;

            dist_sq[t] = eve::fma(diff, diff, dist_sq[t]);
        }
    }

    const auto current_min_dist = eve::load(
        eve::as_aligned(min_sq_dist.data() + n)
    );

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        cost_accumulators[t] += eve::min(current_min_dist, dist_sq[t]);
    }
}

template<std::size_t K_COUNT, std::size_t D, std::size_t LOCAL_TRIAL_TILE, typename Ignore>
inline void evaluate_dynamic_kmeans_pp_candidate_tile_one_vector_masked(
    samples_soa_view<D> samples,
    std::span<const float> min_sq_dist,
    const std::array<float, D * LOCAL_TRIAL_TILE>& candidate_pack,
    std::size_t n,
    Ignore ignore,
    std::array<wide_f, LOCAL_TRIAL_TILE>& cost_accumulators
) {
    std::array<wide_f, K_COUNT> dist_sq;

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        dist_sq[t] = wide_zero_f;
    }

    for (std::size_t d = 0; d < D; ++d) {
        const auto x = eve::load[ignore](
            eve::as_aligned(samples.dimension(d) + n)
        );

        for (std::size_t t = 0; t < K_COUNT; ++t) {
            const wide_f c(candidate_pack[d * LOCAL_TRIAL_TILE + t]);
            const auto diff = x - c;

            dist_sq[t] = eve::fma(diff, diff, dist_sq[t]);
        }
    }

    const auto current_min_dist = eve::load[ignore](
        eve::as_aligned(min_sq_dist.data() + n)
    );

    const auto mask = ignore.mask(eve::as<wide_f>());

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        const auto trial_min = eve::min(current_min_dist, dist_sq[t]);

        cost_accumulators[t] += eve::if_else(
            mask,
            trial_min,
            wide_zero_f
        );
    }
}

template<std::size_t K_COUNT, std::size_t D, std::size_t LOCAL_TRIAL_TILE, std::size_t N_VECTORS>
inline void evaluate_dynamic_kmeans_pp_candidate_tile_costs(
    samples_soa_view<D> samples,
    std::span<const float> min_sq_dist,
    const std::array<float, D * LOCAL_TRIAL_TILE>& candidate_pack,
    std::span<float> trial_costs
) {
    static_assert(K_COUNT > 0);
    static_assert(K_COUNT <= LOCAL_TRIAL_TILE);
    static_assert(N_VECTORS > 0);

    constexpr std::size_t card = simd_cardinal();
    constexpr std::size_t sample_group_size = N_VECTORS * card;

    std::array<wide_f, LOCAL_TRIAL_TILE> cost_accumulators;

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        cost_accumulators[t] = wide_zero_f;
    }

    const std::size_t N = samples.N;
    std::size_t n = 0;

    for (; n + sample_group_size <= N; n += sample_group_size) {
        evaluate_dynamic_kmeans_pp_candidate_tile_full_vectors<K_COUNT, D, LOCAL_TRIAL_TILE, N_VECTORS>(
            samples,
            min_sq_dist,
            candidate_pack,
            n,
            cost_accumulators
        );
    }

    for (; n + card <= N; n += card) {
        evaluate_dynamic_kmeans_pp_candidate_tile_one_vector_full<K_COUNT, D, LOCAL_TRIAL_TILE>(
            samples,
            min_sq_dist,
            candidate_pack,
            n,
            cost_accumulators
        );
    }

    if (n < N) {
        const std::size_t valid = N - n;
        const std::size_t ignored_lanes = card - valid;

        evaluate_dynamic_kmeans_pp_candidate_tile_one_vector_masked<K_COUNT, D, LOCAL_TRIAL_TILE>(
            samples,
            min_sq_dist,
            candidate_pack,
            n,
            eve::ignore_last(ignored_lanes),
            cost_accumulators
        );
    }

    for (std::size_t t = 0; t < K_COUNT; ++t) {
        trial_costs[t] = eve::reduce(cost_accumulators[t]);
    }
}

template<std::size_t TAIL, std::size_t D, std::size_t LOCAL_TRIAL_TILE, std::size_t N_VECTORS>
inline void evaluate_dynamic_kmeans_pp_candidate_tile_tail_exact(
    samples_soa_view<D> samples,
    std::span<const float> min_sq_dist,
    const std::array<float, D * LOCAL_TRIAL_TILE>& candidate_pack,
    std::span<float> trial_costs,
    std::size_t remaining
) {
    if (remaining == TAIL) {
        evaluate_dynamic_kmeans_pp_candidate_tile_costs<TAIL, D, LOCAL_TRIAL_TILE, N_VECTORS>(
            samples,
            min_sq_dist,
            candidate_pack,
            trial_costs
        );
    } else if constexpr (TAIL > 1) {
        evaluate_dynamic_kmeans_pp_candidate_tile_tail_exact<TAIL - 1, D, LOCAL_TRIAL_TILE, N_VECTORS>(
            samples,
            min_sq_dist,
            candidate_pack,
            trial_costs,
            remaining
        );
    }
}

template<std::size_t D>
inline wide_f compute_dynamic_kmeans_pp_simd_dist_sq_full(
    samples_soa_view<D> samples,
    std::size_t n,
    const float* centroid
) {
    auto dist_sq = wide_zero_f;

    for (std::size_t d = 0; d < D; ++d) {
        const auto x = eve::load(
            eve::as_aligned(samples.dimension(d) + n)
        );
        const wide_f c(centroid[d]);
        const auto diff = x - c;

        dist_sq = eve::fma(diff, diff, dist_sq);
    }

    return dist_sq;
}

template<std::size_t D, typename Ignore>
inline wide_f compute_dynamic_kmeans_pp_simd_dist_sq_masked(
    samples_soa_view<D> samples,
    std::size_t n,
    const float* centroid,
    Ignore ignore
) {
    auto dist_sq = wide_zero_f;

    for (std::size_t d = 0; d < D; ++d) {
        const auto x = eve::load[ignore](
            eve::as_aligned(samples.dimension(d) + n)
        );
        const wide_f c(centroid[d]);
        const auto diff = x - c;

        dist_sq = eve::fma(diff, diff, dist_sq);
    }

    return dist_sq;
}

template<std::size_t D>
inline void update_dynamic_kmeans_pp_min_distances(
    samples_soa_view<D> samples,
    std::span<float> min_sq_dist,
    const float* new_centroid
) {
    constexpr std::size_t card = simd_cardinal();

    const std::size_t N = samples.N;
    std::size_t n = 0;

    for (; n + card <= N; n += card) {
        auto current_min_dist = eve::load(
            eve::as_aligned(min_sq_dist.data() + n)
        );

        auto dist_sq = compute_dynamic_kmeans_pp_simd_dist_sq_full(
            samples,
            n,
            new_centroid
        );

        auto new_min = eve::min(current_min_dist, dist_sq);
        eve::store(new_min, eve::as_aligned(min_sq_dist.data() + n));
    }

    if (n < N) {
        const std::size_t valid = N - n;
        const std::size_t ignored_lanes = card - valid;
        auto ignore = eve::ignore_last(ignored_lanes);

        auto current_min_dist = eve::load[ignore](
            eve::as_aligned(min_sq_dist.data() + n)
        );

        auto dist_sq = compute_dynamic_kmeans_pp_simd_dist_sq_masked(
            samples,
            n,
            new_centroid,
            ignore
        );

        auto new_min = eve::min(current_min_dist, dist_sq);
        eve::store[ignore](new_min, eve::as_aligned(min_sq_dist.data() + n));
    }
}

template<
    std::size_t D,
    std::size_t N_VECTORS = dynamic_kmeans_pp_default_n_vectors,
    std::size_t LOCAL_TRIAL_TILE = dynamic_kmeans_pp_default_local_trial_tile
>
struct dynamic_kmeans_pp_backend {
    using centroids_type = centroids_storage<D>;

    static constexpr std::size_t local_trial_tile_size = LOCAL_TRIAL_TILE;

    samples_soa_view<D> samples;

    std::size_t N() const {
        return samples.N;
    }

    centroids_type make_centroids(std::size_t K) const {
        centroids_type centroids;
        centroids.resize(K);
        return centroids;
    }

    void set_centroid_from_sample(
        centroids_type& centroids,
        std::size_t k,
        std::size_t n
    ) const {
        write_dynamic_centroid_from_sample<D>(
            samples,
            n,
            centroids,
            k
        );
    }

    void update_min_distances(
        std::span<float> min_sq_dist,
        const centroids_type& centroids,
        std::size_t centroid_k
    ) const {
        update_dynamic_kmeans_pp_min_distances<D>(
            samples,
            min_sq_dist,
            centroids.row_major.data() + centroid_k * D
        );
    }

    void evaluate_candidate_costs(
        std::span<const float> min_sq_dist,
        std::span<const std::size_t> candidate_indices,
        std::span<float> trial_costs
    ) const {
        static_assert(D > 0);
        static_assert(N_VECTORS > 0);
        static_assert(LOCAL_TRIAL_TILE > 0);

        std::array<float, D * LOCAL_TRIAL_TILE> candidate_pack;

        pack_dynamic_kmeans_pp_candidate_tile<D, LOCAL_TRIAL_TILE>(
            samples,
            candidate_indices,
            candidate_pack
        );

        if (candidate_indices.size() == LOCAL_TRIAL_TILE) {
            evaluate_dynamic_kmeans_pp_candidate_tile_costs<LOCAL_TRIAL_TILE, D, LOCAL_TRIAL_TILE, N_VECTORS>(
                samples,
                min_sq_dist,
                candidate_pack,
                trial_costs
            );
        } else if constexpr (LOCAL_TRIAL_TILE == 1) {
            evaluate_dynamic_kmeans_pp_candidate_tile_costs<1, D, LOCAL_TRIAL_TILE, N_VECTORS>(
                samples,
                min_sq_dist,
                candidate_pack,
                trial_costs
            );
        } else {
            evaluate_dynamic_kmeans_pp_candidate_tile_tail_exact<LOCAL_TRIAL_TILE - 1, D, LOCAL_TRIAL_TILE, N_VECTORS>(
                samples,
                min_sq_dist,
                candidate_pack,
                trial_costs,
                candidate_indices.size()
            );
        }
    }
};

template<
    std::size_t D,
    std::size_t N_VECTORS = dynamic_kmeans_pp_default_n_vectors,
    std::size_t LOCAL_TRIAL_TILE = dynamic_kmeans_pp_default_local_trial_tile
>
centroids_storage<D> greedy_kmeans_pp_init_dynamic(
    samples_soa_view<D> samples,
    int K,
    int num_local_trials = 0,
    unsigned int seed = std::random_device{}()
) {
    dynamic_kmeans_pp_backend<D, N_VECTORS, LOCAL_TRIAL_TILE> backend{ samples };

    return kmeans_pp::greedy_kmeans_pp_core(
        backend,
        K,
        num_local_trials,
        seed
    );
}
