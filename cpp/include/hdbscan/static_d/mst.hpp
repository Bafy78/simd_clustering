#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../../simd.hpp"

struct mst_edge {
    std::int32_t current_node;
    std::int32_t next_node;
    float distance;
};

inline float reduce_min_wide_f(wide_f values) {
    float reduced = std::numeric_limits<float>::infinity();
    for (std::size_t lane = 0; lane < wide_f::size(); ++lane) {
        reduced = std::min(reduced, values.get(lane));
    }
    return reduced;
}

inline void update_mst_candidates_from_current(
    const float* current_row,
    std::span<const float> attached,
    std::span<float> best_distance,
    std::size_t current,
    std::size_t N
) {
    auto row_range = eve::algo::as_range(current_row, current_row + N);
    auto attached_range = eve::algo::as_range(attached.data(), attached.data() + N);
    auto best_range = eve::algo::as_range(best_distance.data(), best_distance.data() + N);
    auto zipped = eve::views::zip(row_range, attached_range, best_range);

    eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
        zipped,
        [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
            auto [candidate_it, attached_it, best_it] = it;
            const auto candidate = eve::load[ignore](candidate_it);
            const auto attached_lanes = eve::load[ignore](attached_it);
            const auto best = eve::load[ignore](best_it);

            const auto active = ignore.mask(eve::as<wide_f>()) && (attached_lanes == wide_zero_f);
            const auto updated = eve::if_else(active && (candidate < best), candidate, best);
            eve::store[ignore](updated, best_it);
        }
    );

    // The current vertex is now in the tree and must never be selected again.
    best_distance[current] = std::numeric_limits<float>::infinity();
}

inline float vector_min_unattached(
    std::span<const float> best_distance,
    std::span<const float> attached,
    std::size_t N
) {
    auto best_range = eve::algo::as_range(best_distance.data(), best_distance.data() + N);
    auto attached_range = eve::algo::as_range(attached.data(), attached.data() + N);
    auto zipped = eve::views::zip(best_range, attached_range);
    wide_f minima(std::numeric_limits<float>::infinity());

    eve::algo::for_each[eve::algo::force_cardinal<cardinal{}()>](
        zipped,
        [&](eve::algo::iterator auto it, eve::relative_conditional_expr auto ignore) {
            auto [best_it, attached_it] = it;
            const auto best = eve::load[ignore](best_it);
            const auto attached_lanes = eve::load[ignore](attached_it);
            const auto active = ignore.mask(eve::as<wide_f>()) && (attached_lanes == wide_zero_f);
            minima = eve::min(minima, eve::if_else(active, best, wide_f(std::numeric_limits<float>::infinity())));
        }
    );

    return reduce_min_wide_f(minima);
}

inline std::size_t first_unattached_index_with_distance(
    std::span<const float> best_distance,
    std::span<const float> attached,
    std::size_t N,
    float target
) {
    for (std::size_t i = 0; i < N; ++i) {
        if (attached[i] == 0.0f && best_distance[i] == target) {
            return i;
        }
    }
    throw std::runtime_error("Could not select next HDBSCAN MST vertex");
}

inline void minimum_spanning_tree_edges(
    std::span<const float> mutual_reachability,
    std::size_t N,
    std::vector<mst_edge>& edges
) {
    if (mutual_reachability.size() != N * N) {
        throw std::runtime_error("HDBSCAN MST stage expected a dense N x N mutual reachability matrix");
    }

    edges.clear();
    if (N == 0) {
        return;
    }
    edges.reserve(N - 1);

    std::vector<float, eve::aligned_allocator<float>> attached(N, 0.0f);
    std::vector<float, eve::aligned_allocator<float>> best_distance(
        N,
        std::numeric_limits<float>::infinity()
    );

    // Match scikit-learn's dense private mst_from_mutual_reachability behavior:
    // the emitted endpoints are the previous selected vertex and the newly selected
    // vertex, while the emitted distance is the current best crossing-edge weight.
    // The endpoint pair is therefore an implementation-stage edge list, not always
    // the actual parent edge in the complete mutual-reachability graph.
    std::size_t current = 0;
    for (std::size_t edge_index = 0; edge_index + 1 < N; ++edge_index) {
        attached[current] = 1.0f;

        update_mst_candidates_from_current(
            mutual_reachability.data() + current * N,
            std::span<const float>(attached.data(), attached.size()),
            std::span<float>(best_distance.data(), best_distance.size()),
            current,
            N
        );

        const float next_distance = vector_min_unattached(
            std::span<const float>(best_distance.data(), best_distance.size()),
            std::span<const float>(attached.data(), attached.size()),
            N
        );
        if (!std::isfinite(next_distance)) {
            throw std::runtime_error("HDBSCAN MST graph is disconnected or contains no selectable edge");
        }

        const std::size_t next = first_unattached_index_with_distance(
            std::span<const float>(best_distance.data(), best_distance.size()),
            std::span<const float>(attached.data(), attached.size()),
            N,
            next_distance
        );

        edges.push_back(mst_edge{
            static_cast<std::int32_t>(current),
            static_cast<std::int32_t>(next),
            next_distance,
        });
        current = next;
    }
}

