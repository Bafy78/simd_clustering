#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <eve/module/core.hpp>
#include <eve/memory/aligned_allocator.hpp>

#include "../../simd.hpp"

struct mst_edge {
    std::int32_t current_node;
    std::int32_t next_node;
    float distance;
};

struct mst_step_result {
    std::size_t next;
    float distance;
};

inline wide_i mst_lane_offsets() {
    return wide_i([](auto lane, auto) {
        return static_cast<int>(lane);
    });
}

inline mst_step_result reduce_min_index_wide(wide_f distances, wide_i indices) {
    float best_distance = std::numeric_limits<float>::infinity();
    std::size_t best_index = 0;

    for (std::size_t lane = 0; lane < wide_f::size(); ++lane) {
        const float lane_distance = distances.get(lane);
        const std::size_t lane_index = static_cast<std::size_t>(indices.get(lane));

        if (lane_distance < best_distance
            || (lane_distance == best_distance && lane_index < best_index)) {
            best_distance = lane_distance;
            best_index = lane_index;
        }
    }

    return mst_step_result{best_index, best_distance};
}

inline mst_step_result update_mst_candidates_and_select_next(
    const float* current_row,
    std::span<float> best_distance,
    std::size_t current,
    std::size_t N
) {
    constexpr std::size_t card = wide_f::size();
    constexpr float selected_sentinel = -1.0f;
    const float infinity = std::numeric_limits<float>::infinity();

    // Selected vertices are encoded directly in best_distance. Mutual reachability
    // distances for Euclidean HDBSCAN are non-negative, so a negative value is a
    // compact active-state sentinel and avoids a separate attached[] stream.
    best_distance[current] = selected_sentinel;

    const wide_i lane_offsets = mst_lane_offsets();
    wide_f vector_min_distances(infinity);
    wide_i vector_min_indices(0);

    auto process_block = [&](std::size_t j, auto ignore) {
        const auto candidates = eve::load[ignore](current_row + j);
        const auto previous_best = eve::load[ignore](best_distance.data() + j);

        const auto active = ignore.mask(eve::as<wide_f>()) && (previous_best >= wide_zero_f);
        const auto updated_best = eve::if_else(
            active && (candidates < previous_best),
            candidates,
            previous_best
        );
        eve::store[ignore](updated_best, best_distance.data() + j);

        const auto selectable_distance = eve::if_else(active, updated_best, wide_f(infinity));
        const wide_i candidate_indices = wide_i(static_cast<int>(j)) + lane_offsets;
        const auto better = selectable_distance < vector_min_distances;

        vector_min_distances = eve::if_else(better, selectable_distance, vector_min_distances);
        vector_min_indices = eve::if_else(better, candidate_indices, vector_min_indices);
    };

    std::size_t j = 0;
    for (; j + card <= N; j += card) {
        process_block(j, eve::ignore_none);
    }

    if (j < N) {
        const std::size_t valid_lanes = N - j;
        const std::size_t ignored_lanes = card - valid_lanes;
        process_block(j, eve::ignore_last(ignored_lanes));
    }

    return reduce_min_index_wide(vector_min_distances, vector_min_indices);
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
        const mst_step_result next_step = update_mst_candidates_and_select_next(
            mutual_reachability.data() + current * N,
            std::span<float>(best_distance.data(), best_distance.size()),
            current,
            N
        );

        if (!std::isfinite(next_step.distance)) {
            throw std::runtime_error("HDBSCAN MST graph is disconnected or contains no selectable edge");
        }

        edges.push_back(mst_edge{
            static_cast<std::int32_t>(current),
            static_cast<std::int32_t>(next_step.next),
            next_step.distance,
        });
        current = next_step.next;
    }
}
