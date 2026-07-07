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

#include "types.hpp"

struct mst_edge {
    std::int32_t current_node;
    std::int32_t next_node;
    double distance;
};

struct mst_step_result {
    std::size_t next;
    double distance;
};

inline hdbscan_wide_i mst_lane_offsets() {
    return hdbscan_wide_i([](auto lane, auto) {
        return static_cast<int>(lane);
    });
}

inline mst_step_result reduce_min_index_wide(hdbscan_wide distances, hdbscan_wide_i indices) {
    double best_distance = std::numeric_limits<double>::infinity();
    std::size_t best_index = 0;

    for (std::size_t lane = 0; lane < hdbscan_wide::size(); ++lane) {
        const double lane_distance = distances.get(lane);
        const std::size_t lane_index = static_cast<std::size_t>(indices.get(lane));

        if (lane_distance < best_distance
            || (lane_distance == best_distance && lane_index < best_index)) {
            best_distance = lane_distance;
            best_index = lane_index;
        }
    }

    return mst_step_result{best_index, best_distance};
}

__attribute__((always_inline)) inline mst_step_result update_mst_candidates_and_select_next(
    const double* current_distance_row,
    std::span<const double> core_distances,
    std::span<double> best_distance,
    std::span<std::uint8_t> selected_count_per_packet,
    std::size_t& full_packet_count,
    std::size_t current,
    std::size_t N,
    bool enable_packet_skip_time_gate
) {
    constexpr std::size_t card = hdbscan_wide::size();
    constexpr double selected_sentinel = -1.0;
    const double infinity = std::numeric_limits<double>::infinity();

    // Selected vertices are encoded directly in best_distance. Squared mutual
    // reachability weights are non-negative, so a negative value is a compact
    // active-state sentinel and avoids a separate attached[] stream.
    best_distance[current] = selected_sentinel;

    const std::size_t packet_count = selected_count_per_packet.size();
    const std::size_t current_packet = current / card;
    ++selected_count_per_packet[current_packet];

    // Only enable packet skipping once there are enough actually-full
    // packets to justify paying the packet-count branch in the row scan.
    const std::size_t tail_lanes = N % card;
    const std::size_t current_packet_capacity =
        (current_packet + 1 == packet_count && tail_lanes != 0)
            ? tail_lanes
            : card;

    if (selected_count_per_packet[current_packet] == current_packet_capacity) {
        ++full_packet_count;
    }

    const bool enable_packet_skip =
        enable_packet_skip_time_gate
        && (full_packet_count * 32 > packet_count);

    const hdbscan_wide_i lane_offsets = mst_lane_offsets();
    const hdbscan_wide current_core(core_distances[current]);
    const double* const core_ptr = core_distances.data();
    double* const best_ptr = best_distance.data();
    hdbscan_wide vector_min_distances(infinity);
    hdbscan_wide_i vector_min_indices(0);

    auto process_block = [&](std::size_t j, auto ignore) {
        const auto previous_best = eve::load[ignore](best_ptr + j);
        const auto core_j = eve::load[ignore](core_ptr + j);
        const auto core_bound = eve::max(current_core, core_j);

        const auto active = ignore.mask(eve::as<hdbscan_wide>()) && (previous_best >= hdbscan_wide_zero);
        const auto can_improve_without_distance = active && (core_bound < previous_best);

        auto updated_best = previous_best;
        if (eve::any(can_improve_without_distance)) {
            const auto raw_distance = eve::load[ignore](current_distance_row + j);
            const auto candidate = eve::max(core_bound, raw_distance);
            updated_best = eve::min(candidate, previous_best);
            eve::store[ignore](updated_best, best_ptr + j);
        }

        const auto selectable_distance = eve::if_else(active, updated_best, hdbscan_wide(infinity));

        const auto better = selectable_distance < vector_min_distances;
        if (eve::any(better)) {
            const hdbscan_wide_i candidate_indices = hdbscan_wide_i(static_cast<int>(j)) + lane_offsets;
            vector_min_distances = eve::if_else(better, selectable_distance, vector_min_distances);
            vector_min_indices = eve::if_else(better, candidate_indices, vector_min_indices);
        }
    };

    auto process_packet_if_live = [&](std::size_t packet, std::size_t offset) {
        if (!enable_packet_skip || selected_count_per_packet[packet] != card) {
            process_block(offset, eve::ignore_none);
        }
    };

    std::size_t packet_index = 0;
    std::size_t j = 0;

    // Unroll-4 over full SIMD packets
    for (; j + 4 * card <= N; j += 4 * card, packet_index += 4) {
        process_packet_if_live(packet_index + 0, j + 0 * card);
        process_packet_if_live(packet_index + 1, j + 1 * card);
        process_packet_if_live(packet_index + 2, j + 2 * card);
        process_packet_if_live(packet_index + 3, j + 3 * card);
    }

    // Remaining unroll-2 chunk, if any
    for (; j + 2 * card <= N; j += 2 * card, packet_index += 2) {
        process_packet_if_live(packet_index + 0, j + 0 * card);
        process_packet_if_live(packet_index + 1, j + 1 * card);
    }

    // Remaining full packet, if any
    for (; j + card <= N; j += card, ++packet_index) {
        process_packet_if_live(packet_index, j);
    }

    // Tail packet
    if (j < N) {
        const std::size_t valid_lanes = N - j;

        if (!enable_packet_skip || selected_count_per_packet[packet_index] != valid_lanes) {
            const std::size_t ignored_lanes = card - valid_lanes;
            process_block(j, eve::ignore_last(ignored_lanes));
        }
    }

    return reduce_min_index_wide(vector_min_distances, vector_min_indices);
}

inline void minimum_spanning_tree_edges(
    std::span<const double> distance_matrix,
    std::span<const double> core_distances,
    std::size_t N,
    std::vector<mst_edge>& edges
) {
    if (distance_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN MST stage expected a dense N x N squared-distance matrix");
    }
    if (core_distances.size() != N) {
        throw std::runtime_error("HDBSCAN MST stage expected N squared core distances");
    }

    edges.clear();
    if (N == 0) {
        return;
    }
    edges.reserve(N - 1);

    std::vector<double, eve::aligned_allocator<double>> best_distance(
        N,
        std::numeric_limits<double>::infinity()
    );
    constexpr std::size_t card = hdbscan_wide::size();
    const std::size_t packet_count = (N + card - 1) / card;
    std::vector<std::uint8_t> selected_count_per_packet(packet_count, 0);
    std::size_t full_packet_count = 0;

    // Match scikit-learn's dense private mst_from_mutual_reachability traversal
    // convention while computing mutual reachability lazily from the raw squared
    // distance matrix and squared core distances:
    //
    //     w(u, v) = max(core_sq[u], core_sq[v], dist_sq[u, v])
    //
    // The emitted endpoints are the previous selected vertex and the newly
    // selected vertex, while the emitted distance is the current best crossing
    // edge weight. The endpoint pair is therefore an implementation-stage edge
    // list, not always the actual parent edge in the complete graph.
    std::size_t current = 0;
    for (std::size_t edge_index = 0; edge_index + 1 < N; ++edge_index) {
        const bool enable_packet_skip_time_gate = (edge_index + 1) > N / 4;

        const mst_step_result next_step = update_mst_candidates_and_select_next(
            distance_matrix.data() + current * N,
            core_distances,
            std::span<double>(best_distance.data(), best_distance.size()),
            std::span<std::uint8_t>(selected_count_per_packet.data(), selected_count_per_packet.size()),
            full_packet_count,
            current,
            N,
            enable_packet_skip_time_gate
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
