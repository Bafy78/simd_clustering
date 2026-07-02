#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "mst.hpp"

struct single_linkage_row {
    std::int32_t left_node;
    std::int32_t right_node;
    double distance;
    std::int32_t cluster_size;
};

struct union_find_for_linkage {
    std::vector<std::int32_t> parent;
    std::vector<std::int32_t> size;
    std::vector<std::int32_t> label;

    explicit union_find_for_linkage(std::size_t n)
        : parent(n), size(n, 1), label(n) {
        for (std::size_t i = 0; i < n; ++i) {
            parent[i] = static_cast<std::int32_t>(i);
            label[i] = static_cast<std::int32_t>(i);
        }
    }

    std::int32_t find(std::int32_t node) {
        std::int32_t root = node;
        while (parent[static_cast<std::size_t>(root)] != root) {
            root = parent[static_cast<std::size_t>(root)];
        }

        while (parent[static_cast<std::size_t>(node)] != node) {
            const std::int32_t next = parent[static_cast<std::size_t>(node)];
            parent[static_cast<std::size_t>(node)] = root;
            node = next;
        }

        return root;
    }

    std::int32_t merge(
        std::int32_t left_root,
        std::int32_t right_root,
        std::int32_t new_label
    ) {
        const auto left_index = static_cast<std::size_t>(left_root);
        const auto right_index = static_cast<std::size_t>(right_root);
        const std::int32_t merged_size = size[left_index] + size[right_index];

        // Preserve sklearn make_single_linkage row semantics: the emitted left
        // and right labels are the component labels of the edge endpoints before
        // union. The new cluster label is assigned after emitting the row.
        parent[right_index] = left_root;
        size[left_index] = merged_size;
        label[left_index] = new_label;
        return merged_size;
    }
};

inline void single_linkage_tree_from_mst_edges(
    std::span<const mst_edge> input_edges,
    std::size_t N,
    std::vector<single_linkage_row>& tree
) {
    const std::size_t expected_edges = N == 0 ? 0 : N - 1;
    if (input_edges.size() != expected_edges) {
        throw std::runtime_error("HDBSCAN linkage stage expected N - 1 MST edges");
    }

    tree.clear();
    if (N == 0) {
        return;
    }
    tree.reserve(expected_edges);

    std::vector<mst_edge> edges(input_edges.begin(), input_edges.end());
    std::sort(
        edges.begin(),
        edges.end(),
        [](const mst_edge& a, const mst_edge& b) {
            if (a.distance != b.distance) {
                return a.distance < b.distance;
            }
            if (a.current_node != b.current_node) {
                return a.current_node < b.current_node;
            }
            return a.next_node < b.next_node;
        }
    );

    union_find_for_linkage components(N);

    for (std::size_t row = 0; row < edges.size(); ++row) {
        const mst_edge& edge = edges[row];
        if (edge.current_node < 0 || edge.next_node < 0
            || static_cast<std::size_t>(edge.current_node) >= N
            || static_cast<std::size_t>(edge.next_node) >= N) {
            throw std::runtime_error("HDBSCAN linkage stage MST edge endpoint is out of range");
        }

        const std::int32_t left_root = components.find(edge.current_node);
        const std::int32_t right_root = components.find(edge.next_node);
        if (left_root == right_root) {
            throw std::runtime_error("HDBSCAN linkage stage received a cyclic MST edge list");
        }

        const std::int32_t left_label = components.label[static_cast<std::size_t>(left_root)];
        const std::int32_t right_label = components.label[static_cast<std::size_t>(right_root)];
        const std::int32_t new_label = static_cast<std::int32_t>(N + row);
        const std::int32_t merged_size = components.merge(left_root, right_root, new_label);

        tree.push_back(single_linkage_row{
            left_label,
            right_label,
            edge.distance,
            merged_size,
        });
    }
}
