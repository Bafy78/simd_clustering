#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>

#include "mst.hpp"

struct single_linkage_row {
    std::int32_t left_node;
    std::int32_t right_node;
    double distance;
    std::int32_t cluster_size;
};

namespace hdbscan_linkage_detail {

inline std::uint64_t sortable_distance_key(double value) noexcept {
    // Mutual-reachability distances are expected to be finite and non-negative.
    // The zero canonicalization keeps +0.0 and -0.0 tied, matching ordinary
    // floating-point comparison for our distance sort.
    if (value == 0.0) {
        value = 0.0;
    }

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    constexpr std::uint64_t sign_bit = std::uint64_t{1} << 63;

    // Standard sortable IEEE-754 transform. For the non-negative distances used
    // here this is equivalent to ordering by the raw magnitude bits, with all
    // non-negative values placed after negative values if they ever appeared.
    return (bits & sign_bit) ? ~bits : (bits ^ sign_bit);
}

inline double distance_from_sortable_key(std::uint64_t key) noexcept {
    constexpr std::uint64_t sign_bit = std::uint64_t{1} << 63;
    const std::uint64_t bits = (key & sign_bit) ? (key ^ sign_bit) : ~key;
    return std::bit_cast<double>(bits);
}

inline std::uint64_t endpoint_sort_key(
    std::int32_t current_node,
    std::int32_t next_node
) noexcept {
    return (std::uint64_t{static_cast<std::uint32_t>(current_node)} << 32) |
           std::uint64_t{static_cast<std::uint32_t>(next_node)};
}

inline std::int32_t endpoint_key_current_node(std::uint64_t endpoints_key) noexcept {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(endpoints_key >> 32));
}

inline std::int32_t endpoint_key_next_node(std::uint64_t endpoints_key) noexcept {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(endpoints_key));
}

struct linkage_sort_key {
    std::uint64_t distance_key;
    std::uint64_t endpoints_key;
};

inline linkage_sort_key pack_edge_for_linkage_sort(const mst_edge& edge) noexcept {
    return linkage_sort_key{
        sortable_distance_key(edge.distance),
        endpoint_sort_key(edge.current_node, edge.next_node),
    };
}

static_assert(sizeof(linkage_sort_key) == 16);

} // namespace hdbscan_linkage_detail

struct union_find_for_linkage {
    std::vector<std::int32_t> storage;
    std::span<std::int32_t> parent_or_label;
    std::span<std::int32_t> size;

    explicit union_find_for_linkage(std::size_t n)
        : storage(2 * n),
          parent_or_label(storage.data(), n),
          size(storage.data() + n, n) {
        using wide_i32 = eve::wide<std::int32_t>;

        constexpr std::size_t lanes = wide_i32::size();
        const wide_i32 lane_offsets = eve::iota(eve::as<wide_i32>{});
        const wide_i32 one{1};

        std::size_t i = 0;
        for (; i + lanes <= n; i += lanes) {
            const auto labels = lane_offsets + wide_i32{static_cast<std::int32_t>(i)};
            eve::store(-(labels + one), parent_or_label.data() + i);
            eve::store(one, size.data() + i);
        }

        for (; i < n; ++i) {
            parent_or_label[i] = encode_label(static_cast<std::int32_t>(i));
            size[i] = 1;
        }
    }

    static constexpr std::int32_t encode_label(std::int32_t label) noexcept {
        return -label - 1;
    }

    static constexpr std::int32_t decode_label(std::int32_t encoded_label) noexcept {
        return -encoded_label - 1;
    }

    std::int32_t find(std::int32_t node) {
        std::int32_t root = node;
        while (parent_or_label[static_cast<std::size_t>(root)] >= 0) {
            root = parent_or_label[static_cast<std::size_t>(root)];
        }

        while (parent_or_label[static_cast<std::size_t>(node)] >= 0) {
            const std::int32_t next = parent_or_label[static_cast<std::size_t>(node)];
            parent_or_label[static_cast<std::size_t>(node)] = root;
            node = next;
        }

        return root;
    }

    std::int32_t component_label(std::int32_t root) const noexcept {
        return decode_label(parent_or_label[static_cast<std::size_t>(root)]);
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
        parent_or_label[right_index] = left_root;
        size[left_index] = merged_size;
        parent_or_label[left_index] = encode_label(new_label);
        return merged_size;
    }
};

inline void single_linkage_tree_from_mst_edges_inplace(
    std::span<mst_edge> edges,
    std::size_t N,
    std::vector<single_linkage_row>& tree
) {
    using hdbscan_linkage_detail::distance_from_sortable_key;
    using hdbscan_linkage_detail::endpoint_key_current_node;
    using hdbscan_linkage_detail::endpoint_key_next_node;
    using hdbscan_linkage_detail::linkage_sort_key;
    using hdbscan_linkage_detail::pack_edge_for_linkage_sort;

    std::vector<linkage_sort_key> sorted_edges;
    sorted_edges.reserve(edges.size());
    for (const mst_edge& edge : edges) {
        sorted_edges.push_back(pack_edge_for_linkage_sort(edge));
    }

    std::sort(
        sorted_edges.begin(),
        sorted_edges.end(),
        [](const linkage_sort_key& a, const linkage_sort_key& b) {
            if (a.distance_key != b.distance_key) {
                return a.distance_key < b.distance_key;
            }
            return a.endpoints_key < b.endpoints_key;
        }
    );

    tree.clear();
    tree.resize(sorted_edges.size());

    union_find_for_linkage components(N);

    for (std::size_t row = 0; row < sorted_edges.size(); ++row) {
        const linkage_sort_key edge = sorted_edges[row];
        const std::int32_t current_node = endpoint_key_current_node(edge.endpoints_key);
        const std::int32_t next_node = endpoint_key_next_node(edge.endpoints_key);
        const std::int32_t left_root = components.find(current_node);
        const std::int32_t right_root = components.find(next_node);

        const std::int32_t left_label = components.component_label(left_root);
        const std::int32_t right_label = components.component_label(right_root);
        const std::int32_t new_label = static_cast<std::int32_t>(N + row);
        const std::int32_t merged_size = components.merge(left_root, right_root, new_label);

        tree[row] = single_linkage_row{
            left_label,
            right_label,
            distance_from_sortable_key(edge.distance_key),
            merged_size,
        };
    }
}


inline void single_linkage_tree_from_mst_edges(
    std::span<const mst_edge> input_edges,
    std::size_t N,
    std::vector<single_linkage_row>& tree
) {
    std::vector<mst_edge> edges(input_edges.begin(), input_edges.end());
    single_linkage_tree_from_mst_edges_inplace(
        std::span<mst_edge>(edges.data(), edges.size()),
        N,
        tree
    );
}
