#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "linkage.hpp"

struct hdbscan_selection_result {
    std::vector<std::int32_t> labels;
    std::vector<float> probabilities;
};

struct condensed_tree_row {
    std::int32_t parent;
    std::int32_t child;
    double lambda_value;
    std::int32_t cluster_size;
};

struct tree_union_find_for_select {
    std::vector<std::int32_t> parent;
    std::vector<std::int32_t> rank;

    explicit tree_union_find_for_select(std::size_t n)
        : parent(n), rank(n, 0) {
        for (std::size_t i = 0; i < n; ++i) {
            parent[i] = static_cast<std::int32_t>(i);
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

    void unite(std::int32_t left, std::int32_t right) {
        std::int32_t left_root = find(left);
        std::int32_t right_root = find(right);
        if (left_root == right_root) {
            return;
        }

        auto left_index = static_cast<std::size_t>(left_root);
        auto right_index = static_cast<std::size_t>(right_root);
        if (rank[left_index] < rank[right_index]) {
            parent[left_index] = right_root;
        } else if (rank[left_index] > rank[right_index]) {
            parent[right_index] = left_root;
        } else {
            parent[right_index] = left_root;
            ++rank[left_index];
        }
    }
};

inline std::vector<std::int32_t> bfs_from_hierarchy(
    std::span<const single_linkage_row> hierarchy,
    std::int32_t bfs_root
) {
    const std::int32_t n_samples = static_cast<std::int32_t>(hierarchy.size() + 1);
    std::vector<std::int32_t> result;
    std::vector<std::int32_t> queue{bfs_root};

    while (!queue.empty()) {
        result.insert(result.end(), queue.begin(), queue.end());

        std::vector<std::int32_t> next_queue;
        for (const std::int32_t node : queue) {
            if (node < n_samples) {
                continue;
            }
            const std::int32_t row_index = node - n_samples;
            if (row_index < 0 || static_cast<std::size_t>(row_index) >= hierarchy.size()) {
                throw std::runtime_error("HDBSCAN select stage hierarchy node is out of range");
            }
            const single_linkage_row& row = hierarchy[static_cast<std::size_t>(row_index)];
            next_queue.push_back(row.left_node);
            next_queue.push_back(row.right_node);
        }
        queue = std::move(next_queue);
    }

    return result;
}

inline std::vector<condensed_tree_row> condense_tree(
    std::span<const single_linkage_row> hierarchy,
    std::size_t min_cluster_size
) {
    if (hierarchy.empty()) {
        return {};
    }

    const std::int32_t n_samples = static_cast<std::int32_t>(hierarchy.size() + 1);
    const std::int32_t root = static_cast<std::int32_t>(2 * hierarchy.size());
    std::int32_t next_label = n_samples + 1;

    const std::vector<std::int32_t> node_list = bfs_from_hierarchy(hierarchy, root);
    std::vector<std::int32_t> relabel(static_cast<std::size_t>(root + 1), 0);
    std::vector<unsigned char> ignore(static_cast<std::size_t>(root + 1), 0);
    relabel[static_cast<std::size_t>(root)] = n_samples;

    std::vector<condensed_tree_row> result;
    result.reserve(hierarchy.size() + n_samples);

    for (const std::int32_t node : node_list) {
        if (node < 0 || node > root) {
            throw std::runtime_error("HDBSCAN select stage node is out of range while condensing");
        }
        if (ignore[static_cast<std::size_t>(node)] || node < n_samples) {
            continue;
        }

        const single_linkage_row& children = hierarchy[static_cast<std::size_t>(node - n_samples)];
        const std::int32_t left = children.left_node;
        const std::int32_t right = children.right_node;
        const double distance = static_cast<double>(children.distance);
        const double lambda_value = distance > 0.0
            ? 1.0 / distance
            : std::numeric_limits<double>::infinity();

        const std::int32_t left_count = left >= n_samples
            ? hierarchy[static_cast<std::size_t>(left - n_samples)].cluster_size
            : 1;
        const std::int32_t right_count = right >= n_samples
            ? hierarchy[static_cast<std::size_t>(right - n_samples)].cluster_size
            : 1;

        const bool left_large = static_cast<std::size_t>(left_count) >= min_cluster_size;
        const bool right_large = static_cast<std::size_t>(right_count) >= min_cluster_size;
        const std::int32_t parent_label = relabel[static_cast<std::size_t>(node)];

        if (left_large && right_large) {
            relabel[static_cast<std::size_t>(left)] = next_label++;
            result.push_back(condensed_tree_row{
                parent_label,
                relabel[static_cast<std::size_t>(left)],
                lambda_value,
                left_count,
            });

            relabel[static_cast<std::size_t>(right)] = next_label++;
            result.push_back(condensed_tree_row{
                parent_label,
                relabel[static_cast<std::size_t>(right)],
                lambda_value,
                right_count,
            });
        } else if (!left_large && !right_large) {
            for (const std::int32_t sub_node : bfs_from_hierarchy(hierarchy, left)) {
                if (sub_node < n_samples) {
                    result.push_back(condensed_tree_row{parent_label, sub_node, lambda_value, 1});
                }
                if (sub_node >= 0 && sub_node <= root) {
                    ignore[static_cast<std::size_t>(sub_node)] = 1;
                }
            }

            for (const std::int32_t sub_node : bfs_from_hierarchy(hierarchy, right)) {
                if (sub_node < n_samples) {
                    result.push_back(condensed_tree_row{parent_label, sub_node, lambda_value, 1});
                }
                if (sub_node >= 0 && sub_node <= root) {
                    ignore[static_cast<std::size_t>(sub_node)] = 1;
                }
            }
        } else if (!left_large) {
            relabel[static_cast<std::size_t>(right)] = parent_label;
            for (const std::int32_t sub_node : bfs_from_hierarchy(hierarchy, left)) {
                if (sub_node < n_samples) {
                    result.push_back(condensed_tree_row{parent_label, sub_node, lambda_value, 1});
                }
                if (sub_node >= 0 && sub_node <= root) {
                    ignore[static_cast<std::size_t>(sub_node)] = 1;
                }
            }
        } else {
            relabel[static_cast<std::size_t>(left)] = parent_label;
            for (const std::int32_t sub_node : bfs_from_hierarchy(hierarchy, right)) {
                if (sub_node < n_samples) {
                    result.push_back(condensed_tree_row{parent_label, sub_node, lambda_value, 1});
                }
                if (sub_node >= 0 && sub_node <= root) {
                    ignore[static_cast<std::size_t>(sub_node)] = 1;
                }
            }
        }
    }

    return result;
}

inline std::vector<double> compute_stability(
    std::span<const condensed_tree_row> condensed_tree,
    std::int32_t& smallest_cluster_out
) {
    if (condensed_tree.empty()) {
        smallest_cluster_out = 0;
        return {};
    }

    std::int32_t largest_child = condensed_tree.front().child;
    std::int32_t smallest_cluster = condensed_tree.front().parent;
    std::int32_t largest_parent = condensed_tree.front().parent;
    for (const condensed_tree_row& row : condensed_tree) {
        largest_child = std::max(largest_child, row.child);
        smallest_cluster = std::min(smallest_cluster, row.parent);
        largest_parent = std::max(largest_parent, row.parent);
    }

    largest_child = std::max(largest_child, smallest_cluster);
    std::vector<double> births(
        static_cast<std::size_t>(largest_child + 1),
        std::numeric_limits<double>::quiet_NaN()
    );
    for (const condensed_tree_row& row : condensed_tree) {
        births[static_cast<std::size_t>(row.child)] = row.lambda_value;
    }
    births[static_cast<std::size_t>(smallest_cluster)] = 0.0;

    std::vector<double> stability(static_cast<std::size_t>(largest_parent + 1), 0.0);
    for (const condensed_tree_row& row : condensed_tree) {
        const double birth = births[static_cast<std::size_t>(row.parent)];
        stability[static_cast<std::size_t>(row.parent)] +=
            (row.lambda_value - birth) * static_cast<double>(row.cluster_size);
    }

    smallest_cluster_out = smallest_cluster;
    return stability;
}

inline std::vector<condensed_tree_row> cluster_tree_rows(
    std::span<const condensed_tree_row> condensed_tree
) {
    std::vector<condensed_tree_row> cluster_tree;
    for (const condensed_tree_row& row : condensed_tree) {
        if (row.cluster_size > 1) {
            cluster_tree.push_back(row);
        }
    }
    return cluster_tree;
}

inline std::vector<std::int32_t> bfs_from_cluster_tree(
    std::span<const condensed_tree_row> cluster_tree,
    std::int32_t bfs_root
) {
    std::vector<std::int32_t> result;
    std::vector<std::int32_t> queue{bfs_root};

    while (!queue.empty()) {
        result.insert(result.end(), queue.begin(), queue.end());

        std::vector<std::int32_t> next_queue;
        for (const std::int32_t parent : queue) {
            for (const condensed_tree_row& row : cluster_tree) {
                if (row.parent == parent) {
                    next_queue.push_back(row.child);
                }
            }
        }
        queue = std::move(next_queue);
    }

    return result;
}

inline std::vector<double> max_lambdas(
    std::span<const condensed_tree_row> condensed_tree
) {
    if (condensed_tree.empty()) {
        return {};
    }

    std::int32_t largest_parent = condensed_tree.front().parent;
    for (const condensed_tree_row& row : condensed_tree) {
        largest_parent = std::max(largest_parent, row.parent);
    }

    std::vector<double> deaths(static_cast<std::size_t>(largest_parent + 1), 0.0);
    std::int32_t current_parent = condensed_tree.front().parent;
    double max_lambda = condensed_tree.front().lambda_value;

    for (std::size_t idx = 1; idx < condensed_tree.size(); ++idx) {
        const condensed_tree_row& row = condensed_tree[idx];
        if (row.parent == current_parent) {
            max_lambda = std::max(max_lambda, row.lambda_value);
        } else {
            deaths[static_cast<std::size_t>(current_parent)] = max_lambda;
            current_parent = row.parent;
            max_lambda = row.lambda_value;
        }
    }
    deaths[static_cast<std::size_t>(current_parent)] = max_lambda;
    return deaths;
}

inline hdbscan_selection_result labels_and_probabilities_from_condensed_tree(
    std::span<const condensed_tree_row> condensed_tree,
    std::span<const condensed_tree_row> cluster_tree,
    const std::unordered_set<std::int32_t>& clusters,
    const std::map<std::int32_t, std::int32_t>& cluster_label_map,
    std::size_t n_samples
) {
    hdbscan_selection_result result;
    result.labels.assign(n_samples, -1);
    result.probabilities.assign(n_samples, 0.0f);

    if (condensed_tree.empty()) {
        return result;
    }

    std::int32_t root_cluster = condensed_tree.front().parent;
    std::int32_t max_node = 0;
    for (const condensed_tree_row& row : condensed_tree) {
        root_cluster = std::min(root_cluster, row.parent);
        max_node = std::max(max_node, row.parent);
        max_node = std::max(max_node, row.child);
    }

    tree_union_find_for_select union_find(static_cast<std::size_t>(max_node + 1));
    for (const condensed_tree_row& row : condensed_tree) {
        if (clusters.find(row.child) == clusters.end()) {
            union_find.unite(row.parent, row.child);
        }
    }

    for (std::size_t sample = 0; sample < n_samples; ++sample) {
        const std::int32_t cluster = union_find.find(static_cast<std::int32_t>(sample));
        std::int32_t label = -1;
        if (cluster != root_cluster) {
            const auto it = cluster_label_map.find(cluster);
            if (it != cluster_label_map.end()) {
                label = it->second;
            }
        }
        result.labels[sample] = label;
    }

    std::map<std::int32_t, std::int32_t> reverse_cluster_map;
    for (const auto& [cluster, label] : cluster_label_map) {
        reverse_cluster_map[label] = cluster;
    }

    const std::vector<double> deaths = max_lambdas(condensed_tree);
    for (const condensed_tree_row& row : condensed_tree) {
        const std::int32_t point = row.child;
        if (point >= root_cluster || point < 0 || static_cast<std::size_t>(point) >= n_samples) {
            continue;
        }

        const std::int32_t cluster_num = result.labels[static_cast<std::size_t>(point)];
        if (cluster_num == -1) {
            continue;
        }

        const auto reverse_it = reverse_cluster_map.find(cluster_num);
        if (reverse_it == reverse_cluster_map.end()) {
            continue;
        }
        const std::int32_t cluster = reverse_it->second;
        const double max_lambda = cluster >= 0 && static_cast<std::size_t>(cluster) < deaths.size()
            ? deaths[static_cast<std::size_t>(cluster)]
            : 0.0;

        if (max_lambda == 0.0 || std::isinf(row.lambda_value)) {
            result.probabilities[static_cast<std::size_t>(point)] = 1.0f;
        } else {
            const double lambda_value = std::min(row.lambda_value, max_lambda);
            result.probabilities[static_cast<std::size_t>(point)] = static_cast<float>(lambda_value / max_lambda);
        }
    }

    return result;
}

inline hdbscan_selection_result select_clusters_from_single_linkage_tree(
    std::span<const single_linkage_row> single_linkage_tree,
    std::size_t min_cluster_size
) {
    const std::size_t n_samples = single_linkage_tree.size() + 1;
    hdbscan_selection_result empty_result;
    empty_result.labels.assign(n_samples, -1);
    empty_result.probabilities.assign(n_samples, 0.0f);

    if (single_linkage_tree.empty()) {
        return empty_result;
    }

    std::vector<condensed_tree_row> condensed_tree = condense_tree(
        single_linkage_tree,
        min_cluster_size
    );
    if (condensed_tree.empty()) {
        return empty_result;
    }

    std::int32_t smallest_cluster = 0;
    std::vector<double> stability = compute_stability(condensed_tree, smallest_cluster);
    std::vector<condensed_tree_row> cluster_tree = cluster_tree_rows(condensed_tree);

    std::int32_t largest_parent = condensed_tree.front().parent;
    for (const condensed_tree_row& row : condensed_tree) {
        largest_parent = std::max(largest_parent, row.parent);
    }

    // EOM only, allow_single_cluster=False: process all stability keys except
    // the root cluster, descending by cluster id.
    std::map<std::int32_t, bool> is_cluster;
    for (std::int32_t node = largest_parent; node > smallest_cluster; --node) {
        is_cluster[node] = true;
    }

    for (auto it = is_cluster.rbegin(); it != is_cluster.rend(); ++it) {
        const std::int32_t node = it->first;
        double subtree_stability = 0.0;
        for (const condensed_tree_row& row : cluster_tree) {
            if (row.parent == node && row.child >= 0
                && static_cast<std::size_t>(row.child) < stability.size()) {
                subtree_stability += stability[static_cast<std::size_t>(row.child)];
            }
        }

        if (subtree_stability > stability[static_cast<std::size_t>(node)]) {
            it->second = false;
            stability[static_cast<std::size_t>(node)] = subtree_stability;
        } else {
            for (const std::int32_t sub_node : bfs_from_cluster_tree(cluster_tree, node)) {
                if (sub_node != node) {
                    const auto sub_it = is_cluster.find(sub_node);
                    if (sub_it != is_cluster.end()) {
                        sub_it->second = false;
                    }
                }
            }
        }
    }

    std::vector<std::int32_t> selected_clusters_sorted;
    for (const auto& [cluster, selected] : is_cluster) {
        if (selected) {
            selected_clusters_sorted.push_back(cluster);
        }
    }

    std::unordered_set<std::int32_t> selected_clusters(
        selected_clusters_sorted.begin(),
        selected_clusters_sorted.end()
    );
    std::map<std::int32_t, std::int32_t> cluster_label_map;
    for (std::size_t idx = 0; idx < selected_clusters_sorted.size(); ++idx) {
        cluster_label_map[selected_clusters_sorted[idx]] = static_cast<std::int32_t>(idx);
    }

    return labels_and_probabilities_from_condensed_tree(
        condensed_tree,
        cluster_tree,
        selected_clusters,
        cluster_label_map,
        n_samples
    );
}
