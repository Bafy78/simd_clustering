#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "linkage.hpp"

struct hdbscan_selection_result {
    std::vector<std::int32_t> labels;
    std::vector<double> probabilities;
};

struct condensed_tree_row {
    std::int32_t parent;
    std::int32_t child;
    double lambda_value;
    std::int32_t cluster_size;
};

struct cluster_record {
    double stability = 0.0;
    double death = 0.0;
    double birth = std::numeric_limits<double>::quiet_NaN();
    double propagated_score = 0.0;

    std::int32_t first_child = -1;
    std::int32_t second_child = -1;

    std::int32_t owner = -1;
    std::int32_t label = -1;
    unsigned char keep_self = 0;
    unsigned char selected = 0;
};

struct condensation_result {
    // Only sample fall-out rows are stored here. Cluster-to-cluster structure
    // and EOM state are carried by cluster_records.
    std::vector<condensed_tree_row> rows;
    std::vector<cluster_record> clusters;
    std::int32_t smallest_cluster = 0;
    std::int32_t largest_parent = 0;
    std::int32_t max_node = 0;
};

inline void ensure_cluster_records_size(
    condensation_result& result,
    std::int32_t node
) {
    if (node < 0) {
        return;
    }

    const std::size_t needed = static_cast<std::size_t>(node + 1);
    if (result.clusters.size() < needed) {
        result.clusters.resize(needed);
    }
}

inline void update_parent_condensation_metadata(
    condensation_result& result,
    std::int32_t parent,
    std::int32_t child,
    double lambda_value,
    std::int32_t cluster_size
) {
    result.smallest_cluster = std::min(result.smallest_cluster, parent);
    result.largest_parent = std::max(result.largest_parent, parent);
    result.max_node = std::max(result.max_node, parent);
    result.max_node = std::max(result.max_node, child);

    ensure_cluster_records_size(result, std::max(parent, child));

    cluster_record& parent_record = result.clusters[static_cast<std::size_t>(parent)];
    parent_record.death = std::max(parent_record.death, lambda_value);
    parent_record.stability +=
        (lambda_value - parent_record.birth) * static_cast<double>(cluster_size);
}

inline void emit_point_condensed_tree_row(
    condensation_result& result,
    std::int32_t parent,
    std::int32_t child,
    double lambda_value
) {
    result.rows.push_back(condensed_tree_row{parent, child, lambda_value, 1});
    update_parent_condensation_metadata(result, parent, child, lambda_value, 1);
}

inline void attach_cluster_child(
    condensation_result& result,
    std::int32_t parent,
    std::int32_t child
) {
    ensure_cluster_records_size(result, std::max(parent, child));

    cluster_record& parent_record = result.clusters[static_cast<std::size_t>(parent)];
    if (parent_record.first_child < 0) {
        parent_record.first_child = child;
    } else if (parent_record.second_child < 0) {
        parent_record.second_child = child;
    } else {
        throw std::runtime_error("HDBSCAN select binary cluster tree received more than two children");
    }
}

inline void emit_cluster_tree_edge(
    condensation_result& result,
    std::int32_t parent,
    std::int32_t child,
    double lambda_value,
    std::int32_t cluster_size
) {
    attach_cluster_child(result, parent, child);
    update_parent_condensation_metadata(result, parent, child, lambda_value, cluster_size);

    if (cluster_size > 1 && child >= 0) {
        ensure_cluster_records_size(result, child);
        result.clusters[static_cast<std::size_t>(child)].birth = lambda_value;
    }
}

inline std::int32_t hierarchy_node_cluster_size(
    std::span<const single_linkage_row> hierarchy,
    std::int32_t node,
    std::int32_t n_samples
) {
    if (node < n_samples) {
        return 1;
    }

    const std::int32_t row_index = node - n_samples;
    if (row_index < 0 || static_cast<std::size_t>(row_index) >= hierarchy.size()) {
        throw std::runtime_error("HDBSCAN select stage hierarchy node is out of range");
    }

    return hierarchy[static_cast<std::size_t>(row_index)].cluster_size;
}

inline void emit_subtree_sample_leaves_bfs(
    std::span<const single_linkage_row> hierarchy,
    std::int32_t subtree_root,
    std::int32_t n_samples,
    std::int32_t parent_label,
    double lambda_value,
    condensation_result& result,
    std::vector<std::int32_t>& queue
) {
    if (subtree_root < n_samples) {
        emit_point_condensed_tree_row(result, parent_label, subtree_root, lambda_value);
        return;
    }

    queue.clear();
    queue.push_back(subtree_root);

    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
        const std::int32_t node = queue[cursor];
        if (node < n_samples) {
            emit_point_condensed_tree_row(result, parent_label, node, lambda_value);
            continue;
        }

        const std::int32_t row_index = node - n_samples;
        if (row_index < 0 || static_cast<std::size_t>(row_index) >= hierarchy.size()) {
            throw std::runtime_error("HDBSCAN select stage hierarchy node is out of range while emitting leaves");
        }

        const single_linkage_row& row = hierarchy[static_cast<std::size_t>(row_index)];
        if (row.left_node < n_samples) {
            emit_point_condensed_tree_row(result, parent_label, row.left_node, lambda_value);
        } else {
            queue.push_back(row.left_node);
        }

        if (row.right_node < n_samples) {
            emit_point_condensed_tree_row(result, parent_label, row.right_node, lambda_value);
        } else {
            queue.push_back(row.right_node);
        }
    }
}

inline condensation_result condense_tree(
    std::span<const single_linkage_row> hierarchy,
    std::size_t min_cluster_size
) {
    if (hierarchy.empty()) {
        return {};
    }

    const std::int32_t n_samples = static_cast<std::int32_t>(hierarchy.size() + 1);
    const std::int32_t root = static_cast<std::int32_t>(2 * hierarchy.size());
    std::int32_t next_label = n_samples + 1;

    std::vector<std::int32_t> relabel(static_cast<std::size_t>(root + 1), 0);
    relabel[static_cast<std::size_t>(root)] = n_samples;

    condensation_result result;
    result.rows.reserve(n_samples);
    result.smallest_cluster = n_samples;
    result.largest_parent = n_samples;
    result.max_node = n_samples;
    result.clusters.resize(static_cast<std::size_t>(root + 1));
    result.clusters[static_cast<std::size_t>(n_samples)].birth = 0.0;

    std::vector<std::int32_t> active_queue;
    active_queue.reserve(hierarchy.size());
    active_queue.push_back(root);

    std::vector<std::int32_t> leaf_queue;
    const std::size_t leaf_queue_reserve = std::min<std::size_t>(
        n_samples,
        min_cluster_size > 0 ? (2 * min_cluster_size + 2) : 2
    );
    leaf_queue.reserve(leaf_queue_reserve);

    for (std::size_t active_cursor = 0; active_cursor < active_queue.size(); ++active_cursor) {
        const std::int32_t node = active_queue[active_cursor];
        if (node < 0 || node > root) {
            throw std::runtime_error("HDBSCAN select stage node is out of range while condensing");
        }
        if (node < n_samples) {
            continue;
        }

        const single_linkage_row& children = hierarchy[static_cast<std::size_t>(node - n_samples)];
        const std::int32_t left = children.left_node;
        const std::int32_t right = children.right_node;
        const double distance = static_cast<double>(children.distance);
        const double lambda_value = distance > 0.0
            ? 1.0 / distance
            : std::numeric_limits<double>::infinity();

        const std::int32_t left_count = hierarchy_node_cluster_size(hierarchy, left, n_samples);
        const std::int32_t right_count = hierarchy_node_cluster_size(hierarchy, right, n_samples);

        const bool left_large = static_cast<std::size_t>(left_count) >= min_cluster_size;
        const bool right_large = static_cast<std::size_t>(right_count) >= min_cluster_size;
        const std::int32_t parent_label = relabel[static_cast<std::size_t>(node)];

        if (left_large && right_large) {
            relabel[static_cast<std::size_t>(left)] = next_label++;
            emit_cluster_tree_edge(
                result,
                parent_label,
                relabel[static_cast<std::size_t>(left)],
                lambda_value,
                left_count
            );
            if (left >= n_samples) {
                active_queue.push_back(left);
            }

            relabel[static_cast<std::size_t>(right)] = next_label++;
            emit_cluster_tree_edge(
                result,
                parent_label,
                relabel[static_cast<std::size_t>(right)],
                lambda_value,
                right_count
            );
            if (right >= n_samples) {
                active_queue.push_back(right);
            }
        } else if (!left_large && !right_large) {
            emit_subtree_sample_leaves_bfs(
                hierarchy,
                left,
                n_samples,
                parent_label,
                lambda_value,
                result,
                leaf_queue
            );
            emit_subtree_sample_leaves_bfs(
                hierarchy,
                right,
                n_samples,
                parent_label,
                lambda_value,
                result,
                leaf_queue
            );
        } else if (!left_large) {
            relabel[static_cast<std::size_t>(right)] = parent_label;
            if (right >= n_samples) {
                active_queue.push_back(right);
            }
            emit_subtree_sample_leaves_bfs(
                hierarchy,
                left,
                n_samples,
                parent_label,
                lambda_value,
                result,
                leaf_queue
            );
        } else {
            relabel[static_cast<std::size_t>(left)] = parent_label;
            if (left >= n_samples) {
                active_queue.push_back(left);
            }
            emit_subtree_sample_leaves_bfs(
                hierarchy,
                right,
                n_samples,
                parent_label,
                lambda_value,
                result,
                leaf_queue
            );
        }
    }

    return result;
}

inline double cluster_children_propagated_score(
    const std::vector<cluster_record>& clusters,
    std::int32_t node
) {
    if (node < 0 || static_cast<std::size_t>(node) >= clusters.size()) {
        return 0.0;
    }

    const cluster_record& record = clusters[static_cast<std::size_t>(node)];
    double score = 0.0;

    const std::int32_t first = record.first_child;
    if (first >= 0 && static_cast<std::size_t>(first) < clusters.size()) {
        score += clusters[static_cast<std::size_t>(first)].propagated_score;
    }

    const std::int32_t second = record.second_child;
    if (second >= 0 && static_cast<std::size_t>(second) < clusters.size()) {
        score += clusters[static_cast<std::size_t>(second)].propagated_score;
    }

    return score;
}

inline void eom_select_cluster_records(
    std::vector<cluster_record>& clusters,
    std::int32_t smallest_cluster,
    std::int32_t largest_parent,
    std::int32_t max_node
) {
    if (max_node < 0 || largest_parent <= smallest_cluster) {
        return;
    }

    const std::int32_t bounded_largest_parent = std::min<std::int32_t>(
        largest_parent,
        static_cast<std::int32_t>(clusters.size()) - 1
    );

    for (std::int32_t node = bounded_largest_parent; node > smallest_cluster; --node) {
        cluster_record& record = clusters[static_cast<std::size_t>(node)];
        const double child_score = cluster_children_propagated_score(clusters, node);

        if (child_score > record.stability) {
            record.propagated_score = child_score;
            record.keep_self = 0;
        } else {
            record.propagated_score = record.stability;
            record.keep_self = 1;
        }
    }

    std::int32_t next_label = 0;
    const std::int32_t bounded_max_node = std::min<std::int32_t>(
        max_node,
        static_cast<std::int32_t>(clusters.size()) - 1
    );

    for (std::int32_t node = smallest_cluster; node <= bounded_max_node; ++node) {
        cluster_record& record = clusters[static_cast<std::size_t>(node)];
        const std::int32_t inherited_owner = record.owner;
        const bool selected = node > smallest_cluster
            && record.keep_self != 0
            && inherited_owner < 0;

        if (selected) {
            record.selected = 1;
            record.owner = node;
            record.label = next_label++;
        }

        const std::int32_t owner_for_children = selected
            ? node
            : inherited_owner;

        const std::int32_t first = record.first_child;
        if (first >= 0 && static_cast<std::size_t>(first) < clusters.size()) {
            clusters[static_cast<std::size_t>(first)].owner = owner_for_children;
        }

        const std::int32_t second = record.second_child;
        if (second >= 0 && static_cast<std::size_t>(second) < clusters.size()) {
            clusters[static_cast<std::size_t>(second)].owner = owner_for_children;
        }
    }
}

inline hdbscan_selection_result labels_and_probabilities_from_sample_rows(
    std::span<const condensed_tree_row> sample_rows,
    const std::vector<cluster_record>& clusters,
    std::size_t n_samples
) {
    hdbscan_selection_result result;
    result.labels.assign(n_samples, -1);
    result.probabilities.assign(n_samples, 0.0);

    for (const condensed_tree_row& row : sample_rows) {
        if (row.parent < 0 || static_cast<std::size_t>(row.parent) >= clusters.size()) {
            continue;
        }

        const cluster_record& parent_record = clusters[static_cast<std::size_t>(row.parent)];
        const std::int32_t owner = parent_record.owner;
        if (owner < 0 || static_cast<std::size_t>(owner) >= clusters.size()) {
            continue;
        }

        const cluster_record& owner_record = clusters[static_cast<std::size_t>(owner)];
        const std::int32_t label = owner_record.label;
        if (label < 0) {
            continue;
        }

        const std::int32_t child = row.child;
        if (child < 0 || static_cast<std::size_t>(child) >= n_samples) {
            continue;
        }

        const std::size_t point_idx = static_cast<std::size_t>(child);
        result.labels[point_idx] = label;

        const double max_lambda = owner_record.death;
        if (max_lambda == 0.0 || std::isinf(row.lambda_value)) {
            result.probabilities[point_idx] = 1.0;
        } else {
            const double lambda_value = std::min(row.lambda_value, max_lambda);
            result.probabilities[point_idx] = lambda_value / max_lambda;
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
    empty_result.probabilities.assign(n_samples, 0.0);

    if (single_linkage_tree.empty()) {
        return empty_result;
    }

    condensation_result condensed = condense_tree(
        single_linkage_tree,
        min_cluster_size
    );
    if (condensed.rows.empty()) {
        return empty_result;
    }

    eom_select_cluster_records(
        condensed.clusters,
        condensed.smallest_cluster,
        condensed.largest_parent,
        condensed.max_node
    );

    return labels_and_probabilities_from_sample_rows(
        condensed.rows,
        condensed.clusters,
        n_samples
    );
}
