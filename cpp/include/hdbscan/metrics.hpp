#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace hdbscan_metrics {

inline std::uint64_t fnv1a_update(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::uint64_t double_vector_fnv1a(std::span<const double> values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (double value : values) {
        static_assert(sizeof(double) == 8);
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(double));
        hash = fnv1a_update(hash, &bits, sizeof(bits));
    }
    return hash;
}

inline double deterministic_weight(std::size_t index) {
    // A fixed, cheap, non-symmetric weight used to make same-summary collisions less likely.
    const std::uint64_t x = static_cast<std::uint64_t>(index) + 0x9e3779b97f4a7c15ULL;
    const std::uint64_t mixed = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    return static_cast<double>((mixed >> 11) & 0x1fffffULL) / static_cast<double>(0x1fffffULL);
}

inline std::vector<std::size_t> probe_indices(std::size_t value_count) {
    std::vector<std::size_t> indices;
    if (value_count == 0) {
        return indices;
    }

    const std::size_t fixed_count = std::min<std::size_t>(value_count, 32);
    for (std::size_t i = 0; i < fixed_count; ++i) {
        indices.push_back(i);
    }

    const std::size_t tail_start = value_count > 32 ? value_count - 32 : value_count;
    for (std::size_t i = tail_start; i < value_count; ++i) {
        indices.push_back(i);
    }

    std::uint64_t state = 0x243f6a8885a308d3ULL ^ static_cast<std::uint64_t>(value_count);
    for (std::size_t i = 0; i < 64 && value_count > 0; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        indices.push_back(static_cast<std::size_t>(state % value_count));
    }

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

inline void write_double_vector_summary_json(
    std::ostream& out,
    std::span<const double> values,
    const char* indent
) {
    const std::size_t value_count = values.size();

    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_squares = 0.0;
    double weighted_sum = 0.0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    std::size_t finite_count = 0;
    std::size_t nan_count = 0;
    std::size_t pos_inf_count = 0;
    std::size_t neg_inf_count = 0;

    for (std::size_t i = 0; i < value_count; ++i) {
        const double value = values[i];
        if (std::isnan(value)) {
            ++nan_count;
            continue;
        }
        if (value == std::numeric_limits<double>::infinity()) {
            ++pos_inf_count;
            continue;
        }
        if (value == -std::numeric_limits<double>::infinity()) {
            ++neg_inf_count;
            continue;
        }
        ++finite_count;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
        sum_abs += std::abs(value);
        sum_squares += value * value;
        weighted_sum += value * deterministic_weight(i);
    }

    if (finite_count == 0) {
        min_value = std::numeric_limits<double>::quiet_NaN();
        max_value = std::numeric_limits<double>::quiet_NaN();
    }

    out << indent << "\"value_count\": " << value_count << ",\n";
    out << indent << "\"finite_count\": " << finite_count << ",\n";
    out << indent << "\"nan_count\": " << nan_count << ",\n";
    out << indent << "\"pos_inf_count\": " << pos_inf_count << ",\n";
    out << indent << "\"neg_inf_count\": " << neg_inf_count << ",\n";
    out << indent << "\"sum\": " << sum << ",\n";
    out << indent << "\"sum_abs\": " << sum_abs << ",\n";
    out << indent << "\"sum_squares\": " << sum_squares << ",\n";
    out << indent << "\"weighted_sum\": " << weighted_sum << ",\n";
    out << indent << "\"min\": " << min_value << ",\n";
    out << indent << "\"max\": " << max_value << ",\n";
    out << indent << "\"fnv1a64_float64\": \"0x" << std::hex << double_vector_fnv1a(values)
        << std::dec << "\",\n";

    out << indent << "\"probes\": [";
    const std::vector<std::size_t> indices = probe_indices(value_count);
    for (std::size_t probe = 0; probe < indices.size(); ++probe) {
        if (probe != 0) {
            out << ", ";
        }
        const std::size_t index = indices[probe];
        out << "{\"index\": " << index
            << ", \"value\": " << static_cast<double>(values[index]) << "}";
    }
    out << "]\n";
}

inline void write_hdbscan_distance_metrics(
    const std::string& filename,
    std::span<const double> distance_matrix,
    std::span<const double> core_distances,
    std::size_t N,
    std::size_t min_samples
) {
    if (distance_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN distance metrics matrix size does not match N * N");
    }
    if (core_distances.size() != N) {
        throw std::runtime_error("HDBSCAN distance metrics core-distance size does not match N");
    }

    double diagonal_max_abs = 0.0;
    double symmetry_max_abs = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        diagonal_max_abs = std::max(
            diagonal_max_abs,
            std::abs(static_cast<double>(distance_matrix[i * N + i]))
        );
        for (std::size_t j = i + 1; j < N; ++j) {
            const double diff = std::abs(
                static_cast<double>(distance_matrix[i * N + j])
                - static_cast<double>(distance_matrix[j * N + i])
            );
            symmetry_max_abs = std::max(symmetry_max_abs, diff);
        }
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open HDBSCAN metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"phase\": \"hdbscan\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"stage\": \"distance\",\n";
    out << "  \"dtype\": \"float64\",\n";
    out << "  \"n_samples\": " << N << ",\n";
    out << "  \"min_samples\": " << min_samples << ",\n";
    out << "  \"shape\": [" << N << ", " << N << "],\n";
    out << "  \"core_distance_shape\": [" << N << "],\n";
    out << "  \"diagonal_max_abs\": " << diagonal_max_abs << ",\n";
    out << "  \"symmetry_max_abs\": " << symmetry_max_abs << ",\n";
    out << "  \"summary\": {\n";
    write_double_vector_summary_json(out, distance_matrix, "    ");
    out << "  },\n";
    out << "  \"core_distance_summary\": {\n";
    write_double_vector_summary_json(out, core_distances, "    ");
    out << "  }\n";
    out << "}\n";
}

inline void write_hdbscan_mst_metrics(
    const std::string& filename,
    std::span<const double> flat_edges,
    std::span<const double> edge_weights,
    std::size_t N,
    std::size_t min_samples
) {
    const std::size_t expected_edges = N == 0 ? 0 : N - 1;
    if (edge_weights.size() != expected_edges) {
        throw std::runtime_error("HDBSCAN MST metrics edge weight count does not match N - 1");
    }
    if (flat_edges.size() != expected_edges * 3) {
        throw std::runtime_error("HDBSCAN MST metrics flat edge size does not match 3 * (N - 1)");
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open HDBSCAN metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"phase\": \"hdbscan\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"stage\": \"mst\",\n";
    out << "  \"dtype\": \"float64\",\n";
    out << "  \"n_samples\": " << N << ",\n";
    out << "  \"min_samples\": " << min_samples << ",\n";
    out << "  \"edge_count\": " << expected_edges << ",\n";
    out << "  \"shape\": [" << expected_edges << ", 3],\n";
    out << "  \"summary\": {\n";
    write_double_vector_summary_json(out, flat_edges, "    ");
    out << "  },\n";
    out << "  \"weight_summary\": {\n";
    write_double_vector_summary_json(out, edge_weights, "    ");
    out << "  }\n";
    out << "}\n";
}

inline void write_hdbscan_linkage_metrics(
    const std::string& filename,
    std::span<const double> flat_tree,
    std::size_t N,
    std::size_t min_samples
) {
    const std::size_t expected_rows = N == 0 ? 0 : N - 1;
    if (flat_tree.size() != expected_rows * 4) {
        throw std::runtime_error("HDBSCAN linkage metrics flat tree size does not match 4 * (N - 1)");
    }

    std::vector<double> distances;
    std::vector<double> cluster_sizes;
    distances.reserve(expected_rows);
    cluster_sizes.reserve(expected_rows);
    for (std::size_t i = 0; i < expected_rows; ++i) {
        distances.push_back(flat_tree[i * 4 + 2]);
        cluster_sizes.push_back(flat_tree[i * 4 + 3]);
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open HDBSCAN metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"phase\": \"hdbscan\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"stage\": \"linkage\",\n";
    out << "  \"dtype\": \"float64\",\n";
    out << "  \"n_samples\": " << N << ",\n";
    out << "  \"min_samples\": " << min_samples << ",\n";
    out << "  \"row_count\": " << expected_rows << ",\n";
    out << "  \"shape\": [" << expected_rows << ", 4],\n";
    out << "  \"summary\": {\n";
    write_double_vector_summary_json(out, flat_tree, "    ");
    out << "  },\n";
    out << "  \"distance_summary\": {\n";
    write_double_vector_summary_json(out, std::span<const double>(distances.data(), distances.size()), "    ");
    out << "  },\n";
    out << "  \"cluster_size_summary\": {\n";
    write_double_vector_summary_json(out, std::span<const double>(cluster_sizes.data(), cluster_sizes.size()), "    ");
    out << "  }\n";
    out << "}\n";
}


inline void write_hdbscan_label_probability_metrics(
    const std::string& filename,
    const std::string& stage,
    std::span<const std::int32_t> labels,
    std::span<const double> probabilities,
    std::size_t N,
    std::size_t min_samples
) {
    if (labels.size() != N || probabilities.size() != N) {
        throw std::runtime_error("HDBSCAN label/probability metrics size does not match N");
    }

    std::vector<double> flat;
    std::vector<double> label_values;
    flat.reserve(N * 2);
    label_values.reserve(N);

    std::set<std::int32_t> clusters;
    std::size_t noise_count = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const std::int32_t label = labels[i];
        const double label_float = static_cast<double>(label);
        label_values.push_back(label_float);
        flat.push_back(label_float);
        flat.push_back(probabilities[i]);
        if (label < 0) {
            ++noise_count;
        } else {
            clusters.insert(label);
        }
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open HDBSCAN metrics output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"phase\": \"hdbscan\",\n";
    out << "  \"language\": \"cpp\",\n";
    out << "  \"stage\": \"" << stage << "\",\n";
    out << "  \"dtype\": \"float64\",\n";
    out << "  \"n_samples\": " << N << ",\n";
    out << "  \"min_samples\": " << min_samples << ",\n";
    out << "  \"shape\": [" << N << ", 2],\n";
    out << "  \"noise_count\": " << noise_count << ",\n";
    out << "  \"cluster_count\": " << clusters.size() << ",\n";
    out << "  \"summary\": {\n";
    write_double_vector_summary_json(out, std::span<const double>(flat.data(), flat.size()), "    ");
    out << "  },\n";
    out << "  \"label_summary\": {\n";
    write_double_vector_summary_json(out, std::span<const double>(label_values.data(), label_values.size()), "    ");
    out << "  },\n";
    out << "  \"probability_summary\": {\n";
    write_double_vector_summary_json(out, probabilities, "    ");
    out << "  }\n";
    out << "}\n";
}

inline void write_hdbscan_select_metrics(
    const std::string& filename,
    std::span<const std::int32_t> labels,
    std::span<const double> probabilities,
    std::size_t N,
    std::size_t min_samples
) {
    write_hdbscan_label_probability_metrics(
        filename,
        "select",
        labels,
        probabilities,
        N,
        min_samples
    );
}

} // namespace hdbscan_metrics
