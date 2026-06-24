#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
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

inline std::uint64_t float_vector_fnv1a(std::span<const float> values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (float value : values) {
        static_assert(sizeof(float) == 4);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(float));
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

inline void write_float_vector_summary_json(
    std::ostream& out,
    std::span<const float> values,
    const char* indent
) {
    const std::size_t value_count = values.size();

    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_squares = 0.0;
    double weighted_sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    std::size_t finite_count = 0;
    std::size_t nan_count = 0;
    std::size_t pos_inf_count = 0;
    std::size_t neg_inf_count = 0;

    for (std::size_t i = 0; i < value_count; ++i) {
        const float value = values[i];
        if (std::isnan(value)) {
            ++nan_count;
            continue;
        }
        if (value == std::numeric_limits<float>::infinity()) {
            ++pos_inf_count;
            continue;
        }
        if (value == -std::numeric_limits<float>::infinity()) {
            ++neg_inf_count;
            continue;
        }
        ++finite_count;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        const double value64 = static_cast<double>(value);
        sum += value64;
        sum_abs += std::abs(value64);
        sum_squares += value64 * value64;
        weighted_sum += value64 * deterministic_weight(i);
    }

    if (finite_count == 0) {
        min_value = std::numeric_limits<float>::quiet_NaN();
        max_value = std::numeric_limits<float>::quiet_NaN();
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
    out << indent << "\"min\": " << static_cast<double>(min_value) << ",\n";
    out << indent << "\"max\": " << static_cast<double>(max_value) << ",\n";
    out << indent << "\"fnv1a64_float32\": \"0x" << std::hex << float_vector_fnv1a(values)
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
    std::span<const float> distance_matrix,
    std::size_t N,
    std::size_t min_samples
) {
    if (distance_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN distance metrics matrix size does not match N * N");
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
    out << "  \"dtype\": \"float32\",\n";
    out << "  \"n_samples\": " << N << ",\n";
    out << "  \"min_samples\": " << min_samples << ",\n";
    out << "  \"shape\": [" << N << ", " << N << "],\n";
    out << "  \"diagonal_max_abs\": " << diagonal_max_abs << ",\n";
    out << "  \"symmetry_max_abs\": " << symmetry_max_abs << ",\n";
    out << "  \"summary\": {\n";
    write_float_vector_summary_json(out, distance_matrix, "    ");
    out << "  }\n";
    out << "}\n";
}

inline void write_hdbscan_mreach_metrics(
    const std::string& filename,
    std::span<const float> mutual_reachability_matrix,
    std::size_t N,
    std::size_t min_samples
) {
    if (mutual_reachability_matrix.size() != N * N) {
        throw std::runtime_error("HDBSCAN mreach metrics matrix size does not match N * N");
    }

    double symmetry_max_abs = 0.0;
    std::vector<float> diagonal;
    diagonal.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        diagonal.push_back(mutual_reachability_matrix[i * N + i]);
        for (std::size_t j = i + 1; j < N; ++j) {
            const double diff = std::abs(
                static_cast<double>(mutual_reachability_matrix[i * N + j])
                - static_cast<double>(mutual_reachability_matrix[j * N + i])
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
    out << "  \"stage\": \"mreach\",\n";
    out << "  \"dtype\": \"float32\",\n";
    out << "  \"n_samples\": " << N << ",\n";
    out << "  \"min_samples\": " << min_samples << ",\n";
    out << "  \"shape\": [" << N << ", " << N << "],\n";
    out << "  \"symmetry_max_abs\": " << symmetry_max_abs << ",\n";
    out << "  \"summary\": {\n";
    write_float_vector_summary_json(out, mutual_reachability_matrix, "    ");
    out << "  },\n";
    out << "  \"diagonal_summary\": {\n";
    write_float_vector_summary_json(out, std::span<const float>(diagonal.data(), diagonal.size()), "    ");
    out << "  }\n";
    out << "}\n";
}

} // namespace hdbscan_metrics
