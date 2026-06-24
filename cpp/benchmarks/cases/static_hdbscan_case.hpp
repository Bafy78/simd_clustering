#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

#include <eve/memory/aligned_allocator.hpp>
#include <nanobench.h>

#include "../../include/hdbscan/static_d/distance.hpp"
#include "../../include/hdbscan/static_d/mst.hpp"
#include "../../include/hdbscan/static_d/linkage.hpp"
#include "../../include/hdbscan/static_d/select.hpp"
#include "../../include/hdbscan/metrics.hpp"
#include "../../include/io/binary.hpp"
#include "../../include/layout/static_soa.hpp"

struct static_hdbscan_case {
    static constexpr int nanobench_argc = 9;
    static constexpr int callgrind_argc = 6;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <stage> <input_bin> <N> <min_samples> <metrics_json_out>"
              " <nanobench_json_out> <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <stage> <input_bin> <N> <min_samples> <metrics_json_out>";
    }

    static static_hdbscan_case make_for_nanobench(int, char* argv[]) {
        static_hdbscan_case out{
            argv[1],
            argv[2],
            static_cast<std::size_t>(std::stoull(argv[3])),
            static_cast<std::size_t>(std::stoull(argv[4]))
        };

        out.metrics_json_out_ = argv[5];
        out.nanobench_json_out_ = argv[6];
        out.bench_epochs_ = static_cast<std::size_t>(std::stoull(argv[7]));
        out.min_epoch_seconds_ = std::stod(argv[8]);

        return out;
    }

    static static_hdbscan_case make_for_callgrind(int, char* argv[]) {
        static_hdbscan_case out{
            argv[1],
            argv[2],
            static_cast<std::size_t>(std::stoull(argv[3])),
            static_cast<std::size_t>(std::stoull(argv[4]))
        };

        out.metrics_json_out_ = argv[5];
        return out;
    }

    std::string title() const {
        return "EVE HDBSCAN " + std::to_string(D) + "D [" + stage_ + "]";
    }

    std::string run_name() const {
        return "hdbscan_" + stage_ + "_static";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        if (stage_ == "distance") {
            euclidean_distance_matrix<D>(
                samples_,
                distance_matrix_
            );
            return;
        }

        if (stage_ == "mreach") {
            mutual_reachability_matrix(
                std::span<const float>(distance_matrix_input_.data(), distance_matrix_input_.size()),
                N_,
                min_samples_,
                mutual_reachability_matrix_
            );
            return;
        }

        if (stage_ == "mst") {
            minimum_spanning_tree_edges(
                std::span<const float>(mutual_reachability_matrix_input_.data(), mutual_reachability_matrix_input_.size()),
                N_,
                mst_edges_
            );
            return;
        }

        if (stage_ == "linkage") {
            single_linkage_tree_from_mst_edges(
                std::span<const mst_edge>(mst_edges_input_.data(), mst_edges_input_.size()),
                N_,
                single_linkage_tree_
            );
            return;
        }

        if (stage_ == "select") {
            selection_result_ = select_clusters_from_single_linkage_tree(
                std::span<const single_linkage_row>(single_linkage_tree_input_.data(), single_linkage_tree_input_.size()),
                min_samples_
            );
            return;
        }

        throw std::invalid_argument(
            "static_hdbscan_case only implements stages 'distance', 'mreach', 'mst', 'linkage', and 'select' for now"
        );
    }

    void keep_alive() const {
        if (stage_ == "distance") {
            ankerl::nanobench::doNotOptimizeAway(distance_matrix_.data());
            ankerl::nanobench::doNotOptimizeAway(distance_matrix_.size());
        } else if (stage_ == "mreach") {
            ankerl::nanobench::doNotOptimizeAway(mutual_reachability_matrix_.data());
            ankerl::nanobench::doNotOptimizeAway(mutual_reachability_matrix_.size());
        } else if (stage_ == "mst") {
            ankerl::nanobench::doNotOptimizeAway(mst_edges_.data());
            ankerl::nanobench::doNotOptimizeAway(mst_edges_.size());
        } else if (stage_ == "linkage") {
            ankerl::nanobench::doNotOptimizeAway(single_linkage_tree_.data());
            ankerl::nanobench::doNotOptimizeAway(single_linkage_tree_.size());
        } else if (stage_ == "select") {
            ankerl::nanobench::doNotOptimizeAway(selection_result_.labels.data());
            ankerl::nanobench::doNotOptimizeAway(selection_result_.labels.size());
            ankerl::nanobench::doNotOptimizeAway(selection_result_.probabilities.data());
            ankerl::nanobench::doNotOptimizeAway(selection_result_.probabilities.size());
        }
        ankerl::nanobench::doNotOptimizeAway(min_samples_);
    }

    void write_outputs() const {
        if (stage_ == "distance") {
            hdbscan_metrics::write_hdbscan_distance_metrics(
                metrics_json_out_,
                std::span<const float>(distance_matrix_.data(), distance_matrix_.size()),
                N_,
                min_samples_
            );
            return;
        }

        if (stage_ == "mreach") {
            hdbscan_metrics::write_hdbscan_mreach_metrics(
                metrics_json_out_,
                std::span<const float>(mutual_reachability_matrix_.data(), mutual_reachability_matrix_.size()),
                N_,
                min_samples_
            );
            return;
        }

        if (stage_ == "mst") {
            write_mst_outputs();
            return;
        }

        if (stage_ == "linkage") {
            write_linkage_outputs();
            return;
        }

        if (stage_ == "select") {
            hdbscan_metrics::write_hdbscan_select_metrics(
                metrics_json_out_,
                std::span<const std::int32_t>(selection_result_.labels.data(), selection_result_.labels.size()),
                std::span<const float>(selection_result_.probabilities.data(), selection_result_.probabilities.size()),
                N_,
                min_samples_
            );
            return;
        }
    }

    void write_mst_outputs() const {
        std::vector<float> flat_edges;
        std::vector<float> edge_weights;
        flat_edges.reserve(mst_edges_.size() * 3);
        edge_weights.reserve(mst_edges_.size());

        for (const auto& edge : mst_edges_) {
            flat_edges.push_back(static_cast<float>(edge.current_node));
            flat_edges.push_back(static_cast<float>(edge.next_node));
            flat_edges.push_back(edge.distance);
            edge_weights.push_back(edge.distance);
        }

        hdbscan_metrics::write_hdbscan_mst_metrics(
            metrics_json_out_,
            std::span<const float>(flat_edges.data(), flat_edges.size()),
            std::span<const float>(edge_weights.data(), edge_weights.size()),
            N_,
            min_samples_
        );
    }

    void write_linkage_outputs() const {
        std::vector<float> flat_tree;
        flat_tree.reserve(single_linkage_tree_.size() * 4);

        for (const auto& row : single_linkage_tree_) {
            flat_tree.push_back(static_cast<float>(row.left_node));
            flat_tree.push_back(static_cast<float>(row.right_node));
            flat_tree.push_back(row.distance);
            flat_tree.push_back(static_cast<float>(row.cluster_size));
        }

        hdbscan_metrics::write_hdbscan_linkage_metrics(
            metrics_json_out_,
            std::span<const float>(flat_tree.data(), flat_tree.size()),
            N_,
            min_samples_
        );
    }

private:
    static_hdbscan_case(
        const std::string& stage,
        const std::string& input_bin,
        std::size_t N,
        std::size_t min_samples
    )
        : stage_(stage),
          N_(N),
          min_samples_(min_samples),
          samples_(eve::algo::no_init, stage == "distance" ? N : 0) {
        if (stage_ != "distance" && stage_ != "mreach" && stage_ != "mst" && stage_ != "linkage" && stage_ != "select") {
            throw std::invalid_argument(
                "static_hdbscan_case only implements stages 'distance', 'mreach', 'mst', 'linkage', and 'select' for now"
            );
        }

        if (stage_ == "distance") {
            const auto raw_aos_data = read_aos_f32(input_bin, N_, D);
            copy_aos_to_static_samples<D>(raw_aos_data, N_, samples_);
        } else if (stage_ == "mreach") {
            distance_matrix_input_ = read_binary_f32(input_bin, N_ * N_);
        } else if (stage_ == "mst") {
            mutual_reachability_matrix_input_ = read_binary_f32(input_bin, N_ * N_);
        } else if (stage_ == "linkage") {
            mst_edges_input_ = read_mst_edges(input_bin, N_);
        } else if (stage_ == "select") {
            single_linkage_tree_input_ = read_single_linkage_tree(input_bin, N_);
        }
    }

    static std::vector<single_linkage_row> read_single_linkage_tree(
        const std::string& filename,
        std::size_t N
    ) {
        const std::size_t row_count = N == 0 ? 0 : N - 1;
        std::vector<std::int32_t> left(row_count);
        std::vector<std::int32_t> right(row_count);
        std::vector<float> distance(row_count);
        std::vector<std::int32_t> cluster_size(row_count);

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        file.read(
            reinterpret_cast<char*>(left.data()),
            static_cast<std::streamsize>(left.size() * sizeof(std::int32_t))
        );
        file.read(
            reinterpret_cast<char*>(right.data()),
            static_cast<std::streamsize>(right.size() * sizeof(std::int32_t))
        );
        file.read(
            reinterpret_cast<char*>(distance.data()),
            static_cast<std::streamsize>(distance.size() * sizeof(float))
        );
        file.read(
            reinterpret_cast<char*>(cluster_size.data()),
            static_cast<std::streamsize>(cluster_size.size() * sizeof(std::int32_t))
        );

        if (!file) {
            throw std::runtime_error("Error: Could not read expected HDBSCAN linkage data from " + filename);
        }

        std::vector<single_linkage_row> rows;
        rows.reserve(row_count);
        for (std::size_t i = 0; i < row_count; ++i) {
            rows.push_back(single_linkage_row{left[i], right[i], distance[i], cluster_size[i]});
        }
        return rows;
    }

    static std::vector<mst_edge> read_mst_edges(
        const std::string& filename,
        std::size_t N
    ) {
        const std::size_t edge_count = N == 0 ? 0 : N - 1;
        std::vector<std::int32_t> left(edge_count);
        std::vector<std::int32_t> right(edge_count);
        std::vector<float> distance(edge_count);

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        file.read(
            reinterpret_cast<char*>(left.data()),
            static_cast<std::streamsize>(left.size() * sizeof(std::int32_t))
        );
        file.read(
            reinterpret_cast<char*>(right.data()),
            static_cast<std::streamsize>(right.size() * sizeof(std::int32_t))
        );
        file.read(
            reinterpret_cast<char*>(distance.data()),
            static_cast<std::streamsize>(distance.size() * sizeof(float))
        );

        if (!file) {
            throw std::runtime_error("Error: Could not read expected HDBSCAN MST edge data from " + filename);
        }

        std::vector<mst_edge> edges;
        edges.reserve(edge_count);
        for (std::size_t i = 0; i < edge_count; ++i) {
            edges.push_back(mst_edge{left[i], right[i], distance[i]});
        }
        return edges;
    }

    std::string stage_;
    std::size_t N_ = 0;
    std::size_t min_samples_ = 0;
    static_samples_soa_vector<D> samples_;
    std::vector<float> distance_matrix_input_;
    std::vector<float> mutual_reachability_matrix_input_;
    std::vector<mst_edge> mst_edges_input_;
    std::vector<single_linkage_row> single_linkage_tree_input_;
    std::vector<float, eve::aligned_allocator<float>> distance_matrix_;
    std::vector<float, eve::aligned_allocator<float>> mutual_reachability_matrix_;
    std::vector<mst_edge> mst_edges_;
    std::vector<single_linkage_row> single_linkage_tree_;
    hdbscan_selection_result selection_result_;

    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;
};
