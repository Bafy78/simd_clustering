#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <nanobench.h>

#include "../../include/greedy_k-means_pp.hpp"
#include "../../include/k_means/io/binary.hpp"
#include "../../include/k_means/static_d/conversion.hpp"

struct static_pp_case {
    static constexpr int nanobench_argc = 7;
    static constexpr int callgrind_argc = 4;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <binary_file> <n_samples> <n_clusters> <nanobench_json_out>"
              " <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <n_samples> <n_clusters>";
    }

    static static_pp_case make_for_nanobench(int, char* argv[]) {
        static_pp_case out{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3])
        };

        out.nanobench_json_out_ = argv[4];
        out.bench_epochs_ = static_cast<std::size_t>(std::stoull(argv[5]));
        out.min_epoch_seconds_ = std::stod(argv[6]);

        return out;
    }

    static static_pp_case make_for_callgrind(int, char* argv[]) {
        return static_pp_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3])
        };
    }

    std::string title() const {
        return "EVE K-Means " + std::to_string(TUPLE_SIZE) + "D (K-Means++ Init)";
    }

    std::string run_name() const {
        return "kmeans_pp";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        final_centroids_ = greedy_kmeans_pp_init(points_, n_clusters_);
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.data());
    }

    void write_outputs() const {}

private:
    static_pp_case(
        const std::string& dataset_bin,
        std::size_t n_samples,
        int n_clusters
    )
        : n_samples_(n_samples),
          n_clusters_(n_clusters) {
        auto raw_points = read_aos_f32(dataset_bin, n_samples_, TUPLE_SIZE);
        points_ = make_static_points_from_aos<TUPLE_SIZE>(raw_points, n_samples_);
    }

    std::size_t n_samples_ = 0;
    int n_clusters_ = 0;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    static_points_soa_vector<TUPLE_SIZE> points_;
    std::vector<PointType> final_centroids_;
};
