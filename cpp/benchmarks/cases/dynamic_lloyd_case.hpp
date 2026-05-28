#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <nanobench.h>

#include "../../include/k_means/dynamic_d/backend.hpp"
#include "../../include/k_means/dynamic_d/input.hpp"
#include "../../include/io/binary.hpp"
#include "../../include/k_means/metrics.hpp"
#include "../../include/layout/static_soa.hpp"

#ifndef KMEANS_K_TILE
#define KMEANS_K_TILE 5
#endif

#ifndef KMEANS_M_GROUP
#define KMEANS_M_GROUP 2
#endif

struct dynamic_lloyd_case {
    static constexpr int nanobench_argc = 9;
    static constexpr int callgrind_argc = 6;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <binary_file> <n_samples> <n_clusters> <init_centroids_bin>"
              " <metrics_json_out> <nanobench_json_out> <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <n_samples> <n_clusters> <init_centroids_bin> <metrics_json_out>";
    }

    static dynamic_lloyd_case make_for_nanobench(int, char* argv[]) {
        dynamic_lloyd_case out{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5]
        };

        out.nanobench_json_out_ = argv[6];
        out.bench_epochs_ = static_cast<std::size_t>(std::stoull(argv[7]));
        out.min_epoch_seconds_ = std::stod(argv[8]);

        return out;
    }

    static dynamic_lloyd_case make_for_callgrind(int, char* argv[]) {
        return dynamic_lloyd_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5]
        };
    }

    std::string title() const {
        return "EVE Static-D Streamed/Tiled K-Means "
            + std::to_string(TUPLE_SIZE)
            + "D K_TILE="
            + std::to_string(KMEANS_K_TILE)
            + " (Lloyd Iterations)";
    }

    std::string run_name() const {
        return "kmeans_lloyd";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        centroids_storage<TUPLE_SIZE> current_centroids = initial_centroids_;

        auto assignments = k_means_micro_gemm<TUPLE_SIZE, KMEANS_K_TILE, KMEANS_M_GROUP>(
            points_storage_.view(),
            current_centroids,
            iterations_to_converge_
        );

        final_centroids_ = std::move(current_centroids);
        final_assignments_ = std::move(assignments);
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.row_major.data());
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.row_major.size());
        ankerl::nanobench::doNotOptimizeAway(final_assignments_.data());
        ankerl::nanobench::doNotOptimizeAway(final_assignments_.size());
    }

    void write_outputs() const {
        auto final_static_centroids = make_static_centroids_from_dynamic(final_centroids_);

        write_lloyd_metrics(
            metrics_json_out_,
            static_points_for_metrics_,
            final_static_centroids,
            final_assignments_,
            n_clusters_,
            iterations_to_converge_
        );
    }

private:
    dynamic_lloyd_case(
        const std::string& dataset_bin,
        std::size_t n_samples,
        int n_clusters,
        const std::string& init_centroids_bin,
        const std::string& metrics_json_out
    )
        : n_samples_(n_samples),
          n_clusters_(n_clusters),
          metrics_json_out_(metrics_json_out) {
        auto raw_points = read_aos_f32(dataset_bin, n_samples_, TUPLE_SIZE);
        auto raw_initial_centroids = read_aos_f32(
            init_centroids_bin,
            static_cast<std::size_t>(n_clusters_),
            TUPLE_SIZE
        );

        points_storage_ = make_dynamic_points_from_aos<TUPLE_SIZE>(raw_points, n_samples_);
        initial_centroids_ = make_dynamic_centroids_from_aos<TUPLE_SIZE>(
            raw_initial_centroids,
            static_cast<std::size_t>(n_clusters_)
        );

        // Metrics compatibility only. This is intentionally not used to prepare
        // the dynamic algorithm.
        static_points_for_metrics_ = make_static_points_from_aos<TUPLE_SIZE>(
            raw_points,
            n_samples_
        );
    }

    std::size_t n_samples_ = 0;
    int n_clusters_ = 0;
    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    points_soa_storage<TUPLE_SIZE> points_storage_;
    centroids_storage<TUPLE_SIZE> initial_centroids_;
    static_points_soa_vector<TUPLE_SIZE> static_points_for_metrics_;

    centroids_storage<TUPLE_SIZE> final_centroids_;
    aligned_int_vector final_assignments_;
    int iterations_to_converge_ = 0;
};
