#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <nanobench.h>

#include "../../include/io/binary.hpp"
#include "../../include/k_means/metrics.hpp"
#include "../../include/k_means/static_d/backend.hpp"
#include "../../include/k_means/static_d/input.hpp"

struct static_lloyd_case {
    static constexpr int nanobench_argc = 9;
    static constexpr int callgrind_argc = 6;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <binary_file> <N> <K> <init_centroids_bin>"
              " <metrics_json_out> <nanobench_json_out> <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <N> <K> <init_centroids_bin> <metrics_json_out>";
    }

    static static_lloyd_case make_for_nanobench(int, char* argv[]) {
        static_lloyd_case out{
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

    static static_lloyd_case make_for_callgrind(int, char* argv[]) {
        return static_lloyd_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5]
        };
    }

    std::string title() const {
        return "EVE K-Means " + std::to_string(D) + "D (Lloyd Algorithm)";
    }

    std::string run_name() const {
        return "kmeans_lloyd";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        std::vector<SampleType> current_centroids = initial_centroids_;
        auto assignments = k_means(samples_, current_centroids, algorithm_iterations_);

        final_inertia_ = compute_static_lloyd_total_inertia_simd(
            samples_,
            current_centroids,
            assignments
        );

        final_centroids_ = std::move(current_centroids);
        final_assignments_ = std::move(assignments);
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.data());
        ankerl::nanobench::doNotOptimizeAway(final_assignments_.data());
        ankerl::nanobench::doNotOptimizeAway(final_inertia_);
    }

    void write_outputs() const {
        write_lloyd_metrics(
            metrics_json_out_,
            samples_,
            final_centroids_,
            final_assignments_,
            K_,
            algorithm_iterations_
        );
    }

private:
    static_lloyd_case(
        const std::string& dataset_bin,
        std::size_t N,
        int K,
        const std::string& init_centroids_bin,
        const std::string& metrics_json_out
    )
        : N_(N),
          K_(K),
          metrics_json_out_(metrics_json_out) {
        auto raw_samples = read_aos_f32(dataset_bin, N_, D);
        auto raw_initial_centroids = read_aos_f32(
            init_centroids_bin,
            static_cast<std::size_t>(K_),
            D
        );

        samples_ = make_static_samples_from_aos<D>(raw_samples, N_);
        initial_centroids_ = make_static_centroids_from_aos<D>(
            raw_initial_centroids,
            static_cast<std::size_t>(K_)
        );
    }

    std::size_t N_ = 0;
    int K_ = 0;
    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    static_samples_soa_vector<D> samples_;
    std::vector<SampleType> initial_centroids_;

    std::vector<SampleType> final_centroids_;
    kumi_kmeans_backend<SampleType>::assignment_vector final_assignments_;
    double final_inertia_ = 0.0;
    int algorithm_iterations_ = 0;
};
