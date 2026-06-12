#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <nanobench.h>

#include "../../include/io/binary.hpp"
#include "../../include/k_means/dynamic_d/greedy_pp.hpp"
#include "../../include/k_means/dynamic_d/input.hpp"

#ifndef KMEANS_PP_N_VECTORS
#define KMEANS_PP_N_VECTORS 2
#endif

#ifndef KMEANS_PP_LOCAL_TRIAL_TILE
#define KMEANS_PP_LOCAL_TRIAL_TILE 5
#endif

struct dynamic_pp_case {
    static constexpr int nanobench_argc = 7;
    static constexpr int callgrind_argc = 4;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <binary_file> <N> <K> <nanobench_json_out>"
              " <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <N> <K>";
    }

    static dynamic_pp_case make_for_nanobench(int, char* argv[]) {
        dynamic_pp_case out{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3])
        };

        out.nanobench_json_out_ = argv[4];
        out.bench_epochs_ = static_cast<std::size_t>(std::stoull(argv[5]));
        out.min_epoch_seconds_ = std::stod(argv[6]);

        return out;
    }

    static dynamic_pp_case make_for_callgrind(int, char* argv[]) {
        return dynamic_pp_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3])
        };
    }

    std::string title() const {
        return "EVE K-Means "
            + std::to_string(D)
            + "D dynamic-D tiled K-Means++ Init N_VECTORS="
            + std::to_string(KMEANS_PP_N_VECTORS)
            + ", LOCAL_TRIAL_TILE="
            + std::to_string(KMEANS_PP_LOCAL_TRIAL_TILE);
    }

    std::string run_name() const {
        return "kmeans_pp";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        final_centroids_ = greedy_kmeans_pp_init_dynamic<
            D,
            KMEANS_PP_N_VECTORS,
            KMEANS_PP_LOCAL_TRIAL_TILE
        >(samples_storage_.view(), K_);
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.row_major.data());
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.row_major.size());
    }

    void write_outputs() const {}

private:
    dynamic_pp_case(
        const std::string& dataset_bin,
        std::size_t N,
        int K
    )
        : N_(N),
          K_(K) {
        auto raw_samples = read_aos_f32(dataset_bin, N_, D);
        samples_storage_ = make_dynamic_samples_from_aos<D>(raw_samples, N_);
    }

    std::size_t N_ = 0;
    int K_ = 0;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    samples_soa_storage<D> samples_storage_;
    centroids_storage<D> final_centroids_;
};
