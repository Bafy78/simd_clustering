#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobench.h>

#include "../../include/io/binary.hpp"
#include "../../include/k_means/lloyd_dispatch.hpp"
#include "../../include/k_means/metrics.hpp"
#include "../../include/k_means/static_d/backend.hpp"
#include "../../include/k_means/static_d/input.hpp"
#include "../../include/k_means/dynamic_d/input.hpp"

#ifndef KMEANS_K_TILE
#define KMEANS_K_TILE 5
#endif

struct auto_lloyd_case {
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

    static auto_lloyd_case make_for_nanobench(int, char* argv[]) {
        auto_lloyd_case out{
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

    static auto_lloyd_case make_for_callgrind(int, char* argv[]) {
        return auto_lloyd_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5]
        };
    }

    std::string title() const {
        std::string selected_impl;

        if (impl_ == kmeans::lloyd_dispatch::lloyd_impl::dynamic_d_micro_gemm) {
            selected_impl = "dynamic-D micro-GEMM N_VECTORS="
                + std::to_string(
                    kmeans::lloyd_dispatch::micro_gemm_n_vectors_for_register_file()
                )
                + ", K_TILE="
                + std::to_string(KMEANS_K_TILE);
        } else {
            selected_impl = "static-D";
        }

        return "EVE K-Means "
            + std::to_string(D)
            + "D auto Lloyd dispatch ("
            + selected_impl
            + ", threshold="
            + std::to_string(threshold_)
            + ", max_static_D="
            + std::to_string(kmeans::lloyd_dispatch::max_static_lloyd_d)
            + ")";
    }

    std::string run_name() const {
        return "kmeans_lloyd";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        if (impl_ == kmeans::lloyd_dispatch::lloyd_impl::dynamic_d_micro_gemm) {
            run_dynamic_micro_gemm();
        } else {
            run_static();
        }
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.data());
        ankerl::nanobench::doNotOptimizeAway(final_centroids_.size());
        ankerl::nanobench::doNotOptimizeAway(final_assignments_.data());
        ankerl::nanobench::doNotOptimizeAway(final_assignments_.size());
    }

    void write_outputs() const {
        write_lloyd_metrics(
            metrics_json_out_,
            static_samples_for_metrics_,
            final_centroids_,
            final_assignments_,
            K_,
            algorithm_iterations_
        );
    }

private:
    auto_lloyd_case(
        const std::string& dataset_bin,
        std::size_t N,
        int K,
        const std::string& init_centroids_bin,
        const std::string& metrics_json_out
    )
        : N_(N),
          K_(K),
          metrics_json_out_(metrics_json_out) {
        if (K_ <= 0) {
            throw std::runtime_error("Invalid number of clusters");
        }

        kmeans::check_cluster_count(static_cast<std::size_t>(K_), N_);

        auto raw_samples = read_aos_f32(dataset_bin, N_, D);
        auto raw_initial_centroids = read_aos_f32(
            init_centroids_bin,
            static_cast<std::size_t>(K_),
            D
        );

        static_samples_for_metrics_ = make_static_samples_from_aos<D>(
            raw_samples,
            N_
        );

        threshold_ = kmeans::lloyd_dispatch::dynamic_micro_gemm_threshold(
            static_cast<std::size_t>(K_),
            N_
        );

        impl_ = kmeans::lloyd_dispatch::choose_lloyd_impl(
            static_cast<std::size_t>(K_),
            N_,
            D
        );

        if (impl_ == kmeans::lloyd_dispatch::lloyd_impl::dynamic_d_micro_gemm) {
            initialize_dynamic_inputs(raw_samples, raw_initial_centroids);
        } else {
            initialize_static_inputs(raw_initial_centroids);
        }
    }

    void initialize_dynamic_inputs(
        const std::vector<float>& raw_samples,
        const std::vector<float>& raw_initial_centroids
    ) {
        dynamic_samples_storage_ = make_dynamic_samples_from_aos<D>(
            raw_samples,
            N_
        );

        dynamic_initial_centroids_ = make_dynamic_centroids_from_aos<D>(
            raw_initial_centroids,
            static_cast<std::size_t>(K_)
        );
    }

    template<std::size_t Dim = D>
    void initialize_static_inputs(const std::vector<float>& raw_initial_centroids)
        requires(kmeans::lloyd_dispatch::static_lloyd_enabled_v<Dim>)
    {
        static_assert(Dim == D);

        static_initial_centroids_ = make_static_centroids_from_aos<Dim>(
            raw_initial_centroids,
            static_cast<std::size_t>(K_)
        );
    }

    template<std::size_t Dim = D>
    void initialize_static_inputs(const std::vector<float>&)
        requires(!kmeans::lloyd_dispatch::static_lloyd_enabled_v<Dim>)
    {
        throw std::runtime_error("static Lloyd is disabled for D > 50");
    }

    template<std::size_t Dim = D>
    void run_static()
        requires(kmeans::lloyd_dispatch::static_lloyd_enabled_v<Dim>)
    {
        static_assert(Dim == D);

        std::vector<static_sample_type<Dim>> current_centroids = static_initial_centroids_;

        auto assignments = k_means(
            static_samples_for_metrics_,
            current_centroids,
            algorithm_iterations_
        );

        final_centroids_ = std::move(current_centroids);
        final_assignments_ = std::move(assignments);
    }

    template<std::size_t Dim = D>
    void run_static()
        requires(!kmeans::lloyd_dispatch::static_lloyd_enabled_v<Dim>)
    {
        throw std::runtime_error("static Lloyd is disabled for D > 50");
    }

    void run_dynamic_micro_gemm() {
        centroids_storage<D> current_centroids = dynamic_initial_centroids_;

        auto assignments =
            kmeans::lloyd_dispatch::k_means_micro_gemm_auto_n_vectors<D, KMEANS_K_TILE>(
                dynamic_samples_storage_.view(),
                current_centroids,
                algorithm_iterations_
            );

        final_centroids_ = make_static_centroids_from_dynamic(current_centroids);
        final_assignments_ = std::move(assignments);
    }

    std::size_t N_ = 0;
    int K_ = 0;
    int threshold_ = 0;
    kmeans::lloyd_dispatch::lloyd_impl impl_ = kmeans::lloyd_dispatch::lloyd_impl::static_d;

    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    static_samples_soa_vector<D> static_samples_for_metrics_;
    std::vector<SampleType> static_initial_centroids_;

    samples_soa_storage<D> dynamic_samples_storage_;
    centroids_storage<D> dynamic_initial_centroids_;

    std::vector<SampleType> final_centroids_;
    aligned_int_vector final_assignments_;
    int algorithm_iterations_ = 0;
};
