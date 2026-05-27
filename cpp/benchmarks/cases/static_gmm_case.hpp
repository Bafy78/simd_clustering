#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobench.h>

#include "../../include/k_means/io/binary.hpp"
#include "../../include/k_means/static_d/conversion.hpp"

struct static_gmm_case {
    static constexpr int nanobench_argc = 12;
    static constexpr int callgrind_argc = 9;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <n_samples> <n_clusters>"
              " <gmm_weights_bin> <gmm_means_bin> <gmm_precisions_bin>"
              " <covariance_type> <metrics_json_out> <nanobench_json_out>"
              " <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <n_samples> <n_clusters>"
              " <gmm_weights_bin> <gmm_means_bin> <gmm_precisions_bin>"
              " <covariance_type> <metrics_json_out>";
    }

    static static_gmm_case make_for_nanobench(int, char* argv[]) {
        static_gmm_case out{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5],
            argv[6],
            argv[7],
            argv[8]
        };

        out.nanobench_json_out_ = argv[9];
        out.bench_epochs_ = static_cast<std::size_t>(std::stoull(argv[10]));
        out.min_epoch_seconds_ = std::stod(argv[11]);

        return out;
    }

    static static_gmm_case make_for_callgrind(int, char* argv[]) {
        return static_gmm_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5],
            argv[6],
            argv[7],
            argv[8]
        };
    }

    std::string title() const {
        return "EVE GaussianMixture " + std::to_string(TUPLE_SIZE) + "D (EM)";
    }

    std::string run_name() const {
        return "gmm_em";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        throw std::runtime_error(
            "static_gmm_case is scaffold-only. Implement the static spherical EM backend "
            "before enabling the C++ GMM benchmark task."
        );
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(points_.data());
        ankerl::nanobench::doNotOptimizeAway(weights_.data());
        ankerl::nanobench::doNotOptimizeAway(initial_means_.data());
        ankerl::nanobench::doNotOptimizeAway(precisions_.data());
    }

    void write_outputs() const {}

private:
    static std::size_t precision_value_count(
        const std::string& covariance_type,
        std::size_t n_clusters
    ) {
        if (covariance_type == "spherical") {
            return n_clusters;
        }

        if (covariance_type == "diag") {
            return n_clusters * TUPLE_SIZE;
        }

        if (covariance_type == "full") {
            return n_clusters * TUPLE_SIZE * TUPLE_SIZE;
        }

        if (covariance_type == "tied") {
            return TUPLE_SIZE * TUPLE_SIZE;
        }

        throw std::runtime_error("Unsupported GMM covariance_type: " + covariance_type);
    }

    static_gmm_case(
        const std::string& dataset_bin,
        std::size_t n_samples,
        int n_clusters,
        const std::string& gmm_weights_bin,
        const std::string& gmm_means_bin,
        const std::string& gmm_precisions_bin,
        const std::string& covariance_type,
        const std::string& metrics_json_out
    )
        : n_samples_(n_samples),
          n_clusters_(n_clusters),
          covariance_type_(covariance_type),
          metrics_json_out_(metrics_json_out) {
        if (n_clusters_ <= 0) {
            throw std::runtime_error("Invalid number of GMM components");
        }

        auto raw_points = read_aos_f32(dataset_bin, n_samples_, TUPLE_SIZE);
        points_ = make_static_points_from_aos<TUPLE_SIZE>(raw_points, n_samples_);

        weights_ = read_binary_f32(
            gmm_weights_bin,
            static_cast<std::size_t>(n_clusters_)
        );

        auto raw_means = read_aos_f32(
            gmm_means_bin,
            static_cast<std::size_t>(n_clusters_),
            TUPLE_SIZE
        );
        initial_means_ = make_static_centroids_from_aos<TUPLE_SIZE>(
            raw_means,
            static_cast<std::size_t>(n_clusters_)
        );

        precisions_ = read_binary_f32(
            gmm_precisions_bin,
            precision_value_count(covariance_type_, static_cast<std::size_t>(n_clusters_))
        );
    }

    std::size_t n_samples_ = 0;
    int n_clusters_ = 0;
    std::string covariance_type_;
    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    static_points_soa_vector<TUPLE_SIZE> points_;
    std::vector<float> weights_;
    std::vector<PointType> initial_means_;
    std::vector<float> precisions_;
};
