#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobench.h>

#include "../../include/hdbscan/static_d/distance.hpp"
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
            hdbscan_static_euclidean_distance_matrix<D>(
                raw_aos_data_,
                N_,
                distance_matrix_
            );
            return;
        }

        throw std::invalid_argument(
            "static_hdbscan_case only implements stage 'distance' for now"
        );
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(distance_matrix_.data());
        ankerl::nanobench::doNotOptimizeAway(distance_matrix_.size());
        ankerl::nanobench::doNotOptimizeAway(min_samples_);
    }

    void write_outputs() const {
        if (stage_ == "distance") {
            hdbscan_metrics::write_hdbscan_distance_metrics(
                metrics_json_out_,
                distance_matrix_,
                N_,
                min_samples_
            );
        }
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
          min_samples_(min_samples) {
        if (stage_ != "distance") {
            throw std::invalid_argument(
                "static_hdbscan_case only implements stage 'distance' for now"
            );
        }

        raw_aos_data_ = read_aos_f32(input_bin, N_, D);
    }

    std::string stage_;
    std::size_t N_ = 0;
    std::size_t min_samples_ = 0;
    std::vector<float> raw_aos_data_;
    std::vector<float> distance_matrix_;

    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;
};
