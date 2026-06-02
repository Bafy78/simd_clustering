#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <nanobench.h>

#include "../../include/io/binary.hpp"
#include "../../include/layout/static_soa.hpp"

struct static_soa_case {
    static constexpr int nanobench_argc = 6;
    static constexpr int callgrind_argc = 3;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <binary_file> <N> <nanobench_json_out>"
              " <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <N>";
    }

    static static_soa_case make_for_nanobench(int, char* argv[]) {
        static_soa_case out{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2]))
        };

        out.nanobench_json_out_ = argv[3];
        out.bench_epochs_ = static_cast<std::size_t>(std::stoull(argv[4]));
        out.min_epoch_seconds_ = std::stod(argv[5]);

        return out;
    }

    static static_soa_case make_for_callgrind(int, char* argv[]) {
        return static_soa_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2]))
        };
    }

    std::string title() const {
        return "EVE K-Means " + std::to_string(D) + "D (AoS to Native Layout)";
    }

    std::string run_name() const {
        return "aos_to_static_soa";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once() {
        copy_aos_to_static_samples<D>(raw_aos_data_, N_, samples_);
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(samples_);
    }

    void write_outputs() const {}

private:
    static_soa_case(const std::string& dataset_bin, std::size_t N)
        : N_(N),
          raw_aos_data_(read_aos_f32(dataset_bin, N_, D)),
          samples_(eve::algo::no_init, N_) {}

    std::size_t N_ = 0;
    std::vector<float> raw_aos_data_;
    static_samples_soa_vector<D> samples_;

    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;
};
