#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobench.h>

#include "../../include/gmm/covariance_type.hpp"
#include "../../include/gmm/dynamic_d/diagonal_covariance.hpp"
#include "../../include/gmm/dynamic_d/em.hpp"
#include "../../include/gmm/dynamic_d/full_covariance.hpp"
#include "../../include/gmm/dynamic_d/input.hpp"
#include "../../include/gmm/dynamic_d/spherical_covariance.hpp"
#include "../../include/gmm/metrics.hpp"
#include "../../include/io/binary.hpp"

#ifndef GMM_N_GROUP
#define GMM_N_GROUP 2
#endif

#ifndef GMM_K_TILE
#define GMM_K_TILE 5
#endif

struct dynamic_gmm_case {
    static constexpr int nanobench_argc = 12;
    static constexpr int callgrind_argc = 9;

    static std::string nanobench_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <N> <K>"
              " <gmm_weights_bin> <gmm_means_bin> <gmm_precisions_bin>"
              " <covariance_type> <metrics_json_out> <nanobench_json_out>"
              " <bench_epochs> <min_epoch_seconds>";
    }

    static std::string callgrind_usage(const char* program) {
        return std::string("Usage: ") + program
            + " <dataset_bin> <N> <K>"
              " <gmm_weights_bin> <gmm_means_bin> <gmm_precisions_bin>"
              " <covariance_type> <metrics_json_out>";
    }

    static dynamic_gmm_case make_for_nanobench(int, char* argv[]) {
        dynamic_gmm_case out{
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

    static dynamic_gmm_case make_for_callgrind(int, char* argv[]) {
        return dynamic_gmm_case{
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

    static dynamic_gmm_case make_for_spill_detector_spherical(int, char* argv[]) {
        return dynamic_gmm_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5],
            argv[6],
            gmm_covariance_type::spherical,
            argv[8]
        };
    }

    static dynamic_gmm_case make_for_spill_detector_diag(int, char* argv[]) {
        return dynamic_gmm_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5],
            argv[6],
            gmm_covariance_type::diag,
            argv[8]
        };
    }

    static dynamic_gmm_case make_for_spill_detector_full(int, char* argv[]) {
        return dynamic_gmm_case{
            argv[1],
            static_cast<std::size_t>(std::stoull(argv[2])),
            std::stoi(argv[3]),
            argv[4],
            argv[5],
            argv[6],
            gmm_covariance_type::full,
            argv[8]
        };
    }

    std::string title() const {
        return "EVE Dynamic-D Micro-GEMM GaussianMixture "
            + std::to_string(D)
            + "D N_GROUP="
            + std::to_string(GMM_N_GROUP)
            + " K_TILE="
            + std::to_string(GMM_K_TILE)
            + " (EM)";
    }

    std::string run_name() const {
        return "gmm_em";
    }

    const std::string& nanobench_json_out() const { return nanobench_json_out_; }
    std::size_t bench_epochs() const { return bench_epochs_; }
    double min_epoch_seconds() const { return min_epoch_seconds_; }

    void run_once_spherical() {
        auto result = run_dynamic_gmm_micro_gemm_em<D, GMM_N_GROUP, GMM_K_TILE>(
            samples_storage_.view(),
            initial_weights_,
            initial_means_row_major_,
            dynamic_spherical_gmm_micro_gemm_covariance<D, GMM_K_TILE>{
                initial_precisions_,
                static_cast<std::size_t>(K_)
            }
        );
        store_result(std::move(result));
    }

    void run_once_diag() {
        auto result = run_dynamic_gmm_micro_gemm_em<D, GMM_N_GROUP, GMM_K_TILE>(
            samples_storage_.view(),
            initial_weights_,
            initial_means_row_major_,
            dynamic_diagonal_gmm_micro_gemm_covariance<D, GMM_K_TILE>{
                initial_precisions_,
                static_cast<std::size_t>(K_)
            }
        );
        store_result(std::move(result));
    }

    void run_once_full() {
        auto result = run_dynamic_gmm_micro_gemm_em<D, GMM_N_GROUP, GMM_K_TILE>(
            samples_storage_.view(),
            initial_weights_,
            initial_means_row_major_,
            dynamic_full_gmm_micro_gemm_covariance<D, GMM_K_TILE>{
                initial_precisions_,
                static_cast<std::size_t>(K_)
            }
        );
        store_result(std::move(result));
    }

    void run_once() {
        switch (covariance_type_) {
        case gmm_covariance_type::spherical:
            run_once_spherical();
            return;
        case gmm_covariance_type::diag:
            run_once_diag();
            return;
        case gmm_covariance_type::full:
            run_once_full();
            return;
        }

        throw std::runtime_error("Unknown dynamic GMM covariance_type");
    }

    void keep_alive() const {
        ankerl::nanobench::doNotOptimizeAway(samples_storage_.data.data());
        ankerl::nanobench::doNotOptimizeAway(final_weights_.data());
        ankerl::nanobench::doNotOptimizeAway(final_means_row_major_.data());
        ankerl::nanobench::doNotOptimizeAway(final_covariances_.data());
        ankerl::nanobench::doNotOptimizeAway(final_precisions_.data());
    }

    void write_outputs() const {
        auto final_static_means = make_static_gmm_means_from_dynamic<D>(
            final_means_row_major_,
            static_cast<std::size_t>(K_)
        );

        write_gmm_metrics(
            metrics_json_out_,
            final_weights_,
            final_static_means,
            final_covariances_,
            lower_bounds_,
            algorithm_iterations_,
            lower_bound_,
            covariance_type_
        );
    }

private:
    void store_result(dynamic_gmm_result<D>&& result) {
        final_weights_ = std::move(result.weights);
        final_means_row_major_ = std::move(result.means_row_major);
        final_covariances_ = std::move(result.covariances);
        final_precisions_ = std::move(result.precisions);
        lower_bounds_ = std::move(result.lower_bounds);
        algorithm_iterations_ = result.algorithm_iterations;
        lower_bound_ = result.lower_bound;
    }

    dynamic_gmm_case(
        const std::string& dataset_bin,
        std::size_t N,
        int K,
        const std::string& gmm_weights_bin,
        const std::string& gmm_means_bin,
        const std::string& gmm_precisions_bin,
        const std::string& covariance_type,
        const std::string& metrics_json_out
    )
        : dynamic_gmm_case(
            dataset_bin,
            N,
            K,
            gmm_weights_bin,
            gmm_means_bin,
            gmm_precisions_bin,
            parse_gmm_covariance_type(covariance_type),
            metrics_json_out
        ) {}

    dynamic_gmm_case(
        const std::string& dataset_bin,
        std::size_t N,
        int K,
        const std::string& gmm_weights_bin,
        const std::string& gmm_means_bin,
        const std::string& gmm_precisions_bin,
        gmm_covariance_type covariance_type,
        const std::string& metrics_json_out
    )
        : N_(N),
          K_(K),
          covariance_type_(covariance_type),
          metrics_json_out_(metrics_json_out) {
        if (K_ <= 0) {
            throw std::runtime_error("Invalid number of GMM clusters");
        }

        samples_storage_ = read_dynamic_gmm_samples_binary<D>(dataset_bin, N_);

        initial_weights_ = read_binary_f32(
            gmm_weights_bin,
            static_cast<std::size_t>(K_)
        );

        initial_means_row_major_ = read_dynamic_gmm_means_binary<D>(
            gmm_means_bin,
            K_
        );

        initial_precisions_ = read_binary_f32(
            gmm_precisions_bin,
            gmm_precision_value_count(
                covariance_type_,
                static_cast<std::size_t>(K_),
                D
            )
        );
    }

    std::size_t N_ = 0;
    int K_ = 0;
    gmm_covariance_type covariance_type_ = gmm_covariance_type::spherical;
    std::string metrics_json_out_;
    std::string nanobench_json_out_;
    std::size_t bench_epochs_ = 0;
    double min_epoch_seconds_ = 0.0;

    samples_soa_storage<D> samples_storage_;
    std::vector<float> initial_weights_;
    aligned_float_vector initial_means_row_major_;
    std::vector<float> initial_precisions_;

    std::vector<float> final_weights_;
    aligned_float_vector final_means_row_major_;
    std::vector<float> final_covariances_;
    std::vector<float> final_precisions_;
    std::vector<float> lower_bounds_;
    int algorithm_iterations_ = 0;
    float lower_bound_ = 0.0f;
};
