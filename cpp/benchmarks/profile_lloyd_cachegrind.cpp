#include <iostream>
#include <string>

#include <valgrind/cachegrind.h>

#include "../include/io_utils.hpp"
#include "../include/k_means/static_d/backend.hpp"

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr
            << "Usage: " << argv[0]
            << " <dataset_bin> <n_samples> <n_clusters> <init_centroids_bin> <metrics_json_out>\n";
        return 1;
    }

    std::string dataset_bin = argv[1];
    std::size_t n_samples = std::stoull(argv[2]);
    int n_clusters = std::stoi(argv[3]);
    std::string init_centroids_bin = argv[4];
    std::string metrics_json_out = argv[5];

    // Not profiled when Valgrind is run with --instr-at-start=no.
    auto points = read_dataset_soa(dataset_bin, n_samples);
    auto centroids = read_initial_centroids_binary(init_centroids_bin, n_clusters);

    int iterations_to_converge = 0;

    CACHEGRIND_START_INSTRUMENTATION;

    auto assignments = k_means(
        points,
        centroids,
        iterations_to_converge
    );

    CACHEGRIND_STOP_INSTRUMENTATION;

    // Also outside the measured region.
    write_lloyd_metrics(
        metrics_json_out,
        points,
        centroids,
        assignments,
        n_clusters,
        iterations_to_converge
    );

    std::cout << "iterations=" << iterations_to_converge << "\n";
    return 0;
}