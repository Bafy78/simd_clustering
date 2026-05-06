#include <iostream>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "../include/io_utils.hpp"
#include "../include/kmeans_lloyd.hpp"

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <binary_file> <n_samples> <n_clusters> <init_centroids_bin> <output_file> <nanobench_json_out>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::size_t n_samples = std::stoull(argv[2]);
    int n_clusters = std::stoi(argv[3]);
    std::string init_centroids_bin = argv[4];
    std::string out_filename = argv[5];
    std::string nanobench_out = argv[6];

    // Setup: Convert to SoA and read Orchestrator centroids (not benchmarked)
    auto points = read_dataset_soa(filename, n_samples);
    auto initial_centroids = read_initial_centroids_binary(init_centroids_bin, n_clusters);

    ankerl::nanobench::Bench bench;
    bench.title("EVE K-Means " + std::to_string(TUPLE_SIZE) + "D (Lloyd Iterations)")
         .unit("run").warmup(1).epochs(20).performanceCounters(false).output(nullptr);
    
    std::vector<PointType> final_centroids;
    std::vector<int, eve::aligned_allocator<int>> final_assignments;
    int iterations_to_converge = 0;

    bench.run("kmeans_lloyd", [&] {
        // Crucial: Copy the initial state for every epoch
        std::vector<PointType> current_centroids = initial_centroids;
        auto centroid_assignments = k_means(points, current_centroids, iterations_to_converge);

        ankerl::nanobench::doNotOptimizeAway(current_centroids.data());
        ankerl::nanobench::doNotOptimizeAway(centroid_assignments.data());

        final_centroids.swap(current_centroids);
        final_assignments.swap(centroid_assignments);
    });

    std::ofstream bench_out(nanobench_out);
    bench.render(ankerl::nanobench::templates::pyperf(), bench_out);
    
    // Save final results for verification
    write_results(out_filename, final_centroids, final_assignments, n_clusters, iterations_to_converge);

    return 0;
}