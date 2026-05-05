#include <iostream>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "../include/io_utils.hpp"
#include "../include/greedy_k-means_pp.hpp"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <binary_file> <n_samples> <n_clusters> <nanobench_json_out>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::size_t n_samples = std::stoull(argv[2]);
    int n_clusters = std::stoi(argv[3]);
    std::string nanobench_out = argv[4];

    // Read and prepare SoA data (not benchmarked)
    auto points = read_dataset_soa(filename, n_samples);

    ankerl::nanobench::Bench bench;
    bench.title("EVE K-Means " + std::to_string(TUPLE_SIZE) + "D (K-Means++ Init)")
         .unit("run").warmup(1).epochs(20).performanceCounters(false).output(nullptr);
    
    bench.run("kmeans_pp", [&] {
        std::vector<PointType> centroids = greedy_kmeans_pp_init(points, n_clusters);
        ankerl::nanobench::doNotOptimizeAway(centroids.data());
    });
    
    std::ofstream bench_out(nanobench_out);
    bench.render(ankerl::nanobench::templates::pyperf(), bench_out);
    return 0;
}