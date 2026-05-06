#include <iostream>
#include <vector>
#include <chrono>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "../include/io_utils.hpp"

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                << " <binary_file> <n_samples> <nanobench_json_out> <bench_epochs> <min_epoch_seconds>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::size_t n_samples = std::stoull(argv[2]);
    std::string nanobench_out = argv[3];
    std::size_t bench_epochs = std::stoull(argv[4]);
    double min_epoch_seconds = std::stod(argv[5]);

    auto min_epoch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(min_epoch_seconds)
    );

    // Read Raw Binary Data (AoS Format) - Outside the benchmark
    std::vector<float> raw_aos_data(n_samples * TUPLE_SIZE);
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(raw_aos_data.data()), raw_aos_data.size() * sizeof(float));

    ankerl::nanobench::Bench bench;
    bench.title("EVE K-Means " + std::to_string(TUPLE_SIZE) + "D (AoS to SoA)")
         .unit("run").warmup(1).epochs(bench_epochs).minEpochTime(min_epoch_time)
         .performanceCounters(false).output(nullptr);

    eve::algo::soa_vector<PointType> points(n_samples);

    bench.run("aos_to_soa", [&] {
        for (std::size_t i = 0; i < n_samples; ++i) {
            PointType pt;
            kumi::for_each_index([&](auto index, auto& element) {
                element = raw_aos_data[i * TUPLE_SIZE + index];
            }, pt);
            points.set(i, pt);
        }
        ankerl::nanobench::doNotOptimizeAway(points);
    });
    
    std::ofstream bench_out(nanobench_out);
    bench.render(ankerl::nanobench::templates::pyperf(), bench_out);
    return 0;
}