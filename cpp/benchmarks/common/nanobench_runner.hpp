#pragma once

#include <fstream>
#include <iostream>
#include <chrono>

#include <nanobench.h>

namespace kmeans_bench {

inline std::chrono::nanoseconds min_epoch_time_from_seconds(double seconds) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(seconds)
    );
}

template<class Case>
int run_nanobench_main(int argc, char* argv[]) {
    if (argc < Case::nanobench_argc) {
        std::cerr << Case::nanobench_usage(argv[0]) << "\n";
        return 1;
    }

    auto bench_case = Case::make_for_nanobench(argc, argv);

    ankerl::nanobench::Bench bench;
    bench.title(bench_case.title())
        .unit("run")
        .warmup(1)
        .epochs(bench_case.bench_epochs())
        .minEpochTime(min_epoch_time_from_seconds(bench_case.min_epoch_seconds()))
        .performanceCounters(false)
        .output(nullptr);

    bench.run(bench_case.run_name(), [&] {
        bench_case.run_once();
        bench_case.keep_alive();
    });

    std::ofstream bench_out(bench_case.nanobench_json_out());
    bench.render(ankerl::nanobench::templates::pyperf(), bench_out);

    bench_case.write_outputs();

    return 0;
}

} // namespace kmeans_bench
