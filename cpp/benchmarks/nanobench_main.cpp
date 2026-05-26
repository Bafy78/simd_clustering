#define ANKERL_NANOBENCH_IMPLEMENT

#include "common/nanobench_runner.hpp"

#ifndef KMEANS_BENCH_CASE_HEADER
#error "KMEANS_BENCH_CASE_HEADER must be defined to the case header path, e.g. \"cases/static_lloyd_case.hpp\""
#endif

#ifndef KMEANS_BENCH_CASE
#error "KMEANS_BENCH_CASE must be defined to the case type, e.g. static_lloyd_case"
#endif

#include KMEANS_BENCH_CASE_HEADER

int main(int argc, char* argv[]) {
    return kmeans_bench::run_nanobench_main<KMEANS_BENCH_CASE>(argc, argv);
}
