#include "common/callgrind_runner.hpp"

#ifndef BENCH_CASE_HEADER
#error "BENCH_CASE_HEADER must be defined to the case header path, e.g. \"cases/static_lloyd_case.hpp\""
#endif

#ifndef BENCH_CASE
#error "BENCH_CASE must be defined to the case type, e.g. static_lloyd_case"
#endif

#include BENCH_CASE_HEADER

int main(int argc, char* argv[]) {
    return kmeans_bench::run_callgrind_main<BENCH_CASE>(argc, argv);
}
