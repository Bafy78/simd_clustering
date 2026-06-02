#include <iostream>

#ifndef BENCH_CASE_HEADER
#error "BENCH_CASE_HEADER must be defined to the case header path, e.g. \"cases/static_lloyd_case.hpp\""
#endif

#ifndef BENCH_CASE
#error "BENCH_CASE must be defined to the case type, e.g. static_lloyd_case"
#endif

#include BENCH_CASE_HEADER

int main(int argc, char* argv[]) {
    if (argc < BENCH_CASE::callgrind_argc) {
        std::cerr << BENCH_CASE::callgrind_usage(argv[0]) << "\n";
        return 1;
    }

#if defined(SPILL_GMM_COVARIANCE_SPHERICAL)
    auto bench_case = BENCH_CASE::make_for_spill_detector_spherical(argc, argv);
    bench_case.run_once_spherical();
#elif defined(SPILL_GMM_COVARIANCE_DIAG)
    auto bench_case = BENCH_CASE::make_for_spill_detector_diag(argc, argv);
    bench_case.run_once_diag();
#else
    auto bench_case = BENCH_CASE::make_for_callgrind(argc, argv);
    bench_case.run_once();
#endif
    bench_case.keep_alive();

    return 0;
}
