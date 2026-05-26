#pragma once

#include <iostream>

#include <valgrind/callgrind.h>

namespace kmeans_bench {

template<class Case>
int run_callgrind_main(int argc, char* argv[]) {
    if (argc < Case::callgrind_argc) {
        std::cerr << Case::callgrind_usage(argv[0]) << "\n";
        return 1;
    }

    auto bench_case = Case::make_for_callgrind(argc, argv);

    // Setup is intentionally outside the measured region when Valgrind is run
    // with --instr-atstart=no.
    CALLGRIND_START_INSTRUMENTATION;

    bench_case.run_once();

    CALLGRIND_STOP_INSTRUMENTATION;

    bench_case.keep_alive();
    bench_case.write_outputs();

    return 0;
}

} // namespace kmeans_bench
