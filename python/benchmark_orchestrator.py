from benchmark_pipeline.config import default_config
from benchmark_pipeline.runner import (
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)


def cpp_cases_for_dimension(config, _D: int) -> set[str]:
    cpp_cases = set(config.cpp_soa_cases)
    cpp_cases.update(config.cpp_lloyd_cases)
    cpp_cases.update(config.cpp_gmm_cases)

    if config.run_cpp_pp:
        cpp_cases.add("pp")

    return cpp_cases


def main() -> None:
    config = default_config()
    prepare_datasets_dir(config.datasets_dir)

    for D in config.test_Ds:
        compile_cpp_binaries(D, cpp_cases_for_dimension(config, D))

        for N in config.test_Ns:
            for K in config.test_Ks:
                execute_pipeline(
                    D,
                    N,
                    K,
                    config.timing_processes,
                    config.timing_values,
                    config.timing_min_time,
                    gmm_covariance_types=config.gmm_covariance_types,
                    cpp_soa_cases=config.cpp_soa_cases,
                    run_cpp_pp=config.run_cpp_pp,
                    run_python_pp=config.run_python_pp,
                    cpp_lloyd_cases=config.cpp_lloyd_cases,
                    run_python_lloyd=config.run_python_lloyd,
                    cpp_gmm_cases=config.cpp_gmm_cases,
                    run_python_gmm=config.run_python_gmm,
                )

    print("\nAll benchmarking finished successfully!")


if __name__ == "__main__":
    main()
