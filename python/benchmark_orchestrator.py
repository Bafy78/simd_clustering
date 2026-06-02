from benchmark_pipeline.config import default_config
from benchmark_pipeline.runner import (
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)
from benchmark_pipeline.tasks import build_pipeline


def cpp_cases_for_dimension(config, D: int) -> set[str]:
    cpp_cases: set[str] = set()

    for N in config.test_Ns:
        for K in config.test_Ks:
            pipeline = build_pipeline(
                D,
                N,
                K,
                config.timing_processes,
                config.timing_values,
                config.timing_min_time,
                gmm_covariance_type=config.gmm_covariance_type,
            )

            for task in pipeline:
                if task.cpp_case is not None:
                    cpp_cases.add(task.cpp_case)

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
                    config.lloyd_parity_tolerance_pct,
                    gmm_covariance_type=config.gmm_covariance_type,
                )

    print("\nAll benchmarking finished successfully!")


if __name__ == "__main__":
    main()
