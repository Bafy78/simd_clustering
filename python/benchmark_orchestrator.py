from benchmark_pipeline.config import BenchmarkConfig, default_config
from benchmark_pipeline.runner import (
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)
from benchmark_pipeline.tasks import BenchmarkCase


def cpp_cases_for_dimension(config: BenchmarkConfig, _D: int) -> set[str]:
    options = config.pipeline
    cpp_cases = set(options.cpp_soa_cases)
    cpp_cases.update(options.cpp_pp_cases)
    cpp_cases.update(options.cpp_lloyd_cases)
    cpp_cases.update(options.cpp_gmm_cases)

    return cpp_cases


def main() -> None:
    config = default_config()
    datasets_dir = prepare_datasets_dir(config.datasets_dir)

    for D in config.test_Ds:
        compile_cpp_binaries(D, cpp_cases_for_dimension(config, D))

        for N in config.test_Ns:
            for K in config.test_Ks:
                execute_pipeline(
                    BenchmarkCase(D, N, K),
                    config.pipeline,
                    datasets_dir=datasets_dir,
                    keep_inputs=config.keep_inputs,
                )

    print("\nAll benchmarking finished successfully!")


if __name__ == "__main__":
    main()
