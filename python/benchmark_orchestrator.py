from benchmark_pipeline.config import default_config
from benchmark_pipeline.runner import (
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)
from benchmark_pipeline.tasks import build_pipeline


def cpp_cases_for_dimension(config, dim: int) -> set[str]:
    cases: set[str] = set()

    for n_samples in config.test_samples:
        for n_clusters in config.test_clusters:
            pipeline = build_pipeline(
                dim,
                n_samples,
                n_clusters,
                config.bench_processes,
                config.bench_values,
                config.bench_min_time,
            )

            for task in pipeline:
                if task.cpp_case is not None:
                    cases.add(task.cpp_case)

    return cases


def main() -> None:
    config = default_config()
    prepare_datasets_dir(config.datasets_dir)

    for dim in config.test_dimensions:
        compile_cpp_binaries(dim, cpp_cases_for_dimension(config, dim))

        for n_samples in config.test_samples:
            for n_clusters in config.test_clusters:
                execute_pipeline(
                    dim,
                    n_samples,
                    n_clusters,
                    config.bench_processes,
                    config.bench_values,
                    config.bench_min_time,
                    config.lloyd_parity_tolerance_pct,
                )

    print("\nAll benchmarking finished successfully!")


if __name__ == "__main__":
    main()
