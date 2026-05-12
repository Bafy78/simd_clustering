from benchmark_pipeline.config import default_config
from benchmark_pipeline.runner import (
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)


def main() -> None:
    config = default_config()
    prepare_datasets_dir(config.datasets_dir)

    for dim in config.test_dimensions:
        compile_cpp_binaries(dim)

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
