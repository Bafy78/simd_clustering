from dataclasses import dataclass

from benchmark_pipeline.paths import DATASETS_DIR


@dataclass(frozen=True)
class BenchmarkConfig:
    test_dimensions: list[int]
    test_samples: list[int]
    test_clusters: list[int]
    bench_processes: int
    bench_values: int
    bench_min_time: float
    lloyd_parity_tolerance_pct: float
    gmm_covariance_type: str
    datasets_dir: str = str(DATASETS_DIR)


def default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        test_dimensions=[1, 2, 3, 5, 8, 12, 17, 23, 30, 40, 63, 80, 100],
        test_samples=[
            4_000,
            15_000,
            50_000,
            100_000,
            300_000,
            800_000,
            2_000_000,
            10_000_000,
        ],
        test_clusters=[10, 25, 50, 100],
        bench_processes=5,
        bench_values=5,
        bench_min_time=0.05,
        lloyd_parity_tolerance_pct=1e-6,
        gmm_covariance_type="spherical",
    )
