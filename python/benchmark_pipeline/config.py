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
    datasets_dir: str = str(DATASETS_DIR)


def default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        test_dimensions=[2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        test_samples=[
            5000,
            10000,
            50000,
            100000,
            500000,
            1000000,
            1500000,
            2000000,
            4000000,
            4500000,
            6000000,
            10000000,
            20000000,
        ],
        test_clusters=[5, 12, 20],
        bench_processes=3,
        bench_values=5,
        bench_min_time=0.05,
        lloyd_parity_tolerance_pct=1e-6,
    )
