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
        test_dimensions=[2, 3, 5, 8, 12, 17, 23, 30, 40, 63, 70],
        test_samples=[
            4_000,
            10_000,
            50_000,
            100_000,
            400_000,
            1_000_000,
            4_000_000,
            12_000_000,
        ],
        test_clusters=[10, 30, 70],
        bench_processes=3,
        bench_values=5,
        bench_min_time=0.05,
        lloyd_parity_tolerance_pct=1e-6,
    )
