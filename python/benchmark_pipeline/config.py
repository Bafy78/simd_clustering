from dataclasses import dataclass

from benchmark_pipeline.paths import DATASETS_DIR


@dataclass(frozen=True)
class BenchmarkConfig:
    test_Ds: list[int]
    test_Ns: list[int]
    test_Ks: list[int]
    timing_processes: int
    timing_values: int
    timing_min_time: float
    gmm_covariance_type: str
    datasets_dir: str = str(DATASETS_DIR)


def default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        test_Ds=[3, 5, 7, 9, 12, 15, 20, 30],
        test_Ns=[
            4_000,
            15_000,
            50_000,
            100_000,
            300_000,
            800_000,
            2_000_000,
            10_000_000,
        ],
        test_Ks=[10, 25, 50, 100],
        timing_processes=8,
        timing_values=6,
        timing_min_time=0.05,
        gmm_covariance_type="spherical",
    )
