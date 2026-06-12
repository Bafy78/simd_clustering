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
    gmm_covariance_types: tuple[str, ...]
    cpp_soa_cases: tuple[str, ...]
    run_cpp_pp: bool
    run_python_pp: bool
    cpp_lloyd_cases: tuple[str, ...]
    run_python_lloyd: bool
    cpp_gmm_cases: tuple[str, ...]
    run_python_gmm: bool
    datasets_dir: str = str(DATASETS_DIR)
    keep_inputs: bool = False


def default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        test_Ds=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 17, 23, 30, 40, 63, 80, 100, 150],
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
        test_Ks=[10, 25, 50],
        timing_processes=8,
        timing_values=6,
        timing_min_time=0.05,
        gmm_covariance_types=(),
        cpp_soa_cases=("soa_static", "soa_dynamic"),
        run_cpp_pp=True,
        run_python_pp=True,
        cpp_lloyd_cases=("lloyd_dynamic", "lloyd_static", "lloyd_auto"),
        run_python_lloyd=True,
        cpp_gmm_cases=(),
        run_python_gmm=False,
    )
