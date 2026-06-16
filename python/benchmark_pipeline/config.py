from dataclasses import dataclass

from benchmark_pipeline.exclusions import BenchmarkExclusionRule
from benchmark_pipeline.paths import DATASETS_DIR


@dataclass(frozen=True)
class PipelineOptions:
    timing_processes: int
    timing_values: int
    timing_min_time: float
    gmm_covariance_types: tuple[str, ...]
    cpp_soa_cases: tuple[str, ...]
    cpp_pp_cases: tuple[str, ...]
    run_python_pp: bool
    cpp_lloyd_cases: tuple[str, ...]
    run_python_lloyd: bool
    cpp_gmm_cases: tuple[str, ...]
    run_python_gmm: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    test_Ds: list[int]
    test_Ns: list[int]
    test_Ks: list[int]
    pipeline: PipelineOptions
    exclusion_rules: tuple[BenchmarkExclusionRule, ...] = ()
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
        pipeline=PipelineOptions(
            timing_processes=8,
            timing_values=6,
            timing_min_time=0.05,
            gmm_covariance_types=(),
            cpp_soa_cases=("soa_static", "soa_dynamic"),
            cpp_pp_cases=(),
            run_python_pp=False,
        exclusion_rules=(
            BenchmarkExclusionRule(
                phase_keys=("lloyd",),
                dimensions=(100,),
                samples=(10_000_000,),
                reason=(
                    "Excluded because the scikit-learn Lloyd reference no longer "
                    "fits in RAM, which would produce misleading speedups."
                ),
            ),
            BenchmarkExclusionRule(
                phase_keys=("gmm",),
                min_clusters=11,
                min_samples=300_001,
                reason="Excluded because GMM doesn't scale well with K and N, so the "
                       "pipeline would simply take too long to run",
            ),
            BenchmarkExclusionRule(
                phase_keys=("gmm",),
                dimensions=(90,),
                samples=(4_000,),
                clusters=(50,),
                reason=(
                    "Excluded because full-covariance estimation is underdetermined: "
                    "each component has fewer samples than dimensions (`N / K <= D`), "
                    "producing rank-deficient covariance matrices that are only marginally "
                    "regularized and can fail positive-definiteness checks."
                ),
            ),
        ),
    )
