from dataclasses import dataclass

from benchmark_pipeline.cachegrind import CACHEGRIND_RESULTS_DIR
from benchmark_pipeline.exclusions import (
    BenchmarkExclusionRule,
    CachegrindExclusionRule,
)
from benchmark_pipeline.paths import DATASETS_DIR, DOWNLOADS_DIR


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
    cpp_hdbscan_cases: tuple[str, ...]
    run_python_hdbscan: bool
    hdbscan_references: tuple[str, ...]
    hdbscan_stages: tuple[str, ...]
    run_cachegrind: bool
    cachegrind_I1: str | None
    cachegrind_D1: str | None
    cachegrind_LL: str | None
    cachegrind_exclusion_rules: tuple[CachegrindExclusionRule, ...]
    cachegrind_results_dir: str = str(CACHEGRIND_RESULTS_DIR)


@dataclass(frozen=True)
class RealDatasetConfig:
    key: str
    D: int
    N: int
    K: int = 10
    source: str = "local"

    # Local/direct URL dense inputs
    path: str | None = None
    url: str | None = None
    format: str = "npy"

    # OpenML. Prefer data_id when known; otherwise use name/version
    data_id: int | None = None
    name: str | None = None
    version: int | str | None = None

    # UCI ML Repository
    dataset_id: int | None = None

    # Hugging Face Datasets
    repo: str | None = None
    hf_config: str | None = None
    split: str = "train"
    feature_column: str | None = None
    feature_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkConfig:
    test_Ds: list[int]
    test_Ns: list[int]
    test_Ks: list[int]
    pipeline: PipelineOptions
    real_datasets: tuple[RealDatasetConfig, ...] = ()
    exclusion_rules: tuple[BenchmarkExclusionRule, ...] = ()
    datasets_dir: str = str(DATASETS_DIR)
    downloads_dir: str = str(DOWNLOADS_DIR)
    keep_inputs: bool = False


def default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        test_Ds=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 17, 23, 30, 40, 60, 90],
        test_Ns=[
            128,
            256,
            512,
            1024,
            2048,
            4_096,
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
            timing_values=8,
            timing_min_time=0.05,
            gmm_covariance_types=("spherical", "diag", "full",),
            cpp_soa_cases=("soa_static", "soa_dynamic",),
            cpp_pp_cases=("pp_static", "pp_dynamic",),
            run_python_pp=True,
            cpp_lloyd_cases=("lloyd_static", "lloyd_dynamic", "lloyd_auto",),
            run_python_lloyd=True,
            cpp_gmm_cases=("gmm_static", "gmm_dynamic",),
            run_python_gmm=True,
            cpp_hdbscan_cases=("hdbscan_static",),
            run_python_hdbscan=True,
            hdbscan_references=("sklearn_brute", "hdbscan_contrib",),
            hdbscan_stages=("distance", "mst", "linkage", "select", "full",),
            run_cachegrind=True,
            cachegrind_I1="32768,8,64",
            cachegrind_D1="49152,12,64",
            cachegrind_LL="100663296,24,64",
            cachegrind_exclusion_rules=(
                CachegrindExclusionRule(
                    phase_keys=("soa", "pp"),
                    reason=(
                        "Excluded cuz we don't care about them."
                    ),
                ),
                CachegrindExclusionRule(
                    min_samples=500_000,
                    reason=("Too slow and we can infer from the rest of the data well enough"),
                ),
                CachegrindExclusionRule(
                    min_dimensions=50,
                    reason=("Too slow and we can infer from the rest of the data well enough"),
                )
            ),
        ),
        real_datasets=(
            RealDatasetConfig(
                key="letter",
                source="openml",
                data_id=6,
                D=17,
                N=20_000,
                K=26,
            ),
            RealDatasetConfig(
                key="mnist",
                source="openml",
                data_id=554,
                D=784,
                N=70000,
                K=10,
            ),
            RealDatasetConfig(
                key="USCensus1990",
                source="uci",
                dataset_id=116,
                D=68,
                N=2_458_285,
                K=27,
            ),
            RealDatasetConfig(
                key="sift1m",
                source="huggingface",
                repo="open-vdb/sift-128-euclidean",
                hf_config="train",
                split="train",
                feature_column="emb",
                D=128,
                N=1_000_000,
                K=256,
            ),
        ),
        exclusion_rules=(
            BenchmarkExclusionRule(
                phase_keys=("lloyd", "pp", "soa"),
                dimensions=(90,),
                samples=(10_000_000,),
                reason=(
                    "Excluded because the scikit-learn Lloyd reference no longer "
                    "fits in RAM, which would produce misleading speedups."
                ),
            ),
            BenchmarkExclusionRule(
                phase_keys=("gmm",),
                min_samples=300_001,
                reason=(
                    "Excluded because GMM doesn't scale well with K and N, so the "
                    "pipeline would simply take too long to run"
                ),
            ),
            BenchmarkExclusionRule(
                phase_keys=("gmm",),
                dimensions=(90,),
                max_samples=4_600,
                clusters=(50,),
                reason=(
                    "Excluded because full-covariance estimation is underdetermined: "
                    "each component has fewer samples than dimensions (`N / K <= D`), "
                    "producing rank-deficient covariance matrices that are only marginally "
                    "regularized and can fail positive-definiteness checks."
                ),
            ),
            BenchmarkExclusionRule(
                phase_keys=("hdbscan",),
                min_samples=5000,
                max_samples=19_999,
                reason=(
                    "Too high for hdbscan quadratic complexity"
                ),
            ),
            BenchmarkExclusionRule(
                phase_keys=("hdbscan",),
                min_samples=20_001,
                reason=(
                    "Too high for hdbscan quadratic complexity"
                ),
            ), # hack to run the letter dataset on hdbscan anyway
            BenchmarkExclusionRule(
                phase_keys=("pp", "gmm", "lloyd",),
                max_samples=4000,
                reason=(
                    "Too small to matter for these algorithms"
                ),
            ),
        ),
    )
