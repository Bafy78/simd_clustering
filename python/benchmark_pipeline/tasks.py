import sys
from dataclasses import dataclass
from typing import Optional

from benchmark_pipeline.cpp_cases import nanobench_binary_path
from benchmark_pipeline.paths import DATASETS_DIR, repo_path


@dataclass
class Task:
    name: str
    command: list[str]
    cpp_case: Optional[str] = None
    cpp_json_arg: Optional[int] = None
    cpp_metrics_arg: Optional[int] = None


def config_id(D: int, N: int, K: int) -> str:
    return f"{D}D_{N}N_{K}K"


def configuration_label(D: int, N: int, K: int) -> str:
    return f"{D}D | {N}N | {K}K"


def dataset_path(filename: str) -> str:
    return str(DATASETS_DIR / filename)


def build_pipeline(
    D: int,
    N: int,
    K: int,
    bench_processes: int,
    bench_values: int,
    bench_min_time: float,
    gmm_covariance_type: str = "spherical",
) -> list[Task]:
    """Defines the strict contract of tasks for a single D/N/K configuration."""
    case_id = config_id(D, N, K)

    dataset_bin = dataset_path(f"data_{case_id}.bin")
    init_centroids_bin = dataset_path(f"init_{case_id}.bin")
    gmm_weights_bin = dataset_path(f"gmm_weights_{case_id}.bin")
    gmm_means_bin = dataset_path(f"gmm_means_{case_id}.bin")
    gmm_precisions_bin = dataset_path(f"gmm_precisions_{case_id}.bin")

    return [
        Task(
            name="Setup: Generate Dataset, K-Means++ Init & GMM Init",
            command=[
                sys.executable,
                repo_path("python", "benchmark_pipeline", "tools", "dataset_gen.py"),
                "--D",
                str(D),
                "--N",
                str(N),
                "--K",
                str(K),
                "--dataset-out",
                dataset_bin,
                "--centroids-out",
                init_centroids_bin,
                "--gmm-weights-out",
                gmm_weights_bin,
                "--gmm-means-out",
                gmm_means_bin,
                "--gmm-precisions-out",
                gmm_precisions_bin,
                "--gmm-covariance-type",
                gmm_covariance_type,
            ],
        ),
        Task(
            name="C++: AoS to Native Layout",
            command=[
                nanobench_binary_path("soa_dynamic"),
                dataset_bin,
                str(N),
                dataset_path(f"soa_cpp_{case_id}.json"),
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_case="soa_dynamic",
            cpp_json_arg=3,
        ),
        # Task(
        #     name="C++: K-Means++ Initialization",
        #     command=[
        #         nanobench_binary_path("pp"),
        #         dataset_bin,
        #         str(N),
        #         str(K),
        #         dataset_path(f"pp_cpp_{case_id}.json"),
        #         str(bench_values),
        #         str(bench_min_time),
        #     ],
        #     cpp_case="pp",
        #     cpp_json_arg=4,
        # ),
        # Task(
        #     name="Python: K-Means++ Initialization",
        #     command=[
        #         sys.executable,
        #         repo_path("python", "benchmark_pipeline", "benches", "bench_pp.py"),
        #         "--dataset-bin",
        #         dataset_bin,
        #         "--D",
        #         str(D),
        #         "--N",
        #         str(N),
        #         "--K",
        #         str(K),
        #         "--processes",
        #         str(bench_processes),
        #         "--values",
        #         str(bench_values),
        #         "--min-time",
        #         str(bench_min_time),
        #         "--output",
        #         dataset_path(f"pp_py_{case_id}.json"),
        #     ],
        # ),
        Task(
            name="C++: Lloyd Iterations",
            command=[
                nanobench_binary_path("lloyd_dynamic"),
                dataset_bin,
                str(N),
                str(K),
                init_centroids_bin,
                dataset_path(f"lloyd_metrics_cpp_{case_id}.json"),
                dataset_path(f"lloyd_cpp_{case_id}.json"),
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_case="lloyd_dynamic",
            cpp_metrics_arg=5,
            cpp_json_arg=6,
        ),
        Task(
            name="Python: Lloyd Iterations",
            command=[
                sys.executable,
                repo_path("python", "benchmark_pipeline", "benches", "bench_lloyd.py"),
                "--dataset-bin",
                dataset_bin,
                "--D",
                str(D),
                "--N",
                str(N),
                "--K",
                str(K),
                "--init-centroids-bin",
                init_centroids_bin,
                "--metrics-file",
                dataset_path(f"lloyd_metrics_py_{case_id}.json"),
                "--processes",
                str(bench_processes),
                "--values",
                str(bench_values),
                "--min-time",
                str(bench_min_time),
                "--output",
                dataset_path(f"lloyd_py_{case_id}.json"),
            ],
        ),
        # Task(
        #     name="C++: GaussianMixture EM",
        #     command=[
        #         nanobench_binary_path("gmm_static"),
        #         dataset_bin,
        #         str(N),
        #         str(K),
        #         gmm_weights_bin,
        #         gmm_means_bin,
        #         gmm_precisions_bin,
        #         gmm_covariance_type,
        #         dataset_path(f"gmm_metrics_cpp_{case_id}.json"),
        #         dataset_path(f"gmm_cpp_{case_id}.json"),
        #         str(bench_values),
        #         str(bench_min_time),
        #     ],
        #     cpp_case="gmm_static",
        #     cpp_metrics_arg=8,
        #     cpp_json_arg=9,
        # ),
        # Task(
        #     name="Python: GaussianMixture EM",
        #     command=[
        #         sys.executable,
        #         repo_path("python", "benchmark_pipeline", "benches", "bench_gmm.py"),
        #         "--dataset-bin",
        #         dataset_bin,
        #         "--D",
        #         str(D),
        #         "--N",
        #         str(N),
        #         "--K",
        #         str(K),
        #         "--covariance-type",
        #         gmm_covariance_type,
        #         "--gmm-weights-bin",
        #         gmm_weights_bin,
        #         "--gmm-means-bin",
        #         gmm_means_bin,
        #         "--gmm-precisions-bin",
        #         gmm_precisions_bin,
        #         "--metrics-file",
        #         dataset_path(f"gmm_metrics_py_{case_id}.json"),
        #         "--processes",
        #         str(bench_processes),
        #         "--values",
        #         str(bench_values),
        #         "--min-time",
        #         str(bench_min_time),
        #         "--output",
        #         dataset_path(f"gmm_py_{case_id}.json"),
        #     ],
        # )
    ]
