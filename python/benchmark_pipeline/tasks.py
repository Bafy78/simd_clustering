import sys
from dataclasses import dataclass
from typing import Optional

from benchmark_pipeline.paths import BIN_DIR, DATASETS_DIR, repo_path


@dataclass
class Task:
    name: str
    command: list[str]
    cpp_json_arg: Optional[int] = None
    cpp_metrics_arg: Optional[int] = None


def config_id(dim: int, n_samples: int, n_clusters: int) -> str:
    return f"{dim}D_{n_samples}S_{n_clusters}K"


def dataset_path(filename: str) -> str:
    return str(DATASETS_DIR / filename)


def binary_path(filename: str) -> str:
    return str(BIN_DIR / filename)


def build_pipeline(
    dim: int,
    n_samples: int,
    n_clusters: int,
    bench_processes: int,
    bench_values: int,
    bench_min_time: float,
) -> list[Task]:
    """Defines the strict contract of tasks for a single configuration."""
    case_id = config_id(dim, n_samples, n_clusters)

    dataset_bin = dataset_path(f"data_{case_id}.bin")
    init_centroids_bin = dataset_path(f"init_{case_id}.bin")

    return [
        Task(
            name="Setup: Generate Dataset & Initial Centroids",
            command=[
                sys.executable,
                repo_path("python", "benchmark_pipeline", "tools", "dataset_gen.py"),
                "--n-samples",
                str(n_samples),
                "--n-features",
                str(dim),
                "--n-clusters",
                str(n_clusters),
                "--dataset-out",
                dataset_bin,
                "--centroids-out",
                init_centroids_bin,
            ],
        ),
        Task(
            name="C++: AoS to SoA Tax",
            command=[
                binary_path("bench_soa.bin"),
                dataset_bin,
                str(n_samples),
                dataset_path(f"soa_cpp_{case_id}.json"),
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_json_arg=3,
        ),
        Task(
            name="C++: K-Means++ Initialization",
            command=[
                binary_path("bench_pp.bin"),
                dataset_bin,
                str(n_samples),
                str(n_clusters),
                dataset_path(f"pp_cpp_{case_id}.json"),
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_json_arg=4,
        ),
        Task(
            name="Python: K-Means++ Initialization",
            command=[
                sys.executable,
                repo_path("python", "benchmark_pipeline", "benches", "bench_pp.py"),
                "--dataset-bin",
                dataset_bin,
                "--n-samples",
                str(n_samples),
                "--n-features",
                str(dim),
                "--n-clusters",
                str(n_clusters),
                "--processes",
                str(bench_processes),
                "--values",
                str(bench_values),
                "--min-time",
                str(bench_min_time),
                "--output",
                dataset_path(f"pp_py_{case_id}.json"),
            ],
        ),
        Task(
            name="C++: Lloyd Iterations",
            command=[
                binary_path("bench_lloyd.bin"),
                dataset_bin,
                str(n_samples),
                str(n_clusters),
                init_centroids_bin,
                dataset_path(f"lloyd_metrics_cpp_{case_id}.json"),
                dataset_path(f"lloyd_cpp_{case_id}.json"),
                str(bench_values),
                str(bench_min_time),
            ],
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
                "--n-samples",
                str(n_samples),
                "--n-features",
                str(dim),
                "--n-clusters",
                str(n_clusters),
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
    ]
