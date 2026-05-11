import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Optional
import shutil


@dataclass
class Task:
    name: str
    command: List[str]
    cpp_json_arg: Optional[int] = None


def run_command(task_name: str, command: List[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\nTask '{task_name}' FAILED!")
        print(f"Command: {' '.join(command)}")
        print(f"Return Code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
        sys.exit(1)


def run_cpp_task_with_processes(task: Task, bench_processes: int) -> None:
    assert task.cpp_json_arg is not None

    final_json = task.command[task.cpp_json_arg]
    final_json_dir = os.path.dirname(final_json) or "."
    final_json_base = os.path.basename(final_json)
    final_json_stem, final_json_ext = os.path.splitext(final_json_base)

    temp_dir = os.path.join(final_json_dir, ".cpp_process_runs")
    os.makedirs(temp_dir, exist_ok=True)

    process_jsons: list[str] = []

    for process_index in range(bench_processes):
        process_json = os.path.join(
            temp_dir,
            f"{final_json_stem}.process_{process_index}{final_json_ext}",
        )

        command = list(task.command)
        command[task.cpp_json_arg] = process_json

        print(
            f"[{task.name}] Running C++ process {process_index + 1}/{bench_processes}..."
        )
        run_command(task.name, command)

        process_jsons.append(process_json)

    merge_command = [
        sys.executable,
        "python/merge_pyperf_runs.py",
        "--output",
        final_json,
        *process_jsons,
    ]

    print(f"[{task.name}] Merging C++ pyperf JSON runs...")
    run_command(f"{task.name}: Merge pyperf JSON runs", merge_command)

    for path in process_jsons:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def compile_cpp_binaries(dim: int):
    """Compiles the independent C++ benchmarks for a specific dimension."""
    print(f"\n{'=' * 50}")
    print(f"--- Compiling C++ Binaries for {dim}D ---")
    print(f"{'=' * 50}")

    os.makedirs("./bin", exist_ok=True)

    cpp_targets = [
        {"src": "cpp/benchmarks/bench_soa.cpp", "bin": "./bin/bench_soa.bin"},
        {"src": "cpp/benchmarks/bench_pp.cpp", "bin": "./bin/bench_pp.bin"},
        {"src": "cpp/benchmarks/bench_lloyd.cpp", "bin": "./bin/bench_lloyd.bin"},
    ]

    for target in cpp_targets:
        cmd = [
            "g++-14",
            "-O3",
            "-march=native",
            "-std=c++20",
            "-I../eve/include",
            "-I../nanobench/src/include",
            f"-DTUPLE_SIZE={dim}",
            target["src"],
            "-o",
            target["bin"],
        ]

        print(f"Compiling {target['src']}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Compilation failed for {target['src']}:\n{result.stderr}")
            sys.exit(1)


def build_pipeline(
    dim: int,
    n_samples: int,
    n_clusters: int,
    bench_processes: int,
    bench_values: int,
    bench_min_time: float,
) -> List[Task]:
    """Defines the strict 'contract' of tasks for a single configuration."""
    config_id = f"{dim}D_{n_samples}S_{n_clusters}K"

    dataset_bin = f"./datasets/data_{config_id}.bin"
    init_centroids_bin = f"./datasets/init_{config_id}.bin"

    tasks: list[Task] = []

    tasks.append(
        Task(
            name="Setup: Generate Dataset & Initial Centroids",
            command=[
                sys.executable,
                "python/dataset_gen.py",
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
        )
    )

    tasks.append(
        Task(
            name="C++: AoS to SoA Tax",
            command=[
                "./bin/bench_soa.bin",
                dataset_bin,
                str(n_samples),
                f"./datasets/soa_cpp_{config_id}.json",
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_json_arg=3,
        )
    )

    tasks.append(
        Task(
            name="C++: K-Means++ Initialization",
            command=[
                "./bin/bench_pp.bin",
                dataset_bin,
                str(n_samples),
                str(n_clusters),
                f"./datasets/pp_cpp_{config_id}.json",
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_json_arg=4,
        )
    )

    tasks.append(
        Task(
            name="Python: K-Means++ Initialization",
            command=[
                sys.executable,
                "python/bench_pp.py",
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
                f"./datasets/pp_py_{config_id}.json",
            ],
        )
    )

    tasks.append(
        Task(
            name="C++: Lloyd Iterations",
            command=[
                "./bin/bench_lloyd.bin",
                dataset_bin,
                str(n_samples),
                str(n_clusters),
                init_centroids_bin,
                f"./datasets/results_cpp_{config_id}.txt",
                f"./datasets/lloyd_cpp_{config_id}.json",
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_json_arg=6,
        )
    )

    tasks.append(
        Task(
            name="Python: Lloyd Iterations",
            command=[
                sys.executable,
                "python/bench_lloyd.py",
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
                "--result-file",
                f"./datasets/results_py_{config_id}.txt",
                "--processes",
                str(bench_processes),
                "--values",
                str(bench_values),
                "--min-time",
                str(bench_min_time),
                "--output",
                f"./datasets/lloyd_py_{config_id}.json",
            ],
        )
    )

    return tasks


def execute_pipeline(
    dim: int,
    n_samples: int,
    n_clusters: int,
    bench_processes: int,
    bench_values: int,
    bench_min_time: float,
):
    """Executes the task list for a specific configuration."""
    print(
        f"\n--- Running Config: {dim}D | {n_samples} Samples | {n_clusters} Clusters ---"
    )

    pipeline = build_pipeline(
        dim,
        n_samples,
        n_clusters,
        bench_processes,
        bench_values,
        bench_min_time,
    )

    for task in pipeline:
        if task.cpp_json_arg is not None:
            run_cpp_task_with_processes(task, bench_processes)
        else:
            print(f"[{task.name}] Running...")
            run_command(task.name, task.command)


if __name__ == "__main__":
    datasets_dir = "./datasets"

    if os.path.exists(datasets_dir):
        print(f"Cleaning {datasets_dir}...")
        shutil.rmtree(datasets_dir)

    os.makedirs(datasets_dir, exist_ok=True)

    test_dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    test_samples = [
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
    ]
    test_clusters = [3, 5, 8, 12, 15, 20]

    bench_processes = 3
    bench_values = 3
    bench_min_time = 0.1

    for dim in test_dimensions:
        compile_cpp_binaries(dim)

        for n in test_samples:
            for k in test_clusters:
                execute_pipeline(
                    dim,
                    n,
                    k,
                    bench_processes,
                    bench_values,
                    bench_min_time,
                )

    print("\nAll benchmarking finished successfully!")
