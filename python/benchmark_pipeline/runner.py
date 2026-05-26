import os
import shutil
import subprocess
import sys
from collections.abc import Iterable

from benchmark_pipeline.metrics import (
    compute_lloyd_parity,
    validate_cpp_process_metrics,
    write_json,
)
from benchmark_pipeline.cpp_cases import (
    cpp_compile_command,
    nanobench_binary_path,
)
from benchmark_pipeline.paths import DATASETS_DIR, REPO_ROOT, repo_path
from benchmark_pipeline.tasks import Task, build_pipeline, config_id, dataset_path


def prepare_datasets_dir(datasets_dir: str) -> None:
    if os.path.exists(datasets_dir):
        print(f"Cleaning {datasets_dir}...")
        shutil.rmtree(datasets_dir)

    os.makedirs(datasets_dir, exist_ok=True)


def run_command(task_name: str, command: list[str]) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

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
    process_metrics: list[str] = []

    final_metrics = None

    if task.cpp_metrics_arg is not None:
        final_metrics = task.command[task.cpp_metrics_arg]
        final_metrics_base = os.path.basename(final_metrics)
        final_metrics_stem, final_metrics_ext = os.path.splitext(final_metrics_base)

    for process_index in range(bench_processes):
        process_json = os.path.join(
            temp_dir,
            f"{final_json_stem}.process_{process_index}{final_json_ext}",
        )

        command = list(task.command)
        command[task.cpp_json_arg] = process_json

        if task.cpp_metrics_arg is not None:
            process_metric = os.path.join(
                temp_dir,
                f"{final_metrics_stem}.process_{process_index}{final_metrics_ext}",
            )

            command[task.cpp_metrics_arg] = process_metric
            process_metrics.append(process_metric)

        print(
            f"[{task.name}] Running C++ process {process_index + 1}/{bench_processes}..."
        )
        run_command(task.name, command)

        process_jsons.append(process_json)

    merge_command = [
        sys.executable,
        repo_path("python", "benchmark_pipeline", "tools", "merge_pyperf_runs.py"),
        "--output",
        final_json,
        *process_jsons,
    ]

    print(f"[{task.name}] Merging C++ pyperf JSON runs...")
    run_command(f"{task.name}: Merge pyperf JSON runs", merge_command)

    if task.cpp_metrics_arg is not None:
        assert final_metrics is not None

        print(f"[{task.name}] Validating C++ process metrics...")
        canonical_metrics = validate_cpp_process_metrics(process_metrics)
        write_json(final_metrics, canonical_metrics)

    for path in process_jsons:
        delete_if_exists(path, label=None)

    for path in process_metrics:
        delete_if_exists(path, label=None)


def compile_cpp_binaries(dim: int, cpp_cases: Iterable[str]):
    """Compiles the C++ nanobench cases for a specific dimension."""
    cases = sorted(set(cpp_cases))

    if not cases:
        return

    print(f"\n{'=' * 50}")
    print(f"--- Compiling C++ Binaries for {dim}D ---")
    print(f"{'=' * 50}")

    os.makedirs(os.path.dirname(nanobench_binary_path("lloyd_static")), exist_ok=True)

    for alg in cases:
        cmd = cpp_compile_command(dim=dim, alg=alg, mode="nanobench")

        print(f"Compiling C++ nanobench case '{alg}'...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

        if result.returncode != 0:
            print(
                f"Compilation failed for C++ nanobench case '{alg}':\n{result.stderr}"
            )
            sys.exit(1)


def delete_if_exists(path: str, *, label: str | None = "intermediate artifact") -> None:
    try:
        os.remove(path)
        if label is not None:
            print(f"Deleted {label}: {path}")
    except FileNotFoundError:
        pass


def cleanup_config_inputs(case_id: str) -> None:
    for path in [
        dataset_path(f"data_{case_id}.bin"),
        dataset_path(f"init_{case_id}.bin"),
    ]:
        delete_if_exists(path, label="temporary input")


def finalize_config(
    *,
    dim: int,
    n_samples: int,
    n_clusters: int,
    lloyd_parity_tolerance_pct: float,
) -> None:
    case_id = config_id(dim, n_samples, n_clusters)

    try:
        cpp_metrics_file = dataset_path(f"lloyd_metrics_cpp_{case_id}.json")
        py_metrics_file = dataset_path(f"lloyd_metrics_py_{case_id}.json")
        parity_file = dataset_path(f"lloyd_parity_{case_id}.json")

        print(f"[Validate: Lloyd Parity] {case_id}...")
        compute_lloyd_parity(
            config_id=case_id,
            cpp_metrics_file=cpp_metrics_file,
            py_metrics_file=py_metrics_file,
            output_file=parity_file,
            tolerance_pct=lloyd_parity_tolerance_pct,
        )

    finally:
        cleanup_config_inputs(case_id)

        delete_if_exists(cpp_metrics_file)
        delete_if_exists(py_metrics_file)


def execute_pipeline(
    dim: int,
    n_samples: int,
    n_clusters: int,
    bench_processes: int,
    bench_values: int,
    bench_min_time: float,
    lloyd_parity_tolerance_pct: float,
):
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

    finalize_config(
        dim=dim,
        n_samples=n_samples,
        n_clusters=n_clusters,
        lloyd_parity_tolerance_pct=lloyd_parity_tolerance_pct,
    )
