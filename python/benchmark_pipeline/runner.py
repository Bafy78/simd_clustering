import os
import shutil
import subprocess
import sys
from collections.abc import Iterable

from benchmark_pipeline.metrics import (
    compute_lloyd_parity,
    validate_cpp_timing_process_metrics,
    write_json,
)
from benchmark_pipeline.cpp_cases import (
    cpp_compile_command,
    nanobench_binary_path,
)
from benchmark_pipeline.paths import REPO_ROOT, repo_path
from benchmark_pipeline.tasks import (
    Task,
    build_pipeline,
    config_id,
    configuration_label,
    dataset_path,
)


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


def run_cpp_task_with_timing_processes(task: Task, timing_processes: int) -> None:
    assert task.cpp_json_arg is not None

    final_json = task.command[task.cpp_json_arg]
    final_json_dir = os.path.dirname(final_json) or "."
    final_json_base = os.path.basename(final_json)
    final_json_stem, final_json_ext = os.path.splitext(final_json_base)

    temp_dir = os.path.join(final_json_dir, ".cpp_timing_process_runs")
    os.makedirs(temp_dir, exist_ok=True)

    timing_process_jsons: list[str] = []
    timing_process_metrics: list[str] = []

    final_metrics = None

    if task.cpp_metrics_arg is not None:
        final_metrics = task.command[task.cpp_metrics_arg]
        final_metrics_base = os.path.basename(final_metrics)
        final_metrics_stem, final_metrics_ext = os.path.splitext(final_metrics_base)

    for timing_process_index in range(timing_processes):
        timing_process_json = os.path.join(
            temp_dir,
            f"{final_json_stem}.timing_process_{timing_process_index}{final_json_ext}",
        )

        command = list(task.command)
        command[task.cpp_json_arg] = timing_process_json

        if task.cpp_metrics_arg is not None:
            timing_process_metric = os.path.join(
                temp_dir,
                f"{final_metrics_stem}.timing_process_{timing_process_index}{final_metrics_ext}",
            )

            command[task.cpp_metrics_arg] = timing_process_metric
            timing_process_metrics.append(timing_process_metric)

        print(
            f"[{task.name}] Running C++ timing process {timing_process_index + 1}/{timing_processes}..."
        )
        run_command(task.name, command)

        timing_process_jsons.append(timing_process_json)

    merge_command = [
        sys.executable,
        repo_path("python", "benchmark_pipeline", "tools", "merge_pyperf_runs.py"),
        "--output",
        final_json,
        *timing_process_jsons,
    ]

    print(f"[{task.name}] Merging C++ pyperf JSON runs...")
    run_command(f"{task.name}: Merge pyperf JSON runs", merge_command)

    if task.cpp_metrics_arg is not None:
        assert final_metrics is not None

        print(f"[{task.name}] Validating C++ timing-process metrics...")
        canonical_metrics = validate_cpp_timing_process_metrics(timing_process_metrics)
        write_json(final_metrics, canonical_metrics)

    for path in timing_process_jsons:
        delete_if_exists(path, label=None)

    for path in timing_process_metrics:
        delete_if_exists(path, label=None)


def compile_cpp_binaries(D: int, cpp_cases: Iterable[str]) -> None:
    """Compiles the C++ nanobench cases for a specific dimension D."""
    cases = sorted(set(cpp_cases))

    if not cases:
        return

    print(f"\n{'=' * 50}")
    print(f"--- Compiling C++ Binaries for {D}D ---")
    print(f"{'=' * 50}")

    os.makedirs(os.path.dirname(nanobench_binary_path("lloyd_static")), exist_ok=True)

    for cpp_case in cases:
        cmd = cpp_compile_command(D=D, cpp_case=cpp_case, mode="nanobench")

        print(f"Compiling C++ nanobench case '{cpp_case}'...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

        if result.returncode != 0:
            print(
                f"Compilation failed for C++ nanobench case '{cpp_case}':\n{result.stderr}"
            )
            sys.exit(1)


def delete_if_exists(path: str, *, label: str | None = "intermediate artifact") -> None:
    try:
        os.remove(path)
        if label is not None:
            print(f"Deleted {label}: {path}")
    except FileNotFoundError:
        pass


def task_references_path(task: Task, path: str) -> bool:
    return any(arg == path for arg in task.command)


def cleanup_config_inputs(case_id: str) -> None:
    for path in [
        dataset_path(f"data_{case_id}.bin"),
        dataset_path(f"init_{case_id}.bin"),
        dataset_path(f"gmm_weights_{case_id}.bin"),
        dataset_path(f"gmm_means_{case_id}.bin"),
        dataset_path(f"gmm_precisions_{case_id}.bin"),
    ]:
        delete_if_exists(path, label="temporary input")


def finalize_config(
    *,
    D: int,
    N: int,
    K: int,
    lloyd_parity_tolerance_pct: float,
    validate_lloyd_parity: bool,
) -> None:
    case_id = config_id(D, N, K)

    cpp_metrics_file = dataset_path(f"lloyd_metrics_cpp_{case_id}.json")
    py_metrics_file = dataset_path(f"lloyd_metrics_py_{case_id}.json")
    parity_file = dataset_path(f"lloyd_parity_{case_id}.json")

    try:
        if validate_lloyd_parity:
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
    D: int,
    N: int,
    K: int,
    timing_processes: int,
    timing_values: int,
    timing_min_time: float,
    lloyd_parity_tolerance_pct: float,
    gmm_covariance_type: str = "spherical",
):
    print(f"\n--- Running Config: {configuration_label(D, N, K)} ---")

    pipeline = build_pipeline(
        D,
        N,
        K,
        timing_processes,
        timing_values,
        timing_min_time,
        gmm_covariance_type=gmm_covariance_type,
    )

    case_id = config_id(D, N, K)

    lloyd_cpp_metrics_file = dataset_path(f"lloyd_metrics_cpp_{case_id}.json")
    lloyd_py_metrics_file = dataset_path(f"lloyd_metrics_py_{case_id}.json")
    validate_lloyd_parity = any(
        task_references_path(task, lloyd_cpp_metrics_file) for task in pipeline
    ) and any(task_references_path(task, lloyd_py_metrics_file) for task in pipeline)

    for task in pipeline:
        if task.cpp_json_arg is not None:
            run_cpp_task_with_timing_processes(task, timing_processes)
        else:
            print(f"[{task.name}] Running...")
            run_command(task.name, task.command)

    finalize_config(
        D=D,
        N=N,
        K=K,
        lloyd_parity_tolerance_pct=lloyd_parity_tolerance_pct,
        validate_lloyd_parity=validate_lloyd_parity,
    )
