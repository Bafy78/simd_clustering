import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from benchmark_pipeline.compile_artifacts import (
    compile_artifact_record,
    compile_artifacts_path,
    write_compile_artifact_record,
)
from benchmark_pipeline.cachegrind import (
    build_cachegrind_record,
    parse_cachegrind_summary_events,
    profile_events_string,
    write_json as write_cachegrind_json,
)
from benchmark_pipeline.config import PipelineOptions
from benchmark_pipeline.exclusions import BenchmarkExclusionRule
from benchmark_pipeline.metrics import (
    validate_cpp_timing_process_metrics,
    write_json,
)
from benchmark_pipeline.cpp_cases import (
    callgrind_binary_path,
    cpp_compile_command,
    nanobench_binary_path,
)
from benchmark_pipeline.paths import (
    BIN_DIR,
    DATASETS_DIR,
    PYTHON_DIR,
    REPO_ROOT,
    repo_path,
    repo_relative_path,
)
from benchmark_pipeline.tasks import (
    BenchmarkArtifacts,
    BenchmarkCase,
    Task,
    build_pipeline,
)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _validate_datasets_dir_for_cleaning(datasets_dir: Path) -> None:
    protected_paths = {
        Path(datasets_dir.anchor).resolve(),
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        REPO_ROOT.resolve(),
        PYTHON_DIR.resolve(),
        BIN_DIR.resolve(),
        REPO_ROOT.parent.resolve(),
    }

    if datasets_dir in protected_paths:
        raise ValueError(f"Refusing to clean protected directory: {datasets_dir}")

    if _is_relative_to(REPO_ROOT.resolve(), datasets_dir):
        raise ValueError(
            f"Refusing to clean {datasets_dir}: it contains the repository root."
        )

    if datasets_dir.exists() and not datasets_dir.is_dir():
        raise ValueError(f"Datasets path exists but is not a directory: {datasets_dir}")

    if datasets_dir.is_symlink():
        raise ValueError(f"Refusing to clean symlinked datasets directory: {datasets_dir}")


def prepare_datasets_dir(datasets_dir: str | Path) -> Path:
    datasets_dir = repo_relative_path(datasets_dir)
    _validate_datasets_dir_for_cleaning(datasets_dir)

    if datasets_dir.exists():
        print(f"Cleaning {datasets_dir}...")
        shutil.rmtree(datasets_dir)

    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


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


def run_and_capture(
    task_name: str,
    command: list[str],
    *,
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> None:
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    if result.returncode != 0:
        print(f"\nTask '{task_name}' FAILED!")
        print(f"Command: {' '.join(command)}")
        print(f"Return Code: {result.returncode}")
        print(f"Stdout file: {stdout_path}")
        print(f"Stderr file: {stderr_path}")
        print(f"Stderr:\n{result.stderr}")
        sys.exit(result.returncode)


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


def run_cachegrind_task(task: Task) -> None:
    info = task.cachegrind
    if info is None:
        raise ValueError("Cachegrind task is missing metadata.")

    print(f"[{task.name}] Running one Cachegrind pass...")
    run_and_capture(
        task.name,
        task.command,
        stdout_path=info.stdout_path,
        stderr_path=info.stderr_path,
    )

    events = parse_cachegrind_summary_events(info.raw_output)

    annotate_command = [
        "callgrind_annotate",
        f"--show={profile_events_string()}",
        "--sort=D1mr,D1mw,DLmr,DLmw,Ir",
        "--inclusive=yes",
        "--tree=both",
        "--threshold=0.0",
        "--context=8",
        info.raw_output,
    ]

    print(f"[{task.name}] Annotating Cachegrind output...")
    run_and_capture(
        f"{task.name}: callgrind_annotate",
        annotate_command,
        stdout_path=info.annotated_output,
        stderr_path=info.annotate_stderr_path,
    )

    record = build_cachegrind_record(
        cpp_case=info.cpp_case,
        D=info.D,
        N=info.N,
        K=info.K,
        params_key=info.params_key,
        cache_model=info.cache_model,
        events=events,
        raw_output=info.raw_output,
        annotated_output=info.annotated_output,
        stdout_path=info.stdout_path,
        stderr_path=info.stderr_path,
        annotate_stderr_path=info.annotate_stderr_path,
        metrics_path=info.metrics_path,
    )
    write_cachegrind_json(info.summary_path, record)


def compile_cpp_binaries(
    D: int,
    cpp_cases: Iterable[str],
    *,
    datasets_dir: str | Path = DATASETS_DIR,
) -> None:
    """Compiles the C++ nanobench cases for a specific dimension D."""
    cases = sorted(set(cpp_cases))

    if not cases:
        return

    print(f"\n{'=' * 50}")
    print(f"--- Compiling C++ Binaries for {D}D ---")
    print(f"{'=' * 50}")

    os.makedirs(os.path.dirname(nanobench_binary_path(cases[0])), exist_ok=True)

    artifact_json = compile_artifacts_path(datasets_dir)

    for cpp_case in cases:
        cmd = cpp_compile_command(D=D, cpp_case=cpp_case, mode="nanobench")

        print(f"Compiling C++ nanobench case '{cpp_case}'...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

        if result.returncode != 0:
            print(
                f"Compilation failed for C++ nanobench case '{cpp_case}':\n{result.stderr}"
            )
            sys.exit(1)

        write_compile_artifact_record(
            artifact_json,
            compile_artifact_record(
                D=D,
                cpp_case=cpp_case,
                command=cmd,
                binary_path=nanobench_binary_path(cpp_case),
            ),
        )


def compile_callgrind_binaries(
    D: int,
    cpp_cases: Iterable[str],
) -> None:
    """Compile the C++ Callgrind entry points for a specific dimension D."""
    cases = sorted(set(cpp_cases))

    if not cases:
        return

    print(f"\n{'=' * 50}")
    print(f"--- Compiling C++ Callgrind Binaries for {D}D ---")
    print(f"{'=' * 50}")

    os.makedirs(os.path.dirname(callgrind_binary_path(cases[0], D)), exist_ok=True)

    for cpp_case in cases:
        cmd = cpp_compile_command(D=D, cpp_case=cpp_case, mode="callgrind")

        print(f"Compiling C++ Callgrind case '{cpp_case}'...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

        if result.returncode != 0:
            print(f"Compilation failed for C++ Callgrind case '{cpp_case}':\n{result.stderr}")
            sys.exit(1)


def delete_if_exists(path: str, label: str | None = "intermediate artifact") -> None:
    try:
        os.remove(path)
        if label is not None:
            print(f"Deleted {label}: {path}")
    except FileNotFoundError:
        pass


def cleanup_config_inputs(
    case_id: str,
    datasets_dir: str | Path = DATASETS_DIR,
) -> None:
    datasets_dir = repo_relative_path(datasets_dir)
    artifacts = BenchmarkArtifacts(case_id, datasets_dir)

    for path in [
        artifacts.dataset_bin,
        artifacts.init_centroids_bin,
        artifacts.gmm_weights_bin,
        artifacts.gmm_means_bin,
    ]:
        delete_if_exists(path, label="temporary input")

    for path in datasets_dir.glob(f"gmm_precisions_*_{case_id}.bin"):
        delete_if_exists(str(path), label="temporary input")


def execute_pipeline(
    case: BenchmarkCase,
    options: PipelineOptions,
    datasets_dir: str | Path = DATASETS_DIR,
    keep_inputs: bool = False,
    exclusion_rules: tuple[BenchmarkExclusionRule, ...] = (),
) -> None:
    print(f"\n--- Running Config: {case.label} ---")

    pipeline = build_pipeline(
        case,
        options,
        datasets_dir=datasets_dir,
        exclusion_rules=exclusion_rules,
    )

    if not pipeline:
        print(f"Skipping {case.label}: all enabled phases are excluded.")
        return

    for task in pipeline:
        if task.kind == "cpp_timing":
            run_cpp_task_with_timing_processes(task, options.timing_processes)
        elif task.kind == "cachegrind":
            run_cachegrind_task(task)
        else:
            print(f"[{task.name}] Running...")
            run_command(task.name, task.command)

    if not keep_inputs:
        cleanup_config_inputs(case.case_id, datasets_dir)
