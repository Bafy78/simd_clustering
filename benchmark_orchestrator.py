import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Optional
import shutil
import json


@dataclass
class Task:
    name: str
    command: List[str]
    cpp_json_arg: Optional[int] = None
    cpp_metrics_arg: Optional[int] = None


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def assert_close(name: str, a: float, b: float, *, rel_tol=1e-10, abs_tol=1e-8):
    if a == b:
        return

    scale = max(abs(a), abs(b), 1.0)
    diff = abs(a - b)

    if diff > max(abs_tol, rel_tol * scale):
        raise RuntimeError(
            f"C++ metrics mismatch for {name}: {a} vs {b} " f"(abs diff={diff})"
        )


def validate_cpp_process_metrics(process_metrics: list[str]) -> dict:
    if not process_metrics:
        raise RuntimeError("No C++ process metrics files to validate")

    reference = load_json(process_metrics[0])

    for path in process_metrics[1:]:
        candidate = load_json(path)

        if candidate.get("schema_version") != reference.get("schema_version"):
            raise RuntimeError(f"schema_version mismatch in {path}")

        if candidate.get("language") != reference.get("language"):
            raise RuntimeError(f"language mismatch in {path}")

        if int(candidate["iterations"]) != int(reference["iterations"]):
            raise RuntimeError(
                f"iteration mismatch in {path}: "
                f"{candidate['iterations']} vs {reference['iterations']}"
            )

        if candidate["cluster_counts"] != reference["cluster_counts"]:
            raise RuntimeError(f"cluster_counts mismatch in {path}")

        assert_close(
            f"inertia in {path}",
            float(candidate["inertia"]),
            float(reference["inertia"]),
        )

        if len(candidate["cluster_inertia"]) != len(reference["cluster_inertia"]):
            raise RuntimeError(f"cluster_inertia length mismatch in {path}")

        if len(candidate.get("centroids", [])) != len(reference.get("centroids", [])):
            raise RuntimeError(f"centroid count mismatch in {path}")

        for i, (a, b) in enumerate(
            zip(candidate["cluster_inertia"], reference["cluster_inertia"])
        ):
            assert_close(
                f"cluster_inertia[{i}] in {path}",
                float(a),
                float(b),
            )

        for k, (cand_centroid, ref_centroid) in enumerate(
            zip(candidate.get("centroids", []), reference.get("centroids", []))
        ):
            if len(cand_centroid) != len(ref_centroid):
                raise RuntimeError(f"centroid[{k}] dimensionality mismatch in {path}")

            for d, (a, b) in enumerate(zip(cand_centroid, ref_centroid)):
                assert_close(
                    f"centroid[{k}][{d}] in {path}",
                    float(a),
                    float(b),
                )

    reference["process_metrics_verified"] = True
    reference["process_metrics_count"] = len(process_metrics)

    return reference


def write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def compute_lloyd_parity(
    *,
    config_id: str,
    cpp_metrics_file: str,
    py_metrics_file: str,
    output_file: str,
    tolerance_pct: float,
) -> dict:
    cpp = load_json(cpp_metrics_file)
    py = load_json(py_metrics_file)

    cpp_inertia = float(cpp["inertia"])
    py_inertia = float(py["inertia"])

    inertia_diff_abs = abs(cpp_inertia - py_inertia)
    scale = max(abs(cpp_inertia), abs(py_inertia))
    if scale > 0.0:
        inertia_diff_pct = inertia_diff_abs / scale * 100.0
    else:
        inertia_diff_pct = 0.0

    parity = {
        "schema_version": 1,
        "config_id": config_id,
        "cpp_iterations": int(cpp["iterations"]),
        "python_iterations": int(py["iterations"]),
        "cpp_inertia": cpp_inertia,
        "python_inertia": py_inertia,
        "inertia_diff_abs": inertia_diff_abs,
        "inertia_diff_pct": inertia_diff_pct,
        "tolerance_pct": tolerance_pct,
        "status": "PASS" if inertia_diff_pct <= tolerance_pct else "FAIL",
        "cpp_cluster_counts": cpp.get("cluster_counts"),
        "python_cluster_counts": py.get("cluster_counts"),
        "cpp_cluster_inertia": cpp.get("cluster_inertia"),
        "python_cluster_inertia": py.get("cluster_inertia"),
    }

    write_json(output_file, parity)

    if parity["status"] != "PASS":
        print(
            f"WARNING: Lloyd parity failed for {config_id}: "
            f"{inertia_diff_pct:.12g}% > {tolerance_pct}%"
        )

    return parity


def cleanup_config_inputs(config_id: str) -> None:
    paths = [
        f"./datasets/data_{config_id}.bin",
        f"./datasets/init_{config_id}.bin",
    ]

    for path in paths:
        try:
            os.remove(path)
            print(f"Deleted temporary input: {path}")
        except FileNotFoundError:
            pass


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
        "python/merge_pyperf_runs.py",
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
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    for path in process_metrics:
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


def delete_if_exists(path: str) -> None:
    try:
        os.remove(path)
        print(f"Deleted intermediate artifact: {path}")
    except FileNotFoundError:
        pass


def finalize_config(
    *,
    dim: int,
    n_samples: int,
    n_clusters: int,
    lloyd_parity_tolerance_pct: float,
) -> None:
    config_id = f"{dim}D_{n_samples}S_{n_clusters}K"

    try:
        cpp_metrics_file = f"./datasets/lloyd_metrics_cpp_{config_id}.json"
        py_metrics_file = f"./datasets/lloyd_metrics_py_{config_id}.json"
        parity_file = f"./datasets/lloyd_parity_{config_id}.json"

        print(f"[Validate: Lloyd Parity] {config_id}...")
        compute_lloyd_parity(
            config_id=config_id,
            cpp_metrics_file=cpp_metrics_file,
            py_metrics_file=py_metrics_file,
            output_file=parity_file,
            tolerance_pct=lloyd_parity_tolerance_pct,
        )

    finally:
        cleanup_config_inputs(config_id)

        delete_if_exists(cpp_metrics_file)
        delete_if_exists(py_metrics_file)


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
                f"./datasets/lloyd_metrics_cpp_{config_id}.json",
                f"./datasets/lloyd_cpp_{config_id}.json",
                str(bench_values),
                str(bench_min_time),
            ],
            cpp_metrics_arg=5,
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
                "--metrics-file",
                f"./datasets/lloyd_metrics_py_{config_id}.json",
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


if __name__ == "__main__":
    datasets_dir = "./datasets"

    if os.path.exists(datasets_dir):
        print(f"Cleaning {datasets_dir}...")
        shutil.rmtree(datasets_dir)

    os.makedirs(datasets_dir, exist_ok=True)

    test_dimensions = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
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
    test_clusters = [5, 12, 20]

    bench_processes = 3
    bench_values = 5
    bench_min_time = 0.05
    lloyd_parity_tolerance_pct = 1e-6

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
                    lloyd_parity_tolerance_pct,
                )

    print("\nAll benchmarking finished successfully!")
