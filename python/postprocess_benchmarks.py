import argparse
import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

PHASE_MAP = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Iterations",
}

LANG_MAP = {
    "cpp": "C++",
    "py": "Python",
}


BENCHMARK_JSON_RE = re.compile(
    r"^(?P<phase>soa|pp|lloyd)_(?P<lang>cpp|py)_(?P<dim>\d+)D_(?P<samples>\d+)S_(?P<clusters>\d+)K\.json$"
)
LLOYD_PARITY_JSON_RE = re.compile(
    r"^lloyd_parity_(?P<dim>\d+)D_(?P<samples>\d+)S_(?P<clusters>\d+)K\.json$"
)


def parse_benchmark_filename(path: Path) -> dict[str, Any] | None:
    match = BENCHMARK_JSON_RE.match(path.name)
    if not match:
        return None

    phase_key = match.group("phase")
    lang_key = match.group("lang")
    dim = int(match.group("dim"))
    samples = int(match.group("samples"))
    clusters = int(match.group("clusters"))

    return {
        "phase_key": phase_key,
        "phase": PHASE_MAP[phase_key],
        "language_key": lang_key,
        "language": LANG_MAP[lang_key],
        "dimensions": dim,
        "samples": samples,
        "clusters": clusters,
        "config_id": f"{dim}D_{samples}S_{clusters}K",
        "configuration": f"{dim}D | {samples}S | {clusters}K",
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def lloyd_iteration_count(
    lloyd_parity: dict[str, dict[str, Any]],
    *,
    config_id: str,
    lang_key: str,
) -> int:
    parity = lloyd_parity.get(config_id)

    if parity is None:
        raise RuntimeError(
            f"Missing Lloyd parity file for {config_id}. "
            f"The orchestrator should finalize each config before post-processing."
        )

    if lang_key == "cpp":
        return int(parity["cpp_iterations"])

    if lang_key == "py":
        return int(parity["python_iterations"])

    raise RuntimeError(f"Unexpected Lloyd language key: {lang_key!r}")


def normalize_lloyd_parity_record(
    record: dict[str, Any],
    *,
    config_id: str,
) -> dict[str, Any]:
    required_fields = [
        "cpp_iterations",
        "python_iterations",
        "cpp_inertia",
        "python_inertia",
        "inertia_diff_abs",
        "inertia_diff_pct",
        "tolerance_pct",
        "status",
    ]

    for field in required_fields:
        if field not in record:
            raise RuntimeError(f"Missing {field!r} in Lloyd parity for {config_id}")

    record_config_id = record.get("config_id")

    if record_config_id is not None and record_config_id != config_id:
        raise RuntimeError(
            f"Lloyd parity config mismatch: file is for {record_config_id}, "
            f"expected {config_id}"
        )

    return {
        **record,
        "config_id": config_id,
        "cpp_iterations": int(record["cpp_iterations"]),
        "python_iterations": int(record["python_iterations"]),
        "cpp_inertia": float(record["cpp_inertia"]),
        "python_inertia": float(record["python_inertia"]),
        "inertia_diff_abs": float(record["inertia_diff_abs"]),
        "inertia_diff_pct": float(record["inertia_diff_pct"]),
        "tolerance_pct": float(record["tolerance_pct"]),
        "status": str(record["status"]),
    }


def load_lloyd_parity_map(data_dir: Path) -> dict[str, dict[str, Any]]:
    parity_records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for path in sorted(data_dir.glob("lloyd_parity_*.json")):
        match = LLOYD_PARITY_JSON_RE.match(path.name)

        if not match:
            print(f"Skipping malformed Lloyd parity filename: {path.name}")
            continue

        dim = int(match.group("dim"))
        samples = int(match.group("samples"))
        clusters = int(match.group("clusters"))
        config_id = f"{dim}D_{samples}S_{clusters}K"

        parity = normalize_lloyd_parity_record(
            load_json(path),
            config_id=config_id,
        )

        if int(parity.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported Lloyd parity schema in {path}")

        parity_records[config_id] = parity

        if parity["status"] != "PASS":
            failures.append(parity)

    if failures:
        failed_ids = ", ".join(record["config_id"] for record in failures[:10])

        print(
            f"WARNING: Loaded {len(failures)} Lloyd parity failures. "
            f"First failures: {failed_ids}"
        )

    pass_count = sum(
        1 for record in parity_records.values() if record["status"] == "PASS"
    )
    fail_count = sum(
        1 for record in parity_records.values() if record["status"] != "PASS"
    )

    print(
        f"Loaded {len(parity_records)} Lloyd parity records "
        f"({pass_count} PASS, {fail_count} non-PASS)."
    )

    return parity_records


def iter_pyperf_values(path: Path):
    """
    Yield values while preserving pyperf run/process grouping.

    Output:
        {
            "benchmark_index": int,
            "run_index": int,
            "process_index": int,
            "value_index": int,
            "time_s": float,
        }
    """
    data = load_json(path)

    for benchmark_index, benchmark in enumerate(data.get("benchmarks", [])):
        runs = benchmark.get("runs", [])

        for run_index, run in enumerate(runs):
            metadata = run.get("metadata", {})

            # For merged C++ files, the merge script writes process_index.
            # For native pyperf Python files, run_index is the process grouping.
            process_index = metadata.get("process_index", run_index)

            values = run.get("values", [])

            for value_index, value in enumerate(values):
                yield {
                    "benchmark_index": benchmark_index,
                    "run_index": run_index,
                    "process_index": int(process_index),
                    "value_index": value_index,
                    "time_s": float(value),
                }


def load_process_aware_records(
    data_dir: Path,
    *,
    lloyd_parity: dict[str, dict[str, Any]],
    completed_config_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for json_path in sorted(data_dir.glob("*.json")):
        parsed = parse_benchmark_filename(json_path)
        if parsed is None:
            known_artifact_prefixes = (
                "lloyd_metrics_",
                "lloyd_parity_",
            )

            if json_path.name == "benchmark_summary.json" or json_path.name.startswith(
                known_artifact_prefixes
            ):
                continue

            print(f"Skipping non-benchmark JSON: {json_path.name}")
            continue

        phase_key = parsed["phase_key"]
        lang_key = parsed["language_key"]
        config_id = parsed["config_id"]

        if completed_config_ids is not None and config_id not in completed_config_ids:
            continue

        iterations: int | None

        if phase_key == "lloyd":
            iterations = lloyd_iteration_count(
                lloyd_parity,
                config_id=config_id,
                lang_key=lang_key,
            )
        else:
            iterations = 1

        for value_record in iter_pyperf_values(json_path):
            time_s = value_record["time_s"]

            records.append(
                {
                    **parsed,
                    "source_json": json_path.name,
                    "benchmark_index": value_record["benchmark_index"],
                    "run_index": value_record["run_index"],
                    "process_index": value_record["process_index"],
                    "value_index": value_record["value_index"],
                    "iterations": iterations,
                    "time_s": time_s,
                    "time_per_iteration_s": time_s / iterations,
                }
            )

    return records


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summary_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)

    if arr.size == 0:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "stddev": None,
            "mad": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
        }

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))

    if arr.size >= 2:
        stddev = float(np.std(arr, ddof=1))
    else:
        stddev = 0.0

    return {
        "n": int(arr.size),
        "median": median,
        "mean": float(np.mean(arr)),
        "stddev": stddev,
        "mad": mad,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": percentile(arr, 5),
        "p25": percentile(arr, 25),
        "p50": percentile(arr, 50),
        "p75": percentile(arr, 75),
        "p95": percentile(arr, 95),
    }


def stable_child_seed(base_seed: int, *parts: Any) -> int:
    """
    Create a deterministic child seed so bootstrap results do not depend on
    dict/list iteration accidents.
    """
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "parts": parts,
        },
        sort_keys=True,
    ).encode("utf-8")

    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def statistic_value(values: np.ndarray, statistic: str) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute statistic on empty values")

    if statistic == "median":
        return float(np.median(values))

    if statistic == "mean":
        return float(np.mean(values))

    raise ValueError(f"Unsupported statistic: {statistic}")


def values_by_process(
    records: list[dict[str, Any]],
    value_field: str,
) -> list[np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)

    for record in records:
        grouped[record["process_index"]].append(float(record[value_field]))

    process_values = [
        np.asarray(grouped[process_id], dtype=np.float64)
        for process_id in sorted(grouped)
    ]

    if not process_values:
        raise ValueError("No process groups found")

    for values in process_values:
        if values.size == 0:
            raise ValueError("Found empty process group")

    return process_values


def flatten_process_values(process_values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(process_values)


def clustered_resample_values(
    process_values: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Cluster bootstrap: resample whole processes/runs with replacement.
    Values inside each selected process remain grouped together.
    """
    n_processes = len(process_values)
    selected = rng.integers(0, n_processes, size=n_processes)

    return np.concatenate([process_values[i] for i in selected])


def clustered_bootstrap_speedup(
    cpp_records: list[dict[str, Any]],
    py_records: list[dict[str, Any]],
    *,
    value_field: str,
    statistic: str,
    bootstrap_iterations: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    """
    Speedup definition:

        Python time / C++ time

    So higher is better for C++.
    """
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be > 0")

    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    cpp_process_values = values_by_process(cpp_records, value_field)
    py_process_values = values_by_process(py_records, value_field)

    cpp_all = flatten_process_values(cpp_process_values)
    py_all = flatten_process_values(py_process_values)

    cpp_point = statistic_value(cpp_all, statistic)
    py_point = statistic_value(py_all, statistic)

    if cpp_point <= 0.0:
        raise ValueError(f"C++ point estimate must be positive, got {cpp_point}")

    point = py_point / cpp_point

    rng = np.random.default_rng(seed)
    ratios = np.empty(bootstrap_iterations, dtype=np.float64)

    for i in range(bootstrap_iterations):
        cpp_sample = clustered_resample_values(cpp_process_values, rng)
        py_sample = clustered_resample_values(py_process_values, rng)

        cpp_stat = statistic_value(cpp_sample, statistic)
        py_stat = statistic_value(py_sample, statistic)

        if cpp_stat <= 0.0:
            ratios[i] = np.nan
        else:
            ratios[i] = py_stat / cpp_stat

    ratios = ratios[np.isfinite(ratios)]

    if ratios.size == 0:
        raise ValueError("All bootstrap ratios were invalid")

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(ratios, 100.0 * alpha / 2.0))
    ci_high = float(np.percentile(ratios, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "point": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": float(ci_level),
        "bootstrap_iterations": int(bootstrap_iterations),
        "valid_bootstrap_iterations": int(ratios.size),
        "statistic": statistic,
        "value_field": value_field,
        "definition": "python_time / cpp_time",
        "cpp_point": float(cpp_point),
        "python_point": float(py_point),
    }


def get_single_iteration_count(records: list[dict[str, Any]]) -> int:
    iterations = sorted({int(record["iterations"]) for record in records})

    if len(iterations) != 1:
        raise RuntimeError(f"Expected exactly one iteration count, got {iterations}")

    iteration_count = iterations[0]

    if iteration_count <= 0:
        raise RuntimeError(f"Iteration count must be positive, got {iteration_count}")

    return iteration_count


def derive_per_iteration_speedup(
    total_speedup: dict[str, Any],
    *,
    cpp_iterations: int,
    python_iterations: int,
) -> dict[str, Any]:
    if cpp_iterations <= 0:
        raise ValueError(f"cpp_iterations must be positive, got {cpp_iterations}")

    if python_iterations <= 0:
        raise ValueError(f"python_iterations must be positive, got {python_iterations}")

    ratio_scale = cpp_iterations / python_iterations

    return {
        **total_speedup,
        "point": float(total_speedup["point"] * ratio_scale),
        "ci_low": float(total_speedup["ci_low"] * ratio_scale),
        "ci_high": float(total_speedup["ci_high"] * ratio_scale),
        "value_field": "time_per_iteration_s",
        "definition": "python_time_per_iteration / cpp_time_per_iteration",
        "cpp_point": float(total_speedup["cpp_point"] / cpp_iterations),
        "python_point": float(total_speedup["python_point"] / python_iterations),
        "derived_from": {
            "value_field": "time_s",
            "scale": float(ratio_scale),
            "cpp_iterations": int(cpp_iterations),
            "python_iterations": int(python_iterations),
        },
    }


def build_speedup_block(
    cpp_records: list[dict[str, Any]],
    py_records: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    cpp_iterations = get_single_iteration_count(cpp_records)
    python_iterations = get_single_iteration_count(py_records)

    total_median = clustered_bootstrap_speedup(
        cpp_records,
        py_records,
        value_field="time_s",
        statistic="median",
        bootstrap_iterations=bootstrap_iterations,
        ci_level=ci_level,
        seed=stable_child_seed(seed, "time_s", "median"),
    )

    total_mean = clustered_bootstrap_speedup(
        cpp_records,
        py_records,
        value_field="time_s",
        statistic="mean",
        bootstrap_iterations=bootstrap_iterations,
        ci_level=ci_level,
        seed=stable_child_seed(seed, "time_s", "mean"),
    )

    return {
        "time_s": {
            "median_ratio": total_median,
            "mean_ratio": total_mean,
        },
        "time_per_iteration_s": {
            "median_ratio": derive_per_iteration_speedup(
                total_median,
                cpp_iterations=cpp_iterations,
                python_iterations=python_iterations,
            ),
            "mean_ratio": derive_per_iteration_speedup(
                total_mean,
                cpp_iterations=cpp_iterations,
                python_iterations=python_iterations,
            ),
        },
    }


def group_records(records: list[dict[str, Any]]):
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        key = (
            record["dimensions"],
            record["samples"],
            record["clusters"],
            record["phase_key"],
            record["language_key"],
        )
        grouped[key].append(record)

    return grouped


def summarize_language_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    process_ids = sorted({record["process_index"] for record in records})

    values_by_process: dict[int, list[float]] = defaultdict(list)

    for record in records:
        values_by_process[record["process_index"]].append(record["time_s"])

    value_counts_by_process = {
        str(process_id): len(values_by_process[process_id])
        for process_id in process_ids
    }

    iterations = sorted({record["iterations"] for record in records})
    if len(iterations) != 1:
        raise RuntimeError(f"Expected exactly one iteration count, got {iterations}")

    return {
        "iterations": iterations[0],
        "n_processes": len(process_ids),
        "n_values": len(records),
        "values_per_process": value_counts_by_process,
        "time_s": summary_stats([record["time_s"] for record in records]),
        "time_per_iteration_s": summary_stats(
            [record["time_per_iteration_s"] for record in records]
        ),
    }


def build_summary(
    records: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    ci_level: float,
    bootstrap_seed: int,
    lloyd_parity: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_records(records)

    configs: dict[tuple[int, int, int], dict[str, Any]] = {}

    for (
        dim,
        samples,
        clusters,
        phase_key,
        language_key,
    ), group in sorted(grouped.items()):
        config_key = (dim, samples, clusters)

        if config_key not in configs:
            configs[config_key] = {
                "dimensions": dim,
                "samples": samples,
                "clusters": clusters,
                "config_id": f"{dim}D_{samples}S_{clusters}K",
                "configuration": f"{dim}D | {samples}S | {clusters}K",
                "phases": {},
            }

        phase_name = PHASE_MAP[phase_key]
        language_name = LANG_MAP[language_key]

        phase_entry = configs[config_key]["phases"].setdefault(
            phase_name,
            {
                "phase_key": phase_key,
                "languages": {},
            },
        )

        phase_entry["languages"][language_name] = summarize_language_records(group)

    # Add C++ vs Python speedup blocks.
    for config_key, config_entry in configs.items():
        dim, samples, clusters = config_key
        config_id = f"{dim}D_{samples}S_{clusters}K"

        for phase_name, phase_entry in config_entry["phases"].items():
            phase_key = phase_entry["phase_key"]

            cpp_records = grouped.get(
                (dim, samples, clusters, phase_key, "cpp"),
            )
            py_records = grouped.get(
                (dim, samples, clusters, phase_key, "py"),
            )

            if cpp_records and py_records:
                speedup_seed = stable_child_seed(
                    bootstrap_seed,
                    config_id,
                    phase_key,
                )

                phase_entry["speedup"] = build_speedup_block(
                    cpp_records,
                    py_records,
                    bootstrap_iterations=bootstrap_iterations,
                    ci_level=ci_level,
                    seed=speedup_seed,
                )

            # Add Lloyd inertia/parity.
            if phase_key == "lloyd":
                parity = lloyd_parity.get(config_id)

                if parity is None:
                    raise RuntimeError(f"Missing Lloyd parity record for {config_id}")

                phase_entry["languages"].setdefault("C++", {})["inertia"] = parity[
                    "cpp_inertia"
                ]

                phase_entry["languages"].setdefault("Python", {})["inertia"] = parity[
                    "python_inertia"
                ]

                phase_entry["parity"] = {
                    "cpp_iterations": parity["cpp_iterations"],
                    "python_iterations": parity["python_iterations"],
                    "cpp_inertia": parity["cpp_inertia"],
                    "python_inertia": parity["python_inertia"],
                    "inertia_diff_abs": parity["inertia_diff_abs"],
                    "inertia_diff_pct": parity["inertia_diff_pct"],
                    "tolerance_pct": parity["tolerance_pct"],
                    "status": parity["status"],
                }

    return {
        "metadata": {
            "schema_version": 1,
            "description": (
                "Post-processed benchmark summary. "
                "Raw pyperf/nanobench JSON values are grouped by process/run before aggregation."
            ),
            "time_unit": "seconds",
            "time_per_iteration_definition": (
                "For Lloyd, total benchmark time divided by Lloyd iteration count. "
                "For non-Lloyd phases, identical to total time."
            ),
            "speedup_definition": "python_time / cpp_time",
            "bootstrap": {
                "method": "independent clustered bootstrap by process/run",
                "iterations": int(bootstrap_iterations),
                "ci_level": float(ci_level),
                "seed": int(bootstrap_seed),
            },
            "lloyd_parity": {
                "source": "precomputed lloyd_parity_*.json files",
                "inertia_note": (
                    "Inertia is computed during per-config finalization from compact "
                    "Lloyd metrics. The post-processing step does not read data_*.bin files."
                ),
            },
        },
        "configs": list(configs.values()),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./datasets/benchmark_summary.json"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)

    args = parser.parse_args()

    print("Step 1/4: Loading Lloyd parity artifacts...")
    lloyd_parity = load_lloyd_parity_map(args.data_dir)

    print(f"Step 2/4: Loading benchmark records from {args.data_dir}...")
    completed_config_ids = set(lloyd_parity)

    records = load_process_aware_records(
        args.data_dir,
        lloyd_parity=lloyd_parity,
        completed_config_ids=completed_config_ids,
    )

    print("Step 3/4: Building summary and running bootstrap intervals...")
    summary = build_summary(
        records,
        bootstrap_iterations=args.bootstrap_iterations,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
        lloyd_parity=lloyd_parity,
    )

    print(f"Step 4/4: Writing output to {args.output}...")
    write_json(args.output, summary)

    print("\n--- Execution Complete ---")
    print(f"Wrote {args.output}")
    print(f"Configurations: {len(summary['configs'])}")
    print(f"Raw timing values: {len(records)}")
    print(f"Bootstrap iterations: {args.bootstrap_iterations}")

    print(f"Lloyd parity configs: {len(lloyd_parity)}")
