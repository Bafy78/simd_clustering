import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np

from benchmark_postprocess.stats import statistic_value


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
    process_count = len(process_values)
    selected_processes = rng.integers(0, process_count, size=process_count)

    return np.concatenate([process_values[i] for i in selected_processes])


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
