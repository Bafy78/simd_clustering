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


def timing_values_by_timing_process(
    records: list[dict[str, Any]],
    time_field: str,
) -> list[np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)

    for record in records:
        grouped[record["timing_process_index"]].append(float(record[time_field]))

    timing_values_by_timing_process = [
        np.asarray(grouped[timing_process_id], dtype=np.float64)
        for timing_process_id in sorted(grouped)
    ]

    if not timing_values_by_timing_process:
        raise ValueError("No timing process groups found")

    for timing_values in timing_values_by_timing_process:
        if timing_values.size == 0:
            raise ValueError("Found empty timing process group")

    return timing_values_by_timing_process


def flatten_timing_values(
    timing_values_by_timing_process: list[np.ndarray],
) -> np.ndarray:
    return np.concatenate(timing_values_by_timing_process)


def clustered_resample_timing_values(
    timing_values_by_timing_process: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Cluster bootstrap: resample whole timing processes/pyperf runs with replacement.
    Timing values inside each selected timing process remain grouped together.
    """
    timing_process_count = len(timing_values_by_timing_process)
    selected_timing_processes = rng.integers(
        0,
        timing_process_count,
        size=timing_process_count,
    )

    return np.concatenate(
        [
            timing_values_by_timing_process[timing_process_index]
            for timing_process_index in selected_timing_processes
        ]
    )


def clustered_bootstrap_speedup(
    cpp_records: list[dict[str, Any]],
    py_records: list[dict[str, Any]],
    *,
    time_field: str,
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

    cpp_timing_values_by_timing_process = timing_values_by_timing_process(
        cpp_records, time_field
    )
    py_timing_values_by_timing_process = timing_values_by_timing_process(
        py_records, time_field
    )

    cpp_all = flatten_timing_values(cpp_timing_values_by_timing_process)
    py_all = flatten_timing_values(py_timing_values_by_timing_process)

    cpp_point = statistic_value(cpp_all, statistic)
    py_point = statistic_value(py_all, statistic)

    if cpp_point <= 0.0:
        raise ValueError(f"C++ point estimate must be positive, got {cpp_point}")

    point = py_point / cpp_point

    rng = np.random.default_rng(seed)
    ratios = np.empty(bootstrap_iterations, dtype=np.float64)

    for bootstrap_i in range(bootstrap_iterations):
        cpp_sample = clustered_resample_timing_values(
            cpp_timing_values_by_timing_process, rng
        )
        py_sample = clustered_resample_timing_values(
            py_timing_values_by_timing_process, rng
        )

        cpp_stat = statistic_value(cpp_sample, statistic)
        py_stat = statistic_value(py_sample, statistic)

        if cpp_stat <= 0.0:
            ratios[bootstrap_i] = np.nan
        else:
            ratios[bootstrap_i] = py_stat / cpp_stat

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
        "time_field": time_field,
        "definition": "python_time / cpp_time",
        "cpp_point": float(cpp_point),
        "python_point": float(py_point),
    }


def get_single_algorithm_iteration_count(records: list[dict[str, Any]]) -> int:
    algorithm_iterations = sorted(
        {int(record["algorithm_iterations"]) for record in records}
    )

    if len(algorithm_iterations) != 1:
        raise RuntimeError(
            "Expected exactly one algorithm-iteration count, "
            f"got {algorithm_iterations}"
        )

    algorithm_iteration_count = algorithm_iterations[0]

    if algorithm_iteration_count <= 0:
        raise RuntimeError(
            "Algorithm-iteration count must be positive, "
            f"got {algorithm_iteration_count}"
        )

    return algorithm_iteration_count


def derive_per_algorithm_iteration_speedup(
    total_speedup: dict[str, Any],
    *,
    cpp_algorithm_iterations: int,
    python_algorithm_iterations: int,
) -> dict[str, Any]:
    if cpp_algorithm_iterations <= 0:
        raise ValueError(
            f"cpp_algorithm_iterations must be positive, got {cpp_algorithm_iterations}"
        )

    if python_algorithm_iterations <= 0:
        raise ValueError(
            "python_algorithm_iterations must be positive, "
            f"got {python_algorithm_iterations}"
        )

    ratio_scale = cpp_algorithm_iterations / python_algorithm_iterations

    return {
        **total_speedup,
        "point": float(total_speedup["point"] * ratio_scale),
        "ci_low": float(total_speedup["ci_low"] * ratio_scale),
        "ci_high": float(total_speedup["ci_high"] * ratio_scale),
        "time_field": "time_per_algorithm_iteration_s",
        "definition": "python_time_per_algorithm_iteration / cpp_time_per_algorithm_iteration",
        "cpp_point": float(total_speedup["cpp_point"] / cpp_algorithm_iterations),
        "python_point": float(
            total_speedup["python_point"] / python_algorithm_iterations
        ),
        "derived_from": {
            "time_field": "time_s",
            "scale": float(ratio_scale),
            "cpp_algorithm_iterations": int(cpp_algorithm_iterations),
            "python_algorithm_iterations": int(python_algorithm_iterations),
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
    cpp_algorithm_iterations = get_single_algorithm_iteration_count(cpp_records)
    python_algorithm_iterations = get_single_algorithm_iteration_count(py_records)

    total_median = clustered_bootstrap_speedup(
        cpp_records,
        py_records,
        time_field="time_s",
        statistic="median",
        bootstrap_iterations=bootstrap_iterations,
        ci_level=ci_level,
        seed=stable_child_seed(seed, "time_s", "median"),
    )

    total_mean = clustered_bootstrap_speedup(
        cpp_records,
        py_records,
        time_field="time_s",
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
        "time_per_algorithm_iteration_s": {
            "median_ratio": derive_per_algorithm_iteration_speedup(
                total_median,
                cpp_algorithm_iterations=cpp_algorithm_iterations,
                python_algorithm_iterations=python_algorithm_iterations,
            ),
            "mean_ratio": derive_per_algorithm_iteration_speedup(
                total_mean,
                cpp_algorithm_iterations=cpp_algorithm_iterations,
                python_algorithm_iterations=python_algorithm_iterations,
            ),
        },
    }
