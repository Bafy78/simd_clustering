from typing import Any

import numpy as np


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


def statistic_value(values: np.ndarray, statistic: str) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute statistic on empty values")

    if statistic == "median":
        return float(np.median(values))

    if statistic == "mean":
        return float(np.mean(values))

    raise ValueError(f"Unsupported statistic: {statistic}")
