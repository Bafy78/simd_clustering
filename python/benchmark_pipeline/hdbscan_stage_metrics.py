"""Compact HDBSCAN stage-output metrics.

These summaries are benchmark/postprocess artifacts, not debugging fixtures.
They intentionally summarize float32 stage-boundary outputs without retaining
full matrices in JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


_UINT64_MASK = (1 << 64) - 1
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211


def _fnv1a64_float32(values: NDArray[np.float32]) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    hash_value = _FNV_OFFSET
    for byte in contiguous.view(np.uint8):
        hash_value ^= int(byte)
        hash_value = (hash_value * _FNV_PRIME) & _UINT64_MASK
    return f"0x{hash_value:x}"


def _deterministic_weight(index: int) -> float:
    x = (int(index) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    mixed = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    return float((mixed >> 11) & 0x1FFFFF) / float(0x1FFFFF)


def _probe_indices(value_count: int) -> list[int]:
    if value_count == 0:
        return []

    indices: list[int] = []
    fixed_count = min(value_count, 32)
    indices.extend(range(fixed_count))

    tail_start = value_count - 32 if value_count > 32 else value_count
    indices.extend(range(tail_start, value_count))

    state = (0x243F6A8885A308D3 ^ value_count) & _UINT64_MASK
    for _ in range(64):
        state = (state * 6364136223846793005 + 1442695040888963407) & _UINT64_MASK
        indices.append(state % value_count)

    return sorted(set(indices))


def _float32_summary(values: ArrayLike) -> dict[str, Any]:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    value_count = int(flat.size)

    finite_mask = np.isfinite(flat)
    nan_count = int(np.isnan(flat).sum())
    pos_inf_count = int(np.isposinf(flat).sum())
    neg_inf_count = int(np.isneginf(flat).sum())
    finite_count = int(finite_mask.sum())

    finite_values = flat[finite_mask].astype(np.float64, copy=False)
    if finite_count:
        sum_value = float(finite_values.sum(dtype=np.float64))
        sum_abs = float(np.abs(finite_values).sum(dtype=np.float64))
        sum_squares = float(np.square(finite_values).sum(dtype=np.float64))
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
    else:
        sum_value = 0.0
        sum_abs = 0.0
        sum_squares = 0.0
        min_value = float("nan")
        max_value = float("nan")

    weighted_sum = 0.0
    for index, value in enumerate(flat):
        value64 = float(value)
        if np.isfinite(value64):
            weighted_sum += value64 * _deterministic_weight(index)

    probes = [
        {"index": int(index), "value": float(flat[index])}
        for index in _probe_indices(value_count)
    ]

    return {
        "value_count": value_count,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "sum": sum_value,
        "sum_abs": sum_abs,
        "sum_squares": sum_squares,
        "weighted_sum": weighted_sum,
        "min": min_value,
        "max": max_value,
        "fnv1a64_float32": _fnv1a64_float32(flat),
        "probes": probes,
    }


def _symmetry_max_abs(matrix: NDArray[np.float32]) -> float:
    n_samples = matrix.shape[0]
    max_abs = 0.0
    for row in range(n_samples - 1):
        diff = np.abs(matrix[row, row + 1 :] - matrix[row + 1 :, row])
        if diff.size:
            max_abs = max(max_abs, float(diff.max()))
    return max_abs


def _matrix32(values: ArrayLike) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32, order="C")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2-D matrix, got shape {matrix.shape}")
    return matrix


def _base_payload(
    *,
    stage: str,
    matrix: NDArray[np.float32],
    min_samples: int,
    language: str,
) -> dict[str, Any]:
    n_samples = int(matrix.shape[0])
    return {
        "schema_version": 1,
        "phase": "hdbscan",
        "language": language,
        "stage": stage,
        "dtype": "float32",
        "n_samples": n_samples,
        "min_samples": int(min_samples),
        "shape": [n_samples, n_samples],
        "symmetry_max_abs": _symmetry_max_abs(matrix),
        "summary": _float32_summary(matrix),
    }


def hdbscan_stage_metrics_payload(
    stage: str,
    values: ArrayLike,
    *,
    min_samples: int,
    language: str,
) -> dict[str, Any]:
    matrix = _matrix32(values)

    if stage == "distance":
        payload = _base_payload(
            stage=stage,
            matrix=matrix,
            min_samples=min_samples,
            language=language,
        )
        payload["diagonal_max_abs"] = float(np.abs(np.diag(matrix)).max(initial=0.0))
        return payload

    if stage == "mreach":
        payload = _base_payload(
            stage=stage,
            matrix=matrix,
            min_samples=min_samples,
            language=language,
        )
        payload["diagonal_summary"] = _float32_summary(np.diag(matrix))
        return payload

    raise ValueError(f"Unsupported HDBSCAN metrics stage {stage!r}")


def write_hdbscan_stage_metrics(
    path: str | Path,
    stage: str,
    values: ArrayLike,
    *,
    min_samples: int,
    language: str,
) -> None:
    payload = hdbscan_stage_metrics_payload(
        stage,
        values,
        min_samples=min_samples,
        language=language,
    )
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
        f.write("\n")
