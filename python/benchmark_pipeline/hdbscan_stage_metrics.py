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



def _mst_edges_to_flat_float32(values: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    edges = np.asarray(values)

    if edges.dtype.names is not None:
        left = np.asarray(edges["current_node"], dtype=np.float32)
        right = np.asarray(edges["next_node"], dtype=np.float32)
        weight = np.asarray(edges["distance"], dtype=np.float32)
    else:
        if edges.ndim != 2 or edges.shape[1] < 3:
            raise ValueError(
                "Expected MST edges to be either a structured sklearn edge array "
                f"or a 2-D array with at least 3 columns, got shape {edges.shape}"
            )
        left = np.asarray(edges[:, 0], dtype=np.float32)
        right = np.asarray(edges[:, 1], dtype=np.float32)
        weight = np.asarray(edges[:, 2], dtype=np.float32)

    edge_count = int(weight.size)
    flat = np.empty(edge_count * 3, dtype=np.float32)
    flat[0::3] = left
    flat[1::3] = right
    flat[2::3] = weight
    return np.ascontiguousarray(flat), np.ascontiguousarray(weight)



def _single_linkage_to_flat_float32(values: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    tree = np.asarray(values)

    if tree.dtype.names is not None:
        left = np.asarray(tree["left_node"], dtype=np.float32)
        right = np.asarray(tree["right_node"], dtype=np.float32)
        distance = np.asarray(tree["value"], dtype=np.float32)
        cluster_size = np.asarray(tree["cluster_size"], dtype=np.float32)
    else:
        if tree.ndim != 2 or tree.shape[1] < 4:
            raise ValueError(
                "Expected single linkage tree to be either a structured sklearn tree "
                f"or a 2-D array with at least 4 columns, got shape {tree.shape}"
            )
        left = np.asarray(tree[:, 0], dtype=np.float32)
        right = np.asarray(tree[:, 1], dtype=np.float32)
        distance = np.asarray(tree[:, 2], dtype=np.float32)
        cluster_size = np.asarray(tree[:, 3], dtype=np.float32)

    row_count = int(distance.size)
    flat = np.empty(row_count * 4, dtype=np.float32)
    flat[0::4] = left
    flat[1::4] = right
    flat[2::4] = distance
    flat[3::4] = cluster_size
    return (
        np.ascontiguousarray(flat),
        np.ascontiguousarray(distance),
        np.ascontiguousarray(cluster_size),
    )


def _select_to_flat_float32(values: object) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValueError("Expected select stage output to be a (labels, probabilities) tuple")

    labels = np.ascontiguousarray(values[0], dtype=np.int32)
    probabilities = np.ascontiguousarray(values[1], dtype=np.float32)
    if labels.shape != probabilities.shape:
        raise ValueError(
            "Expected select labels and probabilities to have identical shape, "
            f"got {labels.shape} and {probabilities.shape}"
        )

    label_values = np.ascontiguousarray(labels.astype(np.float32), dtype=np.float32)
    flat = np.empty(labels.size * 2, dtype=np.float32)
    flat[0::2] = label_values
    flat[1::2] = probabilities
    return flat, label_values, probabilities

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
    if stage == "mst":
        flat_edges, weights = _mst_edges_to_flat_float32(values)
        edge_count = int(weights.size)
        return {
            "schema_version": 1,
            "phase": "hdbscan",
            "language": language,
            "stage": stage,
            "dtype": "float32",
            "n_samples": edge_count + 1 if edge_count else 0,
            "min_samples": int(min_samples),
            "edge_count": edge_count,
            "shape": [edge_count, 3],
            "summary": _float32_summary(flat_edges),
            "weight_summary": _float32_summary(weights),
        }

    if stage == "linkage":
        flat_tree, distances, cluster_sizes = _single_linkage_to_flat_float32(values)
        row_count = int(distances.size)
        return {
            "schema_version": 1,
            "phase": "hdbscan",
            "language": language,
            "stage": stage,
            "dtype": "float32",
            "n_samples": row_count + 1 if row_count else 0,
            "min_samples": int(min_samples),
            "row_count": row_count,
            "shape": [row_count, 4],
            "summary": _float32_summary(flat_tree),
            "distance_summary": _float32_summary(distances),
            "cluster_size_summary": _float32_summary(cluster_sizes),
        }

    if stage == "select":
        flat, labels, probabilities = _select_to_flat_float32(values)
        n_samples = int(labels.size)
        unique_clusters = np.unique(labels[labels >= 0])
        return {
            "schema_version": 1,
            "phase": "hdbscan",
            "language": language,
            "stage": stage,
            "dtype": "float32",
            "n_samples": n_samples,
            "min_samples": int(min_samples),
            "shape": [n_samples, 2],
            "noise_count": int(np.count_nonzero(labels < 0)),
            "cluster_count": int(unique_clusters.size),
            "summary": _float32_summary(flat),
            "label_summary": _float32_summary(labels),
            "probability_summary": _float32_summary(probabilities),
        }

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
