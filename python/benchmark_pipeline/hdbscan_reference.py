"""scikit-learn HDBSCAN brute-path stage wrappers.

The wrappers in this module intentionally call scikit-learn's public estimator
and private HDBSCAN helpers instead of reimplementing those stages in Python.
They are the reference surface used while developing independently benchmarkable
C++ HDBSCAN stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.cluster._hdbscan.hdbscan import (  # type: ignore[reportPrivateImportUsage]
    _brute_mst,
    _hdbscan_brute,
    _process_mst,
    mutual_reachability_graph,
    tree_to_labels,
)
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from benchmark_metadata import (
    FULL_STAGE_KEY,
    HDBSCAN_CORE_STAGE_KEY,
    HDBSCAN_DISTANCE_STAGE_KEY,
    HDBSCAN_LINKAGE_STAGE_KEY,
    HDBSCAN_MREACH_STAGE_KEY,
    HDBSCAN_MST_STAGE_KEY,
    HDBSCAN_SELECT_STAGE_KEY,
    HDBSCAN_STAGE_KEYS,
    SKLEARN_BRUTE_REFERENCE,
)


SUPPORTED_HDBSCAN_REFERENCE_KEYS = (SKLEARN_BRUTE_REFERENCE,)


def validate_hdbscan_reference_key(reference_key: str) -> str:
    if reference_key not in SUPPORTED_HDBSCAN_REFERENCE_KEYS:
        valid = ", ".join(SUPPORTED_HDBSCAN_REFERENCE_KEYS)
        raise ValueError(
            f"Unsupported HDBSCAN reference {reference_key!r}. "
            f"Currently implemented references: {valid}"
        )
    return reference_key

HdbscanStageKey = Literal[
    "distance",
    "core",
    "mreach",
    "mst",
    "linkage",
    "select",
    "full",
]


@dataclass(frozen=True)
class HdbscanFullResult:
    labels: NDArray[np.int32]
    probabilities: NDArray[np.float64]
    single_linkage_tree: NDArray[Any]


@dataclass(frozen=True)
class HdbscanComposedResult:
    distance_matrix: NDArray[np.float64]
    core_distances: NDArray[np.float64]
    mutual_reachability_matrix: NDArray[np.float64]
    mst_edges: NDArray[Any]
    single_linkage_tree: NDArray[Any]
    labels: NDArray[np.int32]
    probabilities: NDArray[np.float64]


def validate_min_samples(min_samples: int, n_samples: int | None = None) -> None:
    if min_samples < 2:
        raise ValueError("HDBSCAN min_samples must be at least 2")
    if n_samples is not None and min_samples > n_samples:
        raise ValueError(
            f"HDBSCAN min_samples ({min_samples}) must be at most n_samples ({n_samples})"
        )


def validate_stage_key(stage_key: str) -> HdbscanStageKey:
    if stage_key not in HDBSCAN_STAGE_KEYS:
        valid = ", ".join(HDBSCAN_STAGE_KEYS)
        raise ValueError(f"Unknown HDBSCAN stage {stage_key!r}. Valid stages: {valid}")
    return stage_key  # type: ignore[return-value]


def as_sklearn_brute_input(X: ArrayLike) -> NDArray[np.float64]:
    """Match sklearn.HDBSCAN.fit's dense non-precomputed dtype/order path."""
    return np.asarray(X, dtype=np.float64, order="C")


def sklearn_brute_distance_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """distance: X -> dense Euclidean distance matrix."""
    X64 = as_sklearn_brute_input(X)
    with threadpool_limits(limits=1):
        distances = pairwise_distances(X64, metric="euclidean", n_jobs=1)
    return np.asarray(distances, dtype=np.float64, order="C")


def _sklearn_mutual_reachability_from_distance_matrix(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    distances = np.asarray(distance_matrix, dtype=np.float64, order="C").copy()
    validate_min_samples(min_samples, distances.shape[0])
    with threadpool_limits(limits=1):
        result = mutual_reachability_graph(
            distances,
            min_samples=min_samples,
            max_distance=0.0,
        )
    return np.asarray(result, dtype=np.float64, order="C")


def sklearn_brute_core_distances(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """core: distance matrix -> core distances.

    scikit-learn does not expose core-distance extraction as a standalone helper
    on the brute path. The private mutual_reachability_graph helper computes the
    same core distances internally and stores them on the diagonal of the dense
    mutual-reachability matrix, so the wrapper calls that helper and reads the
    diagonal instead of duplicating sklearn logic in Python.
    """
    mutual_reachability = _sklearn_mutual_reachability_from_distance_matrix(
        distance_matrix,
        min_samples=min_samples,
    )
    return np.ascontiguousarray(np.diag(mutual_reachability), dtype=np.float64)


def sklearn_brute_mutual_reachability_matrix(
    distance_matrix: ArrayLike,
    core_distances: ArrayLike | None = None,
    *,
    min_samples: int,
    validate_core: bool = True,
) -> NDArray[np.float64]:
    """mreach: distance matrix + core distances -> mutual reachability matrix."""
    mutual_reachability = _sklearn_mutual_reachability_from_distance_matrix(
        distance_matrix,
        min_samples=min_samples,
    )

    if validate_core and core_distances is not None:
        expected_core = np.asarray(core_distances, dtype=np.float64)
        observed_core = np.diag(mutual_reachability)
        np.testing.assert_allclose(observed_core, expected_core, rtol=0.0, atol=0.0)

    return mutual_reachability


def sklearn_brute_mst_edges(
    mutual_reachability_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[Any]:
    """mst: mutual reachability matrix -> MST edge list."""
    mutual_reachability = np.asarray(
        mutual_reachability_matrix,
        dtype=np.float64,
        order="C",
    )
    validate_min_samples(min_samples, mutual_reachability.shape[0])
    with threadpool_limits(limits=1):
        return _brute_mst(mutual_reachability, min_samples=min_samples)


def sklearn_brute_single_linkage_tree(mst_edges: NDArray[Any]) -> NDArray[Any]:
    """linkage: MST edge list -> single linkage tree."""
    with threadpool_limits(limits=1):
        return _process_mst(np.asarray(mst_edges).copy())


def sklearn_brute_select_clusters(
    single_linkage_tree: NDArray[Any],
    *,
    min_samples: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """select: single linkage tree -> labels and probabilities."""
    with threadpool_limits(limits=1):
        labels, probabilities = tree_to_labels(
            single_linkage_tree,
            min_samples,
            "eom",
            False,
            0.0,
            None,
        )
    return (
        np.asarray(labels, dtype=np.int32, order="C"),
        np.asarray(probabilities, dtype=np.float64, order="C"),
    )


def sklearn_brute_full(X: ArrayLike, *, min_samples: int) -> HdbscanFullResult:
    """full: X -> labels, probabilities via sklearn.cluster.HDBSCAN."""
    X64 = as_sklearn_brute_input(X)
    validate_min_samples(min_samples, X64.shape[0])
    with threadpool_limits(limits=1):
        estimator = SklearnHDBSCAN(
            min_cluster_size=min_samples,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="brute",
            alpha=1.0,
            n_jobs=1,
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            max_cluster_size=None,
            store_centers=None,
            copy=False,
        ).fit(X64)
    return HdbscanFullResult(
        labels=np.asarray(estimator.labels_, dtype=np.int32, order="C"),
        probabilities=np.asarray(estimator.probabilities_, dtype=np.float64, order="C"),
        single_linkage_tree=np.asarray(estimator._single_linkage_tree_),
    )


def sklearn_brute_private_full_linkage(
    X: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[Any]:
    """Full sklearn brute linkage helper without cluster selection."""
    X64 = as_sklearn_brute_input(X)
    validate_min_samples(min_samples, X64.shape[0])
    with threadpool_limits(limits=1):
        return _hdbscan_brute(
            X64,
            min_samples=min_samples,
            alpha=1.0,
            metric="euclidean",
            n_jobs=1,
            copy=False,
        )


def compose_sklearn_brute_stages(
    X: ArrayLike,
    *,
    min_samples: int,
) -> HdbscanComposedResult:
    """Run all standalone sklearn brute wrappers and return every stage output."""
    distance_matrix = sklearn_brute_distance_matrix(X)
    core_distances = sklearn_brute_core_distances(
        distance_matrix,
        min_samples=min_samples,
    )
    mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
        distance_matrix,
        core_distances,
        min_samples=min_samples,
    )
    mst_edges = sklearn_brute_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )
    single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)
    labels, probabilities = sklearn_brute_select_clusters(
        single_linkage_tree,
        min_samples=min_samples,
    )
    return HdbscanComposedResult(
        distance_matrix=distance_matrix,
        core_distances=core_distances,
        mutual_reachability_matrix=mutual_reachability_matrix,
        mst_edges=mst_edges,
        single_linkage_tree=single_linkage_tree,
        labels=labels,
        probabilities=probabilities,
    )


def assert_stage_composition_matches_full(
    X: ArrayLike,
    *,
    min_samples: int,
) -> None:
    """Verify that the callable stages compose to sklearn's brute estimator."""
    composed = compose_sklearn_brute_stages(X, min_samples=min_samples)
    private_full_linkage = sklearn_brute_private_full_linkage(X, min_samples=min_samples)
    full = sklearn_brute_full(X, min_samples=min_samples)

    np.testing.assert_array_equal(composed.single_linkage_tree, private_full_linkage)
    np.testing.assert_array_equal(composed.single_linkage_tree, full.single_linkage_tree)
    np.testing.assert_array_equal(composed.labels, full.labels)
    np.testing.assert_allclose(composed.probabilities, full.probabilities, rtol=0.0, atol=0.0)


def _fnv1a64_bytes(data: bytes) -> str:
    hash_value = 0xCBF29CE484222325
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hex(hash_value)


def _deterministic_weight(index: int) -> float:
    x = (int(index) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    mixed = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    return float((mixed >> 11) & 0x1FFFFF) / float(0x1FFFFF)


def _probe_indices(value_count: int) -> list[int]:
    if value_count <= 0:
        return []

    indices: list[int] = []
    fixed_count = min(value_count, 32)
    indices.extend(range(fixed_count))

    tail_start = value_count - 32 if value_count > 32 else value_count
    indices.extend(range(tail_start, value_count))

    state = (0x243F6A8885A308D3 ^ int(value_count)) & 0xFFFFFFFFFFFFFFFF
    for _ in range(64):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        indices.append(int(state % value_count))

    return sorted(set(indices))


def _float32_array_summary(values: NDArray[np.float32]) -> dict[str, Any]:
    flat = np.ascontiguousarray(values, dtype=np.float32).ravel()
    finite = np.isfinite(flat)
    finite_values = flat[finite].astype(np.float64, copy=False)
    finite_indices = np.flatnonzero(finite)

    if finite_values.size:
        weights = np.fromiter(
            (_deterministic_weight(int(index)) for index in finite_indices),
            dtype=np.float64,
            count=finite_values.size,
        )
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        sum_value = float(np.sum(finite_values, dtype=np.float64))
        sum_abs = float(np.sum(np.abs(finite_values), dtype=np.float64))
        sum_squares = float(np.dot(finite_values, finite_values))
        weighted_sum = float(np.dot(finite_values, weights))
    else:
        min_value = float("nan")
        max_value = float("nan")
        sum_value = 0.0
        sum_abs = 0.0
        sum_squares = 0.0
        weighted_sum = 0.0

    probe_indices = _probe_indices(flat.size)
    return {
        "value_count": int(flat.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nan_count": int(np.count_nonzero(np.isnan(flat))),
        "pos_inf_count": int(np.count_nonzero(flat == np.float32(np.inf))),
        "neg_inf_count": int(np.count_nonzero(flat == np.float32(-np.inf))),
        "sum": sum_value,
        "sum_abs": sum_abs,
        "sum_squares": sum_squares,
        "weighted_sum": weighted_sum,
        "min": min_value,
        "max": max_value,
        "fnv1a64_float32": _fnv1a64_bytes(flat.tobytes(order="C")),
        "probes": [
            {"index": int(index), "value": float(flat[index])}
            for index in probe_indices
        ],
    }


def hdbscan_distance_stage_metrics(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
    language: str,
) -> dict[str, Any]:
    matrix = np.ascontiguousarray(distance_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("HDBSCAN distance matrix metrics require a square 2-D matrix")

    diagonal_max_abs = float(np.max(np.abs(np.diag(matrix)))) if matrix.size else 0.0
    symmetry_max_abs = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0

    return {
        "schema_version": 1,
        "phase": "hdbscan",
        "language": language,
        "stage": "distance",
        "dtype": "float32",
        "n_samples": int(matrix.shape[0]),
        "min_samples": int(min_samples),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "diagonal_max_abs": diagonal_max_abs,
        "symmetry_max_abs": symmetry_max_abs,
        "summary": _float32_array_summary(matrix),
    }


def write_hdbscan_distance_stage_metrics(
    path: str,
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
    language: str,
) -> None:
    import json
    from pathlib import Path

    payload = hdbscan_distance_stage_metrics(
        distance_matrix,
        min_samples=min_samples,
        language=language,
    )
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
        f.write("\n")


def run_sklearn_brute_stage(
    stage_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> object:
    """Run one named sklearn brute stage, including predecessors as setup."""
    stage_key = validate_stage_key(stage_key)

    if stage_key == HDBSCAN_DISTANCE_STAGE_KEY:
        return sklearn_brute_distance_matrix(X)

    if stage_key == FULL_STAGE_KEY:
        return sklearn_brute_full(X, min_samples=min_samples)

    distance_matrix = sklearn_brute_distance_matrix(X)

    if stage_key == HDBSCAN_CORE_STAGE_KEY:
        return sklearn_brute_core_distances(distance_matrix, min_samples=min_samples)

    core_distances = sklearn_brute_core_distances(distance_matrix, min_samples=min_samples)

    if stage_key == HDBSCAN_MREACH_STAGE_KEY:
        return sklearn_brute_mutual_reachability_matrix(
            distance_matrix,
            core_distances,
            min_samples=min_samples,
        )

    mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
        distance_matrix,
        core_distances,
        min_samples=min_samples,
    )

    if stage_key == HDBSCAN_MST_STAGE_KEY:
        return sklearn_brute_mst_edges(
            mutual_reachability_matrix,
            min_samples=min_samples,
        )

    mst_edges = sklearn_brute_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )

    if stage_key == HDBSCAN_LINKAGE_STAGE_KEY:
        return sklearn_brute_single_linkage_tree(mst_edges)

    single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)

    if stage_key == HDBSCAN_SELECT_STAGE_KEY:
        return sklearn_brute_select_clusters(single_linkage_tree, min_samples=min_samples)

    raise AssertionError(f"Unhandled HDBSCAN stage {stage_key!r}")


def run_hdbscan_reference_stage(
    reference_key: str,
    stage_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> object:
    """Run one reference implementation stage, including predecessors as setup."""
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return run_sklearn_brute_stage(stage_key, X, min_samples=min_samples)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")
