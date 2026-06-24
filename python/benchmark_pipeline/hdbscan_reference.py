"""scikit-learn HDBSCAN brute-path stage wrappers.

The wrappers in this module intentionally call scikit-learn's private HDBSCAN
helpers instead of reimplementing those stages in Python. The public reference
contract is float32 at dense stage boundaries: input data, distance matrices,
and mutual-reachability matrices are float32. Some scikit-learn private helpers
still require float64 internally; those wrappers promote only at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster._hdbscan.hdbscan import (  # type: ignore[reportPrivateImportUsage]
    HIERARCHY_dtype,
    MST_edge_dtype,
    _brute_mst,
    _process_mst,
    mutual_reachability_graph,
    tree_to_labels,
)
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from benchmark_metadata import (
    FULL_STAGE_KEY,
    HDBSCAN_DISTANCE_STAGE_KEY,
    HDBSCAN_LINKAGE_STAGE_KEY,
    HDBSCAN_MREACH_STAGE_KEY,
    HDBSCAN_MST_STAGE_KEY,
    HDBSCAN_SELECT_STAGE_KEY,
    HDBSCAN_STAGE_KEYS,
    SKLEARN_BRUTE_REFERENCE,
)


SUPPORTED_HDBSCAN_REFERENCE_KEYS = (SKLEARN_BRUTE_REFERENCE,)


HdbscanStageKey = Literal[
    "distance",
    "mreach",
    "mst",
    "linkage",
    "select",
    "full",
]


@dataclass(frozen=True)
class HdbscanFullResult:
    labels: NDArray[np.int32]
    probabilities: NDArray[np.float32]
    single_linkage_tree: NDArray[Any]


def validate_hdbscan_reference_key(reference_key: str) -> str:
    if reference_key not in SUPPORTED_HDBSCAN_REFERENCE_KEYS:
        valid = ", ".join(SUPPORTED_HDBSCAN_REFERENCE_KEYS)
        raise ValueError(
            f"Unsupported HDBSCAN reference {reference_key!r}. "
            f"Currently implemented references: {valid}"
        )
    return reference_key


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


def as_sklearn_brute_input(X: ArrayLike) -> NDArray[np.float32]:
    """Return the float32 dense input used by the staged sklearn reference."""
    return np.asarray(X, dtype=np.float32, order="C")


def sklearn_brute_distance_matrix(X: ArrayLike) -> NDArray[np.float32]:
    """distance: X -> dense Euclidean distance matrix."""
    X32 = as_sklearn_brute_input(X)
    with threadpool_limits(limits=1):
        distances = pairwise_distances(X32, metric="euclidean", n_jobs=1)
    return np.asarray(distances, dtype=np.float32, order="C")


def sklearn_brute_mutual_reachability_matrix(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float32]:
    """mreach: distance matrix -> mutual reachability matrix.

    This stage includes scikit-learn's internal core-distance computation. The
    private helper stores those core distances on the diagonal of the returned
    mutual-reachability matrix.
    """
    distances = np.asarray(distance_matrix, dtype=np.float32, order="C").copy()
    validate_min_samples(min_samples, distances.shape[0])
    with threadpool_limits(limits=1):
        result = mutual_reachability_graph(
            distances,
            min_samples=min_samples,
            max_distance=0.0,
        )
    return np.asarray(result, dtype=np.float32, order="C")


def as_float32_mst_edges(mst_edges: ArrayLike) -> NDArray[Any]:
    """Return sklearn MST-edge dtype with float32-rounded edge weights."""
    edges = np.asarray(mst_edges)
    rounded = np.empty(edges.shape[0], dtype=MST_edge_dtype)

    if edges.dtype.names is not None:
        left = edges["current_node"]
        right = edges["next_node"]
        distance = edges["distance"]
    else:
        if edges.ndim != 2 or edges.shape[1] < 3:
            raise ValueError(f"Expected MST edges with at least 3 columns, got shape {edges.shape}")
        left = edges[:, 0]
        right = edges[:, 1]
        distance = edges[:, 2]

    rounded["current_node"] = np.asarray(left, dtype=np.intp)
    rounded["next_node"] = np.asarray(right, dtype=np.intp)
    rounded["distance"] = np.asarray(distance, dtype=np.float32).astype(np.float64)
    return rounded


def as_float32_single_linkage_tree(single_linkage_tree: ArrayLike) -> NDArray[Any]:
    """Return sklearn hierarchy dtype with float32-rounded merge distances."""
    tree = np.asarray(single_linkage_tree)
    rounded = np.empty(tree.shape[0], dtype=HIERARCHY_dtype)

    if tree.dtype.names is not None:
        left = tree["left_node"]
        right = tree["right_node"]
        distance = tree["value"]
        cluster_size = tree["cluster_size"]
    else:
        if tree.ndim != 2 or tree.shape[1] < 4:
            raise ValueError(f"Expected single linkage tree with at least 4 columns, got shape {tree.shape}")
        left = tree[:, 0]
        right = tree[:, 1]
        distance = tree[:, 2]
        cluster_size = tree[:, 3]

    rounded["left_node"] = np.asarray(left, dtype=np.intp)
    rounded["right_node"] = np.asarray(right, dtype=np.intp)
    rounded["value"] = np.asarray(distance, dtype=np.float32).astype(np.float64)
    rounded["cluster_size"] = np.asarray(cluster_size, dtype=np.intp)
    return rounded


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
        mst_edges = _brute_mst(mutual_reachability, min_samples=min_samples)
    return as_float32_mst_edges(mst_edges)


def sklearn_brute_single_linkage_tree(mst_edges: NDArray[Any]) -> NDArray[Any]:
    """linkage: MST edge list -> single linkage tree."""
    with threadpool_limits(limits=1):
        tree = _process_mst(as_float32_mst_edges(mst_edges))
    return as_float32_single_linkage_tree(tree)


def sklearn_brute_select_clusters(
    single_linkage_tree: NDArray[Any],
    *,
    min_samples: int,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """select: single linkage tree -> labels and probabilities."""
    tree = as_float32_single_linkage_tree(single_linkage_tree)
    with threadpool_limits(limits=1):
        labels, probabilities = tree_to_labels(
            tree,
            min_samples,
            "eom",
            False,
            0.0,
            None,
        )
    return (
        np.asarray(labels, dtype=np.int32, order="C"),
        np.asarray(probabilities, dtype=np.float32, order="C"),
    )


def sklearn_brute_full(X: ArrayLike, *, min_samples: int) -> HdbscanFullResult:
    """full: X -> labels and probabilities via the staged float32 reference."""
    distance_matrix = sklearn_brute_distance_matrix(X)
    mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
        distance_matrix,
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
    return HdbscanFullResult(
        labels=labels,
        probabilities=probabilities,
        single_linkage_tree=single_linkage_tree,
    )


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

    if stage_key == HDBSCAN_MREACH_STAGE_KEY:
        return sklearn_brute_mutual_reachability_matrix(
            distance_matrix,
            min_samples=min_samples,
        )

    mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
        distance_matrix,
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
