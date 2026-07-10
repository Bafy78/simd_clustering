"""HDBSCAN reference-stage wrappers.

The wrappers in this module intentionally call implementation internals where
possible, with small normalization layers around staged HDBSCAN boundaries. The
dense staged contract keeps squared Euclidean weights through distance, core-
distance, implicit mutual-reachability MST, and linkage stages. Before
condensed-tree / selection logic, linkage distances are converted back to true-
distance scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster._hdbscan.hdbscan import (
    HIERARCHY_dtype,
    MST_edge_dtype,
    _brute_mst,
    tree_to_labels,
)
from sklearn.cluster._hdbscan._linkage import make_single_linkage
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.metrics import pairwise_distances

from benchmark_metadata import (
    FULL_STAGE_KEY,
    HDBSCAN_CONTRIB_REFERENCE,
    HDBSCAN_DISTANCE_STAGE_KEY,
    HDBSCAN_LINKAGE_STAGE_KEY,
    HDBSCAN_MST_STAGE_KEY,
    HDBSCAN_SELECT_STAGE_KEY,
    HDBSCAN_STAGE_KEYS,
    REFERENCE_KEYS_BY_PHASE,
    SKLEARN_BRUTE_REFERENCE,
)


@dataclass(frozen=True)
class HdbscanDistanceResult:
    distance_matrix: NDArray[np.float64]
    core_distances: NDArray[np.float64]


@dataclass(frozen=True)
class HdbscanFullResult:
    labels: NDArray[np.int32]
    probabilities: NDArray[np.float64]
    single_linkage_tree: NDArray[Any]


SUPPORTED_HDBSCAN_REFERENCE_KEYS = REFERENCE_KEYS_BY_PHASE["hdbscan"]


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


def validate_stage_key(stage_key: str) -> str:
    if stage_key not in HDBSCAN_STAGE_KEYS:
        valid = ", ".join(HDBSCAN_STAGE_KEYS)
        raise ValueError(f"Unknown HDBSCAN stage {stage_key!r}. Valid stages: {valid}")
    return stage_key  # type: ignore[return-value]


def sqrt_single_linkage_tree_distances_inplace(
    single_linkage_tree: NDArray[Any],
) -> NDArray[Any]:
    """Convert linkage edge weights to true-distance scale in place."""
    tree = np.asarray(single_linkage_tree)
    if not tree.flags.writeable:
        raise ValueError("Expected a writable single-linkage tree for in-place sqrt")

    if tree.dtype.names is not None:
        if "value" in tree.dtype.names:
            distance_name = "value"
        elif "distance" in tree.dtype.names:
            distance_name = "distance"
        else:
            raise ValueError("Structured linkage tree has no value/distance field")
        tree[distance_name] = np.sqrt(
            np.maximum(np.asarray(tree[distance_name], dtype=np.float64), 0.0)
        )
        return tree

    if tree.ndim != 2 or tree.shape[1] < 4:
        raise ValueError(f"Expected single linkage tree with at least 4 columns, got shape {tree.shape}")
    if tree.dtype != np.float64 or not tree.flags.c_contiguous:
        raise ValueError(
            "Expected a C-contiguous float64 single-linkage tree for in-place sqrt"
        )
    tree[:, 2] = np.sqrt(np.maximum(tree[:, 2], 0.0))
    return tree


def sqrt_single_linkage_tree_distances(single_linkage_tree: ArrayLike) -> NDArray[Any]:
    """Return a copy whose linkage edge weights are on true-distance scale."""
    tree = np.asarray(single_linkage_tree)
    if tree.dtype.names is not None:
        out = tree.copy()
    else:
        if tree.ndim != 2 or tree.shape[1] < 4:
            raise ValueError(
                f"Expected single linkage tree with at least 4 columns, got shape {tree.shape}"
            )
        out = np.asarray(tree, dtype=np.float64, order="C").copy()
    return sqrt_single_linkage_tree_distances_inplace(out)


# ---------------------------------------------------------------------------
# scikit-learn brute reference
# ---------------------------------------------------------------------------


def as_sklearn_brute_input(X: ArrayLike) -> NDArray[np.float64]:
    """Return the float64 dense input used by the staged sklearn reference."""
    return np.asarray(X, dtype=np.float64, order="C")


def sklearn_brute_distance_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """distance: X -> dense squared Euclidean distance matrix."""
    X64 = as_sklearn_brute_input(X)
    distances = pairwise_distances(X64, metric="sqeuclidean", n_jobs=1)
    return np.asarray(distances, dtype=np.float64, order="C")


def core_distances_from_squared_distance_matrix(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """core: squared-distance matrix -> squared core-distance vector."""
    distances = np.asarray(distance_matrix, dtype=np.float64, order="C")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Expected a square distance matrix, got shape {distances.shape}")
    validate_min_samples(min_samples, distances.shape[0])
    k_zero_based = min_samples - 1
    return np.ascontiguousarray(
        np.partition(distances, kth=k_zero_based, axis=1)[:, k_zero_based],
        dtype=np.float64,
    )


def sklearn_brute_core_distances(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    return core_distances_from_squared_distance_matrix(
        distance_matrix,
        min_samples=min_samples,
    )


def sklearn_brute_distance_stage_output(
    X: ArrayLike,
    *,
    min_samples: int,
) -> HdbscanDistanceResult:
    """distance: X -> squared distance matrix plus squared core distances."""
    distance_matrix = sklearn_brute_distance_matrix(X)
    core_distances = sklearn_brute_core_distances(
        distance_matrix,
        min_samples=min_samples,
    )
    return HdbscanDistanceResult(
        distance_matrix=distance_matrix,
        core_distances=core_distances,
    )


def sklearn_brute_mutual_reachability_matrix(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """mreach: squared-distance matrix -> squared mutual reachability matrix.

    The helper stores squared core distances on the diagonal of the returned
    mutual-reachability matrix, matching the staged dense C++ contract.
    """
    distances = np.asarray(distance_matrix, dtype=np.float64, order="C").copy()
    return sklearn_brute_mutual_reachability_matrix_inplace(
        distances,
        min_samples=min_samples,
    )


def sklearn_brute_mutual_reachability_matrix_inplace(
    distance_or_mreach_matrix: NDArray[np.float64],
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """Overwrite a dense squared-distance matrix with sklearn squared mreach.

    sklearn's reachability kernel is scale-agnostic once the dense distance
    matrix is supplied: it selects core values with np.partition and applies
    max(core_i, core_j, distance_ij). Feeding squared distances therefore
    yields the staged squared mutual-reachability contract directly.
    """
    if (
        distance_or_mreach_matrix.dtype != np.float64
        or not distance_or_mreach_matrix.flags.c_contiguous
        or not distance_or_mreach_matrix.flags.writeable
    ):
        raise ValueError(
            "Expected a writable C-contiguous float64 squared-distance matrix "
            "for in-place mreach"
        )
    validate_min_samples(min_samples, distance_or_mreach_matrix.shape[0])
    result = mutual_reachability_graph(
        distance_or_mreach_matrix,
        min_samples=min_samples,
    )
    return np.asarray(result, dtype=np.float64, order="C")


def as_cpp_prim_order_mst_edges(mst_edges: ArrayLike) -> NDArray[np.float64]:
    """Return contrib MST edges normalized to the C++ staged MST convention.

    hdbscan-contrib's mst_linkage_core emits true Prim parent edges:

        source_node, new_node, distance

    where source_node is the previously attached vertex that currently gives
    new_node its best crossing-edge distance.

    The C++ staged MST intentionally emits the Prim traversal edge instead:

        previous_selected_node, new_node, distance

    This is enough for the downstream single-linkage stage because both endpoints
    are already in / entering the same growing MST component at that merge
    distance. However, it makes raw MST arrays fail parity against contrib even
    when the selected nodes and edge weights agree.

    Normalize only the staged benchmark representation here. This is not claiming
    contrib's MST parent endpoint is wrong; it just makes the contrib reference
    use the same endpoint convention as the C++ artifact being compared.
    """
    plain = as_float64_plain_mst_edges(mst_edges).copy()
    return normalize_cpp_prim_order_mst_edges_inplace(plain)


def normalize_cpp_prim_order_mst_edges_inplace(
    mst_edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize owned contrib MST edges to the C++ convention in place."""
    plain = np.asarray(mst_edges)
    if (
        plain.dtype != np.float64
        or plain.ndim != 2
        or plain.shape[1] != 3
        or not plain.flags.c_contiguous
        or not plain.flags.writeable
    ):
        raise ValueError(
            "Expected a writable C-contiguous float64 MST edge array with three columns"
        )
    if plain.shape[0] == 0:
        return plain

    previous_selected = np.empty(plain.shape[0], dtype=np.float64)
    previous_selected[0] = 0.0
    previous_selected[1:] = plain[:-1, 1]
    plain[:, 0] = previous_selected
    return np.ascontiguousarray(plain)


def as_float64_mst_edges(mst_edges: ArrayLike) -> NDArray[Any]:
    """Return sklearn MST-edge dtype with float64 edge weights.

    Avoid copying when the input is already sklearn's native MST_edge_dtype.
    """
    edges = np.asarray(mst_edges)

    if edges.dtype == MST_edge_dtype:
        return edges

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
    rounded["distance"] = np.asarray(distance, dtype=np.float64)
    return rounded


def as_float64_single_linkage_tree(single_linkage_tree: ArrayLike) -> NDArray[Any]:
    """Return sklearn hierarchy dtype with float64 merge distances.

    Avoid copying when the input is already sklearn's native HIERARCHY_dtype.
    """
    tree = np.asarray(single_linkage_tree)

    if tree.dtype == HIERARCHY_dtype:
        return tree

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
    rounded["value"] = np.asarray(distance, dtype=np.float64)
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
    return _brute_mst(mutual_reachability, min_samples=min_samples)


def sklearn_brute_single_linkage_tree(mst_edges: NDArray[Any]) -> NDArray[Any]:
    """linkage: MST edge list -> single linkage tree."""
    edges = as_float64_mst_edges(mst_edges)

    # Override sklearn's private _process_mst edge ordering.
    #
    # _process_mst sorts only by distance via np.argsort(distance). For equal
    # MST edge weights, that leaves the order to NumPy's default sort behavior,
    # which is deterministic in a given environment but not a meaningful HDBSCAN
    # contract and not portable to the C++ implementation.
    #
    # Equal mutual-reachability weights are common on low-dimensional / blob-like
    # datasets, and different valid orders produce different internal linkage
    # node IDs even when the MST weights are equivalent. To make the staged
    # sklearn reference comparable to the C++ stage, keep sklearn's Cython
    # make_single_linkage implementation but feed it a canonical edge order that
    # matches the C++ linkage comparator:
    #
    #     distance, then current_node, then next_node
    #
    # This is intentionally a benchmark/reference normalization, not a claim that
    # sklearn's default tie order is algorithmically wrong.
    row_order = np.lexsort(
        (
            edges["next_node"],
            edges["current_node"],
            edges["distance"],
        )
    )
    return make_single_linkage(edges[row_order])


def sklearn_brute_select_clusters(
    single_linkage_tree: NDArray[Any],
    *,
    min_samples: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """select: single linkage tree -> labels and probabilities."""
    tree = as_float64_single_linkage_tree(single_linkage_tree)
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
        np.asarray(probabilities, dtype=np.float64, order="C"),
    )


def sklearn_brute_full(X: ArrayLike, *, min_samples: int) -> HdbscanFullResult:
    """full: X -> labels and probabilities via the staged float64 reference."""
    # Match sklearn's brute fit ownership model: the full path creates one
    # distance matrix, lets the reachability kernel overwrite it, and computes
    # core distances only inside that kernel. The isolated distance/mreach
    # helpers remain non-destructive because their predecessor artifacts may be
    # reused by other staged benchmarks.
    distance_matrix = sklearn_brute_distance_matrix(X)
    mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix_inplace(
        distance_matrix,
        min_samples=min_samples,
    )
    mst_edges = sklearn_brute_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )
    single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)
    sqrt_single_linkage_tree_distances_inplace(single_linkage_tree)
    labels, probabilities = sklearn_brute_select_clusters(
        single_linkage_tree,
        min_samples=min_samples,
    )
    return HdbscanFullResult(
        labels=labels,
        probabilities=probabilities,
        single_linkage_tree=single_linkage_tree,
    )


# ---------------------------------------------------------------------------
# scikit-learn-contrib/hdbscan adapter
# ---------------------------------------------------------------------------


def _contrib_hdbscan_module() -> Any:
    """Import hdbscan lazily so sklearn-only workflows can still import this module."""
    try:
        import hdbscan.hdbscan_ as hdbscan_contrib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The hdbscan_contrib reference requires the `hdbscan` package. "
            "Install the project requirements or remove `hdbscan_contrib` from "
            "hdbscan_references."
        ) from exc
    return hdbscan_contrib


def contrib_internal_min_samples(min_samples: int) -> int:
    """Map this project's self-inclusive min_samples to contrib internals.

    The project and sklearn-brute contract count the sample itself. The contrib
    generic internals use the number of non-self neighbors, so we subtract one.
    """
    validate_min_samples(min_samples)
    return max(1, int(min_samples) - 1)


def as_hdbscan_contrib_input(X: ArrayLike) -> NDArray[np.float64]:
    """Return the float64 dense input used by the hdbscan-contrib adapter."""
    return np.asarray(X, dtype=np.float64, order="C")


def hdbscan_contrib_distance_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """distance: X -> dense squared Euclidean distance matrix for contrib/generic."""
    X64 = as_hdbscan_contrib_input(X)
    distances = pairwise_distances(X64, metric="sqeuclidean", n_jobs=1)
    return np.asarray(distances, dtype=np.float64, order="C")


def hdbscan_contrib_core_distances(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    return core_distances_from_squared_distance_matrix(
        distance_matrix,
        min_samples=min_samples,
    )


def hdbscan_contrib_distance_stage_output(
    X: ArrayLike,
    *,
    min_samples: int,
) -> HdbscanDistanceResult:
    """distance: X -> squared distance matrix plus squared core distances."""
    distance_matrix = hdbscan_contrib_distance_matrix(X)
    core_distances = hdbscan_contrib_core_distances(
        distance_matrix,
        min_samples=min_samples,
    )
    return HdbscanDistanceResult(
        distance_matrix=distance_matrix,
        core_distances=core_distances,
    )


def hdbscan_contrib_mutual_reachability_matrix(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """mreach: squared-distance matrix -> squared mutual reachability matrix.

    hdbscan-contrib's dense reachability kernel is also scale-agnostic for
    alpha=1.0: it selects core values from the supplied matrix and applies
    max(core_i, core_j, distance_ij). The project does not support alpha != 1.
    """
    distances = np.asarray(distance_matrix, dtype=np.float64, order="C").copy()
    return hdbscan_contrib_mutual_reachability_matrix_inplace(
        distances,
        min_samples=min_samples,
    )


def hdbscan_contrib_mutual_reachability_matrix_inplace(
    distance_or_mreach_matrix: NDArray[np.float64],
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """Overwrite a dense squared-distance matrix with contrib squared mreach."""
    if (
        distance_or_mreach_matrix.dtype != np.float64
        or not distance_or_mreach_matrix.flags.c_contiguous
        or not distance_or_mreach_matrix.flags.writeable
    ):
        raise ValueError(
            "Expected a writable C-contiguous float64 squared-distance matrix "
            "for in-place contrib mreach"
        )
    validate_min_samples(min_samples, distance_or_mreach_matrix.shape[0])
    hdbscan_contrib = _contrib_hdbscan_module()
    result = hdbscan_contrib.mutual_reachability(
        distance_or_mreach_matrix,
        min_points=contrib_internal_min_samples(min_samples),
        alpha=1.0,
    )
    return np.asarray(result, dtype=np.float64, order="C")


def as_float64_plain_mst_edges(mst_edges: ArrayLike) -> NDArray[np.float64]:
    """Return an hdbscan-contrib-style MST array with float64 values.

    The plain contrib representation is shape (n_edges, 3): left, right, weight.
    """
    edges = np.asarray(mst_edges)

    if (
        edges.dtype == np.float64
        and edges.ndim == 2
        and edges.shape[1] == 3
        and edges.flags.c_contiguous
    ):
        return edges

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

    plain = np.empty((edges.shape[0], 3), dtype=np.float64)
    plain[:, 0] = np.asarray(left, dtype=np.float64)
    plain[:, 1] = np.asarray(right, dtype=np.float64)
    plain[:, 2] = np.asarray(distance, dtype=np.float64)
    return np.ascontiguousarray(plain)


def as_float64_plain_single_linkage_tree(single_linkage_tree: ArrayLike) -> NDArray[np.float64]:
    """Return a contrib-style single linkage tree with float64 values.

    The plain contrib representation is shape (n_merges, 4): left, right,
    distance, cluster_size.
    """
    tree = np.asarray(single_linkage_tree)

    if (
        tree.dtype == np.float64
        and tree.ndim == 2
        and tree.shape[1] == 4
        and tree.flags.c_contiguous
    ):
        return tree

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

    plain = np.empty((tree.shape[0], 4), dtype=np.float64)
    plain[:, 0] = np.asarray(left, dtype=np.float64)
    plain[:, 1] = np.asarray(right, dtype=np.float64)
    plain[:, 2] = np.asarray(distance, dtype=np.float64)
    plain[:, 3] = np.asarray(cluster_size, dtype=np.float64)
    return np.ascontiguousarray(plain)


def hdbscan_contrib_mst_edges(
    mutual_reachability_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    """mst: mutual reachability matrix -> normalized contrib MST edge list."""
    mutual_reachability = np.asarray(
        mutual_reachability_matrix,
        dtype=np.float64,
        order="C",
    )
    validate_min_samples(min_samples, mutual_reachability.shape[0])
    hdbscan_contrib = _contrib_hdbscan_module()
    mst_edges = hdbscan_contrib.mst_linkage_core(mutual_reachability)
    owned_edges = as_float64_plain_mst_edges(mst_edges)
    return normalize_cpp_prim_order_mst_edges_inplace(owned_edges)


def hdbscan_contrib_single_linkage_tree(mst_edges: ArrayLike) -> NDArray[np.float64]:
    """linkage: contrib MST edge list -> contrib single linkage tree."""
    hdbscan_contrib = _contrib_hdbscan_module()
    edges = np.asarray(as_float64_plain_mst_edges(mst_edges), dtype=np.float64, order="C")

    # Override contrib's private linkage tie ordering.
    #
    # hdbscan-contrib sorts MST rows by weight only before calling label(...):
    #
    #     min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    #
    # For equal mutual-reachability weights, that leaves the row order to
    # NumPy's default argsort tie behavior. Low-dimensional blob datasets create
    # many such ties, and different valid orders produce different internal
    # linkage node IDs, condensed-tree stability accounting, labels, and
    # probabilities.
    #
    # Keep contrib's Cython label(...) implementation, but feed it the same
    # canonical edge order used by the C++ linkage stage:
    #
    #     distance, then current_node, then next_node
    #
    # This is a benchmark normalization for deterministic staged parity, not a
    # claim that contrib's default tie order is algorithmically wrong.
    row_order = np.lexsort(
        (
            edges[:, 1].astype(np.intp, copy=False),
            edges[:, 0].astype(np.intp, copy=False),
            edges[:, 2],
        )
    )

    tree = hdbscan_contrib.label(edges[row_order, :])

    return as_float64_plain_single_linkage_tree(tree)


def hdbscan_contrib_select_clusters(
    single_linkage_tree: ArrayLike,
    *,
    min_samples: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """select: contrib single linkage tree -> labels and probabilities."""
    tree = np.asarray(
        as_float64_plain_single_linkage_tree(single_linkage_tree),
        dtype=np.float64,
        order="C",
    )
    validate_min_samples(min_samples, tree.shape[0] + 1)
    # _tree_to_labels only uses X for its sample count in current contrib. Keep a
    # dummy dense array instead of threading the original X into this isolated stage.
    dummy_X = np.empty((tree.shape[0] + 1, 0), dtype=np.float64)
    hdbscan_contrib = _contrib_hdbscan_module()
    labels, probabilities, *_ = hdbscan_contrib._tree_to_labels(
        dummy_X,
        tree,
        min_cluster_size=min_samples,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        match_reference_implementation=False,
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        max_cluster_size=0,
        cluster_selection_epsilon_max=float("inf"),
    )
    return (
        np.asarray(labels, dtype=np.int32, order="C"),
        np.asarray(probabilities, dtype=np.float64, order="C"),
    )


def hdbscan_contrib_full(X: ArrayLike, *, min_samples: int) -> HdbscanFullResult:
    X64 = as_hdbscan_contrib_input(X)
    validate_min_samples(min_samples, X64.shape[0])

    # Match contrib's generic fit ownership model: the full path owns its
    # distance matrix, so mutual_reachability may overwrite it and compute core
    # distances exactly once. Staged helpers keep their preservation copies.
    distance_matrix = hdbscan_contrib_distance_matrix(X64)
    mutual_reachability_matrix = hdbscan_contrib_mutual_reachability_matrix_inplace(
        distance_matrix,
        min_samples=min_samples,
    )
    mst_edges = hdbscan_contrib_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )
    single_linkage_tree = hdbscan_contrib_single_linkage_tree(mst_edges)
    sqrt_single_linkage_tree_distances_inplace(single_linkage_tree)
    labels, probabilities = hdbscan_contrib_select_clusters(
        single_linkage_tree,
        min_samples=min_samples,
    )

    return HdbscanFullResult(
        labels=labels,
        probabilities=probabilities,
        single_linkage_tree=single_linkage_tree,
    )


# ---------------------------------------------------------------------------
# Generic reference dispatch used by benchmarks.
# ---------------------------------------------------------------------------


def run_sklearn_brute_stage(
    stage_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> object:
    """Run one named sklearn brute stage, including predecessors as setup."""
    stage_key = validate_stage_key(stage_key)

    if stage_key == HDBSCAN_DISTANCE_STAGE_KEY:
        return sklearn_brute_distance_stage_output(X, min_samples=min_samples)

    if stage_key == FULL_STAGE_KEY:
        return sklearn_brute_full(X, min_samples=min_samples)

    distance_output = sklearn_brute_distance_stage_output(X, min_samples=min_samples)
    distance_matrix = distance_output.distance_matrix

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

    squared_single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)
    single_linkage_tree = sqrt_single_linkage_tree_distances(squared_single_linkage_tree)

    if stage_key == HDBSCAN_SELECT_STAGE_KEY:
        return sklearn_brute_select_clusters(single_linkage_tree, min_samples=min_samples)

    raise AssertionError(f"Unhandled HDBSCAN stage {stage_key!r}")


def run_hdbscan_contrib_stage(
    stage_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> object:
    """Run one named hdbscan-contrib stage, including predecessors as setup."""
    stage_key = validate_stage_key(stage_key)

    if stage_key == HDBSCAN_DISTANCE_STAGE_KEY:
        return hdbscan_contrib_distance_stage_output(X, min_samples=min_samples)

    if stage_key == FULL_STAGE_KEY:
        return hdbscan_contrib_full(X, min_samples=min_samples)

    distance_output = hdbscan_contrib_distance_stage_output(X, min_samples=min_samples)
    distance_matrix = distance_output.distance_matrix

    mutual_reachability_matrix = hdbscan_contrib_mutual_reachability_matrix(
        distance_matrix,
        min_samples=min_samples,
    )

    if stage_key == HDBSCAN_MST_STAGE_KEY:
        return hdbscan_contrib_mst_edges(
            mutual_reachability_matrix,
            min_samples=min_samples,
        )

    mst_edges = hdbscan_contrib_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )

    if stage_key == HDBSCAN_LINKAGE_STAGE_KEY:
        return hdbscan_contrib_single_linkage_tree(mst_edges)

    squared_single_linkage_tree = hdbscan_contrib_single_linkage_tree(mst_edges)
    single_linkage_tree = sqrt_single_linkage_tree_distances(squared_single_linkage_tree)

    if stage_key == HDBSCAN_SELECT_STAGE_KEY:
        return hdbscan_contrib_select_clusters(single_linkage_tree, min_samples=min_samples)

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
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return run_hdbscan_contrib_stage(stage_key, X, min_samples=min_samples)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def prepare_hdbscan_reference_stage_input(
    reference_key: str,
    stage_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> tuple[object, ...]:
    """Build predecessor artifacts outside a measured reference-stage function."""
    reference_key = validate_hdbscan_reference_key(reference_key)
    stage_key = validate_stage_key(stage_key)

    if stage_key in {HDBSCAN_DISTANCE_STAGE_KEY, FULL_STAGE_KEY}:
        return (X,)

    distance_output = reference_distance_stage_output(
        reference_key,
        X,
        min_samples=min_samples,
    )
    mutual_reachability_matrix = reference_mutual_reachability_matrix(
        reference_key,
        distance_output.distance_matrix,
        min_samples=min_samples,
    )
    if stage_key == HDBSCAN_MST_STAGE_KEY:
        return (mutual_reachability_matrix,)

    mst_edges = reference_mst_edges(
        reference_key,
        mutual_reachability_matrix,
        min_samples=min_samples,
    )
    if stage_key == HDBSCAN_LINKAGE_STAGE_KEY:
        return (mst_edges,)

    squared_single_linkage_tree = reference_single_linkage_tree(reference_key, mst_edges)
    if stage_key == HDBSCAN_SELECT_STAGE_KEY:
        return (sqrt_single_linkage_tree_distances(squared_single_linkage_tree),)

    raise AssertionError(f"Unhandled HDBSCAN stage {stage_key!r}")


def run_prepared_hdbscan_reference_stage(
    reference_key: str,
    stage_key: str,
    prepared_input: tuple[object, ...],
    *,
    min_samples: int,
) -> object:
    """Run one prepared reference stage without recomputing predecessors."""
    reference_key = validate_hdbscan_reference_key(reference_key)
    stage_key = validate_stage_key(stage_key)

    if stage_key == HDBSCAN_DISTANCE_STAGE_KEY:
        (X,) = prepared_input
        return reference_distance_stage_output(
            reference_key,
            X,
            min_samples=min_samples,
        )

    if stage_key == HDBSCAN_MST_STAGE_KEY:
        (mutual_reachability_matrix,) = prepared_input
        return reference_mst_edges(
            reference_key,
            mutual_reachability_matrix,
            min_samples=min_samples,
        )

    if stage_key == HDBSCAN_LINKAGE_STAGE_KEY:
        (mst_edges,) = prepared_input
        return reference_single_linkage_tree(reference_key, mst_edges)

    if stage_key == HDBSCAN_SELECT_STAGE_KEY:
        (single_linkage_tree,) = prepared_input
        return reference_select_clusters(
            reference_key,
            single_linkage_tree,
            min_samples=min_samples,
        )

    if stage_key == FULL_STAGE_KEY:
        (X,) = prepared_input
        return reference_full(reference_key, X, min_samples=min_samples)

    raise AssertionError(f"Unhandled HDBSCAN stage {stage_key!r}")


def reference_distance_matrix(reference_key: str, X: ArrayLike) -> NDArray[np.float64]:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_distance_matrix(X)
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_distance_matrix(X)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_distance_stage_output(
    reference_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> HdbscanDistanceResult:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_distance_stage_output(X, min_samples=min_samples)
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_distance_stage_output(X, min_samples=min_samples)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_mutual_reachability_matrix(
    reference_key: str,
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float64]:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_mutual_reachability_matrix(
            distance_matrix,
            min_samples=min_samples,
        )
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_mutual_reachability_matrix(
            distance_matrix,
            min_samples=min_samples,
        )
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_mst_edges(
    reference_key: str,
    mutual_reachability_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[Any]:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_mst_edges(
            mutual_reachability_matrix,
            min_samples=min_samples,
        )
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_mst_edges(
            mutual_reachability_matrix,
            min_samples=min_samples,
        )
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_single_linkage_tree(
    reference_key: str,
    mst_edges: ArrayLike,
) -> NDArray[Any]:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_single_linkage_tree(mst_edges)
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_single_linkage_tree(mst_edges)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_select_clusters(
    reference_key: str,
    single_linkage_tree: ArrayLike,
    *,
    min_samples: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_select_clusters(
            single_linkage_tree,
            min_samples=min_samples,
        )
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_select_clusters(
            single_linkage_tree,
            min_samples=min_samples,
        )
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_full(
    reference_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> HdbscanFullResult:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_full(X, min_samples=min_samples)
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_full(X, min_samples=min_samples)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")
