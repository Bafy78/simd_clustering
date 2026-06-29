"""HDBSCAN reference-stage wrappers.

The wrappers in this module intentionally call implementation internals rather
than reimplementing HDBSCAN stages in Python. The public reference contract is
float32 at dense stage boundaries: input data, distance matrices, and mutual-
reachability matrices are float32. Some private helpers require float64
internally; those wrappers promote only at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster._hdbscan.hdbscan import (
    HIERARCHY_dtype,
    MST_edge_dtype,
    _brute_mst,
    mutual_reachability_graph,
    tree_to_labels,
)
from sklearn.cluster._hdbscan._linkage import make_single_linkage
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from benchmark_metadata import (
    FULL_STAGE_KEY,
    HDBSCAN_CONTRIB_REFERENCE,
    HDBSCAN_DISTANCE_STAGE_KEY,
    HDBSCAN_LINKAGE_STAGE_KEY,
    HDBSCAN_MREACH_STAGE_KEY,
    HDBSCAN_MST_STAGE_KEY,
    HDBSCAN_SELECT_STAGE_KEY,
    HDBSCAN_STAGE_KEYS,
    SKLEARN_BRUTE_REFERENCE,
)


SUPPORTED_HDBSCAN_REFERENCE_KEYS = (
    SKLEARN_BRUTE_REFERENCE,
    HDBSCAN_CONTRIB_REFERENCE,
)


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


# ---------------------------------------------------------------------------
# scikit-learn brute reference
# ---------------------------------------------------------------------------


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


def as_cpp_prim_order_mst_edges(mst_edges: ArrayLike) -> NDArray[np.float32]:
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
    plain = as_float32_plain_mst_edges(mst_edges).copy()
    if plain.shape[0] == 0:
        return plain

    previous_selected = np.empty(plain.shape[0], dtype=np.float32)
    previous_selected[0] = 0.0
    previous_selected[1:] = plain[:-1, 1]
    plain[:, 0] = previous_selected
    return np.ascontiguousarray(plain)


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
    edges = as_float32_mst_edges(mst_edges)

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

    with threadpool_limits(limits=1):
        tree = make_single_linkage(edges[row_order])

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


def as_hdbscan_contrib_input(X: ArrayLike) -> NDArray[np.float32]:
    """Return the float32 dense input used by the hdbscan-contrib adapter."""
    return np.asarray(X, dtype=np.float32, order="C")


def hdbscan_contrib_distance_matrix(X: ArrayLike) -> NDArray[np.float32]:
    """distance: X -> dense Euclidean distance matrix for contrib/generic."""
    X32 = as_hdbscan_contrib_input(X)
    with threadpool_limits(limits=1):
        distances = pairwise_distances(X32, metric="euclidean", n_jobs=1)
    return np.asarray(distances, dtype=np.float32, order="C")


def hdbscan_contrib_mutual_reachability_matrix(
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float32]:
    """mreach: distance matrix -> hdbscan-contrib mutual reachability matrix."""
    distances = np.asarray(distance_matrix, dtype=np.float64, order="C")
    validate_min_samples(min_samples, distances.shape[0])
    hdbscan_contrib = _contrib_hdbscan_module()
    with threadpool_limits(limits=1):
        result = hdbscan_contrib.mutual_reachability(
            distances,
            min_points=contrib_internal_min_samples(min_samples),
            alpha=1.0,
        )
    return np.asarray(result, dtype=np.float32, order="C")


def as_float32_plain_mst_edges(mst_edges: ArrayLike) -> NDArray[np.float32]:
    """Return an hdbscan-contrib-style MST array with float32 values.

    The plain contrib representation is shape (n_edges, 3): left, right, weight.
    """
    edges = np.asarray(mst_edges)

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

    plain = np.empty((edges.shape[0], 3), dtype=np.float32)
    plain[:, 0] = np.asarray(left, dtype=np.float32)
    plain[:, 1] = np.asarray(right, dtype=np.float32)
    plain[:, 2] = np.asarray(distance, dtype=np.float32)
    return np.ascontiguousarray(plain)


def as_float32_plain_single_linkage_tree(single_linkage_tree: ArrayLike) -> NDArray[np.float32]:
    """Return a contrib-style single linkage tree with float32 values.

    The plain contrib representation is shape (n_merges, 4): left, right,
    distance, cluster_size.
    """
    tree = np.asarray(single_linkage_tree)

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

    plain = np.empty((tree.shape[0], 4), dtype=np.float32)
    plain[:, 0] = np.asarray(left, dtype=np.float32)
    plain[:, 1] = np.asarray(right, dtype=np.float32)
    plain[:, 2] = np.asarray(distance, dtype=np.float32)
    plain[:, 3] = np.asarray(cluster_size, dtype=np.float32)
    return np.ascontiguousarray(plain)


def hdbscan_contrib_mst_edges(
    mutual_reachability_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float32]:
    """mst: mutual reachability matrix -> normalized contrib MST edge list."""
    mutual_reachability = np.asarray(
        mutual_reachability_matrix,
        dtype=np.float64,
        order="C",
    )
    validate_min_samples(min_samples, mutual_reachability.shape[0])
    hdbscan_contrib = _contrib_hdbscan_module()
    with threadpool_limits(limits=1):
        mst_edges = hdbscan_contrib.mst_linkage_core(mutual_reachability)

    return as_cpp_prim_order_mst_edges(mst_edges)


def hdbscan_contrib_single_linkage_tree(mst_edges: ArrayLike) -> NDArray[np.float32]:
    """linkage: contrib MST edge list -> contrib single linkage tree."""
    hdbscan_contrib = _contrib_hdbscan_module()
    edges = np.asarray(as_float32_plain_mst_edges(mst_edges), dtype=np.float64, order="C")

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

    with threadpool_limits(limits=1):
        tree = hdbscan_contrib.label(edges[row_order, :])

    return as_float32_plain_single_linkage_tree(tree)


def hdbscan_contrib_select_clusters(
    single_linkage_tree: ArrayLike,
    *,
    min_samples: int,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """select: contrib single linkage tree -> labels and probabilities."""
    tree = np.asarray(
        as_float32_plain_single_linkage_tree(single_linkage_tree),
        dtype=np.float64,
        order="C",
    )
    validate_min_samples(min_samples, tree.shape[0] + 1)
    # _tree_to_labels only uses X for its sample count in current contrib. Keep a
    # dummy dense array instead of threading the original X into this isolated
    # stage.
    dummy_X = np.empty((tree.shape[0] + 1, 0), dtype=np.float64)
    hdbscan_contrib = _contrib_hdbscan_module()
    with threadpool_limits(limits=1):
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
        np.asarray(probabilities, dtype=np.float32, order="C"),
    )


def hdbscan_contrib_full(X: ArrayLike, *, min_samples: int) -> HdbscanFullResult:
    X32 = as_hdbscan_contrib_input(X)
    validate_min_samples(min_samples, X32.shape[0])

    distance_matrix = hdbscan_contrib_distance_matrix(X32)
    mutual_reachability_matrix = hdbscan_contrib_mutual_reachability_matrix(
        distance_matrix,
        min_samples=min_samples,
    )
    mst_edges = hdbscan_contrib_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )
    single_linkage_tree = hdbscan_contrib_single_linkage_tree(mst_edges)
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


def run_hdbscan_contrib_stage(
    stage_key: str,
    X: ArrayLike,
    *,
    min_samples: int,
) -> object:
    """Run one named hdbscan-contrib stage, including predecessors as setup."""
    stage_key = validate_stage_key(stage_key)

    if stage_key == HDBSCAN_DISTANCE_STAGE_KEY:
        return hdbscan_contrib_distance_matrix(X)

    if stage_key == FULL_STAGE_KEY:
        return hdbscan_contrib_full(X, min_samples=min_samples)

    distance_matrix = hdbscan_contrib_distance_matrix(X)

    if stage_key == HDBSCAN_MREACH_STAGE_KEY:
        return hdbscan_contrib_mutual_reachability_matrix(
            distance_matrix,
            min_samples=min_samples,
        )

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

    single_linkage_tree = hdbscan_contrib_single_linkage_tree(mst_edges)

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

    distance_matrix = reference_distance_matrix(reference_key, X)
    if stage_key == HDBSCAN_MREACH_STAGE_KEY:
        return (distance_matrix,)

    mutual_reachability_matrix = reference_mutual_reachability_matrix(
        reference_key,
        distance_matrix,
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

    single_linkage_tree = reference_single_linkage_tree(reference_key, mst_edges)
    if stage_key == HDBSCAN_SELECT_STAGE_KEY:
        return (single_linkage_tree,)

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
        return reference_distance_matrix(reference_key, X)

    if stage_key == HDBSCAN_MREACH_STAGE_KEY:
        (distance_matrix,) = prepared_input
        return reference_mutual_reachability_matrix(
            reference_key,
            distance_matrix,
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


def reference_distance_matrix(reference_key: str, X: ArrayLike) -> NDArray[np.float32]:
    reference_key = validate_hdbscan_reference_key(reference_key)
    if reference_key == SKLEARN_BRUTE_REFERENCE:
        return sklearn_brute_distance_matrix(X)
    if reference_key == HDBSCAN_CONTRIB_REFERENCE:
        return hdbscan_contrib_distance_matrix(X)
    raise AssertionError(f"Unhandled HDBSCAN reference {reference_key!r}")


def reference_mutual_reachability_matrix(
    reference_key: str,
    distance_matrix: ArrayLike,
    *,
    min_samples: int,
) -> NDArray[np.float32]:
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
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
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
