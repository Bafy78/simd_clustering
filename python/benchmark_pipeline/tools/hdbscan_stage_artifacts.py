from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmark_pipeline.hdbscan_reference import (
    sklearn_brute_core_distances,
    sklearn_brute_distance_matrix,
    sklearn_brute_mst_edges,
    sklearn_brute_mutual_reachability_matrix,
    sklearn_brute_single_linkage_tree,
    validate_min_samples,
)


def ensure_parent_dir(path: str | Path | None) -> None:
    if path is not None:
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def write_float32_array(path: str | Path, values: Any) -> None:
    ensure_parent_dir(path)
    np.ascontiguousarray(values, dtype=np.float32).tofile(path)


def write_mst_edges(path: str | Path, mst_edges: NDArray[Any]) -> None:
    """Write MST edges as left[int32], right[int32], weight[float32]."""
    ensure_parent_dir(path)
    edges = np.asarray(mst_edges)

    if edges.dtype.names is not None:
        left_values = edges["current_node"]
        right_values = edges["next_node"]
        weight_values = edges["distance"]
    else:
        if edges.ndim != 2 or edges.shape[1] < 3:
            raise ValueError(
                "Expected MST edges to be either sklearn's structured edge array "
                "or a 2-D array with at least 3 columns "
                f"(got shape {edges.shape})"
            )
        left_values = edges[:, 0]
        right_values = edges[:, 1]
        weight_values = edges[:, 2]

    left = np.ascontiguousarray(left_values, dtype=np.int32)
    right = np.ascontiguousarray(right_values, dtype=np.int32)
    weight = np.ascontiguousarray(weight_values, dtype=np.float32)

    with open(path, "wb") as f:
        left.tofile(f)
        right.tofile(f)
        weight.tofile(f)


def write_single_linkage_tree(path: str | Path, tree: NDArray[Any]) -> None:
    """Write linkage rows as left[int32], right[int32], distance[float32], size[int32]."""
    ensure_parent_dir(path)
    linkage = np.asarray(tree)

    if linkage.dtype.names is not None:
        left_values = linkage["left_node"]
        right_values = linkage["right_node"]
        distance_values = linkage["value"]
        size_values = linkage["cluster_size"]
    else:
        if linkage.ndim != 2 or linkage.shape[1] < 4:
            raise ValueError(
                "Expected single linkage tree to be either sklearn's structured tree "
                "or a 2-D array with at least 4 columns "
                f"(got shape {linkage.shape})"
            )
        left_values = linkage[:, 0]
        right_values = linkage[:, 1]
        distance_values = linkage[:, 2]
        size_values = linkage[:, 3]

    left = np.ascontiguousarray(left_values, dtype=np.int32)
    right = np.ascontiguousarray(right_values, dtype=np.int32)
    distance = np.ascontiguousarray(distance_values, dtype=np.float32)
    size = np.ascontiguousarray(size_values, dtype=np.int32)

    with open(path, "wb") as f:
        left.tofile(f)
        right.tofile(f)
        distance.tofile(f)
        size.tofile(f)


def load_dataset(path: str | Path, *, n_samples: int, n_features: int) -> NDArray[np.float32]:
    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(n_samples, n_features),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical sklearn-brute HDBSCAN predecessor artifacts for "
            "isolated C++ stage benchmarks."
        )
    )
    parser.add_argument("--dataset-bin", required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument("--distance-matrix-out")
    parser.add_argument("--core-distances-out")
    parser.add_argument("--mreach-matrix-out")
    parser.add_argument("--mst-edges-out")
    parser.add_argument("--single-linkage-tree-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_outputs = {
        "distance_matrix": args.distance_matrix_out,
        "core_distances": args.core_distances_out,
        "mreach_matrix": args.mreach_matrix_out,
        "mst_edges": args.mst_edges_out,
        "single_linkage_tree": args.single_linkage_tree_out,
    }
    if not any(requested_outputs.values()):
        raise ValueError("At least one HDBSCAN stage artifact output path is required.")

    validate_min_samples(args.min_samples, args.N)
    X = load_dataset(args.dataset_bin, n_samples=args.N, n_features=args.D)

    needs_distance = any(
        requested_outputs[key]
        for key in (
            "distance_matrix",
            "core_distances",
            "mreach_matrix",
            "mst_edges",
            "single_linkage_tree",
        )
    )
    needs_core = any(
        requested_outputs[key]
        for key in (
            "core_distances",
            "mreach_matrix",
            "mst_edges",
            "single_linkage_tree",
        )
    )
    needs_mreach = any(
        requested_outputs[key]
        for key in ("mreach_matrix", "mst_edges", "single_linkage_tree")
    )
    needs_mst = any(
        requested_outputs[key]
        for key in ("mst_edges", "single_linkage_tree")
    )

    distance_matrix = None
    core_distances = None
    mutual_reachability_matrix = None
    mst_edges = None

    if needs_distance:
        distance_matrix = np.ascontiguousarray(sklearn_brute_distance_matrix(X), dtype=np.float32)
        if args.distance_matrix_out:
            write_float32_array(args.distance_matrix_out, distance_matrix)

    if needs_core:
        assert distance_matrix is not None
        core_distances = sklearn_brute_core_distances(
            distance_matrix,
            min_samples=args.min_samples,
        )
        if args.core_distances_out:
            write_float32_array(args.core_distances_out, core_distances)

    if needs_mreach:
        assert distance_matrix is not None
        assert core_distances is not None
        mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
            distance_matrix,
            core_distances,
            min_samples=args.min_samples,
            validate_core=False,
        )
        if args.mreach_matrix_out:
            write_float32_array(args.mreach_matrix_out, mutual_reachability_matrix)

    if needs_mst:
        assert mutual_reachability_matrix is not None
        mst_edges = sklearn_brute_mst_edges(
            mutual_reachability_matrix,
            min_samples=args.min_samples,
        )
        if args.mst_edges_out:
            write_mst_edges(args.mst_edges_out, mst_edges)

    if args.single_linkage_tree_out:
        assert mst_edges is not None
        single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)
        write_single_linkage_tree(args.single_linkage_tree_out, single_linkage_tree)


if __name__ == "__main__":
    main()
