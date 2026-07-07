"""Generate canonical HDBSCAN predecessor artifacts for C++ staged benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from benchmark_pipeline.hdbscan_reference import (
    sklearn_brute_distance_stage_output,
    sklearn_brute_mst_edges,
    sklearn_brute_mutual_reachability_matrix,
    sklearn_brute_single_linkage_tree,
    sqrt_single_linkage_tree_distances,
    validate_min_samples,
)


def load_dataset(path: str, *, n_samples: int, n_features: int) -> NDArray[np.float64]:
    return np.memmap(
        path,
        dtype=np.float64,
        mode="r",
        shape=(n_samples, n_features),
    )


def write_float64_array(path: str, values: NDArray[np.float64]) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    contiguous.tofile(path)


def write_mst_edges(path: str, edges) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    edge_array = np.asarray(edges)
    edge_count = edge_array.shape[0]

    if edge_array.dtype.names is not None:
        left = np.asarray(edge_array["current_node"], dtype=np.int32)
        right = np.asarray(edge_array["next_node"], dtype=np.int32)
        distance = np.asarray(edge_array["distance"], dtype=np.float64)
    else:
        if edge_array.ndim != 2 or edge_array.shape[1] < 3:
            raise ValueError(f"Expected MST edge array with at least 3 columns, got {edge_array.shape}")
        left = np.asarray(edge_array[:, 0], dtype=np.int32)
        right = np.asarray(edge_array[:, 1], dtype=np.int32)
        distance = np.asarray(edge_array[:, 2], dtype=np.float64)

    if left.shape != (edge_count,) or right.shape != (edge_count,) or distance.shape != (edge_count,):
        raise ValueError("Malformed MST edge arrays")

    with open(path, "wb") as f:
        np.ascontiguousarray(left).tofile(f)
        np.ascontiguousarray(right).tofile(f)
        np.ascontiguousarray(distance).tofile(f)


def write_single_linkage_tree(path: str, tree) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    tree_array = np.asarray(tree)
    row_count = tree_array.shape[0]

    if tree_array.dtype.names is not None:
        left = np.asarray(tree_array["left_node"], dtype=np.int32)
        right = np.asarray(tree_array["right_node"], dtype=np.int32)
        if "value" in tree_array.dtype.names:
            distance = np.asarray(tree_array["value"], dtype=np.float64)
        elif "distance" in tree_array.dtype.names:
            distance = np.asarray(tree_array["distance"], dtype=np.float64)
        else:
            raise ValueError("Structured linkage tree has no value/distance field")
        cluster_size = np.asarray(tree_array["cluster_size"], dtype=np.int32)
    else:
        if tree_array.ndim != 2 or tree_array.shape[1] < 4:
            raise ValueError(
                f"Expected single linkage tree array with at least 4 columns, got {tree_array.shape}"
            )
        left = np.asarray(tree_array[:, 0], dtype=np.int32)
        right = np.asarray(tree_array[:, 1], dtype=np.int32)
        distance = np.asarray(tree_array[:, 2], dtype=np.float64)
        cluster_size = np.asarray(tree_array[:, 3], dtype=np.int32)

    expected_shape = (row_count,)
    if (
        left.shape != expected_shape
        or right.shape != expected_shape
        or distance.shape != expected_shape
        or cluster_size.shape != expected_shape
    ):
        raise ValueError("Malformed linkage tree arrays")

    with open(path, "wb") as f:
        np.ascontiguousarray(left).tofile(f)
        np.ascontiguousarray(right).tofile(f)
        np.ascontiguousarray(distance).tofile(f)
        np.ascontiguousarray(cluster_size).tofile(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical sklearn-brute HDBSCAN predecessor artifacts for "
            "isolated C++ stage benchmarks. Distance/core/MST artifacts use "
            "squared weights; select-stage linkage input is written on true-distance scale."
        )
    )
    parser.add_argument("--dataset-bin", required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=False)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument("--distance-matrix-out")
    parser.add_argument("--core-distances-out")
    parser.add_argument("--mst-edges-out")
    parser.add_argument("--single-linkage-tree-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_outputs = {
        "distance_matrix": args.distance_matrix_out,
        "core_distances": args.core_distances_out,
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
            "mst_edges",
            "single_linkage_tree",
        )
    )
    needs_mst = any(
        requested_outputs[key]
        for key in ("mst_edges", "single_linkage_tree")
    )

    distance_matrix = None
    core_distances = None
    mst_edges = None

    if needs_distance:
        distance_output = sklearn_brute_distance_stage_output(X, min_samples=args.min_samples)
        distance_matrix = np.ascontiguousarray(distance_output.distance_matrix, dtype=np.float64)
        core_distances = np.ascontiguousarray(distance_output.core_distances, dtype=np.float64)
        if args.distance_matrix_out:
            write_float64_array(args.distance_matrix_out, distance_matrix)
        if args.core_distances_out:
            write_float64_array(args.core_distances_out, core_distances)

    if needs_mst:
        assert distance_matrix is not None
        mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
            distance_matrix,
            min_samples=args.min_samples,
        )
        mst_edges = sklearn_brute_mst_edges(
            mutual_reachability_matrix,
            min_samples=args.min_samples,
        )
        if args.mst_edges_out:
            write_mst_edges(args.mst_edges_out, mst_edges)

    if args.single_linkage_tree_out:
        assert mst_edges is not None
        squared_single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)
        single_linkage_tree = sqrt_single_linkage_tree_distances(squared_single_linkage_tree)
        write_single_linkage_tree(args.single_linkage_tree_out, single_linkage_tree)


if __name__ == "__main__":
    main()
