from __future__ import annotations

import pyperf

from benchmark_metadata import HDBSCAN_STAGE_KEYS, SKLEARN_BRUTE_REFERENCE
from benchmark_pipeline.hdbscan_reference import (
    compose_sklearn_brute_stages,
    sklearn_brute_distance_matrix,
    sklearn_brute_full,
    sklearn_brute_mst_edges,
    sklearn_brute_mutual_reachability_matrix,
    sklearn_brute_select_clusters,
    sklearn_brute_single_linkage_tree,
    validate_hdbscan_reference_key,
    validate_min_samples,
    validate_stage_key,
    write_hdbscan_stage_metrics,
)

np = None


def import_runtime_deps():
    global np

    import numpy as _np

    np = _np


def load_dataset(args):
    return np.memmap(
        args.dataset_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.D),
    )


def prepare_stage_input(X, stage_key: str, min_samples: int):
    """Build predecessor artifacts outside the measured pyperf function."""
    stage_key = validate_stage_key(stage_key)

    if stage_key == "distance":
        return (X,)

    if stage_key == "full":
        return (X,)

    distance_matrix = np.ascontiguousarray(sklearn_brute_distance_matrix(X), dtype=np.float32)
    if stage_key == "mreach":
        return (distance_matrix,)

    mutual_reachability_matrix = sklearn_brute_mutual_reachability_matrix(
        distance_matrix,
        min_samples=min_samples,
    )
    if stage_key == "mst":
        return (mutual_reachability_matrix,)

    mst_edges = sklearn_brute_mst_edges(
        mutual_reachability_matrix,
        min_samples=min_samples,
    )
    if stage_key == "linkage":
        return (mst_edges,)

    single_linkage_tree = sklearn_brute_single_linkage_tree(mst_edges)
    if stage_key == "select":
        return (single_linkage_tree,)

    raise AssertionError(f"Unhandled HDBSCAN stage {stage_key!r}")


def run_prepared_stage(stage_key: str, prepared_input, min_samples: int):
    if stage_key == "distance":
        (X,) = prepared_input
        return sklearn_brute_distance_matrix(X)

    if stage_key == "mreach":
        (distance_matrix,) = prepared_input
        return sklearn_brute_mutual_reachability_matrix(
            distance_matrix,
            min_samples=min_samples,
        )

    if stage_key == "mst":
        (mutual_reachability_matrix,) = prepared_input
        return sklearn_brute_mst_edges(
            mutual_reachability_matrix,
            min_samples=min_samples,
        )

    if stage_key == "linkage":
        (mst_edges,) = prepared_input
        return sklearn_brute_single_linkage_tree(mst_edges)

    if stage_key == "select":
        (single_linkage_tree,) = prepared_input
        return sklearn_brute_select_clusters(
            single_linkage_tree,
            min_samples=min_samples,
        )

    if stage_key == "full":
        (X,) = prepared_input
        return sklearn_brute_full(X, min_samples=min_samples)

    raise AssertionError(f"Unhandled HDBSCAN stage {stage_key!r}")


def append_custom_args(cmd, args):
    cmd.extend(["--dataset-bin", args.dataset_bin])
    cmd.extend(["--D", str(args.D)])
    cmd.extend(["--N", str(args.N)])
    cmd.extend(["--K", str(args.K)])
    cmd.extend(["--stage", args.stage])
    cmd.extend(["--reference", args.reference])
    cmd.extend(["--min-samples", str(args.min_samples)])
    cmd.extend(["--metrics-file", args.metrics_file])


def main() -> None:
    runner = pyperf.Runner(
        add_cmdline_args=append_custom_args,
        warmups=1,
    )

    runner.argparser.add_argument("--dataset-bin", required=True)
    runner.argparser.add_argument("--D", type=int, required=True)
    runner.argparser.add_argument("--N", type=int, required=True)
    runner.argparser.add_argument("--K", type=int, required=True)
    runner.argparser.add_argument("--stage", choices=HDBSCAN_STAGE_KEYS, required=True)
    runner.argparser.add_argument("--reference", default=SKLEARN_BRUTE_REFERENCE)
    runner.argparser.add_argument("--min-samples", type=int, required=True)
    runner.argparser.add_argument("--metrics-file", required=True)
    runner.argparser.add_argument(
        "--verify-composition",
        action="store_true",
        help="Verify that staged sklearn brute wrappers match sklearn.HDBSCAN.fit.",
    )

    args = runner.parse_args()
    validate_stage_key(args.stage)
    validate_hdbscan_reference_key(args.reference)
    validate_min_samples(args.min_samples, args.N)

    if getattr(args, "worker", False):
        import_runtime_deps()
        X = load_dataset(args)
        if args.reference != SKLEARN_BRUTE_REFERENCE:
            raise ValueError(f"Unsupported HDBSCAN reference for staged benchmark: {args.reference!r}")
        prepared_input = prepare_stage_input(X, args.stage, args.min_samples)
    else:
        X = None
        prepared_input = None

    runner.bench_func(
        f"hdbscan_{args.stage}_{args.reference}_py",
        run_prepared_stage,
        args.stage,
        prepared_input,
        args.min_samples,
    )

    if not getattr(args, "worker", False):
        import_runtime_deps()
        X = load_dataset(args)
        if args.reference != SKLEARN_BRUTE_REFERENCE:
            raise ValueError(f"Unsupported HDBSCAN reference for staged benchmark: {args.reference!r}")
        prepared_input = prepare_stage_input(X, args.stage, args.min_samples)
        result = run_prepared_stage(args.stage, prepared_input, args.min_samples)

        if args.stage in {"distance", "mreach"}:
            write_hdbscan_stage_metrics(
                args.metrics_file,
                args.stage,
                result,
                min_samples=args.min_samples,
                language="py",
            )

        if args.verify_composition:
            composed = compose_sklearn_brute_stages(X, min_samples=args.min_samples)
            full = sklearn_brute_full(X, min_samples=args.min_samples)
            np.testing.assert_array_equal(composed.single_linkage_tree, full.single_linkage_tree)
            np.testing.assert_array_equal(composed.labels, full.labels)
            np.testing.assert_allclose(composed.probabilities, full.probabilities, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    main()
