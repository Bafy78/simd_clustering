from __future__ import annotations

import pyperf

from benchmark_metadata import HDBSCAN_STAGE_KEYS, SKLEARN_BRUTE_REFERENCE, REFERENCE_KEYS_BY_PHASE
SUPPORTED_HDBSCAN_REFERENCE_KEYS = REFERENCE_KEYS_BY_PHASE["hdbscan"]

prepare_hdbscan_reference_stage_input = None
run_prepared_hdbscan_reference_stage = None
write_hdbscan_stage_metrics = None
threadpool_limits = None
np = None


def import_runtime_deps(include_metrics=False):
    global threadpool_limits, np, prepare_hdbscan_reference_stage_input
    global run_prepared_hdbscan_reference_stage
    global write_hdbscan_stage_metrics

    import numpy as _np
    from threadpoolctl import threadpool_limits as _threadpool_limits
    from benchmark_pipeline import hdbscan_reference as _hdbscan_reference

    np = _np
    threadpool_limits = _threadpool_limits
    prepare_hdbscan_reference_stage_input = (
        _hdbscan_reference.prepare_hdbscan_reference_stage_input
    )
    run_prepared_hdbscan_reference_stage = (
        _hdbscan_reference.run_prepared_hdbscan_reference_stage
    )

    if include_metrics:
        from benchmark_pipeline.hdbscan_stage_metrics import (
            write_hdbscan_stage_metrics as _write_hdbscan_stage_metrics,
        )
        write_hdbscan_stage_metrics = _write_hdbscan_stage_metrics


def load_dataset(args):
    return np.memmap(
        args.dataset_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.D),
    )


def prepare_stage_input(X, stage_key: str, reference_key: str, min_samples: int):
    """Build predecessor artifacts outside the measured pyperf function."""
    return prepare_hdbscan_reference_stage_input(
        reference_key,
        stage_key,
        X,
        min_samples=min_samples,
    )


def run_prepared_stage(
    stage_key: str,
    reference_key: str,
    prepared_input,
    min_samples: int,
):
    return run_prepared_hdbscan_reference_stage(
        reference_key,
        stage_key,
        prepared_input,
        min_samples=min_samples,
    )


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
    args = runner.parse_args()

    is_worker = getattr(args, "worker", False)

    if is_worker:
        import_runtime_deps()
        X = load_dataset(args)
        prepared_input = prepare_stage_input(
            X,
            args.stage,
            args.reference,
            args.min_samples,
        )            
    else:
        prepared_input = None

    def bench() -> None:
        runner.bench_func(
                f"hdbscan_{args.stage}_{args.reference}_py",
                run_prepared_stage,
                args.stage,
                args.reference,
                prepared_input,
                args.min_samples,
            )
        
    if is_worker:
        with threadpool_limits(limits=1):
            bench()
    else:
        bench()

    if not is_worker:
        import_runtime_deps(include_metrics=True)
        X = load_dataset(args)
        prepared_input = prepare_stage_input(
            X,
            args.stage,
            args.reference,
            args.min_samples,
        )
        result = run_prepared_stage(
            args.stage,
            args.reference,
            prepared_input,
            args.min_samples,
        )

        if args.stage in {"distance", "mreach", "mst", "linkage", "select", "full"}:
            write_hdbscan_stage_metrics(
                args.metrics_file,
                args.stage,
                result,
                min_samples=args.min_samples,
                language="py",
            )


if __name__ == "__main__":
    main()
