import argparse
from pathlib import Path

from benchmark_postprocess.compile_artifacts import build_compile_artifact_summary
from benchmark_postprocess.cachegrind import build_cachegrind_summary
from benchmark_postprocess.io import write_json
from benchmark_postprocess.parity import (
    completed_metric_keys,
    gmm_completed_config_ids,
    hdbscan_completed_config_ids,
    lloyd_completed_config_ids,
    load_gmm_metrics_map,
    load_hdbscan_metrics_map,
    load_lloyd_metrics_map,
)
from benchmark_postprocess.records import load_timing_process_aware_records
from benchmark_postprocess.summary import build_summary
from benchmark_pipeline.exclusions import EXCLUSIONS_FILENAME, load_exclusion_manifest
from benchmark_pipeline.paths import repo_path, repo_relative_path
from spill_detector import (
    SPILL_DETECTOR_PATTERN,
    SpillDetectorError,
    benchmark_record_scan_targets,
    compile_command_map_from_artifacts,
    scan_targets,
    spill_detection_status,
    summary_payload as spill_summary_payload,
)


def build_spill_detection_summary(
    records: list[dict[str, object]],
    *,
    compile_artifacts: dict[str, object],
    out_dir: Path,
    rg: str,
    pattern: str,
) -> dict[str, object]:
    targets = benchmark_record_scan_targets(records)
    base_payload: dict[str, object] = {
        "enabled": True,
        "scan_target_count": len(targets),
        "scan_targets": [
            {
                "cpp_case": target.cpp_case,
                "D": target.D,
                "gmm_covariance_type": target.gmm_covariance_type,
            }
            for target in targets
        ],
        "out_dir": str(out_dir),
        "compile_command_source": "compile_artifacts",
    }

    try:
        benchmark_compile_commands = compile_command_map_from_artifacts(
            compile_artifacts
        )
        results = scan_targets(
            targets,
            out_dir=out_dir,
            rg=rg,
            pattern=pattern,
            benchmark_compile_commands=benchmark_compile_commands,
        )
    except Exception as exc:
        error: dict[str, object] = {
            **base_payload,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "results": [],
            "total_candidate_reload_pairs": None,
        }
        if isinstance(exc, SpillDetectorError):
            error["exit_code"] = exc.exit_code
        return error

    payload = spill_summary_payload(results)
    return {
        **base_payload,
        **payload,
        "status": spill_detection_status(payload),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--spill-detection",
        action="store_true",
        help=(
            "Run the spill detector for C++ cases present in the benchmark records "
            "and append its results to the benchmark summary."
        ),
    )
    parser.add_argument(
        "--spill-detection-out-dir",
        type=Path,
        default=repo_path("spill_detector_results"),
        help="Directory for spill detector assembly and rg outputs.",
    )
    parser.add_argument(
        "--spill-detection-rg",
        default="rg",
        help="ripgrep executable for spill detection. Defaults to rg.",
    )
    parser.add_argument(
        "--spill-detection-pattern",
        default=SPILL_DETECTOR_PATTERN,
        help="Override the default spill detector PCRE pattern.",
    )
    parser.add_argument(
        "--cachegrind-results-dir",
        type=Path,
        default=repo_path("callgrind_results"),
        help="Directory containing Cachegrind JSON records.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = repo_relative_path(args.data_dir)
    args.output = (
        repo_relative_path(args.output)
        if args.output is not None
        else args.data_dir / "benchmark_summary.json"
    )
    args.spill_detection_out_dir = repo_relative_path(args.spill_detection_out_dir)
    args.cachegrind_results_dir = repo_relative_path(args.cachegrind_results_dir)

    print("Step 1/4: Loading Lloyd, GMM, and HDBSCAN metrics artifacts...")
    lloyd_metrics = load_lloyd_metrics_map(args.data_dir)
    gmm_metrics = load_gmm_metrics_map(args.data_dir)
    hdbscan_metrics = load_hdbscan_metrics_map(args.data_dir)

    print(f"Step 2/4: Loading benchmark records from {args.data_dir}...")
    lloyd_config_ids = lloyd_completed_config_ids(lloyd_metrics)
    gmm_config_ids = gmm_completed_config_ids(gmm_metrics)
    hdbscan_config_ids = hdbscan_completed_config_ids(hdbscan_metrics)
    lloyd_metric_keys = completed_metric_keys(lloyd_metrics)
    gmm_metric_keys = completed_metric_keys(gmm_metrics)
    hdbscan_metric_keys = completed_metric_keys(hdbscan_metrics)
    completed_algorithm_config_ids = lloyd_config_ids | gmm_config_ids | hdbscan_config_ids
    completed_config_ids_by_phase = {}
    if completed_algorithm_config_ids:
        completed_config_ids_by_phase["soa"] = completed_algorithm_config_ids

    completed_metric_keys_by_phase = {
        "lloyd": lloyd_metric_keys,
        "gmm": gmm_metric_keys,
        "hdbscan": hdbscan_metric_keys,
    }

    records = load_timing_process_aware_records(
        args.data_dir,
        lloyd_metrics=lloyd_metrics,
        gmm_metrics=gmm_metrics,
        completed_config_ids_by_phase=completed_config_ids_by_phase,
        completed_metric_keys_by_phase=completed_metric_keys_by_phase,
    )

    exclusions = load_exclusion_manifest(args.data_dir / EXCLUSIONS_FILENAME)[
        "exclusions"
    ]

    print("Step 3/4: Building summary and running bootstrap intervals...")
    compile_artifacts = build_compile_artifact_summary(
        records,
        data_dir=args.data_dir,
    )
    cachegrind = build_cachegrind_summary(args.cachegrind_results_dir)
    summary = build_summary(
        records,
        bootstrap_iterations=args.bootstrap_iterations,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
        lloyd_metrics=lloyd_metrics,
        compile_artifacts=compile_artifacts,
        cachegrind=cachegrind,
        gmm_metrics=gmm_metrics,
        hdbscan_metrics=hdbscan_metrics,
        exclusions=exclusions,
    )

    if args.spill_detection:
        print("Running spill detector for benchmarked C++ cases...")
        summary["spill_detection"] = build_spill_detection_summary(
            records,
            compile_artifacts=compile_artifacts,
            out_dir=args.spill_detection_out_dir,
            rg=args.spill_detection_rg,
            pattern=args.spill_detection_pattern,
        )
        print(
            "Spill detection status: "
            f"{summary['spill_detection'].get('status')} "
            f"({summary['spill_detection'].get('scan_target_count')} target(s))"
        )

    print(f"Step 4/4: Writing output to {args.output}...")
    write_json(args.output, summary)

    print("\n--- Execution Complete ---")
    print(f"Wrote {args.output}")
    print(f"Configurations: {len(summary['configs'])}")
    print(f"Raw timing values: {len(records)}")
    print(f"Bootstrap iterations: {args.bootstrap_iterations}")
    print(f"Lloyd metrics records: {len(lloyd_metrics)}")
    print(f"Lloyd completed configs: {len(lloyd_config_ids)}")
    print(f"Lloyd completed config/stage/params keys: {len(lloyd_metric_keys)}")
    print(f"GMM metrics records: {len(gmm_metrics)}")
    print(f"GMM completed configs: {len(gmm_config_ids)}")
    print(f"GMM completed config/stage/params keys: {len(gmm_metric_keys)}")
    print(f"HDBSCAN metrics records: {len(hdbscan_metrics)}")
    print(f"HDBSCAN completed configs: {len(hdbscan_config_ids)}")
    print(f"HDBSCAN completed config/stage/params keys: {len(hdbscan_metric_keys)}")
    print(f"Configured benchmark phase/stage exclusions: {len(exclusions)}")
    print(
        "C++ compile artifact records: "
        f"{summary['compile_artifacts'].get('record_count', 0)}"
    )
    print(
        "Cachegrind records: "
        f"{summary['cachegrind'].get('record_count', 0)}"
    )


if __name__ == "__main__":
    main()
