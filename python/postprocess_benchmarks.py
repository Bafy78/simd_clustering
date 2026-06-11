import argparse
from pathlib import Path

from benchmark_postprocess.io import write_json
from benchmark_postprocess.parity import (
    completed_metric_keys,
    gmm_completed_config_ids,
    lloyd_completed_config_ids,
    load_gmm_metrics_map,
    load_lloyd_metrics_map,
)
from benchmark_postprocess.records import load_timing_process_aware_records
from benchmark_postprocess.summary import build_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./datasets/benchmark_summary.json"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1/4: Loading Lloyd and GMM metrics artifacts...")
    lloyd_metrics = load_lloyd_metrics_map(args.data_dir)
    gmm_metrics = load_gmm_metrics_map(args.data_dir)

    print(f"Step 2/4: Loading benchmark records from {args.data_dir}...")
    lloyd_config_ids = lloyd_completed_config_ids(lloyd_metrics)
    gmm_config_ids = gmm_completed_config_ids(gmm_metrics)
    lloyd_metric_keys = completed_metric_keys(lloyd_metrics)
    gmm_metric_keys = completed_metric_keys(gmm_metrics)
    completed_config_ids_by_phase = {
        "soa": lloyd_config_ids | gmm_config_ids,
        "pp": lloyd_config_ids | gmm_config_ids,
    }
    completed_metric_keys_by_phase = {
        "lloyd": lloyd_metric_keys,
        "gmm": gmm_metric_keys,
    }

    records = load_timing_process_aware_records(
        args.data_dir,
        lloyd_metrics=lloyd_metrics,
        gmm_metrics=gmm_metrics,
        completed_config_ids_by_phase=completed_config_ids_by_phase,
        completed_metric_keys_by_phase=completed_metric_keys_by_phase,
    )

    print("Step 3/4: Building summary and running bootstrap intervals...")
    summary = build_summary(
        records,
        bootstrap_iterations=args.bootstrap_iterations,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
        lloyd_metrics=lloyd_metrics,
        gmm_metrics=gmm_metrics,
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
    print(f"Lloyd completed config/params keys: {len(lloyd_metric_keys)}")
    print(f"GMM metrics records: {len(gmm_metrics)}")
    print(f"GMM completed configs: {len(gmm_config_ids)}")
    print(f"GMM completed config/params keys: {len(gmm_metric_keys)}")


if __name__ == "__main__":
    main()
