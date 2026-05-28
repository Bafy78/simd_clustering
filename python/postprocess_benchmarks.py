import argparse
from pathlib import Path

from benchmark_postprocess.io import write_json
from benchmark_postprocess.parity import (
    gmm_completed_config_ids,
    load_gmm_metrics_map,
    load_lloyd_parity_map,
)
from benchmark_postprocess.records import load_process_aware_records
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

    print("Step 1/4: Loading Lloyd parity and GMM metrics artifacts...")
    lloyd_parity = load_lloyd_parity_map(args.data_dir)
    gmm_metrics = load_gmm_metrics_map(args.data_dir)

    print(f"Step 2/4: Loading benchmark records from {args.data_dir}...")
    lloyd_config_ids = set(lloyd_parity)
    gmm_config_ids = gmm_completed_config_ids(gmm_metrics)
    completed_config_ids_by_phase = {
        # AoS/SoA and k-means++ are shared setup phases. Keep them whenever
        # at least one iterative algorithm for the config has complete metrics.
        "soa": lloyd_config_ids | gmm_config_ids,
        "pp": lloyd_config_ids | gmm_config_ids,
        "lloyd": lloyd_config_ids,
        "gmm": gmm_config_ids,
    }

    records = load_process_aware_records(
        args.data_dir,
        lloyd_parity=lloyd_parity,
        gmm_metrics=gmm_metrics,
        completed_config_ids_by_phase=completed_config_ids_by_phase,
    )

    print("Step 3/4: Building summary and running bootstrap intervals...")
    summary = build_summary(
        records,
        bootstrap_iterations=args.bootstrap_iterations,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
        lloyd_parity=lloyd_parity,
        gmm_metrics=gmm_metrics,
    )

    print(f"Step 4/4: Writing output to {args.output}...")
    write_json(args.output, summary)

    print("\n--- Execution Complete ---")
    print(f"Wrote {args.output}")
    print(f"Configurations: {len(summary['configs'])}")
    print(f"Raw timing values: {len(records)}")
    print(f"Bootstrap iterations: {args.bootstrap_iterations}")
    print(f"Lloyd parity configs: {len(lloyd_parity)}")
    print(f"GMM metrics records: {len(gmm_metrics)}")
    print(f"GMM completed configs: {len(gmm_config_ids)}")


if __name__ == "__main__":
    main()
