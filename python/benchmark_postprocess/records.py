from pathlib import Path
from typing import Any

from benchmark_postprocess.io import load_json
from benchmark_postprocess.naming import parse_benchmark_filename
from benchmark_postprocess.parity import gmm_iteration_count, lloyd_iteration_count


def iter_pyperf_values(path: Path):
    """
    Yield values while preserving pyperf run/process grouping.

    Output:
        {
            "benchmark_index": int,
            "run_index": int,
            "process_index": int,
            "value_index": int,
            "time_s": float,
        }
    """
    data = load_json(path)

    for benchmark_index, benchmark in enumerate(data.get("benchmarks", [])):
        runs = benchmark.get("runs", [])

        for run_index, run in enumerate(runs):
            metadata = run.get("metadata", {})

            # For merged C++ files, the merge script writes process_index.
            # For native pyperf Python files, run_index is the process grouping.
            process_index = metadata.get("process_index", run_index)

            values = run.get("values", [])

            for value_index, value in enumerate(values):
                yield {
                    "benchmark_index": benchmark_index,
                    "run_index": run_index,
                    "process_index": int(process_index),
                    "value_index": value_index,
                    "time_s": float(value),
                }


def load_process_aware_records(
    data_dir: Path,
    *,
    lloyd_parity: dict[str, dict[str, Any]],
    gmm_metrics: dict[tuple[str, str], dict[str, Any]] | None = None,
    completed_config_ids_by_phase: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for json_path in sorted(data_dir.glob("*.json")):
        parsed = parse_benchmark_filename(json_path)
        if parsed is None:
            known_artifact_prefixes = (
                "lloyd_metrics_",
                "lloyd_parity_",
                "gmm_metrics_",
            )

            if json_path.name == "benchmark_summary.json" or json_path.name.startswith(
                known_artifact_prefixes
            ):
                continue

            print(f"Skipping non-benchmark JSON: {json_path.name}")
            continue

        phase_key = parsed["phase_key"]
        lang_key = parsed["language_key"]
        config_id = parsed["config_id"]

        if completed_config_ids_by_phase is not None:
            allowed_config_ids = completed_config_ids_by_phase.get(phase_key)
            if allowed_config_ids is not None and config_id not in allowed_config_ids:
                continue

        if phase_key == "lloyd":
            iterations = lloyd_iteration_count(
                lloyd_parity,
                config_id=config_id,
                lang_key=lang_key,
            )
        elif phase_key == "gmm":
            if gmm_metrics is None:
                raise RuntimeError("GMM metrics map is required to process GMM records")

            iterations = gmm_iteration_count(
                gmm_metrics,
                config_id=config_id,
                lang_key=lang_key,
            )
        else:
            iterations = 1

        for value_record in iter_pyperf_values(json_path):
            time_s = value_record["time_s"]

            records.append(
                {
                    **parsed,
                    "source_json": json_path.name,
                    "benchmark_index": value_record["benchmark_index"],
                    "run_index": value_record["run_index"],
                    "process_index": value_record["process_index"],
                    "value_index": value_record["value_index"],
                    "iterations": iterations,
                    "time_s": time_s,
                    "time_per_iteration_s": time_s / iterations,
                }
            )

    return records
