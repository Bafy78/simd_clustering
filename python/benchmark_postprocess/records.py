from pathlib import Path
from typing import Any

from benchmark_postprocess.io import load_json
from benchmark_pipeline.compile_artifacts import COMPILE_ARTIFACTS_FILENAME
from benchmark_postprocess.naming import parse_benchmark_filename, parse_metrics_filename
from benchmark_postprocess.parity import (
    MetricsKey,
    gmm_algorithm_iteration_count,
    lloyd_algorithm_iteration_count,
)
from benchmark_pipeline.exclusions import EXCLUSIONS_FILENAME


def iter_timing_values(path: Path):
    """
    Yield timing values while preserving pyperf run/timing-process grouping.

    Output:
        {
            "benchmark_index": int,
            "pyperf_run_index": int,
            "timing_process_index": int,
            "timing_value_index": int,
            "time_s": float,
        }
    """
    data = load_json(path)

    for benchmark_index, benchmark in enumerate(data.get("benchmarks", [])):
        pyperf_runs = benchmark.get("runs", [])

        for pyperf_run_index, pyperf_run in enumerate(pyperf_runs):
            metadata = pyperf_run.get("metadata", {})

            # For merged C++ files, the merge script writes timing_process_index.
            # For native pyperf Python files, pyperf_run_index is the timing-process grouping.
            timing_process_index = metadata.get(
                "timing_process_index", pyperf_run_index
            )

            timing_values = pyperf_run.get("values", [])

            for timing_value_index, timing_value in enumerate(timing_values):
                yield {
                    "benchmark_index": benchmark_index,
                    "pyperf_run_index": pyperf_run_index,
                    "timing_process_index": int(timing_process_index),
                    "timing_value_index": timing_value_index,
                    "time_s": float(timing_value),
                }


def load_timing_process_aware_records(
    data_dir: Path,
    lloyd_metrics: dict[MetricsKey, dict[str, Any]],
    gmm_metrics: dict[MetricsKey, dict[str, Any]] | None = None,
    completed_config_ids_by_phase: dict[str, set[str]] | None = None,
    completed_metric_keys_by_phase: dict[str, set[tuple[str, str, str]]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for json_path in sorted(data_dir.glob("*.json")):
        if json_path.name in {
            "benchmark_summary.json",
            EXCLUSIONS_FILENAME,
            COMPILE_ARTIFACTS_FILENAME,
        }:
            continue

        if parse_metrics_filename(json_path, "lloyd") is not None:
            continue
        if parse_metrics_filename(json_path, "gmm") is not None:
            continue

        identity = parse_benchmark_filename(json_path)
        if identity is None:
            print(f"Skipping non-benchmark JSON: {json_path.name}")
            continue

        phase_key = identity.phase_key
        lang_key = identity.language_key
        variant_key = identity.variant_key
        config_id = identity.config_id
        stage_key = identity.stage_key
        params_key = identity.params_key

        if completed_metric_keys_by_phase is not None and phase_key in completed_metric_keys_by_phase:
            allowed_metric_keys = completed_metric_keys_by_phase[phase_key]
            if (config_id, stage_key, params_key) not in allowed_metric_keys:
                continue

        if completed_config_ids_by_phase is not None:
            allowed_config_ids = completed_config_ids_by_phase.get(phase_key)
            if allowed_config_ids is not None and config_id not in allowed_config_ids:
                continue

        if phase_key == "lloyd":
            algorithm_iterations = lloyd_algorithm_iteration_count(
                lloyd_metrics,
                config_id=config_id,
                stage_key=stage_key,
                variant_key=variant_key,
                lang_key=lang_key,
                params_key=params_key,
            )
        elif phase_key == "gmm":
            if gmm_metrics is None:
                raise RuntimeError("GMM metrics map is required to process GMM records")

            algorithm_iterations = gmm_algorithm_iteration_count(
                gmm_metrics,
                config_id=config_id,
                stage_key=stage_key,
                variant_key=variant_key,
                lang_key=lang_key,
                params_key=params_key,
            )
        else:
            algorithm_iterations = 1

        for timing_value_record in iter_timing_values(json_path):
            time_s = timing_value_record["time_s"]

            records.append(
                {
                    **identity.as_record_fields(),
                    "source_json": json_path.name,
                    "benchmark_index": timing_value_record["benchmark_index"],
                    "pyperf_run_index": timing_value_record["pyperf_run_index"],
                    "timing_process_index": timing_value_record["timing_process_index"],
                    "timing_value_index": timing_value_record["timing_value_index"],
                    "algorithm_iterations": algorithm_iterations,
                    "time_s": time_s,
                    "time_per_algorithm_iteration_s": time_s / algorithm_iterations,
                }
            )

    return records
