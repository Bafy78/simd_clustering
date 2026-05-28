from collections import defaultdict
from typing import Any

from benchmark_postprocess.naming import LANG_MAP, PHASE_MAP
from benchmark_postprocess.parity import compute_gmm_comparison
from benchmark_postprocess.speedup import build_speedup_block, stable_child_seed
from benchmark_postprocess.stats import summary_stats


def group_records(records: list[dict[str, Any]]):
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        key = (
            record["dimensions"],
            record["samples"],
            record["clusters"],
            record["phase_key"],
            record["language_key"],
        )
        grouped[key].append(record)

    return grouped


def summarize_language_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    process_ids = sorted({record["process_index"] for record in records})

    time_values_by_process: dict[int, list[float]] = defaultdict(list)

    for record in records:
        time_values_by_process[record["process_index"]].append(record["time_s"])

    value_counts_by_process = {
        str(process_id): len(time_values_by_process[process_id])
        for process_id in process_ids
    }

    iterations = sorted({record["iterations"] for record in records})
    if len(iterations) != 1:
        raise RuntimeError(f"Expected exactly one iteration count, got {iterations}")

    return {
        "iterations": iterations[0],
        "n_processes": len(process_ids),
        "n_values": len(records),
        "values_per_process": value_counts_by_process,
        "time_s": summary_stats([record["time_s"] for record in records]),
        "time_per_iteration_s": summary_stats(
            [record["time_per_iteration_s"] for record in records]
        ),
    }


def build_summary(
    records: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    ci_level: float,
    bootstrap_seed: int,
    lloyd_parity: dict[str, dict[str, Any]],
    gmm_metrics: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grouped = group_records(records)

    configs: dict[tuple[int, int, int], dict[str, Any]] = {}

    for (
        dim,
        samples,
        clusters,
        phase_key,
        language_key,
    ), group in sorted(grouped.items()):
        config_key = (dim, samples, clusters)

        if config_key not in configs:
            configs[config_key] = {
                "dimensions": dim,
                "samples": samples,
                "clusters": clusters,
                "config_id": f"{dim}D_{samples}S_{clusters}K",
                "configuration": f"{dim}D | {samples}S | {clusters}K",
                "phases": {},
            }

        phase_name = PHASE_MAP[phase_key]
        language_name = LANG_MAP[language_key]

        phase_entry = configs[config_key]["phases"].setdefault(
            phase_name,
            {
                "phase_key": phase_key,
                "languages": {},
            },
        )

        phase_entry["languages"][language_name] = summarize_language_records(group)

    # Add C++ vs Python speedup blocks.
    for config_key, config_entry in configs.items():
        dim, samples, clusters = config_key
        config_id = f"{dim}D_{samples}S_{clusters}K"

        for phase_name, phase_entry in config_entry["phases"].items():
            phase_key = phase_entry["phase_key"]

            cpp_records = grouped.get(
                (dim, samples, clusters, phase_key, "cpp"),
            )
            py_records = grouped.get(
                (dim, samples, clusters, phase_key, "py"),
            )

            if cpp_records and py_records:
                speedup_seed = stable_child_seed(
                    bootstrap_seed,
                    config_id,
                    phase_key,
                )

                phase_entry["speedup"] = build_speedup_block(
                    cpp_records,
                    py_records,
                    bootstrap_iterations=bootstrap_iterations,
                    ci_level=ci_level,
                    seed=speedup_seed,
                )

            # Add Lloyd inertia/parity.
            if phase_key == "lloyd":
                parity = lloyd_parity.get(config_id)

                if parity is None:
                    raise RuntimeError(f"Missing Lloyd parity record for {config_id}")

                phase_entry["languages"].setdefault("C++", {})["inertia"] = parity[
                    "cpp_inertia"
                ]

                phase_entry["languages"].setdefault("Python", {})["inertia"] = parity[
                    "python_inertia"
                ]

                phase_entry["parity"] = {
                    "cpp_iterations": parity["cpp_iterations"],
                    "python_iterations": parity["python_iterations"],
                    "cpp_inertia": parity["cpp_inertia"],
                    "python_inertia": parity["python_inertia"],
                    "inertia_diff_abs": parity["inertia_diff_abs"],
                    "inertia_diff_pct": parity["inertia_diff_pct"],
                    "tolerance_pct": parity["tolerance_pct"],
                    "status": parity["status"],
                }

            if phase_key == "gmm" and gmm_metrics is not None:
                for lang_key, language_name in LANG_MAP.items():
                    metrics = gmm_metrics.get((config_id, lang_key))
                    if metrics is None:
                        continue

                    phase_entry["languages"].setdefault(language_name, {}).update(
                        {
                            "covariance_type": metrics["covariance_type"],
                            "converged": metrics["converged"],
                            "lower_bound": metrics["lower_bound"],
                        }
                    )

                if (config_id, "cpp") in gmm_metrics and (config_id, "py") in gmm_metrics:
                    phase_entry["parity"] = compute_gmm_comparison(
                        gmm_metrics,
                        config_id=config_id,
                    )

    return {
        "metadata": {
            "schema_version": 1,
            "description": (
                "Post-processed benchmark summary. "
                "Raw pyperf/nanobench JSON values are grouped by process/run before aggregation."
            ),
            "time_unit": "seconds",
            "time_per_iteration_definition": (
                "For Lloyd and GMM, total benchmark time divided by algorithm iteration count. "
                "For non-iterative phases, identical to total time."
            ),
            "speedup_definition": "python_time / cpp_time",
            "bootstrap": {
                "method": "independent clustered bootstrap by process/run",
                "iterations": int(bootstrap_iterations),
                "ci_level": float(ci_level),
                "seed": int(bootstrap_seed),
            },
            "algorithm_metrics": {
                "lloyd_source": "precomputed lloyd_parity_*.json files",
                "gmm_source": "precomputed gmm_metrics_{cpp,py}_*.json files",
                "inertia_note": (
                    "Inertia is Lloyd-specific and is computed during per-config "
                    "finalization from compact Lloyd metrics. The post-processing step "
                    "does not read data_*.bin files."
                ),
                "gmm_note": (
                    "GMM uses lower-bound/convergence/parameter metrics instead of "
                    "inertia. Timing summaries and speedups use the same clustered "
                    "aggregation path as Lloyd."
                ),
            },
        },
        "configs": list(configs.values()),
    }
