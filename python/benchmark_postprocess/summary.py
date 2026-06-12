from collections import defaultdict
from typing import Any

from benchmark_postprocess.naming import (
    LANG_MAP,
    BenchmarkIdentity,
    MetricsKey,
    params_display_name,
)
from benchmark_postprocess.parity import (
    compute_gmm_comparison,
    compute_lloyd_comparison,
)
from benchmark_postprocess.speedup import build_speedup_block, stable_child_seed
from benchmark_postprocess.stats import summary_stats

RecordGroupKey = BenchmarkIdentity


def group_records(records: list[dict[str, Any]]):
    grouped: dict[RecordGroupKey, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        grouped[BenchmarkIdentity.from_record(record)].append(record)

    return grouped


def _single_value(records: list[dict[str, Any]], field: str, default: Any = None) -> Any:
    sentinel = object()
    value: Any = sentinel

    for record in records:
        candidate = record.get(field, default)
        if value is sentinel:
            value = candidate
        elif candidate != value:
            raise RuntimeError(
                f"Expected exactly one value for {field!r}, got at least "
                f"{value!r} and {candidate!r}"
            )

    return default if value is sentinel else value


def summarize_language_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    timing_process_ids = sorted({record["timing_process_index"] for record in records})

    time_values_by_timing_process: dict[int, list[float]] = defaultdict(list)

    for record in records:
        time_values_by_timing_process[record["timing_process_index"]].append(
            record["time_s"]
        )

    timing_value_counts_by_process = {
        str(timing_process_id): len(time_values_by_timing_process[timing_process_id])
        for timing_process_id in timing_process_ids
    }

    algorithm_iterations = sorted(
        {record["algorithm_iterations"] for record in records}
    )
    if len(algorithm_iterations) != 1:
        raise RuntimeError(
            "Expected exactly one algorithm-iteration count, "
            f"got {algorithm_iterations}"
        )

    params_key = _single_value(records, "params_key", "default")

    return {
        "variant_key": _single_value(records, "variant_key"),
        "variant": _single_value(records, "variant"),
        "params_key": params_key,
        "params": params_display_name(params_key),
        "source_json": sorted({record["source_json"] for record in records}),
        "algorithm_iterations": algorithm_iterations[0],
        "timing_process_count": len(timing_process_ids),
        "timing_value_count": len(records),
        "timing_values_per_process": timing_value_counts_by_process,
        "time_s": summary_stats([record["time_s"] for record in records]),
        "time_per_algorithm_iteration_s": summary_stats(
            [record["time_per_algorithm_iteration_s"] for record in records]
        ),
    }


def _config_entry(
    configs: dict[tuple[int, int, int], dict[str, Any]],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    if identity.config_key not in configs:
        configs[identity.config_key] = {
            "dimensions": identity.dimensions,
            "samples": identity.samples,
            "clusters": identity.clusters,
            "config_id": identity.config_id,
            "phases": {},
        }

    return configs[identity.config_key]


def _phase_entry(
    config_entry: dict[str, Any],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    return config_entry["phases"].setdefault(
        identity.phase,
        {
            "phase_key": identity.phase_key,
            "variants": {},
        },
    )


def _variant_entry(
    phase_entry: dict[str, Any],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    return phase_entry["variants"].setdefault(
        identity.variant,
        {
            "variant_key": identity.variant_key,
            "variant": identity.variant,
            "parameterizations": {},
        },
    )


def _parameterization_entry(
    variant_entry: dict[str, Any],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    return variant_entry["parameterizations"].setdefault(
        identity.params,
        {
            "params_key": identity.params_key,
            "params": identity.params,
            "languages": {},
        },
    )


def _py_reference_records(
    grouped: dict[RecordGroupKey, list[dict[str, Any]]],
    identity: BenchmarkIdentity,
) -> list[dict[str, Any]] | None:
    records = grouped.get(identity.python_reference())
    if records:
        return records

    return None


def _metrics_for(
    metrics: dict[MetricsKey, dict[str, Any]],
    identity: BenchmarkIdentity,
) -> dict[str, Any] | None:
    return metrics.get(identity.metrics_key)


def _append_parity_status(records: list[dict[str, Any]], phase_name: str) -> None:
    if not records:
        return

    failures = [parity for parity in records if parity["status"] != "PASS"]

    if failures:
        failed_ids = ", ".join(
            f"{parity['config_id']}:{parity.get('variant_key')}:{parity.get('params_key')}"
            for parity in failures[:10]
        )
        print(
            f"WARNING: Computed {len(failures)} {phase_name} parity failures. "
            f"First failures: {failed_ids}"
        )

    pass_count = sum(1 for parity in records if parity["status"] == "PASS")
    fail_count = sum(1 for parity in records if parity["status"] != "PASS")
    print(
        f"Computed {len(records)} {phase_name} parity records "
        f"({pass_count} PASS, {fail_count} non-PASS)."
    )


def build_summary(
    records: list[dict[str, Any]],
    bootstrap_iterations: int,
    ci_level: float,
    bootstrap_seed: int,
    lloyd_metrics: dict[MetricsKey, dict[str, Any]],
    gmm_metrics: dict[MetricsKey, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grouped = group_records(records)

    configs: dict[tuple[int, int, int], dict[str, Any]] = {}
    lloyd_comparison_records: list[dict[str, Any]] = []
    gmm_parity_records: list[dict[str, Any]] = []

    # First add all concrete C++ variants. Python reference timing is attached
    # to each C++ variant later, so one Python run can be compared to many C++
    # implementations for the same parameterization.
    for identity, group in sorted(grouped.items()):
        if identity.is_python_reference:
            continue

        config_entry = _config_entry(configs, identity)
        phase_entry = _phase_entry(config_entry, identity)
        variant_entry = _variant_entry(phase_entry, identity)
        parameterization_entry = _parameterization_entry(variant_entry, identity)
        parameterization_entry["languages"][identity.language] = (
            summarize_language_records(group)
        )

    # Attach the shared Python reference to every concrete C++ variant and build
    # speedup/parity per config × phase × variant × parameterization.
    for config_key, config_entry in configs.items():
        D, N, K = config_key

        for phase_entry in config_entry["phases"].values():
            phase_key = phase_entry["phase_key"]

            for variant_entry in phase_entry.get("variants", {}).values():
                variant_key = variant_entry["variant_key"]

                for parameterization_entry in variant_entry.get(
                    "parameterizations", {}
                ).values():
                    identity = BenchmarkIdentity(
                        dimensions=D,
                        samples=N,
                        clusters=K,
                        phase_key=phase_key,
                        variant_key=variant_key,
                        language_key="cpp",
                        params_key=parameterization_entry["params_key"],
                    )
                    cpp_records = grouped.get(identity)
                    py_records = _py_reference_records(grouped, identity)

                    if py_records:
                        parameterization_entry["languages"][LANG_MAP["py"]] = (
                            summarize_language_records(py_records)
                        )

                    if cpp_records and py_records:
                        speedup_seed = stable_child_seed(
                            bootstrap_seed,
                            identity.config_id,
                            identity.phase_key,
                            identity.variant_key,
                            identity.params_key,
                        )

                        parameterization_entry["speedup"] = build_speedup_block(
                            cpp_records,
                            py_records,
                            bootstrap_iterations=bootstrap_iterations,
                            ci_level=ci_level,
                            seed=speedup_seed,
                        )

                    if identity.phase_key == "lloyd" and cpp_records and py_records:
                        parity = compute_lloyd_comparison(
                            lloyd_metrics,
                            config_id=identity.config_id,
                            cpp_variant_key=identity.variant_key,
                            params_key=identity.params_key,
                        )

                        parameterization_entry["languages"].setdefault(LANG_MAP["cpp"], {})[
                            "inertia"
                        ] = parity["cpp_inertia"]
                        parameterization_entry["languages"].setdefault(LANG_MAP["py"], {})[
                            "inertia"
                        ] = parity["python_inertia"]

                        parameterization_entry["parity"] = parity
                        lloyd_comparison_records.append(parity)

                    if identity.phase_key == "gmm" and gmm_metrics is not None:
                        cpp_metrics = _metrics_for(gmm_metrics, identity)
                        py_metrics = _metrics_for(gmm_metrics, identity.python_reference())

                        if cpp_metrics is not None:
                            parameterization_entry["languages"].setdefault(
                                LANG_MAP["cpp"], {}
                            ).update(
                                {
                                    "covariance_type": cpp_metrics["covariance_type"],
                                    "lower_bound": cpp_metrics["lower_bound"],
                                }
                            )

                        if py_metrics is not None:
                            parameterization_entry["languages"].setdefault(
                                LANG_MAP["py"], {}
                            ).update(
                                {
                                    "covariance_type": py_metrics["covariance_type"],
                                    "lower_bound": py_metrics["lower_bound"],
                                }
                            )

                        if cpp_records and py_records:
                            parity = compute_gmm_comparison(
                                gmm_metrics,
                                config_id=identity.config_id,
                                cpp_variant_key=identity.variant_key,
                                params_key=identity.params_key,
                            )
                            parameterization_entry["parity"] = parity
                            gmm_parity_records.append(parity)

    _append_parity_status(lloyd_comparison_records, "Lloyd")
    _append_parity_status(gmm_parity_records, "GMM")

    return {
        "metadata": {
            "schema_version": 3,
            "description": (
                "Post-processed benchmark summary. Raw pyperf/nanobench JSON "
                "values are grouped by timing process/pyperf run before aggregation. "
                "C++ variants and algorithm parameterizations are represented explicitly; "
                "Python reference runs are attached to each comparable C++ variant with "
                "the same parameterization."
            ),
            "time_unit": "seconds",
            "time_per_algorithm_iteration_definition": (
                "For Lloyd and GMM, total benchmark time divided by algorithm-iteration count. "
                "For non-iterative phases, identical to total time."
            ),
            "speedup_definition": "python_time / cpp_time",
            "bootstrap": {
                "method": "independent clustered bootstrap by timing process/pyperf run",
                "bootstrap_iterations": int(bootstrap_iterations),
                "ci_level": float(ci_level),
                "seed": int(bootstrap_seed),
            },
            "algorithm_metrics": {
                "lloyd_source": "precomputed lloyd_metrics_{variant}_{cpp,py}_*.json files",
                "gmm_source": "precomputed gmm_metrics_{variant}_{covariance_type}_{cpp,py}_*.json files",
                "parity_note": (
                    "Lloyd and GMM parity are computed per C++ variant against the "
                    "shared sklearn reference for the same parameterization using "
                    "the current thresholds in benchmark_postprocess.parity. "
                    "Reporting reads only this summary."
                ),
                "inertia_note": (
                    "Inertia is Lloyd-specific and is read from compact Lloyd metrics. "
                    "The post-processing step does not read data_*.bin files."
                ),
                "gmm_note": (
                    "GMM uses lower-bound/parameter metrics instead of inertia. "
                    "Timing summaries and speedups use the same clustered aggregation path as Lloyd."
                ),
            },
        },
        "configs": list(configs.values()),
    }
