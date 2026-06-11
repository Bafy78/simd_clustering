from collections import defaultdict
from typing import Any

from benchmark_postprocess.naming import (
    LANG_MAP,
    LANGUAGE_REFERENCE_VARIANT,
    PHASE_MAP,
    format_config_id,
    params_display_name,
    variant_display_name,
)
from benchmark_postprocess.parity import (
    MetricsKey,
    compute_gmm_comparison,
    compute_lloyd_comparison,
    metrics_key,
)
from benchmark_postprocess.speedup import build_speedup_block, stable_child_seed
from benchmark_postprocess.stats import summary_stats

RecordGroupKey = tuple[int, int, int, str, str, str, str]


def group_records(records: list[dict[str, Any]]):
    grouped: dict[RecordGroupKey, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        key = (
            record["dimensions"],
            record["samples"],
            record["clusters"],
            record["phase_key"],
            record["variant_key"],
            record["language_key"],
            record["params_key"],
        )
        grouped[key].append(record)

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


def _config_entry(configs: dict[tuple[int, int, int], dict[str, Any]], D: int, N: int, K: int):
    config_key = (D, N, K)

    if config_key not in configs:
        configs[config_key] = {
            "dimensions": D,
            "samples": N,
            "clusters": K,
            "config_id": format_config_id(D, N, K),
            "phases": {},
        }

    return configs[config_key]


def _phase_entry(config_entry: dict[str, Any], phase_key: str) -> dict[str, Any]:
    phase_name = PHASE_MAP[phase_key]

    return config_entry["phases"].setdefault(
        phase_name,
        {
            "phase_key": phase_key,
            "variants": {},
        },
    )


def _variant_entry(phase_entry: dict[str, Any], variant_key: str) -> dict[str, Any]:
    variant_name = variant_display_name(variant_key)

    return phase_entry["variants"].setdefault(
        variant_name,
        {
            "variant_key": variant_key,
            "variant": variant_name,
            "parameterizations": {},
        },
    )


def _parameterization_entry(
    variant_entry: dict[str, Any],
    params_key: str,
) -> dict[str, Any]:
    params_name = params_display_name(params_key)

    return variant_entry["parameterizations"].setdefault(
        params_name,
        {
            "params_key": params_key,
            "params": params_name,
            "languages": {},
        },
    )


def _py_reference_records(
    grouped: dict[RecordGroupKey, list[dict[str, Any]]],
    D: int,
    N: int,
    K: int,
    phase_key: str,
    params_key: str,
) -> list[dict[str, Any]] | None:
    records = grouped.get(
        (D, N, K, phase_key, LANGUAGE_REFERENCE_VARIANT, "py", params_key)
    )
    if records:
        return records

    return None


def _metrics_for(
    metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    variant_key: str,
    lang_key: str,
    params_key: str,
) -> dict[str, Any] | None:
    return metrics.get(metrics_key(config_id, variant_key, lang_key, params_key))


def _py_metrics_for(
    metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    params_key: str,
) -> dict[str, Any] | None:
    return metrics.get(metrics_key(config_id, LANGUAGE_REFERENCE_VARIANT, "py", params_key))


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
    for (
        D,
        N,
        K,
        phase_key,
        variant_key,
        language_key,
        params_key,
    ), group in sorted(grouped.items()):
        if language_key == "py" and variant_key == LANGUAGE_REFERENCE_VARIANT:
            continue

        config_entry = _config_entry(configs, D, N, K)
        phase_entry = _phase_entry(config_entry, phase_key)
        variant_entry = _variant_entry(phase_entry, variant_key)
        parameterization_entry = _parameterization_entry(variant_entry, params_key)
        language_name = LANG_MAP[language_key]
        parameterization_entry["languages"][language_name] = summarize_language_records(group)

    # Attach the shared Python reference to every concrete C++ variant and build
    # speedup/parity per config × phase × variant × parameterization.
    for config_key, config_entry in configs.items():
        D, N, K = config_key
        config_id = format_config_id(D, N, K)

        for phase_entry in config_entry["phases"].values():
            phase_key = phase_entry["phase_key"]

            for variant_entry in phase_entry.get("variants", {}).values():
                variant_key = variant_entry["variant_key"]

                for parameterization_entry in variant_entry.get(
                    "parameterizations", {}
                ).values():
                    params_key = parameterization_entry["params_key"]
                    cpp_records = grouped.get(
                        (D, N, K, phase_key, variant_key, "cpp", params_key)
                    )
                    py_records = _py_reference_records(
                        grouped,
                        D,
                        N,
                        K,
                        phase_key,
                        params_key,
                    )

                    if py_records:
                        parameterization_entry["languages"][LANG_MAP["py"]] = (
                            summarize_language_records(py_records)
                        )

                    if cpp_records and py_records:
                        speedup_seed = stable_child_seed(
                            bootstrap_seed,
                            config_id,
                            phase_key,
                            variant_key,
                            params_key,
                        )

                        parameterization_entry["speedup"] = build_speedup_block(
                            cpp_records,
                            py_records,
                            bootstrap_iterations=bootstrap_iterations,
                            ci_level=ci_level,
                            seed=speedup_seed,
                        )

                    if phase_key == "lloyd" and cpp_records and py_records:
                        parity = compute_lloyd_comparison(
                            lloyd_metrics,
                            config_id=config_id,
                            cpp_variant_key=variant_key,
                            params_key=params_key,
                        )

                        parameterization_entry["languages"].setdefault(LANG_MAP["cpp"], {})[
                            "inertia"
                        ] = parity["cpp_inertia"]
                        parameterization_entry["languages"].setdefault(LANG_MAP["py"], {})[
                            "inertia"
                        ] = parity["python_inertia"]

                        parameterization_entry["parity"] = parity
                        lloyd_comparison_records.append(parity)

                    if phase_key == "gmm" and gmm_metrics is not None:
                        cpp_metrics = _metrics_for(
                            gmm_metrics,
                            config_id,
                            variant_key,
                            "cpp",
                            params_key,
                        )
                        py_metrics = _py_metrics_for(gmm_metrics, config_id, params_key)

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
                                config_id=config_id,
                                cpp_variant_key=variant_key,
                                params_key=params_key,
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
