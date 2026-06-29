from collections import defaultdict
from typing import Any

from benchmark_metadata import (
    DEFAULT_DATASET_KEY,
    LANGUAGE_CPP_KEY,
    LANGUAGE_PY_KEY,
    NO_PARAMS,
    language_display_name,
    reference_display_name,
    stage_display_name,
    format_config_id,
)
from benchmark_postprocess.naming import (
    BenchmarkIdentity,
    MetricsKey,
    params_display_name,
)
from benchmark_postprocess.parity import (
    compute_gmm_comparison,
    compute_hdbscan_comparison,
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

    params_key = _single_value(records, "params_key", NO_PARAMS)

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


def _config_entry_for_values(
    configs: dict[tuple[str, int, int, int], dict[str, Any]],
    *,
    dataset: str = DEFAULT_DATASET_KEY,
    D: int,
    N: int,
    K: int,
) -> dict[str, Any]:
    config_key = (str(dataset), int(D), int(N), int(K))

    if config_key not in configs:
        dataset, D, N, K = config_key
        configs[config_key] = {
            "dataset": dataset,
            "dimensions": D,
            "samples": N,
            "clusters": K,
            "config_id": format_config_id(D, N, K, dataset=dataset),
            "phases": {},
            "excluded_phases": {},
        }

    return configs[config_key]


def _config_entry(
    configs: dict[tuple[str, int, int, int], dict[str, Any]],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    return _config_entry_for_values(
        configs,
        dataset=identity.dataset,
        D=identity.dimensions,
        N=identity.samples,
        K=identity.clusters,
    )


def _phase_entry(
    config_entry: dict[str, Any],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    return config_entry["phases"].setdefault(
        identity.phase,
        {
            "phase_key": identity.phase_key,
            "phase": identity.phase,
            "stages": {},
        },
    )


def _stage_entry(
    phase_entry: dict[str, Any],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    return phase_entry["stages"].setdefault(
        identity.stage,
        {
            "stage_key": identity.stage_key,
            "stage": identity.stage,
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


def _reference_identities_for(
    grouped: dict[RecordGroupKey, list[dict[str, Any]]],
    identity: BenchmarkIdentity,
) -> list[BenchmarkIdentity]:
    references: list[BenchmarkIdentity] = []

    for candidate in grouped:
        if not candidate.is_python_reference:
            continue
        if candidate.config_key != identity.config_key:
            continue
        if candidate.phase_key != identity.phase_key:
            continue
        if candidate.stage_key != identity.stage_key:
            continue
        if candidate.params_key != identity.params_key:
            continue
        references.append(candidate)

    return sorted(references, key=lambda item: item.variant_key)


def _reference_records(
    grouped: dict[RecordGroupKey, list[dict[str, Any]]],
    identity: BenchmarkIdentity,
    reference_key: str,
) -> list[dict[str, Any]] | None:
    records = grouped.get(identity.python_reference(reference_key))
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
            f"{parity['config_id']}:{parity['stage_key']}:{parity.get('variant_key')}:{parity.get('params_key')}"
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


def _append_exclusions(
    configs: dict[tuple[str, int, int, int], dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> None:
    for exclusion in exclusions:
        config_entry = _config_entry_for_values(
            configs,
            dataset=str(exclusion.get("dataset", DEFAULT_DATASET_KEY)),
            D=int(exclusion["dimensions"]),
            N=int(exclusion["samples"]),
            K=int(exclusion["clusters"]),
        )

        phase_key = str(exclusion["phase_key"])
        phase_name = str(exclusion.get("phase", phase_key))
        stage_key = str(exclusion["stage_key"])
        stage_name = str(exclusion.get("stage", stage_display_name(stage_key)))
        phase_exclusion = config_entry.setdefault("excluded_phases", {}).setdefault(
            phase_name,
            {
                "phase_key": phase_key,
                "phase": phase_name,
                "stages": {},
            },
        )
        phase_exclusion["stages"][stage_name] = {
            "phase_key": phase_key,
            "phase": phase_name,
            "stage_key": stage_key,
            "stage": stage_name,
            "reason": exclusion.get("reason", ""),
            "matched_rules": exclusion.get("matched_rules", []),
        }


def _sorted_exclusions(exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        exclusions,
        key=lambda item: (
            str(item.get("dataset", DEFAULT_DATASET_KEY)),
            int(item["dimensions"]),
            int(item["samples"]),
            int(item["clusters"]),
            str(item["phase_key"]),
            str(item["stage_key"]),
        ),
    )


def _sorted_config_entries(
    configs: dict[tuple[str, int, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    return [configs[config_key] for config_key in sorted(configs)]


def _attach_cachegrind_records(
    configs: dict[tuple[str, int, int, int], dict[str, Any]],
    cachegrind: dict[str, Any] | None,
) -> None:
    if not cachegrind:
        return

    for record in cachegrind.get("records", []):
        identity = BenchmarkIdentity(
            dimensions=int(record["D"]),
            samples=int(record["N"]),
            clusters=int(record["K"]),
            dataset=str(record.get("dataset", DEFAULT_DATASET_KEY)),
            phase_key=str(record["phase_key"]),
            stage_key=str(record["stage_key"]),
            variant_key=str(record["variant_key"]),
            language_key=LANGUAGE_CPP_KEY,
            params_key=str(record.get("params_key", NO_PARAMS)),
        )

        config_entry = _config_entry(configs, identity)
        phase_entry = _phase_entry(config_entry, identity)
        stage_entry = _stage_entry(phase_entry, identity)
        variant_entry = _variant_entry(stage_entry, identity)
        parameterization_entry = _parameterization_entry(variant_entry, identity)
        parameterization_entry["cachegrind"] = {
            "cpp_case": record.get("cpp_case"),
            "tool": record.get("tool", "callgrind"),
            "cache_sim": record.get("cache_sim", True),
            "cache_model": record.get("cache_model", {}),
            "events": record.get("events", {}),
            "derived": record.get("derived", {}),
            "files": record.get("files", {}),
        }


def build_summary(
    records: list[dict[str, Any]],
    bootstrap_iterations: int,
    ci_level: float,
    bootstrap_seed: int,
    lloyd_metrics: dict[MetricsKey, dict[str, Any]],
    compile_artifacts: dict[str, Any],
    cachegrind: dict[str, Any] | None = None,
    gmm_metrics: dict[MetricsKey, dict[str, Any]] | None = None,
    hdbscan_metrics: dict[MetricsKey, dict[str, Any]] | None = None,
    exclusions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grouped = group_records(records)
    exclusions = _sorted_exclusions(exclusions or [])

    configs: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    lloyd_comparison_records: list[dict[str, Any]] = []
    gmm_parity_records: list[dict[str, Any]] = []
    hdbscan_parity_records: list[dict[str, Any]] = []

    # First add all concrete C++ variants. Python reference timing is attached
    # to each C++ variant later, so one Python run can be compared to many C++
    # implementations for the same parameterization.
    for identity, group in sorted(grouped.items()):
        if identity.is_python_reference:
            continue

        config_entry = _config_entry(configs, identity)
        phase_entry = _phase_entry(config_entry, identity)
        stage_entry = _stage_entry(phase_entry, identity)
        variant_entry = _variant_entry(stage_entry, identity)
        parameterization_entry = _parameterization_entry(variant_entry, identity)
        parameterization_entry["languages"][identity.language] = (
            summarize_language_records(group)
        )

    # Attach every available Python reference to every concrete C++ variant and
    # build speedup/parity per config × phase × stage × variant × reference.
    for config_key, config_entry in configs.items():
        dataset, D, N, K = config_key

        for phase_entry in config_entry["phases"].values():
            phase_key = phase_entry["phase_key"]

            for stage_entry in phase_entry.get("stages", {}).values():
                stage_key = stage_entry["stage_key"]

                for variant_entry in stage_entry.get("variants", {}).values():
                    variant_key = variant_entry["variant_key"]

                    for parameterization_entry in variant_entry.get(
                        "parameterizations", {}
                    ).values():
                        identity = BenchmarkIdentity(
                            dimensions=D,
                            samples=N,
                            clusters=K,
                            dataset=dataset,
                            phase_key=phase_key,
                            stage_key=stage_key,
                            variant_key=variant_key,
                            language_key=LANGUAGE_CPP_KEY,
                            params_key=parameterization_entry["params_key"],
                        )
                        cpp_records = grouped.get(identity)
                        if not cpp_records:
                            continue

                        comparisons = parameterization_entry.setdefault("comparisons", {})

                        for reference_identity in _reference_identities_for(grouped, identity):
                            reference_key = reference_identity.variant_key
                            py_records = _reference_records(
                                grouped,
                                identity,
                                reference_key,
                            )
                            if not py_records:
                                continue

                            comparison_entry: dict[str, Any] = {
                                "reference_key": reference_key,
                                "reference": reference_display_name(reference_key),
                                "language_key": LANGUAGE_PY_KEY,
                                "language": language_display_name(LANGUAGE_PY_KEY),
                                "timing": summarize_language_records(py_records),
                            }

                            speedup_seed = stable_child_seed(
                                bootstrap_seed,
                                identity.config_id,
                                identity.phase_key,
                                identity.stage_key,
                                identity.variant_key,
                                identity.params_key,
                                reference_key,
                            )
                            comparison_entry["speedup"] = build_speedup_block(
                                cpp_records,
                                py_records,
                                bootstrap_iterations=bootstrap_iterations,
                                ci_level=ci_level,
                                seed=speedup_seed,
                            )

                            if identity.phase_key == "lloyd":
                                parity = compute_lloyd_comparison(
                                    lloyd_metrics,
                                    config_id=identity.config_id,
                                    cpp_variant_key=identity.variant_key,
                                    py_variant_key=reference_key,
                                    params_key=identity.params_key,
                                    stage_key=identity.stage_key,
                                )
                                parity["reference_key"] = reference_key
                                parity["reference"] = reference_display_name(reference_key)
                                comparison_entry["parity"] = parity
                                parameterization_entry["languages"].setdefault(
                                    language_display_name(LANGUAGE_CPP_KEY), {}
                                )["inertia"] = parity["cpp_inertia"]
                                lloyd_comparison_records.append(parity)

                            if identity.phase_key == "gmm" and gmm_metrics is not None:
                                cpp_metrics = _metrics_for(gmm_metrics, identity)
                                py_metrics = _metrics_for(
                                    gmm_metrics,
                                    identity.python_reference(reference_key),
                                )

                                if cpp_metrics is not None:
                                    parameterization_entry["languages"].setdefault(
                                        language_display_name(LANGUAGE_CPP_KEY), {}
                                    ).update(
                                        {
                                            "covariance_type": cpp_metrics["covariance_type"],
                                            "lower_bound": cpp_metrics["lower_bound"],
                                        }
                                    )

                                if py_metrics is not None:
                                    comparison_entry["stage_metrics"] = {
                                        "covariance_type": py_metrics.get("covariance_type"),
                                        "lower_bound": py_metrics.get("lower_bound"),
                                    }

                                if cpp_metrics is not None and py_metrics is not None:
                                    parity = compute_gmm_comparison(
                                        gmm_metrics,
                                        config_id=identity.config_id,
                                        cpp_variant_key=identity.variant_key,
                                        py_variant_key=reference_key,
                                        params_key=identity.params_key,
                                        stage_key=identity.stage_key,
                                    )
                                    parity["reference_key"] = reference_key
                                    parity["reference"] = reference_display_name(reference_key)
                                    comparison_entry["parity"] = parity
                                    gmm_parity_records.append(parity)

                            if identity.phase_key == "hdbscan" and hdbscan_metrics is not None:
                                cpp_metrics = _metrics_for(hdbscan_metrics, identity)
                                py_metrics = _metrics_for(
                                    hdbscan_metrics,
                                    identity.python_reference(reference_key),
                                )

                                if cpp_metrics is not None:
                                    parameterization_entry["languages"].setdefault(
                                        language_display_name(LANGUAGE_CPP_KEY), {}
                                    )["stage_metrics"] = {
                                        "stage": cpp_metrics.get("stage"),
                                        "dtype": cpp_metrics.get("dtype"),
                                        "shape": cpp_metrics.get("shape"),
                                    }

                                if py_metrics is not None:
                                    comparison_entry["stage_metrics"] = {
                                        "stage": py_metrics.get("stage"),
                                        "dtype": py_metrics.get("dtype"),
                                        "shape": py_metrics.get("shape"),
                                    }

                                if cpp_metrics is not None and py_metrics is not None:
                                    parity = compute_hdbscan_comparison(
                                        hdbscan_metrics,
                                        config_id=identity.config_id,
                                        cpp_variant_key=identity.variant_key,
                                        py_variant_key=reference_key,
                                        params_key=identity.params_key,
                                        stage_key=identity.stage_key,
                                    )
                                    parity["reference_key"] = reference_key
                                    parity["reference"] = reference_display_name(reference_key)
                                    comparison_entry["parity"] = parity
                                    hdbscan_parity_records.append(parity)

                            comparisons[reference_key] = comparison_entry

    _append_exclusions(configs, exclusions)
    _attach_cachegrind_records(configs, cachegrind)

    _append_parity_status(lloyd_comparison_records, "Lloyd")
    _append_parity_status(gmm_parity_records, "GMM")
    _append_parity_status(hdbscan_parity_records, "HDBSCAN")

    return {
        "metadata": {
            "schema_version": 10,
            "description": (
                "Post-processed benchmark summary. Raw pyperf/nanobench JSON "
                "values are grouped by timing process/pyperf run before aggregation. "
                "C++ stages, variants, and algorithm parameterizations are represented explicitly; "
                "Reference runs are attached under per-reference comparison blocks for "
                "each comparable C++ variant with the same parameterization. Configured exclusions are preserved as "
                "reported-but-not-run dataset/D/N/K/phase/stage entries."
            ),
            "time_unit": "seconds",
            "time_per_algorithm_iteration_definition": (
                "For Lloyd and GMM, total benchmark time divided by algorithm-iteration count. "
                "For non-iterative phases, identical to total time."
            ),
            "speedup_definition": "reference_time / cpp_time",
            "compile_artifacts": {
                "source": "compile_artifacts.json",
                "compiler_executable_definition": (
                    "The first executable in the C++ compile command captured for the "
                    "nanobench binary, for example g++-14 or clang++-18."
                ),
                "compiler_version_definition": (
                    "The complete --version banner reported by compiler_executable "
                    "when the binary was compiled."
                ),
                "architecture_definition": (
                    "Resolved value of the architecture selection flag used by the C++ "
                    "compile command, so -march=native is reported as the compiler's "
                    "concrete native target rather than as 'native'."
                ),
                "executable_size_definition": (
                    "Size in bytes of the generated nanobench executable captured "
                    "immediately after compiling that D/phase/stage/variant."
                ),
            },
            "cachegrind": {
                "source": "callgrind_results/cachegrind.*.json",
                "cache_sim_source": "Callgrind --cache-sim=yes counters, parsed from each raw callgrind output summary line.",
                "measurement_region": (
                    "The C++ Callgrind runner starts instrumentation immediately before "
                    "one bench_case.run_once() and stops it immediately afterwards. "
                    "Input loading and setup are intentionally outside the measured region."
                ),
            },
            "exclusion_count": len(exclusions),
            "bootstrap": {
                "method": "independent clustered bootstrap by timing process/pyperf run",
                "bootstrap_iterations": int(bootstrap_iterations),
                "ci_level": float(ci_level),
                "seed": int(bootstrap_seed),
            },
            "algorithm_metrics": {
                "lloyd_source": "precomputed lloyd_{stage}_metrics_{variant}_{cpp,py}_*.json files",
                "gmm_source": "precomputed gmm_{stage}_metrics_{variant}_{covariance_type}_{cpp,py}_*.json files",
                "parity_note": (
                    "Lloyd, GMM, and HDBSCAN parity are computed per C++ variant against "
                    "each available reference for the same parameterization using "
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
                "hdbscan_source": "precomputed hdbscan_{stage}_metrics_{variant}_{cpp,py}_*.json files",
                "hdbscan_note": (
                    "HDBSCAN stage parity is computed from compact stage-output summaries. "
                    "For implemented float-output stages, summaries are generated after casting the "
                    "reference output to float32, matching the current C++ implementation scope."
                ),
            },
        },
        "exclusions": exclusions,
        "compile_artifacts": compile_artifacts,
        "cachegrind": cachegrind
        or {
            "enabled": False,
            "record_count": 0,
            "planned_record_count": 0,
            "missing_record_count": 0,
            "exclusion_count": 0,
            "records": [],
            "exclusions": [],
        },
        "configs": _sorted_config_entries(configs),
    }
