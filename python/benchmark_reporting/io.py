from pathlib import Path
import json
from typing import Any, Iterator
import pandas as pd
from html import escape

from benchmark_pipeline.paths import repo_relative_path
from benchmark_reporting.constants import *
from benchmark_metadata import FULL_STAGE_KEY, NO_PARAMS, PHASE_DISPLAY_NAMES, STAGE_DISPLAY_NAMES, stage_display_name

DEFAULT_BENCHMARK_SUMMARY_JSON = Path("datasets/benchmark_summary.json")


def _resolve_summary_json(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> Path:
    path = repo_relative_path(summary_json)

    if path.is_dir():
        raise IsADirectoryError(
            f"Expected benchmark summary JSON file, got directory: {path}"
        )

    return path


def load_benchmark_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> dict[str, Any]:
    path = _resolve_summary_json(summary_json)

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark summary not found: {path}. "
            f"Run the post-processing step first."
        )

    with path.open("r") as f:
        return json.load(f)


def load_exclusion_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load configured benchmark exclusions from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for exclusion in summary.get("exclusions", []):
        phase_key = exclusion.get("phase_key")
        for _stage_key, stage_name in _record_stage_entries(exclusion):
            records.append(
                {
                    COL_PHASE: _phase_display_name(
                        phase_key,
                        exclusion.get("phase", phase_key),
                    ),
                    COL_STAGE: stage_name,
                    COL_DATASET: str(exclusion.get("dataset", "blobs")),
                    COL_DIMENSIONS: int(exclusion["dimensions"]),
                    COL_SAMPLES: int(exclusion["samples"]),
                    COL_CLUSTERS: int(exclusion["clusters"]),
                    COL_EXCLUSION_REASON: exclusion.get("reason", ""),
                    COL_EXCLUSION_RULES: ", ".join(
                        str(rule.get("rule_index"))
                        for rule in exclusion.get("matched_rules", [])
                    ),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)

    return df.sort_values(
        [COL_PHASE, COL_STAGE, COL_DATASET, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
    ).reset_index(drop=True)


def _file_link(path_value: str | None) -> str:
    if not path_value:
        return "-"

    path = Path(path_value)
    label = escape(path.name)

    try:
        href = path.resolve().relative_to(repo_relative_path("./")).as_posix()
    except ValueError:
        href = path.resolve().as_uri()

    return f'<a href="{escape(href, quote=True)}">{label}</a>'


def _cpp_case_phase_variant(cpp_case: str | None) -> tuple[str | None, str | None]:
    if not cpp_case:
        return None, None

    for variant_key in ("static", "dynamic", "auto"):
        suffix = f"_{variant_key}"
        if cpp_case.endswith(suffix):
            return cpp_case[: -len(suffix)], variant_key

    return None, None


def _record_stage_entries(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Return display-ready stage entries for summary records.

    Current summaries usually store ``stage_key`` plus optional ``stage``.
    Some compile-artifact records may store ``stage_keys`` because one binary
    can serve several stages. Legacy summaries had no stage metadata at all;
    reporting treats those records as belonging to the synthetic Full stage.
    """
    if "stage_key" in record:
        stage_key = str(record.get("stage_key") or FULL_STAGE_KEY)
        return [(stage_key, _stage_display_name(stage_key, record.get("stage")))]

    stage_keys = record.get("stage_keys")
    if isinstance(stage_keys, list) and stage_keys:
        return [
            (str(stage_key), _stage_display_name(str(stage_key)))
            for stage_key in stage_keys
        ]

    return [(FULL_STAGE_KEY, _stage_display_name(FULL_STAGE_KEY))]


def load_compile_artifact_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load postprocess-attached C++ compile artifact records."""
    summary = load_benchmark_summary(summary_json)
    compile_artifacts = summary.get("compile_artifacts", {})
    rows: list[dict[str, Any]] = []

    for result in compile_artifacts.get("records", []):
        phase_key = result.get("phase_key")
        variant_key = result.get("variant_key")
        executable_size_bytes = int(result["executable_size_bytes"])

        for _stage_key, stage_name in _record_stage_entries(result):
            rows.append(
                {
                    COL_PHASE: _phase_display_name(phase_key, phase_key or "-"),
                    COL_STAGE: stage_name,
                    COL_VARIANT: _variant_display_name(variant_key),
                    COL_DATASET: str(result.get("dataset", "compile")),
                    COL_DIMENSIONS: int(result["D"]),
                    COL_CPP_CASE: result.get("cpp_case"),
                    COL_COMPILER_EXECUTABLE: result.get("compiler_executable"),
                    COL_COMPILER_VERSION: result.get("compiler_version"),
                    COL_ARCHITECTURE: result.get("architecture"),
                    COL_ARCHITECTURE_FLAG: result.get("architecture_flag"),
                    COL_EXECUTABLE_SIZE_BYTES: executable_size_bytes,
                    COL_EXECUTABLE_SIZE_MIB: executable_size_bytes / (1024 ** 2),
                    "Binary": _file_link(result.get("binary_path")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)

    return df.sort_values(
        [COL_PHASE, COL_STAGE, COL_VARIANT, COL_DATASET, COL_DIMENSIONS]
    ).reset_index(drop=True)


def load_spill_detection_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load postprocess-attached spill detector results from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    spill = summary.get("spill_detection", {})
    rows: list[dict[str, Any]] = []

    for result in spill.get("results", []):
        cpp_case_name = result.get("cpp_case")
        phase_key, variant_key = _cpp_case_phase_variant(cpp_case_name)
        gmm_covariance_type = result.get("gmm_covariance_type")

        rows.append(
            {
                COL_PHASE: _phase_display_name(phase_key, phase_key or "-"),
                COL_STAGE: _stage_display_name(FULL_STAGE_KEY),
                COL_VARIANT: _variant_display_name(variant_key),
                COL_PARAMS: _params_display_name(gmm_covariance_type),
                "C++ Case": cpp_case_name,
                COL_DATASET: str(result.get("dataset", "compile")),
                COL_DIMENSIONS: int(result.get("D")),
                "Candidate Reload Pairs": int(
                    result.get("candidate_reload_pairs", 0)
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)

    return df.sort_values(
        [COL_PHASE, COL_STAGE, COL_VARIANT, COL_PARAMS, COL_DATASET, COL_DIMENSIONS]
    ).reset_index(drop=True)


def load_cachegrind_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load Cachegrind records from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    cachegrind = summary.get("cachegrind", {})
    rows: list[dict[str, Any]] = []

    for result in cachegrind.get("records", []):
        phase_key = result.get("phase_key")
        variant_key = result.get("variant_key")
        params_key = result.get("params_key", NO_PARAMS)
        events = result.get("events", {})
        derived = result.get("derived", {})
        cache_model = result.get("cache_model", {})
        files = result.get("files", {})

        for _stage_key, stage_name in _record_stage_entries(result):
            rows.append(
                {
                    COL_PHASE: _phase_display_name(phase_key, phase_key or "-"),
                    COL_STAGE: stage_name,
                    COL_VARIANT: _variant_display_name(variant_key),
                    COL_PARAMS: _params_display_name(params_key),
                    COL_CPP_CASE: result.get("cpp_case"),
                    COL_DATASET: str(result.get("dataset", "blobs")),
                    COL_DIMENSIONS: int(result["D"]),
                    COL_SAMPLES: int(result["N"]),
                    COL_CLUSTERS: int(result["K"]),
                    COL_CACHEGRIND_I1: cache_model.get("I1"),
                    COL_CACHEGRIND_D1: cache_model.get("D1"),
                    COL_CACHEGRIND_LL: cache_model.get("LL"),
                    COL_CACHEGRIND_IR: int(events.get("Ir", 0)),
                    COL_CACHEGRIND_I1MR: int(events.get("I1mr", 0)),
                    COL_CACHEGRIND_ILMR: int(events.get("ILmr", 0)),
                    COL_CACHEGRIND_DR: int(events.get("Dr", 0)),
                    COL_CACHEGRIND_D1MR: int(events.get("D1mr", 0)),
                    COL_CACHEGRIND_DLMR: int(events.get("DLmr", 0)),
                    COL_CACHEGRIND_DW: int(events.get("Dw", 0)),
                    COL_CACHEGRIND_D1MW: int(events.get("D1mw", 0)),
                    COL_CACHEGRIND_DLMW: int(events.get("DLmw", 0)),
                    COL_CACHEGRIND_DATA_REFS: derived.get("data_refs"),
                    COL_CACHEGRIND_D1_DATA_MISSES: derived.get("d1_data_misses"),
                    COL_CACHEGRIND_LL_DATA_MISSES: derived.get("ll_data_misses"),
                    COL_CACHEGRIND_D1_DATA_MISS_RATE: derived.get("d1_data_miss_rate"),
                    COL_CACHEGRIND_LL_DATA_MISS_RATE: derived.get("ll_data_miss_rate"),
                    COL_CACHEGRIND_I1_MISS_RATE: derived.get("instruction_l1_miss_rate"),
                    COL_CACHEGRIND_ILL_MISS_RATE: derived.get("instruction_ll_miss_rate"),
                    "Raw Cachegrind": _file_link(files.get("raw")),
                    "Annotated Cachegrind": _file_link(files.get("annotated")),
                    "Valgrind stderr": _file_link(files.get("stderr")),
                    "Metrics": _file_link(files.get("metrics")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)

    return df.sort_values(
        [
            COL_PHASE,
            COL_STAGE,
            COL_VARIANT,
            COL_PARAMS,
            COL_DATASET,
            COL_DIMENSIONS,
            COL_SAMPLES,
            COL_CLUSTERS,
        ]
    ).reset_index(drop=True)


def _language_display_name(summary_language_name: str) -> str:
    if summary_language_name == "C++":
        return LANG_CPP
    if summary_language_name == "Python":
        return LANG_PY
    return summary_language_name


def _phase_display_name(phase_key: str | None, fallback: str) -> str:
    return PHASE_DISPLAY_NAMES.get(str(phase_key), fallback) if phase_key else fallback


def _stage_display_name(stage_key: str | None, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    if not stage_key:
        return stage_display_name(FULL_STAGE_KEY)
    return stage_display_name(stage_key)


def _apply_phase_stage_categories(df: pd.DataFrame) -> pd.DataFrame:
    if COL_PHASE in df:
        df[COL_PHASE] = pd.Categorical(
            df[COL_PHASE],
            categories=list(PHASE_DISPLAY_NAMES.values()),
            ordered=True,
        )
    if COL_STAGE in df:
        stage_values = list(STAGE_DISPLAY_NAMES.values())
        extra_stages = sorted(set(df[COL_STAGE].dropna().astype(str)) - set(stage_values))
        df[COL_STAGE] = pd.Categorical(
            df[COL_STAGE],
            categories=stage_values + extra_stages,
            ordered=True,
        )
    return df


def _variant_display_name(variant_key: str | None, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    if not variant_key:
        return "Default"
    return variant_key.replace("_", " ").title()


def _params_display_name(params_key: str | None, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    if not params_key:
        return "Default"
    return params_key.replace("_", " ").title()


def _copy_stats_with_prefix(
    record: dict[str, Any],
    prefix: str,
    stats: dict[str, Any] | None,
) -> None:
    if not stats:
        return

    for key, value in stats.items():
        record[f"{prefix}_{key}"] = value


def _selected_stat(
    language_entry: dict[str, Any],
    time_field: str,
    statistic: str,
) -> float:
    stats = language_entry.get(time_field)

    if stats is None:
        raise KeyError(f"Missing time field {time_field!r} in summary language entry")

    if statistic not in stats:
        raise KeyError(
            f"Missing statistic {statistic!r} in summary time field {time_field!r}"
        )

    value = stats[statistic]

    if value is None:
        raise ValueError(
            f"Statistic {statistic!r} for time field {time_field!r} is null"
        )

    return float(value)


def _iter_phase_variant_parameterizations(
    phase_entry: dict[str, Any],
) -> Iterator[
    tuple[str, dict[str, Any], str, dict[str, Any], str, dict[str, Any]]
]:
    """Yield stage/variant/parameterization entries.

    New summaries nest variants under phase.stages. Legacy summaries that kept
    variants directly under the phase are treated as the Full stage.
    """
    if "stages" in phase_entry:
        stage_items = phase_entry.get("stages", {}).items()
    else:
        stage_items = [(stage_display_name(FULL_STAGE_KEY), {
            "stage_key": FULL_STAGE_KEY,
            "stage": stage_display_name(FULL_STAGE_KEY),
            "variants": phase_entry.get("variants", {}),
        })]

    for stage_name, stage_entry in stage_items:
        stage_key = stage_entry.get("stage_key", FULL_STAGE_KEY)
        stage_name = _stage_display_name(
            stage_key,
            stage_entry.get("stage", stage_name),
        )

        for variant_name, variant_entry in stage_entry.get("variants", {}).items():
            parameterizations = variant_entry.get("parameterizations")
            if parameterizations is None:
                raise RuntimeError(
                    "Expected summary entries with a 'parameterizations' block. "
                    "Run the current post-processing step again."
                )

            for params_name, parameterization_entry in parameterizations.items():
                yield (
                    stage_name,
                    stage_entry,
                    variant_name,
                    variant_entry,
                    params_name,
                    parameterization_entry,
                )


def _iter_comparisons(
    parameterization_entry: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    comparisons = parameterization_entry.get("comparisons")
    if comparisons:
        for reference_key, comparison_entry in comparisons.items():
            yield str(reference_key), comparison_entry
        return

    # Reporting-only support for older summaries. New post-processing emits the
    # per-reference `comparisons` block instead of these top-level fields.
    legacy_entry: dict[str, Any] = {}
    if "speedup" in parameterization_entry:
        legacy_entry["speedup"] = parameterization_entry["speedup"]
    if "parity" in parameterization_entry:
        legacy_entry["parity"] = parameterization_entry["parity"]
    if legacy_entry:
        legacy_entry.setdefault("reference_key", "reference")
        legacy_entry.setdefault("reference", "Reference")
        yield "reference", legacy_entry


def _comparison_reference_name(reference_key: str, comparison_entry: dict[str, Any]) -> str:
    return str(comparison_entry.get("reference") or reference_key.replace("_", " ").title())


def _empty_benchmark_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_PHASE,
            COL_STAGE,
            COL_LANGUAGE,
            COL_VARIANT,
            COL_PARAMS,
            COL_REFERENCE,
            COL_REFERENCE_KEY,
            COL_DATASET,
            COL_DIMENSIONS,
            COL_SAMPLES,
            COL_CLUSTERS,
            COL_ALGORITHM_ITERATIONS,
            COL_TIME_S,
            COL_TIME_FIELD,
            COL_TIME_STATISTIC,
            COL_TIMING_PROCESS_COUNT,
            COL_TIMING_VALUE_COUNT,
            COL_INERTIA,
            COL_COVARIANCE_TYPE,
            COL_LOWER_BOUND,
        ]
    )


def _empty_speedup_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_PHASE,
            COL_STAGE,
            COL_VARIANT,
            COL_PARAMS,
            COL_REFERENCE,
            COL_REFERENCE_KEY,
            COL_DATASET,
            COL_DIMENSIONS,
            COL_SAMPLES,
            COL_CLUSTERS,
            COL_TIME_FIELD,
            COL_SPEEDUP_STATISTIC,
            COL_SPEEDUP,
            COL_SPEEDUP_CI_LOW,
            COL_SPEEDUP_CI_HIGH,
            COL_SPEEDUP_CI_LEVEL,
            COL_CPP_POINT,
            COL_PY_POINT,
        ]
    )

def load_benchmark_data(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
    time_field: str = "time_s",
    statistic: str = "median",
) -> pd.DataFrame:
    """
    Load summarized timing records from benchmark_summary.json.

    Returns one row per:
        config × phase × variant × params × language

    Python reference rows are repeated per C++ variant when the summary used a
    shared reference to compute per-variant speedups.
    """
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        dataset = str(config.get("dataset", "blobs"))
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_name_from_json, phase_entry in config.get("phases", {}).items():
            phase_key = phase_entry.get("phase_key")
            phase_name = _phase_display_name(phase_key, phase_name_from_json)

            for (
                stage_name,
                _stage_entry,
                variant_name_from_json,
                variant_entry,
                params_name_from_json,
                parameterization_entry,
            ) in _iter_phase_variant_parameterizations(phase_entry):
                variant_key = variant_entry.get("variant_key")
                variant_name = _variant_display_name(
                    variant_key,
                    variant_entry.get("variant", variant_name_from_json),
                )
                params_key = parameterization_entry.get("params_key", NO_PARAMS)
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )

                for language_name_from_json, language_entry in parameterization_entry.get(
                    "languages", {}
                ).items():
                    language_name = _language_display_name(language_name_from_json)

                    selected_time = _selected_stat(
                        language_entry,
                        time_field=time_field,
                        statistic=statistic,
                    )

                    algorithm_iterations = int(
                        language_entry.get("algorithm_iterations", 1)
                    )
                    record = {
                        COL_PHASE: phase_name,
                        COL_STAGE: stage_name,
                        COL_LANGUAGE: language_name,
                        COL_VARIANT: variant_name,
                        COL_PARAMS: params_name,
                        COL_REFERENCE: "-",
                        COL_REFERENCE_KEY: "",
                        COL_DATASET: dataset,
                        COL_DIMENSIONS: D,
                        COL_SAMPLES: N,
                        COL_CLUSTERS: K,
                        COL_ALGORITHM_ITERATIONS: algorithm_iterations,
                        COL_TIME_S: selected_time,
                        COL_TIME_FIELD: time_field,
                        COL_TIME_STATISTIC: statistic,
                        COL_TIMING_PROCESS_COUNT: language_entry.get(
                            "timing_process_count"
                        ),
                        COL_TIMING_VALUE_COUNT: language_entry.get("timing_value_count"),
                        COL_INERTIA: language_entry.get("inertia"),
                        COL_COVARIANCE_TYPE: language_entry.get("covariance_type"),
                        COL_LOWER_BOUND: language_entry.get("lower_bound"),
                    }

                    _copy_stats_with_prefix(
                        record,
                        prefix="time_s",
                        stats=language_entry.get("time_s"),
                    )
                    _copy_stats_with_prefix(
                        record,
                        prefix="time_per_algorithm_iteration_s",
                        stats=language_entry.get("time_per_algorithm_iteration_s"),
                    )

                    records.append(record)

                for reference_key, comparison_entry in _iter_comparisons(parameterization_entry):
                    timing_entry = comparison_entry.get("timing")
                    if not timing_entry:
                        continue

                    selected_time = _selected_stat(
                        timing_entry,
                        time_field=time_field,
                        statistic=statistic,
                    )
                    algorithm_iterations = int(timing_entry.get("algorithm_iterations", 1))
                    record = {
                        COL_PHASE: phase_name,
                        COL_STAGE: stage_name,
                        COL_LANGUAGE: LANG_PY,
                        COL_VARIANT: variant_name,
                        COL_PARAMS: params_name,
                        COL_REFERENCE: _comparison_reference_name(reference_key, comparison_entry),
                        COL_REFERENCE_KEY: reference_key,
                        COL_DATASET: dataset,
                        COL_DIMENSIONS: D,
                        COL_SAMPLES: N,
                        COL_CLUSTERS: K,
                        COL_ALGORITHM_ITERATIONS: algorithm_iterations,
                        COL_TIME_S: selected_time,
                        COL_TIME_FIELD: time_field,
                        COL_TIME_STATISTIC: statistic,
                        COL_TIMING_PROCESS_COUNT: timing_entry.get("timing_process_count"),
                        COL_TIMING_VALUE_COUNT: timing_entry.get("timing_value_count"),
                    }
                    _copy_stats_with_prefix(record, prefix="time_s", stats=timing_entry.get("time_s"))
                    _copy_stats_with_prefix(
                        record,
                        prefix="time_per_algorithm_iteration_s",
                        stats=timing_entry.get("time_per_algorithm_iteration_s"),
                    )
                    records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        return _empty_benchmark_dataframe()

    df = _apply_phase_stage_categories(df)

    df[COL_LANGUAGE] = pd.Categorical(
        df[COL_LANGUAGE],
        categories=[LANG_CPP, LANG_PY],
        ordered=True,
    )

    variant_order = ["Static", "Dynamic", "Auto", "Reference"]
    present_variants = [v for v in variant_order if v in set(df[COL_VARIANT])]
    extra_variants = sorted(set(df[COL_VARIANT]) - set(present_variants))
    df[COL_VARIANT] = pd.Categorical(
        df[COL_VARIANT],
        categories=present_variants + extra_variants,
        ordered=True,
    )

    params_order = ["Default"]
    present_params = [p for p in params_order if p in set(df[COL_PARAMS])]
    extra_params = sorted(set(df[COL_PARAMS]) - set(present_params))
    df[COL_PARAMS] = pd.Categorical(
        df[COL_PARAMS],
        categories=present_params + extra_params,
        ordered=True,
    )

    return df.sort_values(
        [COL_PHASE, COL_STAGE, COL_VARIANT, COL_PARAMS, COL_REFERENCE, COL_DATASET, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS, COL_LANGUAGE]
    ).reset_index(drop=True)


def load_speedup_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
    time_field: str = "time_per_algorithm_iteration_s",
    ratio_statistic: str = "median_ratio",
) -> pd.DataFrame:
    """
    Load precomputed Python/C++ speedups and clustered-bootstrap CIs
    from benchmark_summary.json.

    Default uses median speedup on time-per-algorithm-iteration.
    """
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        dataset = str(config.get("dataset", "blobs"))
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_name_from_json, phase_entry in config.get("phases", {}).items():
            phase_key = phase_entry.get("phase_key")
            phase_name = _phase_display_name(phase_key, phase_name_from_json)

            for (
                stage_name,
                _stage_entry,
                variant_name_from_json,
                variant_entry,
                params_name_from_json,
                parameterization_entry,
            ) in _iter_phase_variant_parameterizations(phase_entry):
                variant_key = variant_entry.get("variant_key")
                variant_name = _variant_display_name(
                    variant_key,
                    variant_entry.get("variant", variant_name_from_json),
                )
                params_key = parameterization_entry.get("params_key", NO_PARAMS)
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )
                for reference_key, comparison_entry in _iter_comparisons(parameterization_entry):
                    speedup_entry = (
                        comparison_entry.get("speedup", {})
                        .get(time_field, {})
                        .get(ratio_statistic)
                    )

                    if not speedup_entry:
                        continue

                    records.append(
                        {
                            COL_PHASE: phase_name,
                            COL_STAGE: stage_name,
                            COL_VARIANT: variant_name,
                            COL_PARAMS: params_name,
                            COL_REFERENCE: _comparison_reference_name(reference_key, comparison_entry),
                            COL_REFERENCE_KEY: reference_key,
                            COL_DATASET: dataset,
                            COL_DIMENSIONS: D,
                            COL_SAMPLES: N,
                            COL_CLUSTERS: K,
                            COL_TIME_FIELD: time_field,
                            COL_SPEEDUP_STATISTIC: ratio_statistic,
                            COL_SPEEDUP: speedup_entry["point"],
                            COL_SPEEDUP_CI_LOW: speedup_entry["ci_low"],
                            COL_SPEEDUP_CI_HIGH: speedup_entry["ci_high"],
                            COL_SPEEDUP_CI_LEVEL: speedup_entry["ci_level"],
                            COL_CPP_POINT: speedup_entry.get("cpp_point"),
                            COL_PY_POINT: speedup_entry.get("python_point"),
                        }
                    )

    df = pd.DataFrame(records)

    if df.empty:
        return _empty_speedup_dataframe()

    df = _apply_phase_stage_categories(df)

    variant_order = ["Static", "Dynamic", "Auto", "Reference"]
    present_variants = [v for v in variant_order if v in set(df[COL_VARIANT])]
    extra_variants = sorted(set(df[COL_VARIANT]) - set(present_variants))
    df[COL_VARIANT] = pd.Categorical(
        df[COL_VARIANT],
        categories=present_variants + extra_variants,
        ordered=True,
    )

    params_order = ["Default"]
    present_params = [p for p in params_order if p in set(df[COL_PARAMS])]
    extra_params = sorted(set(df[COL_PARAMS]) - set(present_params))
    df[COL_PARAMS] = pd.Categorical(
        df[COL_PARAMS],
        categories=present_params + extra_params,
        ordered=True,
    )

    return df.sort_values(
        [COL_PHASE, COL_STAGE, COL_VARIANT, COL_PARAMS, COL_REFERENCE, COL_DATASET, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
    ).reset_index(drop=True)


def load_lloyd_parity_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load Lloyd parity/inertia results from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        dataset = str(config.get("dataset", "blobs"))
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "lloyd":
                continue

            for (
                stage_name,
                _stage_entry,
                variant_name_from_json,
                variant_entry,
                params_name_from_json,
                parameterization_entry,
            ) in _iter_phase_variant_parameterizations(phase_entry):
                variant_key = variant_entry.get("variant_key")
                variant_name = _variant_display_name(
                    variant_key,
                    variant_entry.get("variant", variant_name_from_json),
                )
                params_key = parameterization_entry.get("params_key", NO_PARAMS)
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )

                for reference_key, comparison_entry in _iter_comparisons(parameterization_entry):
                    parity = comparison_entry.get("parity")
                    if not parity:
                        continue

                    diff_pct = float(parity["inertia_diff_pct"])
                    thresholds = parity.get("thresholds", {})
                    failure_reasons = parity.get("failure_reasons", [])
                    passed = parity.get("status") == "PASS"

                    records.append(
                        {
                            COL_PHASE: PHASE_DISPLAY_NAMES["lloyd"],
                            COL_STAGE: stage_name,
                            COL_VARIANT: variant_name,
                            COL_PARAMS: params_name,
                            COL_REFERENCE: _comparison_reference_name(reference_key, comparison_entry),
                            COL_REFERENCE_KEY: reference_key,
                            COL_DATASET: dataset,
                            COL_DIMENSIONS: D,
                            COL_SAMPLES: N,
                            COL_CLUSTERS: K,
                            "Diff (%)": diff_pct,
                            "Status": "✅ PASS" if passed else "❌ FAIL",
                            "Failure Reasons": ", ".join(failure_reasons),
                            "Lloyd C++ Algorithm Iterations": parity[
                                "cpp_algorithm_iterations"
                            ],
                            "Lloyd Py Algorithm Iterations": parity[
                                "python_algorithm_iterations"
                            ],
                            "Algorithm Iteration Diff Abs": parity.get(
                                "algorithm_iteration_diff_abs"
                            ),
                            "C++ Inertia": parity["cpp_inertia"],
                            "Py Inertia": parity["python_inertia"],
                            "Inertia Diff Abs": parity["inertia_diff_abs"],
                            "Inertia Diff Threshold (%)": thresholds.get(
                                "inertia_diff_pct"
                            ),
                            "Algorithm Iteration Diff Threshold Abs": thresholds.get(
                                "algorithm_iteration_diff_abs"
                            ),
                        }
                    )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)
    return df.sort_values(
        by=["Diff (%)", COL_DATASET, COL_STAGE, COL_REFERENCE], ascending=[False, True, True, True]
    ).reset_index(drop=True)


def load_gmm_parity_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load GMM C++/Python parity records from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        dataset = str(config.get("dataset", "blobs"))
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "gmm":
                continue

            for (
                stage_name,
                _stage_entry,
                variant_name_from_json,
                variant_entry,
                params_name_from_json,
                parameterization_entry,
            ) in _iter_phase_variant_parameterizations(phase_entry):
                variant_key = variant_entry.get("variant_key")
                variant_name = _variant_display_name(
                    variant_key,
                    variant_entry.get("variant", variant_name_from_json),
                )
                params_key = parameterization_entry.get("params_key", NO_PARAMS)
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )

                for reference_key, comparison_entry in _iter_comparisons(parameterization_entry):
                    parity = comparison_entry.get("parity")
                    if not parity:
                        continue

                    status = parity.get("status", "FAIL")
                    failure_reasons = parity.get("failure_reasons", [])

                    records.append(
                        {
                            COL_PHASE: PHASE_DISPLAY_NAMES["gmm"],
                            COL_STAGE: stage_name,
                            COL_VARIANT: variant_name,
                            COL_PARAMS: params_name,
                            COL_REFERENCE: _comparison_reference_name(reference_key, comparison_entry),
                            COL_REFERENCE_KEY: reference_key,
                            COL_DATASET: dataset,
                            COL_DIMENSIONS: D,
                            COL_SAMPLES: N,
                            COL_CLUSTERS: K,
                            "Status": "✅ PASS" if status == "PASS" else "❌ FAIL",
                            "Failure Reasons": ", ".join(failure_reasons),
                            "Covariance Type": parity.get("covariance_type"),
                            "GMM C++ Algorithm Iterations": parity.get(
                                "cpp_algorithm_iterations"
                            ),
                            "GMM Py Algorithm Iterations": parity.get(
                                "python_algorithm_iterations"
                            ),
                            "Algorithm Iteration Diff Abs": parity.get(
                                "algorithm_iteration_diff_abs"
                            ),
                            "C++ Lower Bound": parity.get("cpp_lower_bound"),
                            "Py Lower Bound": parity.get("python_lower_bound"),
                            "Lower Bound Diff Abs": parity.get("lower_bound_diff_abs"),
                            "Lower Bound Diff (%)": parity.get("lower_bound_diff_pct"),
                            "Weights Max Abs Diff": parity.get("weights_max_abs_diff"),
                            "Means Max Abs Diff": parity.get("means_max_abs_diff"),
                            "Covariances Max Rel Diff": parity.get("covariances_max_rel_diff"),
                            "Lower Bound Diff Abs Threshold": parity.get("thresholds", {}).get(
                                "lower_bound_diff_abs"
                            ),
                            "Weights Max Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "weights_max_abs_diff"
                            ),
                            "Means Max Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "means_max_abs_diff"
                            ),
                            "Covariances Max Rel Diff Threshold": parity.get("thresholds", {}).get(
                                "covariances_max_rel_diff"
                            ),
                            "Algorithm Iteration Diff Threshold Abs": parity.get(
                                "thresholds", {}
                            ).get("algorithm_iteration_diff_abs"),
                        }
                    )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)
    return df.sort_values(
        by=[
            "Status",
            "Lower Bound Diff Abs",
            COL_STAGE,
            COL_VARIANT,
            COL_PARAMS,
            COL_REFERENCE,
            COL_DATASET,
            COL_DIMENSIONS,
            COL_SAMPLES,
            COL_CLUSTERS,
        ],
        ascending=[True, False, True, True, True, True, True, True, True, True],
    ).reset_index(drop=True)


def load_hdbscan_parity_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load HDBSCAN stage parity records from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        dataset = str(config.get("dataset", "blobs"))
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "hdbscan":
                continue

            for (
                stage_name,
                _stage_entry,
                variant_name_from_json,
                variant_entry,
                params_name_from_json,
                parameterization_entry,
            ) in _iter_phase_variant_parameterizations(phase_entry):
                variant_key = variant_entry.get("variant_key")
                variant_name = _variant_display_name(
                    variant_key,
                    variant_entry.get("variant", variant_name_from_json),
                )
                params_key = parameterization_entry.get("params_key", NO_PARAMS)
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )

                for reference_key, comparison_entry in _iter_comparisons(parameterization_entry):
                    parity = comparison_entry.get("parity")
                    if not parity:
                        continue

                    status = parity.get("status", "FAIL")
                    failure_reasons = parity.get("failure_reasons", [])
                    records.append(
                        {
                            COL_PHASE: PHASE_DISPLAY_NAMES["hdbscan"],
                            COL_STAGE: stage_name,
                            COL_VARIANT: variant_name,
                            COL_PARAMS: params_name,
                            COL_REFERENCE: _comparison_reference_name(reference_key, comparison_entry),
                            COL_REFERENCE_KEY: reference_key,
                            COL_DATASET: dataset,
                            COL_DIMENSIONS: D,
                            COL_SAMPLES: N,
                            COL_CLUSTERS: K,
                            "Status": "✅ PASS" if status == "PASS" else "❌ FAIL",
                            "Failure Reasons": ", ".join(failure_reasons),
                            "Shape": parity.get("cpp_shape"),
                            "Reference Shape": parity.get("python_shape"),
                            "Hash Equal": parity.get("hash_equal"),
                            "Summary Scalar Abs Diff Max": parity.get("summary_scalar_abs_diff_max"),
                            "Summary Scalar Rel Diff Max": parity.get("summary_scalar_rel_diff_max"),
                            "Probe Value Max Abs Diff": parity.get("probe_value_max_abs_diff"),
                            "Summary Scalar Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "summary_scalar_abs_diff"
                            ),
                            "Summary Scalar Rel Diff Threshold": parity.get("thresholds", {}).get(
                                "summary_scalar_rel_diff"
                            ),
                            "Probe Value Max Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "probe_value_max_abs_diff"
                            ),
                            "C++ Diagonal Max Abs": parity.get("cpp_diagonal_max_abs"),
                            "Reference Diagonal Max Abs": parity.get("python_diagonal_max_abs"),
                            "C++ Symmetry Max Abs": parity.get("cpp_symmetry_max_abs"),
                            "Reference Symmetry Max Abs": parity.get("python_symmetry_max_abs"),
                            "Diagonal Max Abs Threshold": parity.get("thresholds", {}).get(
                                "diagonal_max_abs"
                            ),
                            "Symmetry Max Abs Threshold": parity.get("thresholds", {}).get(
                                "symmetry_max_abs"
                            ),
                            "C++ Noise Count": parity.get("cpp_noise_count"),
                            "Reference Noise Count": parity.get("python_noise_count"),
                            "C++ Cluster Count": parity.get("cpp_cluster_count"),
                            "Reference Cluster Count": parity.get("python_cluster_count"),
                            "Label Hash Equal": parity.get("label_hash_equal"),
                            "Label Summary Scalar Abs Diff Max": parity.get(
                                "label_summary_scalar_abs_diff_max"
                            ),
                            "Label Summary Scalar Rel Diff Max": parity.get(
                                "label_summary_scalar_rel_diff_max"
                            ),
                            "Label Probe Value Max Abs Diff": parity.get(
                                "label_probe_value_max_abs_diff"
                            ),
                            "Label Summary Scalar Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "summary_scalar_abs_diff"
                            ),
                            "Label Summary Scalar Rel Diff Threshold": parity.get("thresholds", {}).get(
                                "summary_scalar_rel_diff"
                            ),
                            "Label Probe Value Max Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "probe_value_max_abs_diff"
                            ),
                            "Probability Hash Equal": parity.get("probability_hash_equal"),
                            "Probability Summary Scalar Abs Diff Max": parity.get(
                                "probability_summary_scalar_abs_diff_max"
                            ),
                            "Probability Summary Scalar Rel Diff Max": parity.get(
                                "probability_summary_scalar_rel_diff_max"
                            ),
                            "Probability Probe Value Max Abs Diff": parity.get(
                                "probability_probe_value_max_abs_diff"
                            ),
                            "Probability Summary Scalar Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "summary_scalar_abs_diff"
                            ),
                            "Probability Summary Scalar Rel Diff Threshold": parity.get("thresholds", {}).get(
                                "summary_scalar_rel_diff"
                            ),
                            "Probability Probe Value Max Abs Diff Threshold": parity.get("thresholds", {}).get(
                                "probe_value_max_abs_diff"
                            ),
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = _apply_phase_stage_categories(df)
    return df.sort_values(
        by=["Status", COL_STAGE, COL_VARIANT, COL_PARAMS, COL_REFERENCE, COL_DATASET, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS],
        ascending=[True, True, True, True, True, True, True, True, True],
    ).reset_index(drop=True)
