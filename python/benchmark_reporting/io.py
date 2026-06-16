from pathlib import Path
import json
from typing import Any, Iterator
import pandas as pd

from python.benchmark_pipeline.paths import repo_relative_path

from .constants import *

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
        records.append(
            {
                COL_PHASE: _phase_display_name(
                    phase_key,
                    exclusion.get("phase", phase_key),
                ),
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

    df[COL_PHASE] = pd.Categorical(
        df[COL_PHASE],
        categories=list(PHASE_MAP.values()),
        ordered=True,
    )

    return df.sort_values(
        [COL_PHASE, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
    ).reset_index(drop=True)


def _language_display_name(summary_language_name: str) -> str:
    if summary_language_name == "C++":
        return LANG_CPP
    if summary_language_name == "Python":
        return LANG_PY
    return summary_language_name


def _phase_display_name(phase_key: str, fallback: str) -> str:
    return PHASE_MAP.get(phase_key, fallback)


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
) -> Iterator[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    """Yield schema v3 variant/parameterization entries."""
    for variant_name, variant_entry in phase_entry.get("variants", {}).items():
        parameterizations = variant_entry.get("parameterizations")
        if parameterizations is None:
            raise RuntimeError(
                "Expected schema v3 summary entries with a 'parameterizations' block. "
                "Run the current post-processing step again."
            )

        for params_name, parameterization_entry in parameterizations.items():
            yield variant_name, variant_entry, params_name, parameterization_entry


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
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_name_from_json, phase_entry in config.get("phases", {}).items():
            phase_key = phase_entry.get("phase_key")
            phase_name = _phase_display_name(phase_key, phase_name_from_json)

            for (
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
                params_key = parameterization_entry.get("params_key", "default")
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
                        COL_LANGUAGE: language_name,
                        COL_VARIANT: variant_name,
                        COL_PARAMS: params_name,
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

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df[COL_PHASE] = pd.Categorical(
        df[COL_PHASE],
        categories=list(PHASE_MAP.values()),
        ordered=True,
    )

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
        [COL_PHASE, COL_VARIANT, COL_PARAMS, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS, COL_LANGUAGE]
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
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_name_from_json, phase_entry in config.get("phases", {}).items():
            phase_key = phase_entry.get("phase_key")
            phase_name = _phase_display_name(phase_key, phase_name_from_json)

            for (
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
                params_key = parameterization_entry.get("params_key", "default")
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )
                speedup_entry = (
                    parameterization_entry.get("speedup", {})
                    .get(time_field, {})
                    .get(ratio_statistic)
                )

                if not speedup_entry:
                    continue

                records.append(
                    {
                        COL_PHASE: phase_name,
                        COL_VARIANT: variant_name,
                        COL_PARAMS: params_name,
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
        return df

    df[COL_PHASE] = pd.Categorical(
        df[COL_PHASE],
        categories=list(PHASE_MAP.values()),
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
        [COL_PHASE, COL_VARIANT, COL_PARAMS, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
    ).reset_index(drop=True)


def load_lloyd_parity_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load Lloyd parity/inertia results from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "lloyd":
                continue

            for (
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
                params_key = parameterization_entry.get("params_key", "default")
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )
                parity = parameterization_entry.get("parity")
                if not parity:
                    continue

                diff_pct = float(parity["inertia_diff_pct"])
                thresholds = parity.get("thresholds", {})
                failure_reasons = parity.get("failure_reasons", [])
                passed = parity.get("status") == "PASS"

                records.append(
                    {
                        COL_VARIANT: variant_name,
                        COL_PARAMS: params_name,
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

    return df.sort_values(by="Diff (%)", ascending=False).reset_index(drop=True)


def load_gmm_parity_summary(
    summary_json: str | Path = DEFAULT_BENCHMARK_SUMMARY_JSON,
) -> pd.DataFrame:
    """Load GMM C++/Python parity records from benchmark_summary.json."""
    summary = load_benchmark_summary(summary_json)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "gmm":
                continue

            for (
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
                params_key = parameterization_entry.get("params_key", "default")
                params_name = _params_display_name(
                    params_key,
                    parameterization_entry.get("params", params_name_from_json),
                )
                parity = parameterization_entry.get("parity")
                if not parity:
                    continue

                status = parity.get("status", "FAIL")
                failure_reasons = parity.get("failure_reasons", [])

                records.append(
                    {
                        COL_VARIANT: variant_name,
                        COL_PARAMS: params_name,
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

    return df.sort_values(
        by=[
            "Status",
            "Lower Bound Diff Abs",
            COL_VARIANT,
            COL_PARAMS,
            COL_DIMENSIONS,
            COL_SAMPLES,
            COL_CLUSTERS,
        ],
        ascending=[True, False, True, True, True, True, True],
    ).reset_index(drop=True)
