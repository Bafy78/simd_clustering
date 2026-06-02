import re
from pathlib import Path
import json
from typing import Any
import pandas as pd

from .constants import *

SUMMARY_FILENAME = "benchmark_summary.json"


def extract_config_params(filename):
    """Extract Dimensions (D), Samples (N), and Clusters (K) from a config ID."""
    match = re.search(r"(\d+)D_(\d+)N_(\d+)K", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def _summary_path(
    data_dir=Path("./datasets"), summary_filename=SUMMARY_FILENAME
) -> Path:
    data_dir = Path(data_dir)

    if data_dir.is_file():
        return data_dir

    return data_dir / summary_filename


def load_benchmark_summary(
    data_dir=Path("./datasets"),
    summary_filename=SUMMARY_FILENAME,
) -> dict[str, Any]:
    path = _summary_path(data_dir, summary_filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark summary not found: {path}. "
            f"Run the post-processing step first."
        )

    with path.open("r") as f:
        return json.load(f)


def _language_display_name(summary_language_name: str) -> str:
    if summary_language_name == "C++":
        return LANG_CPP
    if summary_language_name == "Python":
        return LANG_PY
    return summary_language_name


def _phase_display_name(phase_key: str, fallback: str) -> str:
    return PHASE_MAP.get(phase_key, fallback)


def _copy_stats_with_prefix(
    record: dict[str, Any],
    *,
    prefix: str,
    stats: dict[str, Any] | None,
) -> None:
    if not stats:
        return

    for key, value in stats.items():
        record[f"{prefix}_{key}"] = value


def _selected_stat(
    language_entry: dict[str, Any],
    *,
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


def load_benchmark_data(
    data_dir=Path("./datasets"),
    *,
    summary_filename=SUMMARY_FILENAME,
    time_field: str = "time_s",
    statistic: str = "median",
) -> pd.DataFrame:
    """
    Compatibility loader for the existing report notebook.

    Returns one row per:
        config × phase × language

    COL_TIME_S is the selected summary statistic, by default median total time.

    Iterative phases such as Lloyd and GMM use their recorded algorithm
    iteration count. Non-iterative phases use 1.
    """
    summary = load_benchmark_summary(data_dir, summary_filename)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        config_id = config["config_id"]
        configuration = config.get(
            "configuration",
            f"{D}D | {N}N | {K}K",
        )

        for phase_name_from_json, phase_entry in config.get("phases", {}).items():
            phase_key = phase_entry.get("phase_key")
            phase_name = _phase_display_name(phase_key, phase_name_from_json)

            for language_name_from_json, language_entry in phase_entry.get(
                "languages", {}
            ).items():
                language_name = _language_display_name(language_name_from_json)

                selected_time = _selected_stat(
                    language_entry,
                    time_field=time_field,
                    statistic=statistic,
                )

                iterations = int(language_entry.get("iterations", 1))

                record = {
                    COL_PHASE: phase_name,
                    COL_LANGUAGE: language_name,
                    COL_DIMENSIONS: D,
                    COL_SAMPLES: N,
                    COL_CLUSTERS: K,
                    COL_ITERATIONS: iterations,
                    COL_TIME_S: selected_time,
                    COL_CONFIGURATION: configuration,
                    COL_CONFIG_ID: config_id,
                    COL_PHASE_KEY: phase_key,
                    COL_TIME_FIELD: time_field,
                    COL_TIME_STATISTIC: statistic,
                    COL_PROCESS_COUNT: language_entry.get("process_count"),
                    COL_TIMING_VALUE_COUNT: language_entry.get("timing_value_count"),
                    COL_INERTIA: language_entry.get("inertia"),
                    COL_COVARIANCE_TYPE: language_entry.get("covariance_type"),
                    COL_CONVERGED: language_entry.get("converged"),
                    COL_LOWER_BOUND: language_entry.get("lower_bound"),
                }

                _copy_stats_with_prefix(
                    record,
                    prefix="time_s",
                    stats=language_entry.get("time_s"),
                )
                _copy_stats_with_prefix(
                    record,
                    prefix="time_per_iteration_s",
                    stats=language_entry.get("time_per_iteration_s"),
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

    return df.sort_values(
        [COL_PHASE, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS, COL_LANGUAGE]
    ).reset_index(drop=True)


def load_speedup_summary(
    data_dir=Path("./datasets"),
    *,
    summary_filename=SUMMARY_FILENAME,
    time_field: str = "time_per_iteration_s",
    ratio_statistic: str = "median_ratio",
) -> pd.DataFrame:
    """
    Load precomputed Python/C++ speedups and clustered-bootstrap CIs
    from benchmark_summary.json.

    Default uses median speedup on time-per-iteration.
    """
    summary = load_benchmark_summary(data_dir, summary_filename)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        config_id = config["config_id"]
        configuration = config.get(
            "configuration",
            f"{D}D | {N}N | {K}K",
        )

        for phase_name_from_json, phase_entry in config.get("phases", {}).items():
            phase_key = phase_entry.get("phase_key")
            phase_name = _phase_display_name(phase_key, phase_name_from_json)

            speedup_entry = (
                phase_entry.get("speedup", {}).get(time_field, {}).get(ratio_statistic)
            )

            if not speedup_entry:
                continue

            records.append(
                {
                    COL_PHASE: phase_name,
                    COL_DIMENSIONS: D,
                    COL_SAMPLES: N,
                    COL_CLUSTERS: K,
                    COL_CONFIGURATION: configuration,
                    COL_CONFIG_ID: config_id,
                    COL_PHASE_KEY: phase_key,
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

    return df.sort_values(
        [COL_PHASE, COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
    ).reset_index(drop=True)


def load_lloyd_parity_summary(
    data_dir=Path("./datasets"),
    *,
    summary_filename=SUMMARY_FILENAME,
    tolerance_pct: float | None = None,
) -> pd.DataFrame:
    """
    Load Lloyd parity/inertia results from benchmark_summary.json.

    Returns a dataframe compatible with the old validation display.
    """
    summary = load_benchmark_summary(data_dir, summary_filename)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        configuration = config.get(
            "configuration",
            f"{D}D | {N}N | {K}K",
        )

        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "lloyd":
                continue

            parity = phase_entry.get("parity")
            if not parity:
                continue

            diff_pct = float(parity["inertia_diff_pct"])
            effective_tolerance = (
                float(tolerance_pct)
                if tolerance_pct is not None
                else float(parity["tolerance_pct"])
            )

            passed = diff_pct <= effective_tolerance

            records.append(
                {
                    COL_CONFIGURATION: configuration,
                    COL_DIMENSIONS: D,
                    COL_SAMPLES: N,
                    COL_CLUSTERS: K,
                    "Diff (%)": diff_pct,
                    "Status": "✅ PASS" if passed else "❌ FAIL",
                    "Lloyd C++ Iteration": parity["cpp_iterations"],
                    "Lloyd Py Iterations": parity["python_iterations"],
                    "C++ Inertia": parity["cpp_inertia"],
                    "Py Inertia": parity["python_inertia"],
                    "Inertia Diff Abs": parity["inertia_diff_abs"],
                    "Tolerance (%)": effective_tolerance,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    return df.sort_values(by="Diff (%)", ascending=False).reset_index(drop=True)


def load_gmm_parity_summary(
    data_dir=Path("./datasets"),
    *,
    summary_filename=SUMMARY_FILENAME,
) -> pd.DataFrame:
    """Load GMM C++/Python parity records from benchmark_summary.json."""
    summary = load_benchmark_summary(data_dir, summary_filename)
    records: list[dict[str, Any]] = []

    for config in summary.get("configs", []):
        D = int(config["dimensions"])
        N = int(config["samples"])
        K = int(config["clusters"])
        configuration = config.get(
            "configuration",
            f"{D}D | {N}N | {K}K",
        )

        for phase_entry in config.get("phases", {}).values():
            if phase_entry.get("phase_key") != "gmm":
                continue

            parity = phase_entry.get("parity")
            if not parity:
                continue

            status = parity.get("status", "FAIL")
            failure_reasons = parity.get("failure_reasons", [])

            records.append(
                {
                    COL_CONFIGURATION: configuration,
                    COL_DIMENSIONS: D,
                    COL_SAMPLES: N,
                    COL_CLUSTERS: K,
                    "Status": "✅ PASS" if status == "PASS" else "❌ FAIL",
                    "Failure Reasons": ", ".join(failure_reasons),
                    "Covariance Type": parity.get("covariance_type"),
                    "Converged Match": parity.get("converged_match"),
                    "GMM C++ Iterations": parity.get("cpp_iterations"),
                    "GMM Py Iterations": parity.get("python_iterations"),
                    "Iteration Diff Abs": parity.get("iteration_diff_abs"),
                    "C++ Converged": parity.get("cpp_converged"),
                    "Py Converged": parity.get("python_converged"),
                    "C++ Lower Bound": parity.get("cpp_lower_bound"),
                    "Py Lower Bound": parity.get("python_lower_bound"),
                    "Lower Bound Diff Abs": parity.get("lower_bound_diff_abs"),
                    "Lower Bound Diff (%)": parity.get("lower_bound_diff_pct"),
                    "Weights Max Abs Diff": parity.get("weights_max_abs_diff"),
                    "Means Max Abs Diff": parity.get("means_max_abs_diff"),
                    "Covariances Max Rel Diff": parity.get("covariances_max_rel_diff"),
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    return df.sort_values(
        by=[
            "Status",
            "Lower Bound Diff Abs",
            COL_DIMENSIONS,
            COL_SAMPLES,
            COL_CLUSTERS,
        ],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
