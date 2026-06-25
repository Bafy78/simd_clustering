from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from benchmark_reporting.constants import *
from benchmark_reporting.io import (
    load_benchmark_data,
    load_benchmark_summary,
    load_cachegrind_summary,
    load_compile_artifact_summary,
    load_gmm_parity_summary,
    load_hdbscan_parity_summary,
    load_lloyd_parity_summary,
    load_speedup_summary,
    load_spill_detection_summary,
)
from benchmark_reporting.plotting import plot_clustered_heatmap_grid

DEFAULT_SELECTED_REFERENCE_KEY = "sklearn_brute"
DEFAULT_REFERENCE_FALLBACK_KEY = "reference"

COL_BASELINE = "Baseline"
COL_CANDIDATE = "Candidate"
COL_BASELINE_LABEL = "Baseline Label"
COL_CANDIDATE_LABEL = "Candidate Label"
COL_COMMON = "Common"
COL_BASELINE_ONLY = "Baseline Only"
COL_CANDIDATE_ONLY = "Candidate Only"
COL_ENTITY = "Entity"
COL_METRIC = "Metric"
COL_VALUE = "Value"
COL_SELECTED_REFERENCE_KEY = "Selected Reference Key"
COL_REQUESTED_REFERENCE_KEY = "Requested Reference Key"
COL_FALLBACK_REFERENCE_KEY = "Fallback Reference Key"
COL_SPEEDUP_BASELINE = "Baseline Speedup (x)"
COL_SPEEDUP_CANDIDATE = "Candidate Speedup (x)"
COL_SPEEDUP_RATIO = "Speedup Ratio"
COL_SPEEDUP_DELTA_PCT = "Speedup Delta (%)"
COL_MATCHED_POINTS = "Matched Points"
COL_MEDIAN_SPEEDUP_RATIO = "Median Speedup Ratio"
COL_MEDIAN_SPEEDUP_DELTA_PCT = "Median Speedup Delta (%)"
COL_MEAN_SPEEDUP_DELTA_PCT = "Mean Speedup Delta (%)"
COL_MIN_SPEEDUP_DELTA_PCT = "Min Speedup Delta (%)"
COL_MAX_SPEEDUP_DELTA_PCT = "Max Speedup Delta (%)"
COL_STATUS_BASELINE = "Baseline Status"
COL_STATUS_CANDIDATE = "Candidate Status"
COL_STATUS_TRANSITION = "Status Transition"
COL_PARITY_PRESSURE = "Parity Pressure"
COL_WORST_CHECK = "Worst Check"
COL_PARITY_PRESSURE_BASELINE = "Baseline Parity Pressure"
COL_PARITY_PRESSURE_CANDIDATE = "Candidate Parity Pressure"
COL_PARITY_PRESSURE_DELTA = "Parity Pressure Delta"
COL_PARITY_PRESSURE_ABS_DELTA = "Parity Pressure Abs Delta"
COL_WORST_CHECK_BASELINE = "Baseline Worst Check"
COL_WORST_CHECK_CANDIDATE = "Candidate Worst Check"
COL_COUNTER = "Counter"
COL_COUNTER_BASELINE = "Baseline Counter"
COL_COUNTER_CANDIDATE = "Candidate Counter"
COL_COUNTER_RATIO = "Counter Ratio"
COL_COUNTER_DELTA_PCT = "Counter Delta (%)"
COL_MEDIAN_COUNTER_DELTA_PCT = "Median Counter Delta (%)"
COL_MEAN_COUNTER_DELTA_PCT = "Mean Counter Delta (%)"
COL_MIN_COUNTER_DELTA_PCT = "Min Counter Delta (%)"
COL_MAX_COUNTER_DELTA_PCT = "Max Counter Delta (%)"
COL_PARITY_NUMERIC_METRIC = "Parity Metric"
COL_PARITY_BASELINE_VALUE = "Baseline Value"
COL_PARITY_CANDIDATE_VALUE = "Candidate Value"
COL_PARITY_DELTA = "Delta"
COL_TRANSITION_COUNT = "Count"
COL_POINT_SET = "Point Set"
COL_DIRECTION = "Direction"
COL_MISSING_LEVEL = "Missing Level"
COL_MISSING_DEPTH = "Missing Depth"
COL_PATH = "Path"
COL_MISSING_POINTS = "Missing Points"
COL_GRID_SUMMARY = "Grid Summary"
COL_NEXT_DETAIL = "Next Detail"

SPEEDUP_MATCH_COLS = [
    COL_PHASE,
    COL_STAGE,
    COL_VARIANT,
    COL_PARAMS,
    COL_REFERENCE_KEY,
    COL_DIMENSIONS,
    COL_SAMPLES,
    COL_CLUSTERS,
]

CACHEGRIND_MATCH_COLS = [
    COL_PHASE,
    COL_STAGE,
    COL_VARIANT,
    COL_PARAMS,
    COL_DIMENSIONS,
    COL_SAMPLES,
    COL_CLUSTERS,
]

PARITY_MATCH_COLS = SPEEDUP_MATCH_COLS

SPILL_MATCH_COLS = [
    COL_PHASE,
    COL_STAGE,
    COL_VARIANT,
    COL_PARAMS,
    COL_CPP_CASE,
    COL_DIMENSIONS,
]

CACHEGRIND_COUNTER_COLS = [
    COL_CACHEGRIND_IR,
    COL_CACHEGRIND_I1MR,
    COL_CACHEGRIND_ILMR,
    COL_CACHEGRIND_DR,
    COL_CACHEGRIND_D1MR,
    COL_CACHEGRIND_DLMR,
    COL_CACHEGRIND_DW,
    COL_CACHEGRIND_D1MW,
    COL_CACHEGRIND_DLMW,
]

SPEEDUP_COMPARISON_DISPLAY_COLS = [
    COL_PHASE,
    COL_STAGE,
    COL_VARIANT,
    COL_PARAMS,
    COL_REFERENCE_KEY,
    COL_DIMENSIONS,
    COL_SAMPLES,
    COL_CLUSTERS,
    COL_SPEEDUP_BASELINE,
    COL_SPEEDUP_CANDIDATE,
    COL_SPEEDUP_RATIO,
    COL_SPEEDUP_DELTA_PCT,
]

CACHEGRIND_COMPARISON_DISPLAY_COUNTERS = [
    COL_CACHEGRIND_IR,
    COL_CACHEGRIND_DR,
    COL_CACHEGRIND_DW,
    COL_CACHEGRIND_D1MR,
    COL_CACHEGRIND_DLMR,
]

SPEEDUP_SUMMARY_GROUPS = {
    "by_phase": [COL_PHASE],
    "by_phase_stage": [COL_PHASE, COL_STAGE],
    "by_phase_stage_variant": [COL_PHASE, COL_STAGE, COL_VARIANT],
    "by_phase_stage_variant_params": [
        COL_PHASE,
        COL_STAGE,
        COL_VARIANT,
        COL_PARAMS,
    ],
    "by_dimensions": [COL_DIMENSIONS],
    "by_samples": [COL_SAMPLES],
    "by_clusters": [COL_CLUSTERS],
}



@dataclass(frozen=True)
class SummaryBundle:
    """Loaded reporting views for one benchmark summary JSON file."""

    path: Path
    label: str
    raw: dict
    benchmark_data: pd.DataFrame
    speedups: pd.DataFrame
    compile_artifacts: pd.DataFrame
    cachegrind: pd.DataFrame
    spills: pd.DataFrame
    lloyd_parity: pd.DataFrame
    gmm_parity: pd.DataFrame
    hdbscan_parity: pd.DataFrame

    @property
    def parity_frames(self) -> Mapping[str, pd.DataFrame]:
        return {
            "Lloyd": self.lloyd_parity,
            "GMM": self.gmm_parity,
            "HDBSCAN": self.hdbscan_parity,
        }

    @property
    def parity(self) -> pd.DataFrame:
        frames = [df for df in self.parity_frames.values() if not df.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)


def load_labeled_summary(
    summary_json: str | Path,
    *,
    benchmark_time_field: str = "time_s",
    benchmark_statistic: str = "median",
    speedup_time_field: str = "time_per_algorithm_iteration_s",
    speedup_ratio_statistic: str = "median_ratio",
) -> SummaryBundle:
    """Load one benchmark summary and infer a compiler/architecture label."""

    path = Path(summary_json)
    raw = load_benchmark_summary(path)
    compile_artifacts = load_compile_artifact_summary(path)

    lloyd_parity = _ensure_phase(
        load_lloyd_parity_summary(path),
        "Lloyd Algorithm",
    )
    gmm_parity = _ensure_phase(
        load_gmm_parity_summary(path),
        "GaussianMixture EM",
    )
    hdbscan_parity = _ensure_phase(
        load_hdbscan_parity_summary(path),
        "HDBSCAN",
    )

    return SummaryBundle(
        path=path,
        label=infer_run_label(compile_artifacts),
        raw=raw,
        benchmark_data=load_benchmark_data(
            path,
            time_field=benchmark_time_field,
            statistic=benchmark_statistic,
        ),
        speedups=load_speedup_summary(
            path,
            time_field=speedup_time_field,
            ratio_statistic=speedup_ratio_statistic,
        ),
        compile_artifacts=compile_artifacts,
        cachegrind=load_cachegrind_summary(path),
        spills=load_spill_detection_summary(path),
        lloyd_parity=lloyd_parity,
        gmm_parity=gmm_parity,
        hdbscan_parity=hdbscan_parity,
    )


def infer_run_label(compile_artifact_df: pd.DataFrame) -> str:
    """Infer a stable run label from compiler and architecture metadata."""

    if compile_artifact_df.empty:
        return "Unknown compiler / unknown architecture"

    executable = _one_or_mixed(compile_artifact_df, COL_COMPILER_EXECUTABLE)
    executable = _compact_executable(executable)

    version = _one_or_mixed(compile_artifact_df, COL_COMPILER_VERSION)
    version = _compact_compiler_version(version)

    architecture = _one_or_mixed(compile_artifact_df, COL_ARCHITECTURE)
    architecture_flag = _one_or_mixed(compile_artifact_df, COL_ARCHITECTURE_FLAG)
    architecture_label = _architecture_label(architecture, architecture_flag)

    compiler_parts = [part for part in (executable, version) if part]
    compiler_label = " ".join(compiler_parts) if compiler_parts else "unknown compiler"

    if architecture_label:
        return f"{compiler_label} / {architecture_label}"
    return compiler_label


def infer_run_context(compile_artifact_df: pd.DataFrame) -> pd.DataFrame:
    """Return compiler/architecture metadata used to explain a run label."""

    context_cols = [
        COL_COMPILER_EXECUTABLE,
        COL_COMPILER_VERSION,
        COL_ARCHITECTURE,
        COL_ARCHITECTURE_FLAG,
        COL_CPP_CASE,
        COL_PHASE,
        COL_STAGE,
        COL_VARIANT,
    ]

    run_context = compile_artifact_df.reindex(columns=context_cols)

    rows = []
    for col in context_cols:
        values = _unique_nonempty_strings(run_context[col])
        rows.append(
            {
                COL_METRIC: col,
                COL_VALUE: _format_value_list(values),
                "Unique Values": len(values),
            }
        )
    return pd.DataFrame(rows)


def available_reference_keys(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return {
        str(value)
        for value in df[COL_REFERENCE_KEY].dropna().unique()
        if str(value)
    }


def resolve_reference_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    selected_key: str = DEFAULT_SELECTED_REFERENCE_KEY,
    fallback_key: str = DEFAULT_REFERENCE_FALLBACK_KEY,
) -> str:
    """Resolve the reference key used for a two-summary comparison."""

    left_keys = available_reference_keys(left)
    right_keys = available_reference_keys(right)
    common_keys = left_keys & right_keys

    if selected_key in common_keys:
        return selected_key
    if fallback_key in common_keys:
        return fallback_key
    if common_keys:
        return sorted(common_keys)[0]
    return selected_key


def filter_reference(df: pd.DataFrame, reference_key: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    keys = df[COL_REFERENCE_KEY].fillna("").astype(str)
    return df[(keys == "") | (keys == reference_key)].copy()


def add_parity_pressure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if COL_PARITY_PRESSURE not in out.columns:
        out[COL_PARITY_PRESSURE] = np.nan
    else:
        out[COL_PARITY_PRESSURE] = pd.to_numeric(
            out[COL_PARITY_PRESSURE],
            errors="coerce",
        )

    if COL_WORST_CHECK not in out.columns:
        out[COL_WORST_CHECK] = ""
    else:
        out[COL_WORST_CHECK] = out[COL_WORST_CHECK].fillna("").astype(str)

    phase = out[COL_PHASE].fillna("").astype(str)

    gmm_mask = phase == "GaussianMixture EM"
    if gmm_mask.any():
        _assign_parity_pressure(
            out,
            gmm_mask,
            {
                "lower_bound": (
                    "Lower Bound Diff Abs",
                    "Lower Bound Diff Abs Threshold",
                ),
                "weights": (
                    "Weights Max Abs Diff",
                    "Weights Max Abs Diff Threshold",
                ),
                "means": (
                    "Means Max Abs Diff",
                    "Means Max Abs Diff Threshold",
                ),
                "covariances": (
                    "Covariances Max Rel Diff",
                    "Covariances Max Rel Diff Threshold",
                ),
            },
            absolute_checks={
                "algorithm_iterations": "Algorithm Iteration Diff Abs",
            },
        )

    lloyd_mask = phase == "Lloyd Algorithm"
    if lloyd_mask.any():
        _assign_parity_pressure(
            out,
            lloyd_mask,
            {
                "inertia": (
                    "Diff (%)",
                    "Inertia Diff Threshold (%)",
                ),
            },
            absolute_checks={
                "algorithm_iterations": "Algorithm Iteration Diff Abs",
            },
        )

    return out


def _assign_parity_pressure(
    df: pd.DataFrame,
    mask: pd.Series,
    ratio_checks: Mapping[str, tuple[str, str]],
    *,
    absolute_checks: Mapping[str, str] | None = None,
) -> None:
    index = df.index[mask]

    checks: dict[str, pd.Series] = {}
    for check_name, (value_col, threshold_col) in ratio_checks.items():
        checks[check_name] = _parity_pressure_ratio(
            df.loc[index, value_col],
            df.loc[index, threshold_col],
        )

    for check_name, value_col in (absolute_checks or {}).items():
        checks[check_name] = pd.to_numeric(
            df.loc[index, value_col],
            errors="coerce",
        )

    ratios = pd.DataFrame(checks, index=index)
    valid = ratios.notna().any(axis=1)
    if not valid.any():
        return

    valid_ratios = ratios.loc[valid]
    df.loc[valid_ratios.index, COL_PARITY_PRESSURE] = valid_ratios.max(axis=1)
    df.loc[valid_ratios.index, COL_WORST_CHECK] = valid_ratios.idxmax(axis=1)


def _parity_pressure_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    values = np.where(
        denominator == 0,
        np.where(numerator == 0, 0.0, np.inf),
        numerator / denominator,
    )
    return pd.Series(values, index=numerator.index, dtype="float64")


def compare_summary_compatibility(
    baseline: SummaryBundle,
    candidate: SummaryBundle,
    *,
    selected_reference_key: str = DEFAULT_SELECTED_REFERENCE_KEY,
    fallback_reference_key: str = DEFAULT_REFERENCE_FALLBACK_KEY,
) -> dict[str, pd.DataFrame]:
    """Build compact compatibility/context tables for two summaries."""

    effective_reference_key = resolve_reference_key(
        baseline.speedups,
        candidate.speedups,
        selected_key=selected_reference_key,
        fallback_key=fallback_reference_key,
    )

    baseline_speedups = filter_reference(baseline.speedups, effective_reference_key)
    candidate_speedups = filter_reference(candidate.speedups, effective_reference_key)
    speedup_points = compare_point_sets(
        baseline_speedups,
        candidate_speedups,
        SPEEDUP_MATCH_COLS,
    )
    cachegrind_points = compare_point_sets(
        baseline.cachegrind,
        candidate.cachegrind,
        CACHEGRIND_MATCH_COLS,
    )
    spill_points = compare_point_sets(
        baseline.spills,
        candidate.spills,
        SPILL_MATCH_COLS,
    )
    parity_points = compare_point_sets(
        filter_reference(baseline.parity, effective_reference_key),
        filter_reference(candidate.parity, effective_reference_key),
        PARITY_MATCH_COLS,
    )

    point_counts = pd.DataFrame(
        [
            _point_count_row("Speedup points", speedup_points),
            _point_count_row("Cachegrind points", cachegrind_points),
            _point_count_row("Parity points", parity_points),
            _point_count_row("Spill-detection points", spill_points),
        ]
    )

    return {
        "overview": _overview_table(baseline, candidate),
        "reference_resolution": pd.DataFrame(
            [
                {
                    COL_REQUESTED_REFERENCE_KEY: selected_reference_key,
                    COL_FALLBACK_REFERENCE_KEY: fallback_reference_key,
                    COL_SELECTED_REFERENCE_KEY: effective_reference_key,
                    f"{COL_BASELINE} Reference Keys": _format_value_list(
                        sorted(available_reference_keys(baseline.speedups))
                    ),
                    f"{COL_CANDIDATE} Reference Keys": _format_value_list(
                        sorted(available_reference_keys(candidate.speedups))
                    ),
                }
            ]
        ),
        "run_context": _run_context_table(baseline, candidate),
        "point_counts": point_counts,
        "missing_hierarchy": make_all_missing_hierarchy_summary(
            baseline,
            candidate,
            selected_reference_key=selected_reference_key,
            fallback_reference_key=fallback_reference_key,
        ),
        "baseline_only_speedup_points": speedup_points[COL_BASELINE_ONLY],
        "candidate_only_speedup_points": speedup_points[COL_CANDIDATE_ONLY],
        "baseline_only_cachegrind_points": cachegrind_points[COL_BASELINE_ONLY],
        "candidate_only_cachegrind_points": cachegrind_points[COL_CANDIDATE_ONLY],
        "baseline_only_parity_points": parity_points[COL_BASELINE_ONLY],
        "candidate_only_parity_points": parity_points[COL_CANDIDATE_ONLY],
        "baseline_only_spill_points": spill_points[COL_BASELINE_ONLY],
        "candidate_only_spill_points": spill_points[COL_CANDIDATE_ONLY],
    }


def compare_point_sets(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    key_cols: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Return common and one-sided strict point sets for two dataframes."""

    key_cols = list(key_cols)
    baseline_keys = _distinct_keys(baseline_df, key_cols)
    candidate_keys = _distinct_keys(candidate_df, key_cols)

    common = baseline_keys.merge(candidate_keys, on=key_cols, how="inner")
    baseline_only = baseline_keys.merge(
        candidate_keys,
        on=key_cols,
        how="left",
        indicator=True,
    )
    baseline_only = baseline_only[baseline_only["_merge"] == "left_only"].drop(columns="_merge")

    candidate_only = candidate_keys.merge(
        baseline_keys,
        on=key_cols,
        how="left",
        indicator=True,
    )
    candidate_only = candidate_only[candidate_only["_merge"] == "left_only"].drop(columns="_merge")

    return {
        COL_COMMON: _sort_if_possible(common, key_cols),
        COL_BASELINE_ONLY: _sort_if_possible(baseline_only, key_cols),
        COL_CANDIDATE_ONLY: _sort_if_possible(candidate_only, key_cols),
    }


def make_speedup_comparison(
    baseline_speedups: pd.DataFrame,
    candidate_speedups: pd.DataFrame,
    *,
    selected_reference_key: str = DEFAULT_SELECTED_REFERENCE_KEY,
    fallback_reference_key: str = DEFAULT_REFERENCE_FALLBACK_KEY,
) -> pd.DataFrame:
    """Compare Python/C++ speedups for exact matched benchmark points."""

    effective_reference_key = resolve_reference_key(
        baseline_speedups,
        candidate_speedups,
        selected_key=selected_reference_key,
        fallback_key=fallback_reference_key,
    )
    baseline = filter_reference(baseline_speedups, effective_reference_key)
    candidate = filter_reference(candidate_speedups, effective_reference_key)

    if baseline.empty or candidate.empty:
        return pd.DataFrame(columns=SPEEDUP_MATCH_COLS)

    baseline = _prepare_for_join(baseline, SPEEDUP_MATCH_COLS)
    candidate = _prepare_for_join(candidate, SPEEDUP_MATCH_COLS)

    value_cols = [
        COL_REFERENCE,
        COL_TIME_FIELD,
        COL_SPEEDUP_STATISTIC,
        COL_SPEEDUP,
        COL_SPEEDUP_CI_LOW,
        COL_SPEEDUP_CI_HIGH,
        COL_CPP_POINT,
        COL_PY_POINT,
    ]
    baseline_cols = SPEEDUP_MATCH_COLS + value_cols
    candidate_cols = SPEEDUP_MATCH_COLS + value_cols

    merged = baseline[baseline_cols].merge(
        candidate[candidate_cols],
        on=SPEEDUP_MATCH_COLS,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )

    merged[COL_SPEEDUP_BASELINE] = pd.to_numeric(
        merged[f"{COL_SPEEDUP}_baseline"],
        errors="coerce",
    )
    merged[COL_SPEEDUP_CANDIDATE] = pd.to_numeric(
        merged[f"{COL_SPEEDUP}_candidate"],
        errors="coerce",
    )
    merged[COL_SPEEDUP_RATIO] = _safe_ratio(
        merged[COL_SPEEDUP_CANDIDATE],
        merged[COL_SPEEDUP_BASELINE],
    )
    merged[COL_SPEEDUP_DELTA_PCT] = 100.0 * (merged[COL_SPEEDUP_RATIO] - 1.0)

    return _sort_if_possible(merged, SPEEDUP_MATCH_COLS).reset_index(drop=True)


def speedup_comparison_display_columns(df: pd.DataFrame) -> list[str]:
    """Return the preferred strict-speedup comparison columns."""

    return list(SPEEDUP_COMPARISON_DISPLAY_COLS)


def summarize_speedup_comparison(
    speedup_comparison: pd.DataFrame,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Summarize exact-match speedup deltas for a grouping."""

    group_cols = list(group_cols)
    output_cols = group_cols + [
        COL_MATCHED_POINTS,
        COL_MEDIAN_SPEEDUP_RATIO,
        COL_MEDIAN_SPEEDUP_DELTA_PCT,
        COL_MEAN_SPEEDUP_DELTA_PCT,
        COL_MIN_SPEEDUP_DELTA_PCT,
        COL_MAX_SPEEDUP_DELTA_PCT,
    ]

    if speedup_comparison.empty:
        return pd.DataFrame(columns=output_cols)

    df = speedup_comparison.copy()
    df[COL_SPEEDUP_RATIO] = pd.to_numeric(df[COL_SPEEDUP_RATIO], errors="coerce")
    df[COL_SPEEDUP_DELTA_PCT] = pd.to_numeric(
        df[COL_SPEEDUP_DELTA_PCT],
        errors="coerce",
    )

    summary = (
        df.groupby(group_cols, observed=True, dropna=False)
        .agg(
            **{
                COL_MATCHED_POINTS: (COL_SPEEDUP_DELTA_PCT, "size"),
                COL_MEDIAN_SPEEDUP_RATIO: (COL_SPEEDUP_RATIO, "median"),
                COL_MEDIAN_SPEEDUP_DELTA_PCT: (COL_SPEEDUP_DELTA_PCT, "median"),
                COL_MEAN_SPEEDUP_DELTA_PCT: (COL_SPEEDUP_DELTA_PCT, "mean"),
                COL_MIN_SPEEDUP_DELTA_PCT: (COL_SPEEDUP_DELTA_PCT, "min"),
                COL_MAX_SPEEDUP_DELTA_PCT: (COL_SPEEDUP_DELTA_PCT, "max"),
            }
        )
        .reset_index()
    )
    return _sort_if_possible(summary, group_cols).reset_index(drop=True)


def summarize_speedup_comparison_default_groups(
    speedup_comparison: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the standard notebook speedup-delta summary tables."""

    return {
        name: summarize_speedup_comparison(speedup_comparison, group_cols)
        for name, group_cols in SPEEDUP_SUMMARY_GROUPS.items()
    }

def plot_clustered_delta_heatmaps(
    comparison_df: pd.DataFrame,
    *,
    value_col: str,
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
    group_cols: Iterable[str],
    title_prefix: str,
    cbar_label: str,
    positive_meaning: str,
    cmap: str = "coolwarm",
    title: str | None = None,
):
    if comparison_df.empty:
        return []

    plot_df = _prepare_clustered_delta_heatmap_data(
        comparison_df,
        value_col=value_col,
        group_cols=group_cols,
    )
    if plot_df.empty:
        return []

    figures = []

    for phase_label, df_phase in _iter_comparison_phase_stage_data(plot_df):
        phase_max_abs = _symmetric_abs_max(df_phase[value_col])
        norm = mcolors.TwoSlopeNorm(
            vmin=-phase_max_abs,
            vcenter=0.0,
            vmax=phase_max_abs,
        )

        for (variant, params), df_variant in df_phase.groupby(
            [COL_VARIANT, COL_PARAMS],
            observed=True,
            sort=False,
        ):
            clusters = _ordered_numeric_values(df_variant[COL_CLUSTERS])
            if not clusters:
                continue

            figure_title = title or _clustered_delta_heatmap_title(
                title_prefix=title_prefix,
                candidate_label=candidate_label,
                baseline_label=baseline_label,
                positive_meaning=positive_meaning,
                phase_label=phase_label,
                variant=variant,
                params=params,
            )

            fig = plot_clustered_heatmap_grid(
                df_variant,
                clusters=clusters,
                value_col=value_col,
                title=figure_title,
                heatmap_kwargs={"norm": norm},
                cbar_kws={
                    "label": cbar_label,
                    "format": mtick.FormatStrFormatter("%+.0f%%"),
                },
                annot=True,
                fmt="+.1f",
                cmap=cmap,
            )
            figures.append(fig)

    return figures


def _prepare_clustered_delta_heatmap_data(
    comparison_df: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Aggregate duplicate strict rows without changing the plotted grid shape."""

    df = comparison_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    for col in CONFIG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS, value_col])
    if df.empty:
        return df

    grouping = list(dict.fromkeys(
        list(group_cols) + [COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
    ))

    return (
        df.groupby(grouping, observed=True, dropna=False, as_index=False)[value_col]
        .median()
        .sort_values(grouping)
        .reset_index(drop=True)
    )


def _clustered_delta_heatmap_title(
    *,
    title_prefix: str,
    candidate_label: str,
    baseline_label: str,
    positive_meaning: str,
    phase_label: str,
    variant,
    params,
) -> str:
    detail = [f"Phase: {phase_label}"]
    if str(variant):
        detail.append(f"Variant: {variant}")
    if str(params):
        detail.append(f"Params: {params}")

    return (
        f"{title_prefix}\n"
        f"{candidate_label} vs {baseline_label}; {positive_meaning}\n"
        + " | ".join(detail)
    )


def _iter_comparison_phase_stage_data(df: pd.DataFrame):
    phase_values = _ordered_present_values(df, COL_PHASE, PHASE_ORDER)

    for phase in phase_values:
        phase_df = df[df[COL_PHASE].astype(str) == str(phase)].copy()
        if phase_df.empty:
            continue

        stage_values = _ordered_present_values(phase_df, COL_STAGE, STAGE_ORDER)
        for stage in stage_values:
            stage_df = phase_df[phase_df[COL_STAGE].astype(str) == str(stage)].copy()
            stage_label = str(stage)
            label = str(phase) if stage_label == STAGE_ORDER[0] else f"{phase} — {stage_label}"
            yield label, stage_df


def _ordered_present_values(
    df: pd.DataFrame,
    col: str,
    preferred_order: Iterable[str],
) -> list[str]:
    if df.empty:
        return []

    present = list(dict.fromkeys(df[col].dropna().astype(str)))
    preferred = [value for value in preferred_order if value in present]
    extras = [value for value in present if value not in set(preferred)]
    return preferred + extras


def _ordered_numeric_values(series: pd.Series) -> list:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return sorted(values.tolist())


def _symmetric_abs_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    max_abs = float(np.nanmax(np.abs(values)))
    if max_abs == 0.0:
        return 1.0
    return max_abs


def make_cachegrind_comparison(
    baseline_cachegrind: pd.DataFrame,
    candidate_cachegrind: pd.DataFrame,
) -> pd.DataFrame:
    """Compare raw Cachegrind counters for exact matched points."""

    if baseline_cachegrind.empty or candidate_cachegrind.empty:
        return pd.DataFrame(columns=CACHEGRIND_MATCH_COLS)

    baseline = _prepare_for_join(baseline_cachegrind, CACHEGRIND_MATCH_COLS)
    candidate = _prepare_for_join(candidate_cachegrind, CACHEGRIND_MATCH_COLS)
    counters = list(CACHEGRIND_COUNTER_COLS)

    merged = baseline[CACHEGRIND_MATCH_COLS + counters].merge(
        candidate[CACHEGRIND_MATCH_COLS + counters],
        on=CACHEGRIND_MATCH_COLS,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )

    for counter in counters:
        base_col = f"{counter}_baseline"
        candidate_col = f"{counter}_candidate"
        ratio_col = cachegrind_counter_ratio_col(counter)
        delta_col = cachegrind_counter_delta_pct_col(counter)
        merged[ratio_col] = _safe_ratio(merged[candidate_col], merged[base_col])
        merged[delta_col] = 100.0 * (merged[ratio_col] - 1.0)

    return _sort_if_possible(merged, CACHEGRIND_MATCH_COLS).reset_index(drop=True)


def cachegrind_counter_ratio_col(counter: str) -> str:
    return f"{counter} Ratio"


def cachegrind_counter_delta_pct_col(counter: str) -> str:
    return f"{counter} Delta (%)"


def cachegrind_available_counters(cachegrind_comparison: pd.DataFrame) -> list[str]:
    return list(CACHEGRIND_COUNTER_COLS) if not cachegrind_comparison.empty else []


def cachegrind_comparison_display_columns(
    cachegrind_comparison: pd.DataFrame,
    *,
    counters: Iterable[str] = CACHEGRIND_COMPARISON_DISPLAY_COUNTERS,
) -> list[str]:
    columns = list(CACHEGRIND_MATCH_COLS)
    for counter in counters:
        columns.extend(
            [
                f"{counter}_baseline",
                f"{counter}_candidate",
                cachegrind_counter_ratio_col(counter),
                cachegrind_counter_delta_pct_col(counter),
            ]
        )
    return columns


def cachegrind_comparison_long(cachegrind_comparison: pd.DataFrame) -> pd.DataFrame:
    """Return one row per matched point and raw Cachegrind counter."""

    counters = cachegrind_available_counters(cachegrind_comparison)
    rows: list[dict] = []
    if cachegrind_comparison.empty:
        return pd.DataFrame(
            columns=CACHEGRIND_MATCH_COLS
            + [
                COL_COUNTER,
                COL_COUNTER_BASELINE,
                COL_COUNTER_CANDIDATE,
                COL_COUNTER_RATIO,
                COL_COUNTER_DELTA_PCT,
            ]
        )

    for _, row in cachegrind_comparison.iterrows():
        key_values = {col: row[col] for col in CACHEGRIND_MATCH_COLS}
        for counter in counters:
            rows.append(
                {
                    **key_values,
                    COL_COUNTER: counter,
                    COL_COUNTER_BASELINE: row.get(f"{counter}_baseline"),
                    COL_COUNTER_CANDIDATE: row.get(f"{counter}_candidate"),
                    COL_COUNTER_RATIO: row.get(cachegrind_counter_ratio_col(counter)),
                    COL_COUNTER_DELTA_PCT: row.get(cachegrind_counter_delta_pct_col(counter)),
                }
            )

    return pd.DataFrame(rows)


def summarize_cachegrind_comparison(
    cachegrind_comparison: pd.DataFrame,
    group_cols: Iterable[str] = (COL_PHASE, COL_STAGE, COL_COUNTER),
) -> pd.DataFrame:
    """Summarize raw Cachegrind counter deltas without normalization."""

    long_df = cachegrind_comparison_long(cachegrind_comparison)
    group_cols = list(group_cols)
    output_cols = group_cols + [
        COL_MATCHED_POINTS,
        COL_MEDIAN_COUNTER_DELTA_PCT,
        COL_MEAN_COUNTER_DELTA_PCT,
        COL_MIN_COUNTER_DELTA_PCT,
        COL_MAX_COUNTER_DELTA_PCT,
    ]

    if long_df.empty:
        return pd.DataFrame(columns=output_cols)

    long_df[COL_COUNTER_DELTA_PCT] = pd.to_numeric(
        long_df[COL_COUNTER_DELTA_PCT],
        errors="coerce",
    )
    summary = (
        long_df.groupby(group_cols, observed=True, dropna=False)
        .agg(
            **{
                COL_MATCHED_POINTS: (COL_COUNTER_DELTA_PCT, "size"),
                COL_MEDIAN_COUNTER_DELTA_PCT: (COL_COUNTER_DELTA_PCT, "median"),
                COL_MEAN_COUNTER_DELTA_PCT: (COL_COUNTER_DELTA_PCT, "mean"),
                COL_MIN_COUNTER_DELTA_PCT: (COL_COUNTER_DELTA_PCT, "min"),
                COL_MAX_COUNTER_DELTA_PCT: (COL_COUNTER_DELTA_PCT, "max"),
            }
        )
        .reset_index()
    )
    return _sort_if_possible(summary, group_cols).reset_index(drop=True)


def make_parity_comparison(
    baseline_parity: pd.DataFrame,
    candidate_parity: pd.DataFrame,
    *,
    selected_reference_key: str = DEFAULT_SELECTED_REFERENCE_KEY,
    fallback_reference_key: str = DEFAULT_REFERENCE_FALLBACK_KEY,
) -> pd.DataFrame:
    """Compare parity status and shared numeric parity columns."""

    if baseline_parity.empty or candidate_parity.empty:
        return pd.DataFrame(columns=PARITY_MATCH_COLS)

    effective_reference_key = resolve_reference_key(
        baseline_parity,
        candidate_parity,
        selected_key=selected_reference_key,
        fallback_key=fallback_reference_key,
    )
    baseline = add_parity_pressure(
        _prepare_for_join(
            filter_reference(baseline_parity, effective_reference_key),
            PARITY_MATCH_COLS,
        )
    )
    candidate = add_parity_pressure(
        _prepare_for_join(
            filter_reference(candidate_parity, effective_reference_key),
            PARITY_MATCH_COLS,
        )
    )

    common_value_cols = [
        col
        for col in baseline.columns
        if col in candidate.columns and col not in PARITY_MATCH_COLS
    ]
    merged = baseline[PARITY_MATCH_COLS + common_value_cols].merge(
        candidate[PARITY_MATCH_COLS + common_value_cols],
        on=PARITY_MATCH_COLS,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )

    merged[COL_STATUS_BASELINE] = merged["Status_baseline"].map(_compact_status)
    merged[COL_STATUS_CANDIDATE] = merged["Status_candidate"].map(_compact_status)
    merged[COL_STATUS_TRANSITION] = (
        merged[COL_STATUS_BASELINE] + " → " + merged[COL_STATUS_CANDIDATE]
    )

    numeric_cols = [
        col
        for col in common_value_cols
        if col != COL_PARITY_PRESSURE
        and _is_numeric_series(baseline[col])
        and _is_numeric_series(candidate[col])
    ]
    for col in numeric_cols:
        base_col = f"{col}_baseline"
        candidate_col = f"{col}_candidate"
        merged[f"{col} Delta"] = (
            pd.to_numeric(merged[candidate_col], errors="coerce")
            - pd.to_numeric(merged[base_col], errors="coerce")
        )

    baseline_pressure_col = f"{COL_PARITY_PRESSURE}_baseline"
    candidate_pressure_col = f"{COL_PARITY_PRESSURE}_candidate"
    merged[COL_PARITY_PRESSURE_BASELINE] = pd.to_numeric(
        merged[baseline_pressure_col],
        errors="coerce",
    )
    merged[COL_PARITY_PRESSURE_CANDIDATE] = pd.to_numeric(
        merged[candidate_pressure_col],
        errors="coerce",
    )
    merged[COL_PARITY_PRESSURE_DELTA] = (
        merged[COL_PARITY_PRESSURE_CANDIDATE]
        - merged[COL_PARITY_PRESSURE_BASELINE]
    )
    merged[COL_PARITY_PRESSURE_ABS_DELTA] = merged[COL_PARITY_PRESSURE_DELTA].abs()

    baseline_worst_col = f"{COL_WORST_CHECK}_baseline"
    candidate_worst_col = f"{COL_WORST_CHECK}_candidate"
    merged[COL_WORST_CHECK_BASELINE] = merged[baseline_worst_col]
    merged[COL_WORST_CHECK_CANDIDATE] = merged[candidate_worst_col]

    return _sort_if_possible(merged, PARITY_MATCH_COLS).reset_index(drop=True)


def parity_comparison_display_columns(parity_comparison: pd.DataFrame) -> list[str]:
    columns = PARITY_MATCH_COLS + [
        COL_STATUS_BASELINE,
        COL_STATUS_CANDIDATE,
        COL_STATUS_TRANSITION,
        COL_PARITY_PRESSURE_BASELINE,
        COL_PARITY_PRESSURE_CANDIDATE,
        COL_PARITY_PRESSURE_DELTA,
        COL_PARITY_PRESSURE_ABS_DELTA,
        COL_WORST_CHECK_BASELINE,
        COL_WORST_CHECK_CANDIDATE,
        "Failure Reasons_baseline",
        "Failure Reasons_candidate",
    ]
    delta_cols = [
        col
        for col in parity_numeric_delta_columns(parity_comparison)
        if col not in columns
    ]
    return _existing_columns(parity_comparison, columns + delta_cols)


def parity_numeric_delta_columns(parity_comparison: pd.DataFrame) -> list[str]:
    """Return signed numeric delta columns, excluding sort-only helper columns."""

    return [
        col
        for col in parity_comparison.columns
        if col.endswith(" Delta") and col != COL_PARITY_PRESSURE_ABS_DELTA
    ]


def filter_parity_status_transitions(parity_comparison: pd.DataFrame) -> pd.DataFrame:
    """Keep only matched parity rows whose compact status actually changed."""

    if parity_comparison.empty:
        return parity_comparison.copy()

    baseline_status = parity_comparison[COL_STATUS_BASELINE].astype(str)
    candidate_status = parity_comparison[COL_STATUS_CANDIDATE].astype(str)
    return parity_comparison[baseline_status != candidate_status].copy()


def sort_parity_by_pressure_delta(
    parity_comparison: pd.DataFrame,
    *,
    transitions_only: bool = True,
) -> pd.DataFrame:
    """Sort matched parity rows by the largest parity-pressure change first."""

    if transitions_only:
        df = filter_parity_status_transitions(parity_comparison)
    else:
        df = parity_comparison.copy()

    if df.empty:
        return df

    sort_cols = [
        COL_PARITY_PRESSURE_ABS_DELTA,
        COL_PARITY_PRESSURE_DELTA,
        COL_PHASE,
        COL_STAGE,
        COL_VARIANT,
        COL_PARAMS,
    ]
    ascending = [False, False, True, True, True, True]

    return (
        df.sort_values(sort_cols, ascending=ascending, na_position="last")
        .reset_index(drop=True)
    )


def summarize_parity_transitions(
    parity_comparison: pd.DataFrame,
    group_cols: Iterable[str] = (COL_PHASE, COL_STAGE, COL_STATUS_TRANSITION),
    *,
    include_unchanged: bool = False,
) -> pd.DataFrame:
    df = parity_comparison if include_unchanged else filter_parity_status_transitions(parity_comparison)
    group_cols = list(group_cols)
    if df.empty:
        return pd.DataFrame(columns=group_cols + [COL_TRANSITION_COUNT])

    summary = (
        df.groupby(group_cols, observed=True, dropna=False)
        .size()
        .reset_index(name=COL_TRANSITION_COUNT)
    )
    return _sort_if_possible(summary, group_cols).reset_index(drop=True)


def parity_numeric_delta_long(parity_comparison: pd.DataFrame) -> pd.DataFrame:
    """Return one row per matched parity point and numeric parity delta."""

    delta_cols = parity_numeric_delta_columns(parity_comparison)
    rows: list[dict] = []
    if parity_comparison.empty or not delta_cols:
        return pd.DataFrame(
            columns=PARITY_MATCH_COLS
            + [
                COL_PARITY_NUMERIC_METRIC,
                COL_PARITY_BASELINE_VALUE,
                COL_PARITY_CANDIDATE_VALUE,
                COL_PARITY_DELTA,
            ]
        )

    for _, row in parity_comparison.iterrows():
        key_values = {col: row[col] for col in PARITY_MATCH_COLS}
        for delta_col in delta_cols:
            metric = delta_col[: -len(" Delta")]
            baseline_col = f"{metric}_baseline"
            candidate_col = f"{metric}_candidate"
            rows.append(
                {
                    **key_values,
                    COL_PARITY_NUMERIC_METRIC: metric,
                    COL_PARITY_BASELINE_VALUE: row.get(baseline_col),
                    COL_PARITY_CANDIDATE_VALUE: row.get(candidate_col),
                    COL_PARITY_DELTA: row.get(delta_col),
                }
            )
    return pd.DataFrame(rows)


PARITY_DELTA_METRIC_EXCLUDE_PATTERNS = (r"Threshold", r"Py", r"C\+\+",)

def filter_relevant_parity_numeric_deltas(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    metric = long_df[COL_PARITY_NUMERIC_METRIC].fillna("").astype(str)

    mask = pd.Series(True, index=long_df.index)
    for pattern in PARITY_DELTA_METRIC_EXCLUDE_PATTERNS:
        mask &= ~metric.str.contains(pattern, regex=True)

    return long_df[mask].copy()


def summarize_parity_numeric_deltas(
    parity_comparison: pd.DataFrame,
    group_cols: Iterable[str] = (COL_PHASE, COL_STAGE, COL_PARITY_NUMERIC_METRIC),
) -> pd.DataFrame:
    long_df = parity_numeric_delta_long(parity_comparison)
    long_df = filter_relevant_parity_numeric_deltas(long_df)
    long_df[COL_PARITY_DELTA] = pd.to_numeric(
        long_df[COL_PARITY_DELTA],
        errors="coerce",
    )
    long_df = long_df.dropna(subset=[COL_PARITY_DELTA])
    group_cols = list(group_cols)
    output_cols = group_cols + [
        COL_MATCHED_POINTS,
        "Median Delta",
        "Mean Delta",
        "Min Delta",
        "Max Delta",
    ]

    if long_df.empty:
        return pd.DataFrame(columns=output_cols)

    summary = (
        long_df.groupby(group_cols, observed=True, dropna=False)
        .agg(
            **{
                COL_MATCHED_POINTS: (COL_PARITY_DELTA, "size"),
                "Median Delta": (COL_PARITY_DELTA, "median"),
                "Mean Delta": (COL_PARITY_DELTA, "mean"),
                "Min Delta": (COL_PARITY_DELTA, "min"),
                "Max Delta": (COL_PARITY_DELTA, "max"),
            }
        )
        .reset_index()
    )
    return _sort_if_possible(summary, group_cols).reset_index(drop=True)


def make_spill_comparison(
    baseline_spills: pd.DataFrame,
    candidate_spills: pd.DataFrame,
) -> pd.DataFrame:
    """Compare spill-detector reload-pair counts for exact matched points."""

    baseline = _prepare_for_join(baseline_spills, SPILL_MATCH_COLS)
    candidate = _prepare_for_join(candidate_spills, SPILL_MATCH_COLS)
    value_cols = ["Candidate Reload Pairs"]
    for df in (baseline, candidate):
        if df.empty:
            df[value_cols[0]] = pd.Series(dtype="Int64")

    merged = baseline[SPILL_MATCH_COLS + value_cols].merge(
        candidate[SPILL_MATCH_COLS + value_cols],
        on=SPILL_MATCH_COLS,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
    )

    merged["Reload Pair Delta"] = (
        merged["Candidate Reload Pairs_candidate"].fillna(0)
        - merged["Candidate Reload Pairs_baseline"].fillna(0)
    )

    return _sort_if_possible(merged, SPILL_MATCH_COLS).reset_index(drop=True)


def _prefix_sets(df: pd.DataFrame, key_cols: list[str]) -> dict[int, set[tuple]]:
    return {
        depth: {
            _row_tuple(row, key_cols[:depth])
            for _, row in df.iterrows()
        }
        for depth in range(1, len(key_cols) + 1)
    }


def _row_tuple(row: pd.Series, cols: list[str]) -> tuple:
    return tuple(_normalize_key_value(row.get(col, "")) for col in cols)


def _normalize_key_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _format_path(cols: Iterable[str], values: Iterable) -> str:
    return " / ".join(
        f"{col}={_format_scalar(value)}"
        for col, value in zip(cols, values)
    )


def _format_scalar(value) -> str:
    if value == "":
        return "<empty>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _grid_summary(df: pd.DataFrame) -> str:
    parts = []
    for col in CONFIG_COLS:
        if col not in df.columns:
            continue
        values = _ordered_unique_values(df[col])
        if values:
            parts.append(f"{col}={_format_value_list([_format_scalar(v) for v in values])}")
    return "; ".join(parts) if parts else "-"


def _ordered_unique_values(series: pd.Series) -> list:
    values = []
    for value in series.dropna().tolist():
        value = _normalize_key_value(value)
        if value not in values:
            values.append(value)

    def sort_key(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return (0, float(value))
        return (1, str(value))

    return sorted(values, key=sort_key)


def _overview_table(baseline: SummaryBundle, candidate: SummaryBundle) -> pd.DataFrame:
    rows = [
        _overview_row("Run label", baseline.label, candidate.label),
        _overview_row("Summary path", str(baseline.path), str(candidate.path)),
        _overview_row("Configurations", _config_count(baseline.raw), _config_count(candidate.raw)),
        _overview_row("Timing rows", len(baseline.benchmark_data), len(candidate.benchmark_data)),
        _overview_row("Speedup rows", len(baseline.speedups), len(candidate.speedups)),
        _overview_row("Compile artifact rows", len(baseline.compile_artifacts), len(candidate.compile_artifacts)),
        _overview_row("Cachegrind rows", len(baseline.cachegrind), len(candidate.cachegrind)),
        _overview_row("Parity rows", len(baseline.parity), len(candidate.parity)),
        _overview_row("Spill-detection rows", len(baseline.spills), len(candidate.spills)),
    ]
    return pd.DataFrame(rows)


def _overview_row(metric: str, baseline_value, candidate_value) -> dict:
    return {
        COL_METRIC: metric,
        COL_BASELINE: baseline_value,
        COL_CANDIDATE: candidate_value,
    }


def _run_context_table(baseline: SummaryBundle, candidate: SummaryBundle) -> pd.DataFrame:
    left = infer_run_context(baseline.compile_artifacts).rename(
        columns={COL_VALUE: COL_BASELINE, "Unique Values": f"{COL_BASELINE} Unique Values"}
    )
    right = infer_run_context(candidate.compile_artifacts).rename(
        columns={COL_VALUE: COL_CANDIDATE, "Unique Values": f"{COL_CANDIDATE} Unique Values"}
    )
    return left.merge(right, on=COL_METRIC, how="outer")


def _point_count_row(entity: str, point_sets: Mapping[str, pd.DataFrame]) -> dict:
    return {
        COL_ENTITY: entity,
        COL_COMMON: len(point_sets[COL_COMMON]),
        COL_BASELINE_ONLY: len(point_sets[COL_BASELINE_ONLY]),
        COL_CANDIDATE_ONLY: len(point_sets[COL_CANDIDATE_ONLY]),
    }


def _config_count(summary: dict) -> int:
    return len(summary.get("configs", []))


def _ensure_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    if COL_PHASE not in result.columns:
        result.insert(0, COL_PHASE, phase)
    elif result[COL_PHASE].isna().all():
        result[COL_PHASE] = phase
    return result


def _compact_executable(value: str | None) -> str | None:
    if not value or value.startswith("mixed("):
        return value
    return Path(value).name


def _compact_compiler_version(value: str | None) -> str | None:
    if not value or value.startswith("mixed("):
        return value

    first_line = next((line.strip() for line in str(value).splitlines() if line.strip()), "")
    if not first_line:
        return None

    version_match = re.search(r"\b\d+(?:\.\d+){1,3}\b", first_line)
    if version_match:
        return version_match.group(0)

    return first_line[:80]


def _architecture_label(
    architecture: str | None,
    architecture_flag: str | None,
) -> str | None:
    if architecture and architecture_flag and architecture != architecture_flag:
        return f"{architecture} ({architecture_flag})"
    return architecture or architecture_flag


def _one_or_mixed(df: pd.DataFrame, col: str) -> str | None:
    values = _unique_nonempty_strings(df[col])
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return f"mixed({', '.join(values[:3])}{'…' if len(values) > 3 else ''})"


def _unique_nonempty_strings(series: pd.Series) -> list[str]:
    values = []
    for value in series.dropna().unique():
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return values


def _format_value_list(values: Iterable[str], *, max_values: int = 12) -> str:
    values = list(values)
    if not values:
        return "-"
    displayed = values[:max_values]
    suffix = "" if len(values) <= max_values else f", … (+{len(values) - max_values})"
    return ", ".join(displayed) + suffix



def _distinct_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=key_cols)
    prepared = _prepare_for_join(df, key_cols)
    return prepared[key_cols].drop_duplicates().reset_index(drop=True)


def _prepare_for_join(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in key_cols:
        if col not in result.columns:
            if not result.empty:
                raise KeyError(f"missing comparison key column: {col}")
            result[col] = pd.NA
        if col in CONFIG_COLS:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
        else:
            result[col] = result[col].astype("object").where(result[col].notna(), "")
            result[col] = result[col].astype(str)
    return result


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _sort_if_possible(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = list(cols)
    if df.empty:
        return df.reset_index(drop=True)
    return df.sort_values(cols).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.divide(denominator.replace({0: np.nan}))


def _compact_status(value) -> str:
    text = str(value)
    if "PASS" in text:
        return "PASS"
    if "FAIL" in text:
        return "FAIL"
    return text


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().any()

COL_STRUCTURAL_SUMMARY = "Structural Summary"
COL_MISSING_PATTERN = "Missing Pattern"


def make_missing_hierarchy_summary(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    key_cols: Iterable[str],
    *,
    point_set: str,
    baseline_direction: str = "Baseline only",
    candidate_direction: str = "Candidate only",
) -> pd.DataFrame:
    """Summarize one-sided strict-match points without dumping leaf keys.

    Missing structural branches still appear at their highest absent level. For
    example, a missing phase is one row. Missing grid slices are additionally
    compacted as cartesian products, so a slice such as ``N=800000`` and ``K=10``
    across every GMM variant/params/D value becomes one row instead of one row per
    dimension.
    """

    key_cols = list(key_cols)

    baseline_keys = _distinct_keys(baseline_df, key_cols)
    candidate_keys = _distinct_keys(candidate_df, key_cols)
    point_sets = compare_point_sets(baseline_df, candidate_df, key_cols)

    rows = []
    rows.extend(
        _compact_one_sided_missing_rows(
            point_set=point_set,
            direction=baseline_direction,
            missing_keys=point_sets[COL_BASELINE_ONLY],
            opposite_keys=candidate_keys,
            key_cols=key_cols,
        )
    )
    rows.extend(
        _compact_one_sided_missing_rows(
            point_set=point_set,
            direction=candidate_direction,
            missing_keys=point_sets[COL_CANDIDATE_ONLY],
            opposite_keys=baseline_keys,
            key_cols=key_cols,
        )
    )

    if not rows:
        return _empty_missing_hierarchy_frame()

    result = pd.DataFrame(rows)
    return _sort_if_possible(
        result,
        [COL_POINT_SET, COL_DIRECTION, COL_MISSING_DEPTH, COL_PATH, COL_MISSING_PATTERN],
    ).reset_index(drop=True)


def make_all_missing_hierarchy_summary(
    baseline: SummaryBundle,
    candidate: SummaryBundle,
    *,
    selected_reference_key: str = DEFAULT_SELECTED_REFERENCE_KEY,
    fallback_reference_key: str = DEFAULT_REFERENCE_FALLBACK_KEY,
) -> pd.DataFrame:
    """Build compact missing-point summaries for notebook compatibility output."""

    effective_reference_key = resolve_reference_key(
        baseline.speedups,
        candidate.speedups,
        selected_key=selected_reference_key,
        fallback_key=fallback_reference_key,
    )
    baseline_speedups = filter_reference(baseline.speedups, effective_reference_key)
    candidate_speedups = filter_reference(candidate.speedups, effective_reference_key)
    baseline_parity = filter_reference(baseline.parity, effective_reference_key)
    candidate_parity = filter_reference(candidate.parity, effective_reference_key)

    speedup_summary = make_missing_hierarchy_summary(
        baseline_speedups,
        candidate_speedups,
        SPEEDUP_MATCH_COLS,
        point_set="Speedup points",
    )
    cachegrind_summary = make_missing_hierarchy_summary(
        baseline.cachegrind,
        candidate.cachegrind,
        CACHEGRIND_MATCH_COLS,
        point_set="Cachegrind points",
    )
    parity_summary = make_missing_hierarchy_summary(
        baseline_parity,
        candidate_parity,
        PARITY_MATCH_COLS,
        point_set="Parity points",
    )
    parity_summary = _omit_missing_rows_already_reported(
        parity_summary,
        speedup_summary,
        source_point_set="Parity points",
        reference_point_set="Speedup points",
    )
    spill_summary = make_missing_hierarchy_summary(
        baseline.spills,
        candidate.spills,
        SPILL_MATCH_COLS,
        point_set="Spill-detection points",
    )

    frames = [
        speedup_summary,
        cachegrind_summary,
        parity_summary,
        spill_summary,
    ]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return _empty_missing_hierarchy_frame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _compact_one_sided_missing_rows(
    *,
    point_set: str,
    direction: str,
    missing_keys: pd.DataFrame,
    opposite_keys: pd.DataFrame,
    key_cols: list[str],
) -> list[dict]:
    if missing_keys.empty:
        return []

    missing = _distinct_keys(missing_keys, key_cols)
    opposite = _distinct_keys(opposite_keys, key_cols)
    prefix_sets = _prefix_sets(opposite, key_cols)
    grid_start = _first_config_col_index(key_cols)

    structural_groups: dict[tuple[int, tuple], list[tuple]] = {}
    residual_grid_keys: list[tuple] = []

    for _, row in missing.iterrows():
        full_key = _row_tuple(row, key_cols)
        missing_depth = len(key_cols)
        for depth in range(1, len(key_cols) + 1):
            if full_key[:depth] not in prefix_sets[depth]:
                missing_depth = depth
                break

        # If the first absent prefix is before the benchmark grid, report the
        # hierarchy branch directly: phase/stage/variant/params/reference/case.
        # Once the absence starts at D/N/K, defer it to cartesian grid compaction.
        if missing_depth < grid_start:
            structural_groups.setdefault(
                (missing_depth, full_key[:missing_depth]),
                [],
            ).append(full_key)
        else:
            residual_grid_keys.append(full_key)

    rows: list[dict] = []
    for (missing_depth, path_values), full_keys in structural_groups.items():
        group_df = pd.DataFrame(full_keys, columns=key_cols)
        rows.append(
            _missing_summary_row(
                point_set=point_set,
                direction=direction,
                key_cols=key_cols,
                path_cols=key_cols[:missing_depth],
                path_values=path_values,
                group_df=group_df,
                missing_level=key_cols[missing_depth - 1],
                missing_depth=missing_depth,
                missing_pattern="absent hierarchy branch",
            )
        )

    if residual_grid_keys:
        residual_df = pd.DataFrame(residual_grid_keys, columns=key_cols)
        rows.extend(
            _compact_grid_missing_rows(
                point_set=point_set,
                direction=direction,
                residual_df=residual_df,
                key_cols=key_cols,
                grid_start=grid_start,
            )
        )

    return rows


def _compact_grid_missing_rows(
    *,
    point_set: str,
    direction: str,
    residual_df: pd.DataFrame,
    key_cols: list[str],
    grid_start: int,
) -> list[dict]:
    context_cols = _grid_context_cols(key_cols, grid_start)
    if not context_cols:
        context_cols = key_cols[:grid_start]

    rows: list[dict] = []
    group_iter = residual_df.groupby(context_cols, observed=True, dropna=False, sort=False)
    for context_values, context_df in group_iter:
        if not isinstance(context_values, tuple):
            context_values = (context_values,)
        rows.extend(
            _cartesian_missing_rows(
                point_set=point_set,
                direction=direction,
                df=context_df,
                key_cols=key_cols,
                path_cols=context_cols,
                path_values=tuple(_normalize_key_value(v) for v in context_values),
            )
        )
    return rows


def _cartesian_missing_rows(
    *,
    point_set: str,
    direction: str,
    df: pd.DataFrame,
    key_cols: list[str],
    path_cols: list[str],
    path_values: tuple,
) -> list[dict]:
    df = _distinct_keys(df, key_cols)
    residual_cols = [col for col in key_cols if col not in path_cols]

    if df.empty:
        return []

    if _is_cartesian_product(df, residual_cols):
        return [
            _missing_summary_row(
                point_set=point_set,
                direction=direction,
                key_cols=key_cols,
                path_cols=path_cols,
                path_values=path_values,
                group_df=df,
                missing_level=_missing_pattern_level(df, residual_cols),
                missing_depth=len(path_cols),
                missing_pattern="complete cartesian product",
            )
        ]

    split_col = _choose_cartesian_split_column(df, residual_cols)
    if split_col is None:
        return [
            _missing_summary_row(
                point_set=point_set,
                direction=direction,
                key_cols=key_cols,
                path_cols=path_cols,
                path_values=path_values,
                group_df=df,
                missing_level="Grid pattern",
                missing_depth=len(path_cols),
                missing_pattern="partial grid slice",
            )
        ]

    rows: list[dict] = []
    for value, sub_df in df.groupby(split_col, observed=True, dropna=False, sort=False):
        rows.extend(
            _cartesian_missing_rows(
                point_set=point_set,
                direction=direction,
                df=sub_df,
                key_cols=key_cols,
                path_cols=path_cols + [split_col],
                path_values=path_values + (_normalize_key_value(value),),
            )
        )
    return rows


def _missing_summary_row(
    *,
    point_set: str,
    direction: str,
    key_cols: list[str],
    path_cols: list[str],
    path_values: tuple,
    group_df: pd.DataFrame,
    missing_level: str,
    missing_depth: int,
    missing_pattern: str,
) -> dict:
    group_df = _distinct_keys(group_df, key_cols)
    path_cols = list(path_cols)
    path_value_by_col = dict(zip(path_cols, path_values))
    path = _format_path(path_cols, path_values) if path_cols else "<all>"

    return {
        COL_POINT_SET: point_set,
        COL_DIRECTION: direction,
        COL_MISSING_LEVEL: missing_level,
        COL_MISSING_DEPTH: missing_depth,
        COL_PATH: path,
        COL_MISSING_POINTS: len(group_df),
        COL_STRUCTURAL_SUMMARY: _structural_summary(group_df, key_cols, path_value_by_col),
        COL_GRID_SUMMARY: _grid_summary(group_df),
        COL_MISSING_PATTERN: missing_pattern,
        COL_NEXT_DETAIL: _compact_next_detail(group_df, key_cols, path_cols),
    }


def _first_config_col_index(key_cols: list[str]) -> int:
    for i, col in enumerate(key_cols):
        if col in CONFIG_COLS:
            return i
    return len(key_cols)


def _grid_context_cols(key_cols: list[str], grid_start: int) -> list[str]:
    prefix = key_cols[:grid_start]
    if COL_PHASE in prefix and COL_STAGE in prefix:
        return [COL_PHASE, COL_STAGE]
    if COL_PHASE in prefix:
        return [COL_PHASE]
    return prefix[:1]


def _is_cartesian_product(df: pd.DataFrame, cols: list[str]) -> bool:
    if df.empty:
        return False
    if not cols:
        return len(df) <= 1

    distinct = _distinct_keys(df, cols)
    product = 1
    for col in cols:
        unique_count = len(_ordered_unique_values(distinct[col]))
        if unique_count == 0:
            return False
        product *= unique_count
        if product > len(distinct):
            return False
    return product == len(distinct)


def _choose_cartesian_split_column(df: pd.DataFrame, residual_cols: list[str]) -> str | None:
    candidates = []
    for index, col in enumerate(residual_cols):
        values = _ordered_unique_values(df[col])
        if len(values) <= 1:
            continue
        # Keep hierarchy readable by preferring earlier non-grid columns, but do
        # not strongly penalize grid columns because splitting N/K is often the
        # clearest way to describe a partial rectangular slice.
        is_grid = col in CONFIG_COLS
        candidates.append((len(values), int(is_grid), index, col))
    if not candidates:
        return None
    return min(candidates)[-1]


def _missing_pattern_level(df: pd.DataFrame, residual_cols: list[str]) -> str:
    fixed_grid_cols = [
        col
        for col in residual_cols
        if col in CONFIG_COLS and len(_ordered_unique_values(df[col])) == 1
    ]
    if fixed_grid_cols:
        return ", ".join(fixed_grid_cols)

    fixed_structural_cols = [
        col
        for col in residual_cols
        if col not in CONFIG_COLS and len(_ordered_unique_values(df[col])) == 1
    ]
    if fixed_structural_cols:
        return ", ".join(fixed_structural_cols)

    grid_cols = [col for col in residual_cols if col in CONFIG_COLS]
    return ", ".join(grid_cols) if grid_cols else "Pattern"


def _structural_summary(
    df: pd.DataFrame,
    key_cols: list[str],
    path_value_by_col: Mapping[str, object],
) -> str:
    parts = []
    for col in key_cols:
        if col in CONFIG_COLS or col in path_value_by_col:
            continue
        values = _ordered_unique_values(df[col])
        if values:
            parts.append(f"{col}={_format_value_list([_format_scalar(v) for v in values])}")
    return "; ".join(parts) if parts else "-"


def _compact_next_detail(df: pd.DataFrame, key_cols: list[str], path_cols: list[str]) -> str:
    variable_structural = []
    for col in key_cols:
        if col in CONFIG_COLS or col in path_cols:
            continue
        values = _ordered_unique_values(df[col])
        if len(values) > 1:
            variable_structural.append(
                f"{col}: {_format_value_list([_format_scalar(v) for v in values])}"
            )
    if variable_structural:
        return "; ".join(variable_structural)
    return "complete across listed values"


def _empty_missing_hierarchy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_POINT_SET,
            COL_DIRECTION,
            COL_MISSING_LEVEL,
            COL_MISSING_DEPTH,
            COL_PATH,
            COL_MISSING_POINTS,
            COL_STRUCTURAL_SUMMARY,
            COL_GRID_SUMMARY,
            COL_MISSING_PATTERN,
            COL_NEXT_DETAIL,
        ]
    )


def _omit_missing_rows_already_reported(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    source_point_set: str,
    reference_point_set: str,
) -> pd.DataFrame:
    """Suppress missing-structure rows already communicated by another point set.

    Parity strict keys often have the same one-sided structure as speedups. In
    that case, repeating the same hierarchy makes the compatibility section much
    longer without adding information. We keep only parity-specific rows and add a
    single pointer row when every parity row was already covered.
    """

    if source.empty or reference.empty:
        return source

    source = source.copy()
    reference = reference.copy()

    reference_signatures = {
        _missing_summary_signature(row)
        for _, row in reference.iterrows()
    }
    duplicate_mask = source.apply(
        lambda row: _missing_summary_signature(row) in reference_signatures,
        axis=1,
    )
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        return source

    remaining = source.loc[~duplicate_mask].copy()
    same_as_row = pd.DataFrame(
        [
            {
                COL_POINT_SET: source_point_set,
                COL_DIRECTION: f"Same as {reference_point_set}",
                COL_MISSING_LEVEL: "Same as speedup",
                COL_MISSING_DEPTH: 0,
                COL_PATH: f"same as {reference_point_set}",
                COL_MISSING_POINTS: int(source.loc[duplicate_mask, COL_MISSING_POINTS].sum()),
                COL_STRUCTURAL_SUMMARY: "-",
                COL_GRID_SUMMARY: "No parity-only missing structure in the omitted rows",
                COL_MISSING_PATTERN: "duplicate structure omitted",
                COL_NEXT_DETAIL: f"{duplicate_count} row(s) already reported under {reference_point_set}",
            }
        ]
    )

    if remaining.empty:
        return same_as_row

    return pd.concat([same_as_row, remaining], ignore_index=True, sort=False)


def _missing_summary_signature(row: pd.Series) -> tuple:
    signature_cols = [
        COL_DIRECTION,
        COL_MISSING_LEVEL,
        COL_PATH,
        COL_MISSING_POINTS,
        COL_STRUCTURAL_SUMMARY,
        COL_GRID_SUMMARY,
        COL_MISSING_PATTERN,
        COL_NEXT_DETAIL,
    ]
    return tuple(str(row.get(col, "")) for col in signature_cols)


def display_wrapped_dataframe(
    df: pd.DataFrame,
    *,
    max_col_width_px: int = 420,
) -> pd.io.formats.style.Styler:
    """Return a Styler that makes list-heavy DataFrames readable in notebooks."""

    def _format_cell(value: object) -> object:
        if isinstance(value, (list, tuple, set, frozenset)):
            values = list(value)
            if isinstance(value, (set, frozenset)):
                values = sorted(values, key=str)

            if not values:
                return "-"

            return "\n".join(f"• {item}" for item in values)

        return value

    formatted = df.map(_format_cell)

    return formatted.style.set_properties(
        **{
            "white-space": "pre-wrap",
            "text-align": "left",
            "vertical-align": "top",
            "max-width": f"{max_col_width_px}px",
        }
    )