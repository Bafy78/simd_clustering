import pandas as pd
import numpy as np

from benchmark_metadata import (
    PHASE_DISPLAY_NAMES,
    PRIMARY_REFERENCE_KEY_BY_PHASE,
    REFERENCE_VARIANT,
)
from benchmark_reporting.constants import *


def _is_sequence_filter(value):
    return isinstance(value, (list, tuple, set, pd.Index, np.ndarray))


def filter_bench(
    df,
    *,
    phase=None,
    stage=None,
    language=None,
    variant=None,
    params=None,
    reference=None,
    dimensions=None,
    samples=None,
    clusters=None,
):
    mask = pd.Series(True, index=df.index)

    filters = {
        COL_PHASE: phase,
        COL_STAGE: stage,
        COL_LANGUAGE: language,
        COL_VARIANT: variant,
        COL_PARAMS: params,
        COL_REFERENCE: reference,
        COL_DIMENSIONS: dimensions,
        COL_SAMPLES: samples,
        COL_CLUSTERS: clusters,
    }

    for col, value in filters.items():
        if value is None:
            continue

        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}")

        if _is_sequence_filter(value):
            mask &= df[col].isin(value)
        else:
            mask &= df[col] == value

    return df.loc[mask].copy()


def filter_primary_references(df, *, overrides=None):
    """Keep the primary Python reference for each benchmark phase.

    ``overrides`` accepts benchmark phase keys (for example ``"hdbscan"``)
    or their display names. Rows without a reference key, such as C++ timing
    rows, are retained.
    """
    if df.empty or COL_REFERENCE_KEY not in df.columns:
        return df.copy()
    if COL_PHASE not in df.columns:
        raise KeyError(f"Missing column {COL_PHASE!r}")

    reference_by_display_phase = {
        PHASE_DISPLAY_NAMES[phase_key]: reference_key
        for phase_key, reference_key in PRIMARY_REFERENCE_KEY_BY_PHASE.items()
    }

    for phase, reference_key in (overrides or {}).items():
        display_phase = PHASE_DISPLAY_NAMES.get(str(phase), str(phase))
        if display_phase not in PHASE_DISPLAY_NAMES.values():
            raise KeyError(f"Unknown benchmark phase {phase!r}")
        reference_by_display_phase[display_phase] = str(reference_key)

    expected_reference = (
        df[COL_PHASE]
        .astype("object")
        .map(reference_by_display_phase)
        .fillna(REFERENCE_VARIANT)
        .astype(str)
    )
    actual_reference = df[COL_REFERENCE_KEY].fillna("").astype(str)
    mask = actual_reference.eq("") | actual_reference.eq(expected_reference)

    result = df.loc[mask].copy()
    for col in result.select_dtypes(include="category").columns:
        result[col] = result[col].cat.remove_unused_categories()
    return result.reset_index(drop=True)


def add_time_per_algorithm_iteration_columns(df):
    result = df.copy()

    if "time_per_algorithm_iteration_s_median" not in result.columns:
        raise KeyError(
            "Expected column 'time_per_algorithm_iteration_s_median' from benchmark_summary.json"
        )

    result[COL_TIME_PER_ALGORITHM_ITER] = result[
        "time_per_algorithm_iteration_s_median"
    ]
    result[COL_TIME_PER_ALGORITHM_ITER_MS] = (
        result[COL_TIME_PER_ALGORITHM_ITER] * 1000.0
    )

    return result


def add_total_time_ms_column(df):
    result = df.copy()
    result["Total_Time_ms"] = result[COL_TIME_S] * 1000.0
    return result


def require_speedup_columns(df):
    required = [
        COL_PHASE,
        COL_DIMENSIONS,
        COL_SAMPLES,
        COL_CLUSTERS,
        COL_SPEEDUP,
        COL_SPEEDUP_CI_LOW,
        COL_SPEEDUP_CI_HIGH,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "Expected a post-processed speedup dataframe from "
            f"load_speedup_summary(); missing columns: {missing}"
        )


def add_ci_error_columns(
    df_speedup,
):
    require_speedup_columns(df_speedup)

    result = df_speedup.copy()
    result[COL_SPEEDUP_CI_LOWER_ERROR] = (
        result[COL_SPEEDUP] - result[COL_SPEEDUP_CI_LOW]
    )
    result[COL_SPEEDUP_CI_UPPER_ERROR] = (
        result[COL_SPEEDUP_CI_HIGH] - result[COL_SPEEDUP]
    )
    result[COL_SPEEDUP_ERROR_WIDTH] = (
        result[COL_SPEEDUP_CI_HIGH] - result[COL_SPEEDUP_CI_LOW]
    )

    return result


def add_speedup_retention(
    df_speedup,
    *,
    base_clusters=None,
    group_cols=None,
):
    require_speedup_columns(df_speedup)

    result = df_speedup.copy()

    if result.empty:
        result[COL_BASE_SPEEDUP] = pd.Series(dtype=float)
        result[COL_RETENTION] = pd.Series(dtype=float)
        return result

    if group_cols is None:
        identity_cols = [COL_PHASE]
        if COL_STAGE in result.columns:
            identity_cols.append(COL_STAGE)
        if COL_VARIANT in result.columns:
            identity_cols.append(COL_VARIANT)
        if COL_PARAMS in result.columns:
            identity_cols.append(COL_PARAMS)
        if COL_REFERENCE in result.columns:
            identity_cols.append(COL_REFERENCE)
        group_cols = identity_cols + [COL_DIMENSIONS, COL_SAMPLES]

    if base_clusters is None:
        base_clusters = int(result[COL_CLUSTERS].min())

    baseline = (
        result[result[COL_CLUSTERS] == base_clusters]
        .loc[:, group_cols + [COL_SPEEDUP]]
        .rename(columns={COL_SPEEDUP: COL_BASE_SPEEDUP})
    )

    result = result.merge(
        baseline,
        on=group_cols,
        how="left",
        validate="many_to_one",
    )

    if result[COL_BASE_SPEEDUP].isna().any():
        missing = result[result[COL_BASE_SPEEDUP].isna()]
        raise RuntimeError(
            "Missing baseline speedup rows for some configurations. "
            f"Example rows:\n{missing.head()}"
        )

    result[COL_RETENTION] = result[COL_SPEEDUP] / result[COL_BASE_SPEEDUP] * 100.0

    return result


def add_time_per_algorithm_iteration_per_sample_columns(
    df,
    *,
    statistic: str = "median",
    spread: str = "iqr",
    scale: float = 1000.0,
):
    """
    Add per-algorithm-iteration-per-sample timing columns from benchmark_summary.json stats.

    The default center is the median time_per_algorithm_iteration_s divided by samples.
    The default spread is IQR, i.e. p25 to p75, which matches a median-centered plot.

    Parameters
    ----------
    statistic:
        Center statistic from time_per_algorithm_iteration_s, for example "median" or "mean".

    spread:
        Timing-run spread to show as error bars.

        Supported values:
        - "iqr" or "p25_p75": p25 to p75
        - "p05_p95": p05 to p95
        - "stddev": center ± stddev
        - "mad": center ± MAD

    scale:
        Unit scale. Use 1000.0 to convert seconds to milliseconds.
    """
    result = df.copy()

    prefix = "time_per_algorithm_iteration_s"
    center_col = f"{prefix}_{statistic}"

    if center_col not in result.columns:
        raise KeyError(
            f"Expected column {center_col!r}. "
            "Did load_benchmark_data() copy time_per_algorithm_iteration_s stats?"
        )

    if COL_SAMPLES not in result.columns:
        raise KeyError(f"Missing required column {COL_SAMPLES!r}")

    result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_MS] = (
        result[center_col] / result[COL_SAMPLES] * scale
    )

    if spread in {"iqr", "p25_p75"}:
        low_col = f"{prefix}_p25"
        high_col = f"{prefix}_p75"
        spread_label = "p25–p75"

        _require_columns(result, [low_col, high_col])

        low = result[low_col] / result[COL_SAMPLES] * scale
        high = result[high_col] / result[COL_SAMPLES] * scale

    elif spread == "p05_p95":
        low_col = f"{prefix}_p05"
        high_col = f"{prefix}_p95"
        spread_label = "p05–p95"

        _require_columns(result, [low_col, high_col])

        low = result[low_col] / result[COL_SAMPLES] * scale
        high = result[high_col] / result[COL_SAMPLES] * scale

    elif spread in {"stddev", "mad"}:
        spread_col = f"{prefix}_{spread}"
        spread_label = f"± {spread}"

        _require_columns(result, [spread_col])

        err = result[spread_col] / result[COL_SAMPLES] * scale
        low = result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_MS] - err
        high = result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_MS] + err

    else:
        raise ValueError(
            "Unsupported spread. Expected one of: "
            "'iqr', 'p25_p75', 'p05_p95', 'stddev', 'mad'."
        )

    result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_LOW_MS] = low.clip(lower=0)
    result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_HIGH_MS] = high

    result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_LOWER_ERROR_MS] = (
        result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_MS]
        - result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_LOW_MS]
    )

    result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_UPPER_ERROR_MS] = (
        result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_HIGH_MS]
        - result[COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_MS]
    )

    result[COL_TIMING_RUN_SPREAD] = spread_label

    return result


def _require_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def available_speedup_phases(df_speedup, *, phase_order=PHASE_ORDER):
    """Return ordered phases that are actually present in a speedup dataframe."""
    require_speedup_columns(df_speedup)

    present = set(df_speedup[COL_PHASE].dropna())

    return [phase for phase in phase_order if phase in present]


def prepare_speedup_comparison_data(
    df_speedup,
    *,
    phases=None,
    add_errors=True,
):
    """
    Canonical prep for any plot comparing speedups.

    By default this keeps every phase that has speedup data.
    Pass phases=... only for a deliberately phase-specific plot.
    """
    require_speedup_columns(df_speedup)

    result = df_speedup.copy()

    if add_errors and COL_SPEEDUP_ERROR_WIDTH not in result.columns:
        result = add_ci_error_columns(result)

    if phases is None:
        phases = available_speedup_phases(result)

    result = filter_bench(result, phase=phases)

    result[COL_PHASE] = pd.Categorical(
        result[COL_PHASE],
        categories=phases,
        ordered=True,
    )

    if COL_STAGE in result.columns:
        present_stages = list(dict.fromkeys(result[COL_STAGE].dropna().astype(str)))
        stage_categories = [stage for stage in STAGE_ORDER if stage in present_stages]
        stage_categories.extend(stage for stage in present_stages if stage not in stage_categories)
        result[COL_STAGE] = pd.Categorical(
            result[COL_STAGE],
            categories=stage_categories,
            ordered=True,
        )

    sort_cols = [COL_PHASE]
    if COL_STAGE in result.columns:
        sort_cols.append(COL_STAGE)
    if COL_VARIANT in result.columns:
        sort_cols.append(COL_VARIANT)
    if COL_PARAMS in result.columns:
        sort_cols.append(COL_PARAMS)
    if COL_REFERENCE in result.columns:
        sort_cols.append(COL_REFERENCE)
    sort_cols += [COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]

    return result.sort_values(sort_cols).reset_index(drop=True)


def iter_speedup_phase_data(df_speedup, *, phases=None):
    """
    Yield one dataframe per speedup phase/stage.

    Existing full-stage data keeps the phase-only label. Non-full stages get a
    ``Phase — Stage`` label so report plots do not collapse distinct stages.
    """
    df_plot = prepare_speedup_comparison_data(df_speedup, phases=phases)

    if phases is None:
        phases = available_speedup_phases(df_plot)

    for phase in phases:
        phase_df = filter_bench(df_plot, phase=phase)
        if phase_df.empty:
            continue

        if COL_STAGE not in phase_df.columns:
            yield phase, phase_df
            continue

        stage_values = [
            stage for stage in STAGE_ORDER if stage in set(phase_df[COL_STAGE].dropna().astype(str))
        ]
        stage_values.extend(
            stage
            for stage in phase_df[COL_STAGE].dropna().astype(str).unique()
            if stage not in set(stage_values)
        )

        for stage in stage_values:
            stage_df = filter_bench(phase_df, stage=stage)
            if stage_df.empty:
                continue
            label = phase if stage == STAGE_ORDER[0] else f"{phase} — {stage}"
            yield label, stage_df


def _safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    values = np.where(
        denominator == 0,
        np.where(numerator == 0, 0.0, np.inf),
        numerator / denominator,
    )
    return pd.Series(values, index=numerator.index, dtype="float64")


def _numeric_column(df, column):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _hdbscan_summary_scalar_pressure(
    df,
    *,
    abs_col,
    rel_col,
    abs_threshold_col,
    rel_threshold_col,
):
    """Pressure for HDBSCAN scalar checks that pass on abs OR rel tolerance."""

    pressures = pd.concat(
        [
            _safe_ratio(_numeric_column(df, abs_col), _numeric_column(df, abs_threshold_col)),
            _safe_ratio(_numeric_column(df, rel_col), _numeric_column(df, rel_threshold_col)),
        ],
        axis=1,
    )
    return pressures.min(axis=1, skipna=True)


def _hdbscan_count_pressure(df, cpp_col, reference_col):
    cpp = _numeric_column(df, cpp_col)
    reference = _numeric_column(df, reference_col)
    valid = cpp.notna() & reference.notna()
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    values.loc[valid] = np.where(cpp.loc[valid] == reference.loc[valid], 0.0, 1.0)
    return values


def _algorithm_iteration_pressure(diff_abs):
    return pd.to_numeric(diff_abs, errors="coerce")


def add_gmm_parity_pressure(df):
    out = df.copy()

    ratios = pd.DataFrame(
        {
            "lower_bound": _safe_ratio(
                out["Lower Bound Diff Abs"],
                out["Lower Bound Diff Abs Threshold"],
            ),
            "weights": _safe_ratio(
                out["Weights Max Abs Diff"],
                out["Weights Max Abs Diff Threshold"],
            ),
            "means": _safe_ratio(
                out["Means Max Abs Diff"],
                out["Means Max Abs Diff Threshold"],
            ),
            "covariances": _safe_ratio(
                out["Covariances Max Rel Diff"],
                out["Covariances Max Rel Diff Threshold"],
            ),
            "algorithm_iterations": _algorithm_iteration_pressure(
                out["Algorithm Iteration Diff Abs"],
            ),
        }
    )

    out["Parity Pressure"] = ratios.max(axis=1)
    out["Worst Check"] = ratios.idxmax(axis=1)

    return out


def add_lloyd_parity_pressure(df):
    out = df.copy()

    ratios = pd.DataFrame(
        {
            "inertia": _safe_ratio(
                out["Diff (%)"],
                out["Inertia Diff Threshold (%)"],
            ),
            "algorithm_iterations": _algorithm_iteration_pressure(
                out["Algorithm Iteration Diff Abs"],
            ),
        }
    )

    out["Parity Pressure"] = ratios.max(axis=1)
    out["Worst Check"] = ratios.idxmax(axis=1)

    return out


def add_hdbscan_full_parity_pressure(df):
    """Add a HDBSCAN Full-stage parity pressure column.

    HDBSCAN summary scalar checks are accepted when either the absolute or the
    relative tolerance passes.  The pressure therefore uses the smaller of the
    two ratios for each scalar-summary check.
    """

    out = df.copy()
    if out.empty:
        return out

    if COL_STAGE in out.columns:
        out = out[out[COL_STAGE].astype(str) == "Full"].copy()
    if out.empty:
        out["Parity Pressure"] = pd.Series(dtype="float64")
        out["Worst Check"] = pd.Series(dtype="object")
        return out

    ratios = pd.DataFrame(
        {
            "summary_scalar": _hdbscan_summary_scalar_pressure(
                out,
                abs_col="Summary Scalar Abs Diff Max",
                rel_col="Summary Scalar Rel Diff Max",
                abs_threshold_col="Summary Scalar Abs Diff Threshold",
                rel_threshold_col="Summary Scalar Rel Diff Threshold",
            ),
            "probe_values": _safe_ratio(
                _numeric_column(out, "Probe Value Max Abs Diff"),
                _numeric_column(out, "Probe Value Max Abs Diff Threshold"),
            ),
            "label_summary_scalar": _hdbscan_summary_scalar_pressure(
                out,
                abs_col="Label Summary Scalar Abs Diff Max",
                rel_col="Label Summary Scalar Rel Diff Max",
                abs_threshold_col="Label Summary Scalar Abs Diff Threshold",
                rel_threshold_col="Label Summary Scalar Rel Diff Threshold",
            ),
            "label_probe_values": _safe_ratio(
                _numeric_column(out, "Label Probe Value Max Abs Diff"),
                _numeric_column(out, "Label Probe Value Max Abs Diff Threshold"),
            ),
            "probability_summary_scalar": _hdbscan_summary_scalar_pressure(
                out,
                abs_col="Probability Summary Scalar Abs Diff Max",
                rel_col="Probability Summary Scalar Rel Diff Max",
                abs_threshold_col="Probability Summary Scalar Abs Diff Threshold",
                rel_threshold_col="Probability Summary Scalar Rel Diff Threshold",
            ),
            "probability_probe_values": _safe_ratio(
                _numeric_column(out, "Probability Probe Value Max Abs Diff"),
                _numeric_column(out, "Probability Probe Value Max Abs Diff Threshold"),
            ),
            "noise_count": _hdbscan_count_pressure(
                out,
                "C++ Noise Count",
                "Reference Noise Count",
            ),
            "cluster_count": _hdbscan_count_pressure(
                out,
                "C++ Cluster Count",
                "Reference Cluster Count",
            ),
        },
        index=out.index,
    )

    valid = ratios.notna().any(axis=1)
    out["Parity Pressure"] = np.nan
    out["Worst Check"] = ""
    if valid.any():
        valid_ratios = ratios.loc[valid]
        out.loc[valid_ratios.index, "Parity Pressure"] = valid_ratios.max(axis=1)
        out.loc[valid_ratios.index, "Worst Check"] = valid_ratios.idxmax(axis=1)

    return out
