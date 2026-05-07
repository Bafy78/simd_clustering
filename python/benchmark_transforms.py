import pandas as pd
import numpy as np

from .benchmark_constants import *


def _is_sequence_filter(value):
    return isinstance(value, (list, tuple, set, pd.Index, np.ndarray))

def filter_bench(
    df,
    *,
    phase=None,
    language=None,
    dimensions=None,
    samples=None,
    clusters=None,
):
    """Return a filtered copy of a benchmark dataframe.

    Each argument can be either a single value or a list/set/tuple of values.
    """
    mask = pd.Series(True, index=df.index)

    filters = {
        COL_PHASE: phase,
        COL_LANGUAGE: language,
        COL_DIMENSIONS: dimensions,
        COL_SAMPLES: samples,
        COL_CLUSTERS: clusters,
    }

    for col, value in filters.items():
        if value is None:
            continue

        if _is_sequence_filter(value):
            mask &= df[col].isin(value)
        else:
            mask &= df[col] == value

    return df.loc[mask].copy()


def mean_time_by_config(df, group_cols=None):
    """Compute mean benchmark time grouped by configuration columns."""
    if group_cols is None:
        group_cols = CONFIG_LANGUAGE_COLS

    return (
        df.groupby(group_cols, observed=True)[COL_TIME_S]
        .mean()
        .reset_index()
    )

def add_speedup(
    df,
    *,
    numerator_col=LANG_PY,
    denominator_col=LANG_CPP,
    out_col=COL_SPEEDUP,
):
    """Return a copy of df with a speedup column added.

    Default interpretation:
    speedup = Python time / C++ time
    """
    result = df.copy()

    missing_cols = [
        col for col in [numerator_col, denominator_col]
        if col not in result.columns
    ]
    if missing_cols:
        raise KeyError(f"Missing required columns for speedup: {missing_cols}")

    result[out_col] = result[numerator_col] / result[denominator_col]
    return result


def compute_language_speedup(
    df,
    *,
    phase=None,
    group_cols=None,
):
    """Compute mean Python-vs-C++ speedup for matching benchmark configs."""
    if group_cols is None:
        group_cols = CONFIG_COLS

    group_cols = list(group_cols)

    if phase is not None:
        df = filter_bench(df, phase=phase)

    df_mean = mean_time_by_config(
        df,
        group_cols=group_cols + [COL_LANGUAGE],
    )

    df_pivot = df_mean.pivot(
        index=group_cols,
        columns=COL_LANGUAGE,
        values=COL_TIME_S,
    ).reset_index()

    return add_speedup(df_pivot)


def compute_cpp_speedup_against_python_mean(
    df,
    *,
    group_cols=None,
):
    """Compare each raw C++ run against the mean Python time for the same config.

    This preserves C++ run-level variance, which is useful for Seaborn error bands.
    """
    if group_cols is None:
        group_cols = PHASE_CONFIG_COLS

    group_cols = list(group_cols)

    py_means = mean_time_by_config(
        filter_bench(df, language=LANG_PY),
        group_cols=group_cols,
    ).rename(columns={COL_TIME_S: COL_PY_MEAN_TIME})

    df_cpp = filter_bench(df, language=LANG_CPP)

    result = pd.merge(
        df_cpp,
        py_means,
        on=group_cols,
    )

    result[COL_SPEEDUP] = result[COL_PY_MEAN_TIME] / result[COL_TIME_S]
    return result