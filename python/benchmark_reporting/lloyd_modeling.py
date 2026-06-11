"""Lloyd timing model helpers for benchmark reporting notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

from .constants import *
from .transforms import filter_bench

COL_WORK_SIZE = "Work Size"
COL_MODEL_NAME = "Model"
COL_NUM_PARAMS = "Num Params"
COL_PRED_TIME_PER_ALGORITHM_ITER_MS = "Predicted Time Per Algorithm Iteration (ms)"
COL_PRED_TOTAL_TIME_MS = "Predicted Total Lloyd Time (ms)"
COL_TOTAL_TIME_MS = "Total Lloyd Time (ms)"
COL_ABS_ERROR_PCT = "Abs Error (%)"
COL_LOG_R2 = "Log-space R²"
COL_MEDIAN_ABS_ERROR_PCT = "Median Abs Error (%)"
COL_P90_ABS_ERROR_PCT = "P90 Abs Error (%)"
COL_MODEL_INTERCEPT_MS = "Intercept (ms)"
COL_MODEL_SLOPE_NS_PER_WORK = "Slope (ns / work unit)"
COL_TERM = "Term"
COL_COMPONENT = "Component"
COL_COEFFICIENT = "Coefficient"

COEF_DISPLAY_THRESHOLD = 1e-5
MAX_TERM_POWER = 2
DEFAULT_VIF_THRESHOLD = 100.0
DEFAULT_SAMPLE_SCALE = 1_000.0

ALL_COMBO_MODEL_NAME = (
    "amortized all-square-combinations model: "
    "setup(all D,N,K monomials up to squares) + "
    "algorithm_iterations × algorithm_iter(all D,N,K monomials up to squares)"
)


@dataclass(frozen=True)
class LloydTimingModelReport:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    coefficients: pd.DataFrame
    coefficients_all: pd.DataFrame
    collinearity_removed: pd.DataFrame


@dataclass(frozen=True)
class _FittedLloydModel:
    predictions: pd.DataFrame
    terms: list[str]
    setup_coef: np.ndarray
    algorithm_iter_coef: np.ndarray
    full_coef: np.ndarray
    collinearity_removed: pd.DataFrame
    keep_mask: np.ndarray
    column_metadata: list[dict]


def prepare_lloyd_model_data(df_lloyd: pd.DataFrame) -> pd.DataFrame:
    """Return the positive-valued Lloyd rows needed by the timing model."""
    df_model = df_lloyd.copy()

    df_model[COL_TIME_PER_ALGORITHM_ITER_MS] = (
        df_model[COL_TIME_S] / df_model[COL_ALGORITHM_ITERATIONS] * 1_000
    )
    df_model[COL_TOTAL_TIME_MS] = df_model[COL_TIME_S] * 1_000
    df_model[COL_WORK_SIZE] = df_model[COL_DIMENSIONS] * df_model[COL_SAMPLES]

    return df_model[
        df_model[COL_TIME_PER_ALGORITHM_ITER_MS].gt(0)
        & df_model[COL_TOTAL_TIME_MS].gt(0)
        & df_model[COL_WORK_SIZE].gt(0)
        & df_model[COL_SAMPLES].gt(0)
        & df_model[COL_ALGORITHM_ITERATIONS].gt(0)
    ].copy()


def fit_lloyd_timing_models(
    df_lloyd: pd.DataFrame,
    *,
    languages: Sequence[str] = (LANG_CPP, LANG_PY),
    group_cols: Sequence[str] | None = None,
    model_name: str = ALL_COMBO_MODEL_NAME,
    vif_threshold: float = DEFAULT_VIF_THRESHOLD,
    coefficient_threshold: float = COEF_DISPLAY_THRESHOLD,
    sample_scale: float = DEFAULT_SAMPLE_SCALE,
) -> LloydTimingModelReport:
    """Fit amortized Lloyd timing models.

    By default, models are fit independently per language. When the dataframe
    contains more than one implementation variant, the default grouping becomes
    language × variant so static and dynamic C++ implementations are not folded
    into one fitted model.
    """
    df_model = prepare_lloyd_model_data(df_lloyd)
    df_model = filter_bench(df_model, language=languages)

    if group_cols is None:
        group_cols = [COL_LANGUAGE]
        if COL_VARIANT in df_model.columns and df_model[COL_VARIANT].nunique() > 1:
            group_cols = [COL_LANGUAGE, COL_VARIANT]
    else:
        group_cols = list(group_cols)

    fit_entries: list[tuple[dict[str, object], _FittedLloydModel]] = []

    for group_values, df_group in df_model.groupby(list(group_cols), observed=True, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_record = dict(zip(group_cols, group_values))
        group_label = " / ".join(str(group_record[col]) for col in group_cols)

        if df_group.empty:
            raise ValueError(f"No Lloyd rows found for group {group_label!r}.")

        fit_entries.append(
            (
                group_record,
                _fit_amortized_model(
                    df_group,
                    model_name=model_name,
                    vif_threshold=vif_threshold,
                    sample_scale=sample_scale,
                ),
            )
        )

    if not fit_entries:
        raise ValueError("No Lloyd rows found for the requested language/variant groups.")

    predictions = pd.concat(
        [fit.predictions for _group_record, fit in fit_entries],
        ignore_index=True,
    )

    summary = _summary_frame(fit_entries, model_name)
    coefficients_all = pd.concat(
        [
            _coefficient_frame(
                group_record,
                model_name,
                fit.terms,
                fit.setup_coef,
                fit.algorithm_iter_coef,
            )
            for group_record, fit in fit_entries
        ],
        ignore_index=True,
    )

    sort_cols = [col for col in [COL_LANGUAGE, COL_VARIANT, COL_COMPONENT, COL_COEFFICIENT] if col in coefficients_all.columns]
    ascending = [True] * len(sort_cols)
    if sort_cols and sort_cols[-1] == COL_COEFFICIENT:
        ascending[-1] = False

    coefficients = coefficients_all[
        coefficients_all[COL_COEFFICIENT].abs().gt(coefficient_threshold)
    ]
    if sort_cols:
        coefficients = coefficients.sort_values(sort_cols, ascending=ascending)
    coefficients = coefficients.reset_index(drop=True)

    collinearity_frames = []
    for group_record, fit in fit_entries:
        if fit.collinearity_removed.empty:
            continue
        collinearity_frames.append(fit.collinearity_removed.assign(**group_record))

    if collinearity_frames:
        collinearity_removed = pd.concat(collinearity_frames, ignore_index=True)
        front_cols = [col for col in [COL_LANGUAGE, COL_VARIANT] if col in collinearity_removed.columns]
        collinearity_removed = collinearity_removed[
            front_cols + ["Removed Component", "Removed Term", "VIF"]
        ]
    else:
        front_cols = list(group_cols)
        collinearity_removed = pd.DataFrame(
            columns=front_cols + ["Removed Component", "Removed Term", "VIF"]
        )

    return LloydTimingModelReport(
        predictions=predictions,
        summary=summary,
        coefficients=coefficients,
        coefficients_all=coefficients_all,
        collinearity_removed=collinearity_removed,
    )

def _summary_frame(
    fit_entries: Sequence[tuple[dict[str, object], _FittedLloydModel]],
    model_name: str,
) -> pd.DataFrame:
    records = []

    for group_record, fit in fit_entries:
        df_group = fit.predictions
        y_true = df_group[COL_TIME_PER_ALGORITHM_ITER_MS].to_numpy(dtype=float)
        y_pred = df_group[COL_PRED_TIME_PER_ALGORITHM_ITER_MS].to_numpy(dtype=float)

        records.append(
            {
                **group_record,
                COL_MODEL_NAME: model_name,
                COL_NUM_PARAMS: len(fit.full_coef),
                COL_LOG_R2: r2_score(np.log(y_true), np.log(y_pred)),
                **_relative_error_metrics(y_true, y_pred),
            }
        )

    return pd.DataFrame(records)

def _relative_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_error_pct = np.abs(y_pred / y_true - 1) * 100

    return {
        COL_MEDIAN_ABS_ERROR_PCT: np.median(abs_error_pct),
        COL_P90_ABS_ERROR_PCT: np.percentile(abs_error_pct, 90),
    }


def _term_complexity_from_powers(
    powers: tuple[int, ...],
) -> tuple[int, int, tuple[int, ...]]:
    return sum(powers), max(powers), tuple(powers)


def _compute_vif_scores(
    x: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VIF for currently kept non-constant columns."""
    x = np.asarray(x, dtype=float)
    kept_indices = np.flatnonzero(keep_mask)
    x_kept = x[:, kept_indices]

    std = x_kept.std(axis=0)
    non_constant_local = std > 1e-12

    z = np.zeros_like(x_kept)
    z[:, non_constant_local] = (
        x_kept[:, non_constant_local] - x_kept[:, non_constant_local].mean(axis=0)
    ) / std[non_constant_local]

    vif = np.full(len(kept_indices), np.nan)

    for local_j, _global_j in enumerate(kept_indices):
        if not non_constant_local[local_j]:
            continue

        other_local = np.flatnonzero(non_constant_local)
        other_local = other_local[other_local != local_j]

        if len(other_local) == 0:
            vif[local_j] = 1.0
            continue

        y = z[:, local_j]
        x_other = z[:, other_local]

        beta = np.linalg.lstsq(x_other, y, rcond=None)[0]
        y_hat = x_other @ beta

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum(y**2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        r2 = np.clip(r2, 0, 1 - 1e-12)

        vif[local_j] = 1 / (1 - r2)

    return kept_indices, vif


def _prune_collinear_columns(
    x: np.ndarray,
    column_metadata: list[dict],
    *,
    vif_threshold: float = DEFAULT_VIF_THRESHOLD,
    mandatory_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Remove columns explainable by other columns, preferring simpler terms."""
    n_cols = x.shape[1]
    keep_mask = np.ones(n_cols, dtype=bool)

    if mandatory_mask is None:
        mandatory_mask = np.zeros(n_cols, dtype=bool)
    else:
        mandatory_mask = np.asarray(mandatory_mask, dtype=bool)

    removed_records = []

    while True:
        kept_indices, vif = _compute_vif_scores(x, keep_mask)
        bad_local = np.flatnonzero(vif > vif_threshold)

        if len(bad_local) == 0:
            break

        removable_local = [j for j in bad_local if not mandatory_mask[kept_indices[j]]]

        if not removable_local:
            break

        def removal_key(local_j: int) -> tuple[int, int, float, tuple[int, ...]]:
            global_j = kept_indices[local_j]
            meta = column_metadata[global_j]
            total_degree, max_power, powers_tuple = _term_complexity_from_powers(
                meta["powers"]
            )

            return (
                total_degree,
                max_power,
                vif[local_j],
                powers_tuple,
            )

        local_to_remove = max(removable_local, key=removal_key)
        global_to_remove = kept_indices[local_to_remove]
        keep_mask[global_to_remove] = False

        removed_records.append(
            {
                "Removed Component": column_metadata[global_to_remove]["component"],
                "Removed Term": column_metadata[global_to_remove]["term"],
                "VIF": vif[local_to_remove],
            }
        )

    return keep_mask, pd.DataFrame(removed_records)


def _power_label(label: str, power: int) -> str:
    if power == 1:
        return label
    if power == 2:
        return f"{label}²"
    return f"{label}^{power}"


def _build_all_square_combination_terms(
    variable_specs: Sequence[tuple[str, np.ndarray]],
    *,
    max_power: int = MAX_TERM_POWER,
) -> tuple[np.ndarray, list[str], list[tuple[int, ...]]]:
    names = [name for name, _ in variable_specs]
    values = [np.asarray(values, dtype=float) for _, values in variable_specs]
    row_count = len(values[0])

    columns = [np.ones(row_count)]
    terms = ["Intercept"]
    powers_list = [tuple(0 for _ in values)]

    exponent_rows = [
        powers
        for powers in product(range(max_power + 1), repeat=len(values))
        if any(powers)
    ]
    exponent_rows.sort(key=lambda powers: (sum(powers), tuple(-p for p in powers)))

    for powers in exponent_rows:
        col = np.ones(row_count)
        pieces = []

        for name, value, power in zip(names, values, powers):
            if power:
                col *= value**power
                pieces.append(_power_label(name, power))

        columns.append(col)
        terms.append(" × ".join(pieces))
        powers_list.append(tuple(powers))

    return np.column_stack(columns), terms, powers_list


def _fit_log_error_positive_linear_model(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit y = X @ beta by minimizing log-space error with positive coefficients."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Improves conditioning for high-degree terms while returning coefficients
    # in the original unscaled variable units.
    col_scale = np.maximum(np.nanmedian(np.abs(x), axis=0), 1.0)
    x_scaled = x / col_scale

    initial_coef_scaled = np.linalg.lstsq(x_scaled, y, rcond=None)[0]
    initial_coef_scaled = np.maximum(initial_coef_scaled, 1e-12)

    def residuals(coef_scaled: np.ndarray) -> np.ndarray:
        y_hat = x_scaled @ coef_scaled
        return np.log(y_hat) - np.log(y)

    result = least_squares(
        residuals,
        x0=initial_coef_scaled,
        bounds=(
            np.full(x_scaled.shape[1], 1e-12),
            np.full(x_scaled.shape[1], np.inf),
        ),
        x_scale="jac",
        max_nfev=50_000,
    )

    coef_scaled = result.x
    coef = coef_scaled / col_scale
    y_hat = x @ coef

    return coef, y_hat


def _fit_amortized_model(
    df_lang: pd.DataFrame,
    *,
    model_name: str,
    vif_threshold: float,
    sample_scale: float,
) -> _FittedLloydModel:
    D = df_lang[COL_DIMENSIONS].to_numpy(dtype=float)
    N = df_lang[COL_SAMPLES].to_numpy(dtype=float) / sample_scale
    K = df_lang[COL_CLUSTERS].to_numpy(dtype=float)
    algorithm_iterations = df_lang[COL_ALGORITHM_ITERATIONS].to_numpy(dtype=float)

    x_base, terms, powers_list = _build_all_square_combination_terms(
        [
            (COL_DIMENSIONS, D),
            (COL_SAMPLES, N),
            (COL_CLUSTERS, K),
        ],
        max_power=MAX_TERM_POWER,
    )

    term_count = len(terms)
    x_full = np.column_stack(
        [
            x_base,
            x_base * algorithm_iterations[:, None],
        ]
    )

    column_metadata = [
        {
            "component": "Setup / amortized component",
            "term": term,
            "powers": powers,
        }
        for term, powers in zip(terms, powers_list)
    ] + [
        {
            "component": "True per-algorithm-iteration component",
            "term": term,
            "powers": powers,
        }
        for term, powers in zip(terms, powers_list)
    ]

    # Usually keep both intercept-like terms.
    mandatory_mask = np.zeros(x_full.shape[1], dtype=bool)
    mandatory_mask[0] = True  # setup intercept
    mandatory_mask[term_count] = True  # algorithm-iteration intercept

    keep_mask, df_collinearity_removed = _prune_collinear_columns(
        x_full,
        column_metadata,
        vif_threshold=vif_threshold,
        mandatory_mask=mandatory_mask,
    )

    x = x_full[:, keep_mask]
    y_total = df_lang[COL_TOTAL_TIME_MS].to_numpy(dtype=float)

    kept_coef, _total_y_hat = _fit_log_error_positive_linear_model(x, y_total)

    full_coef = np.zeros(x_full.shape[1])
    full_coef[keep_mask] = kept_coef

    setup_coef = full_coef[:term_count]
    algorithm_iter_coef = full_coef[term_count:]

    setup_y_hat = x_base @ setup_coef
    algorithm_iter_y_hat = x_base @ algorithm_iter_coef
    total_y_hat = setup_y_hat + algorithm_iterations * algorithm_iter_y_hat
    per_algorithm_iter_y_hat = total_y_hat / algorithm_iterations

    df_out = df_lang.copy()
    df_out[COL_PRED_TOTAL_TIME_MS] = total_y_hat
    df_out[COL_PRED_TIME_PER_ALGORITHM_ITER_MS] = per_algorithm_iter_y_hat
    df_out[COL_MODEL_NAME] = model_name

    return _FittedLloydModel(
        predictions=df_out,
        terms=terms,
        setup_coef=setup_coef,
        algorithm_iter_coef=algorithm_iter_coef,
        full_coef=full_coef,
        collinearity_removed=df_collinearity_removed,
        keep_mask=keep_mask,
        column_metadata=column_metadata,
    )


def _coefficient_frame(
    group_record: dict[str, object],
    model_name: str,
    terms: Sequence[str],
    setup_coef: np.ndarray,
    algorithm_iter_coef: np.ndarray,
) -> pd.DataFrame:
    records = []

    for component, coef_values in [
        ("Setup / amortized component", setup_coef),
        ("True per-algorithm-iteration component", algorithm_iter_coef),
    ]:
        for term, coef in zip(terms, coef_values):
            records.append(
                {
                    **group_record,
                    COL_MODEL_NAME: model_name,
                    COL_COMPONENT: component,
                    COL_TERM: term,
                    COL_COEFFICIENT: coef,
                }
            )

    return pd.DataFrame(records)


__all__ = [
    "ALL_COMBO_MODEL_NAME",
    "COEF_DISPLAY_THRESHOLD",
    "COL_ABS_ERROR_PCT",
    "COL_COEFFICIENT",
    "COL_COMPONENT",
    "COL_LOG_R2",
    "COL_MEDIAN_ABS_ERROR_PCT",
    "COL_MODEL_INTERCEPT_MS",
    "COL_MODEL_NAME",
    "COL_MODEL_SLOPE_NS_PER_WORK",
    "COL_NUM_PARAMS",
    "COL_P90_ABS_ERROR_PCT",
    "COL_PRED_TIME_PER_ALGORITHM_ITER_MS",
    "COL_PRED_TOTAL_TIME_MS",
    "COL_TERM",
    "COL_TOTAL_TIME_MS",
    "COL_WORK_SIZE",
    "LloydTimingModelReport",
    "fit_lloyd_timing_models",
    "prepare_lloyd_model_data",
]
