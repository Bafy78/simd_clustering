from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from benchmark_reporting.constants import (
    COL_CLUSTERS,
    COL_DIMENSIONS,
    COL_SAMPLES,
    CONFIG_COLS,
)


COL_CONFIGURATION_RECTANGLE = "Configuration Rectangle"
COL_CONFIGURATION_COUNT = "Configurations"


@dataclass(frozen=True)
class CartesianRectangle:
    """One Cartesian-product block in a compact cover of tabular points."""

    frame: pd.DataFrame
    path: tuple[tuple[str, object], ...]
    residual_cols: tuple[str, ...]
    is_complete: bool = True


def compact_cartesian_rectangles(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> list[CartesianRectangle]:
    """Cover distinct rows with compact Cartesian-product rectangles.

    The cover uses the same projection-based recursive partitioning for any
    collection of discrete columns. Each complete result represents every
    combination of the values listed for its residual columns.
    """

    columns = list(columns)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing rectangle column(s): {', '.join(missing)}")

    distinct = _normalized_distinct_frame(df, columns)
    if distinct.empty:
        return []

    return _compact_cartesian_rectangles(
        distinct,
        residual_cols=columns,
        path=(),
    )


def summarize_configuration_rectangles(
    df: pd.DataFrame,
    *,
    config_cols: Iterable[str] = CONFIG_COLS,
) -> pd.DataFrame:
    """Return one readable row per compact configuration rectangle."""

    config_cols = list(config_cols)
    output_cols = [COL_CONFIGURATION_COUNT, COL_CONFIGURATION_RECTANGLE]
    if df.empty:
        return pd.DataFrame(columns=output_cols)

    rectangles = compact_cartesian_rectangles(df, config_cols)
    rows = [
        {
            COL_CONFIGURATION_COUNT: len(rectangle.frame),
            COL_CONFIGURATION_RECTANGLE: summarize_columns(
                rectangle.frame,
                config_cols,
            ),
        }
        for rectangle in rectangles
    ]
    return pd.DataFrame(rows, columns=output_cols)


def distinct_keys(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return normalized, distinct rows for the requested columns."""

    columns = list(columns)
    if not columns:
        return pd.DataFrame([{}]) if not df.empty else pd.DataFrame()

    return _normalized_distinct_frame(df[columns], columns)


def normalize_key_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def ordered_unique_values(series: pd.Series) -> list:
    values = []
    for value in series.dropna().tolist():
        value = normalize_key_value(value)
        if value not in values:
            values.append(value)

    def sort_key(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return (0, float(value))
        return (1, str(value))

    return sorted(values, key=sort_key)


def format_scalar(value) -> str:
    if isinstance(value, (list, tuple, set, frozenset)):
        ordered = list(value)
        if isinstance(value, (set, frozenset)):
            ordered = sorted(ordered, key=repr)
        return format_value_list([format_scalar(item) for item in ordered])
    if value == "":
        return "<empty>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_path(columns: Iterable[str], values: Iterable) -> str:
    return " / ".join(
        f"{col}={format_scalar(value)}"
        for col, value in zip(columns, values)
    )


def summarize_columns(df: pd.DataFrame, columns: Iterable[str]) -> str:
    parts = []
    for col in columns:
        if col not in df.columns:
            continue
        values = ordered_unique_values(df[col])
        if values:
            parts.append(
                f"{col}={format_value_list([format_scalar(value) for value in values])}"
            )
    return "; ".join(parts) if parts else "-"


def is_cartesian_product(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    columns = list(columns)
    if df.empty:
        return False
    if not columns:
        return len(df) <= 1

    distinct = distinct_keys(df, columns)
    product = 1
    for col in columns:
        unique_count = len(ordered_unique_values(distinct[col]))
        if unique_count == 0:
            return False
        product *= unique_count
        if product > len(distinct):
            return False
    return product == len(distinct)


def _compact_cartesian_rectangles(
    df: pd.DataFrame,
    *,
    residual_cols: list[str],
    path: tuple[tuple[str, object], ...],
) -> list[CartesianRectangle]:
    df = _normalized_distinct_frame(
        df,
        [col for col, _ in path] + residual_cols,
    )
    if df.empty:
        return []

    if is_cartesian_product(df, residual_cols):
        return [
            CartesianRectangle(
                frame=df,
                path=path,
                residual_cols=tuple(residual_cols),
            )
        ]

    split_col = _choose_cartesian_split_column(df, residual_cols)
    if split_col is None:
        return [
            CartesianRectangle(
                frame=df,
                path=path,
                residual_cols=tuple(residual_cols),
                is_complete=False,
            )
        ]

    rectangles: list[CartesianRectangle] = []
    remaining_cols = [col for col in residual_cols if col != split_col]
    for values, sub_df in _partition_by_projection(df, split_col, remaining_cols):
        rectangles.extend(
            _compact_cartesian_rectangles(
                sub_df,
                residual_cols=remaining_cols,
                path=path + ((split_col, _compact_path_value(values)),),
            )
        )
    return rectangles


def _choose_cartesian_split_column(
    df: pd.DataFrame,
    residual_cols: list[str],
) -> str | None:
    candidates = []
    cost_memo: dict[tuple, int] = {}

    for index, col in enumerate(residual_cols):
        values = ordered_unique_values(df[col])
        if len(values) <= 1:
            continue

        remaining_cols = [
            candidate for candidate in residual_cols if candidate != col
        ]
        partitions = _partition_by_projection(df, col, remaining_cols)
        if len(partitions) <= 1:
            continue

        cost = sum(
            _cartesian_cover_cost(partition_df, remaining_cols, memo=cost_memo)
            for _, partition_df in partitions
        )
        candidates.append(
            (cost, len(partitions), _cartesian_split_priority(col), index, col)
        )

    if not candidates:
        return None
    return min(candidates)[-1]


def _cartesian_cover_cost(
    df: pd.DataFrame,
    residual_cols: list[str],
    *,
    memo: dict[tuple, int],
) -> int:
    distinct = distinct_keys(df, residual_cols)
    key = (
        tuple(residual_cols),
        tuple(
            sorted(
                (_row_tuple(row, residual_cols) for _, row in distinct.iterrows()),
                key=repr,
            )
        ),
    )
    if key in memo:
        return memo[key]

    if distinct.empty or is_cartesian_product(distinct, residual_cols):
        memo[key] = 1
        return 1

    costs = []
    for col in residual_cols:
        if len(ordered_unique_values(distinct[col])) <= 1:
            continue
        remaining_cols = [
            candidate for candidate in residual_cols if candidate != col
        ]
        partitions = _partition_by_projection(distinct, col, remaining_cols)
        if len(partitions) <= 1:
            continue
        costs.append(
            sum(
                _cartesian_cover_cost(partition_df, remaining_cols, memo=memo)
                for _, partition_df in partitions
            )
        )

    cost = min(costs) if costs else 1
    memo[key] = cost
    return cost


def _partition_by_projection(
    df: pd.DataFrame,
    split_col: str,
    remaining_cols: list[str],
) -> list[tuple[tuple, pd.DataFrame]]:
    buckets: dict[frozenset, list[object]] = {}
    first_value_order: dict[frozenset, int] = {}

    normalized_split = df[split_col].map(normalize_key_value)
    ordered_values = ordered_unique_values(normalized_split)

    for order, value in enumerate(ordered_values):
        value_df = df[normalized_split == value]
        projection = frozenset(
            _row_tuple(row, remaining_cols)
            for _, row in distinct_keys(value_df, remaining_cols).iterrows()
        )
        buckets.setdefault(projection, []).append(value)
        first_value_order.setdefault(projection, order)

    partitions = []
    for projection in sorted(buckets, key=lambda key: first_value_order[key]):
        values = tuple(buckets[projection])
        sub_df = df[normalized_split.isin(set(values))].copy()
        partitions.append((values, sub_df))
    return partitions


def _row_tuple(row: pd.Series, columns: Iterable[str]) -> tuple:
    return tuple(normalize_key_value(row.get(col, "")) for col in columns)


def _normalized_distinct_frame(
    df: pd.DataFrame,
    key_columns: Iterable[str],
) -> pd.DataFrame:
    key_columns = list(key_columns)
    result = df.copy()
    for col in key_columns:
        result[col] = result[col].map(normalize_key_value)
    return result.drop_duplicates(subset=key_columns).reset_index(drop=True)


def _compact_path_value(values: tuple):
    if len(values) == 1:
        return values[0]
    return values


def _cartesian_split_priority(col: str) -> int:
    grid_priority = {
        COL_SAMPLES: 0,
        COL_CLUSTERS: 1,
        COL_DIMENSIONS: 2,
    }
    if col in grid_priority:
        return grid_priority[col]
    if col in CONFIG_COLS:
        return 10
    return 20


def format_value_list(values: Iterable[str], *, max_values: int = 12) -> str:
    values = list(values)
    if not values:
        return "-"
    displayed = values[:max_values]
    suffix = (
        ""
        if len(values) <= max_values
        else f", … (+{len(values) - max_values})"
    )
    return ", ".join(displayed) + suffix
