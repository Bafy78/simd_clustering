import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections.abc import Callable
import seaborn as sns
from typing import Any
import re
from matplotlib.lines import Line2D


def create_subplot_grid(n_plots, cols=2, row_height=5, fig_width=12):
    """Handles the boilerplate for dynamic grid layouts and unused axes."""
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(fig_width, row_height * rows)
    )

    axes = axes.flatten() if n_plots > 1 else np.array([axes])

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    return fig, axes


def format_abbrev(num):
    num = float(num)
    if num >= 1e9:
        val = num / 1e9
        suffix = "B"
    elif num >= 1e6:
        val = num / 1e6
        suffix = "M"
    elif num >= 1e3:
        val = num / 1e3
        suffix = "K"
    else:
        val = num
        suffix = ""

    # Format to 1 decimal place, then remove trailing '.0' if it's a whole number
    return f"{val:.1f}".replace(".0", "") + suffix


SMALL_MULTIPLE_TITLE_STYLE = dict(
    boxstyle="round,pad=0.3", facecolor="black", alpha=0.1, edgecolor="none"
)


def _extract_and_remove_figure_legend(
    g,
    *,
    default_title=None,
    label_formatter=None,
):
    """Extract Seaborn figure-level legend handles/labels and remove the legend.

    Useful when we want to place legends manually on selected facets.
    """
    handles, labels, title = [], [], default_title

    if not g.figure.legends:
        return handles, labels, title

    fig_legend = g.figure.legends[0]

    handles = getattr(
        fig_legend,
        "legend_handles",
        getattr(fig_legend, "legendHandles", []),
    )

    labels = [text.get_text() for text in fig_legend.texts]

    if label_formatter is not None:
        labels = [label_formatter(label) for label in labels]

    legend_title = fig_legend.get_title().get_text()
    title = legend_title or default_title

    fig_legend.remove()

    return handles, labels, title


def _put_facet_titles_inside(g):
    """Move Seaborn facet titles inside each subplot with shared styling."""
    for ax in g.axes.flat:
        current_title = ax.get_title()
        ax.set_title("")

        if current_title:
            ax.text(
                0.5,
                0.95,
                current_title,
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=SMALL_MULTIPLE_TITLE_STYLE,
            )

    return g


def _add_legend_to_row_ends(
    g,
    handles,
    labels,
    *,
    title=None,
    col_wrap=3,
    anchor=(1.02, 0.5),
):
    """Place legends on the right-most facet in each row."""
    if not handles:
        return g

    axes = list(g.axes.flat)

    for i, ax in enumerate(axes):
        is_row_end = i % col_wrap == col_wrap - 1
        is_last_axis = i == len(axes) - 1

        if is_row_end or is_last_axis:
            ax.legend(
                handles,
                labels,
                title=title,
                loc="center left",
                bbox_to_anchor=anchor,
            )

    return g


def move_facet_legend_to_row_ends(
    g,
    *,
    default_title,
    label_formatter=None,
    col_wrap=3,
    anchor=(1, 0.5),
):
    handles, labels, title = _extract_and_remove_figure_legend(
        g,
        default_title=default_title,
        label_formatter=label_formatter,
    )

    _add_legend_to_row_ends(
        g,
        handles,
        labels,
        title=title,
        col_wrap=col_wrap,
        anchor=anchor,
    )


def style_facet_grid(
    g,
    *,
    title=None,
    title_y=1.02,
    x_log=False,
    sample_x_axis=False,
    titles_inside=False,
    grid_axis=None,
    integer_x_axis=False,
):
    """Apply common styling to a Seaborn FacetGrid or relplot result."""
    if x_log:
        g.set(xscale="log")

    if not titles_inside:
        g.set_titles(bbox=SMALL_MULTIPLE_TITLE_STYLE)
    else:
        _put_facet_titles_inside(g)

    if title is not None:
        g.figure.suptitle(
            title,
            y=title_y,
            fontsize=16,
            fontweight="bold",
        )

    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True)

        if sample_x_axis:
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: format_abbrev(x))
            )

        if grid_axis is not None:
            ax.grid(axis=grid_axis, linestyle="--", alpha=0.7)

        if integer_x_axis:
            ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    return g


from python.benchmark_reporting.constants import (
    COL_DIMENSIONS,
    COL_SAMPLES,
    COL_CLUSTERS,
)


def make_cluster_pivot(
    subset,
    value_col: str,
    *,
    reference_pivot=None,
):
    pivot = subset.pivot(
        index=COL_DIMENSIONS,
        columns=COL_SAMPLES,
        values=value_col,
    )

    if reference_pivot is not None:
        pivot = pivot.reindex_like(reference_pivot)

    formatted = pivot.copy()
    formatted.columns = [format_abbrev(c) for c in formatted.columns]

    return formatted, pivot


def add_initial_letter_annotations(
    df,
    *,
    source_col: str,
    annot_col: str,
):
    """Add a one-letter annotation column derived from source_col values."""
    out = df.copy()

    labels = list(dict.fromkeys(str(v) for v in out[source_col].dropna()))

    def first_letter(label: str) -> str:
        match = re.search(r"[A-Za-z]", label)
        if match is None:
            raise ValueError(f"Could not derive an annotation letter from {label!r}")
        return match.group(0).lower()

    label_to_letter = {label: first_letter(label) for label in labels}

    letter_to_labels = {}
    for label, letter in label_to_letter.items():
        letter_to_labels.setdefault(letter, []).append(label)

    collisions = {
        letter: labels for letter, labels in letter_to_labels.items() if len(labels) > 1
    }

    if collisions:
        raise ValueError(
            "Duplicate annotation letters found. "
            f"Please rename checks or provide an explicit mapping: {collisions}"
        )

    out[annot_col] = out[source_col].map(
        lambda value: label_to_letter.get(str(value), "")
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker=f"${letter}$",
            linestyle="None",
            color="black",
            markersize=12,
            label=label.replace("_", " "),
        )
        for label, letter in label_to_letter.items()
    ]

    return out, handles, label_to_letter


def plot_clustered_heatmap_grid(
    df,
    *,
    clusters,
    value_col: str,
    title: str,
    heatmap_kwargs: dict[str, Any],
    cbar_kws: dict[str, Any],
    annot_col: str | None = None,
    fmt: str = ".1f",
    legend_handles=None,
    post_heatmap: Callable | None = None,
):
    fig, axes = create_subplot_grid(
        len(clusters),
        cols=2,
        row_height=6,
        fig_width=14,
    )

    axes_flat = axes.flatten()

    for i, cluster in enumerate(clusters):
        ax = axes_flat[i]
        subset = df[df[COL_CLUSTERS] == cluster]

        heat, heat_raw = make_cluster_pivot(
            subset,
            value_col,
        )

        if annot_col is None:
            annot_arg = True
        else:
            annot_arg, _ = make_cluster_pivot(
                subset,
                annot_col,
                reference_pivot=heat_raw,
            )

        show_cbar = i % 2 == 1
        cbar_ax = ax.inset_axes([1.04, 0, 0.05, 1]) if show_cbar else None

        sns.heatmap(
            heat,
            annot=annot_arg,
            fmt=fmt,
            cmap="turbo",
            ax=ax,
            cbar=show_cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws if show_cbar else {},
            linewidths=0.5,
            **heatmap_kwargs,
        )

        if post_heatmap is not None:
            post_heatmap(
                ax=ax,
                subset=subset,
                heat=heat,
                heat_raw=heat_raw,
            )

        ax.set_title(f"Clusters: {cluster}", bbox=SMALL_MULTIPLE_TITLE_STYLE)
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Dimensions" if i % 2 == 0 else "")
        ax.tick_params(axis="x", labelrotation=0)

    for ax in axes_flat[len(clusters) :]:
        ax.set_visible(False)

    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            ncol=min(len(legend_handles), 6),
        )

    plt.suptitle(
        title,
        y=1.0,
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()
