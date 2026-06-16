import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections.abc import Callable
import seaborn as sns
from typing import Any
import re
from matplotlib.lines import Line2D


def create_subplot_grid(plot_count, cols=2, row_height=5, fig_width=12):
    """Handles the boilerplate for dynamic grid layouts and unused axes."""
    rows = math.ceil(plot_count / cols)
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(fig_width, row_height * rows)
    )

    axes = axes.flatten() if plot_count > 1 else np.array([axes])

    for j in range(plot_count, len(axes)):
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
    COL_EQUIVALENT_ALGORITHM_ITERS,
    COL_LANGUAGE,
    COL_PARAMS,
    COL_PHASE,
    COL_TIME_PER_ALGORITHM_ITER,
    COL_TIME_S,
    COL_VARIANT,
    LANG_CPP,
    PHASE_MAP,
)
from python.benchmark_reporting.transforms import filter_bench


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


def _phase_display_name_from_key(phase_key: str) -> str:
    try:
        return PHASE_MAP[phase_key]
    except KeyError as exc:
        valid = ", ".join(sorted(PHASE_MAP))
        raise ValueError(
            f"Unknown phase key {phase_key!r}; valid phase keys: {valid}"
        ) from exc


def _remove_unused_categories(df, columns):
    out = df.copy()

    for col in columns:
        if col in out.columns and hasattr(out[col], "cat"):
            out[col] = out[col].cat.remove_unused_categories()

    return out


def _format_variant_params_title(variant, params):
    if str(params) == "Default":
        return f"Variant: {variant}"
    return f"Variant: {variant}; Params: {params}"


def plot_fixed_costs_vs_algorithm_iteration_time(
    df_bench,
    *,
    algorithm_phase_key: str,
    algorithm_label: str,
    fixed_phase_keys=("soa", "pp"),
    max_samples=None,
    fixed_cost_note: str | None = None,
):
    """Plot fixed C++ setup costs against an algorithm iteration baseline.

    Fixed phases such as SoA conversion and K-Means++ do not have the same
    parameterization space as algorithms. For example, GMM has covariance-type
    parameterizations while SoA and K-Means++ do not. This helper therefore
    pairs fixed costs with algorithm rows on D/N/K/variant and deliberately
    replicates fixed costs across algorithm parameterizations.
    """
    algorithm_phase = _phase_display_name_from_key(algorithm_phase_key)
    fixed_phases = [_phase_display_name_from_key(key) for key in fixed_phase_keys]

    df_cpp = filter_bench(
        df_bench,
        language=LANG_CPP,
    ).copy()

    algorithm_N_values = set(
        filter_bench(
            df_cpp,
            phase=algorithm_phase,
        )[COL_SAMPLES].dropna().unique()
    )

    fixed_N_values = set(
        filter_bench(
            df_cpp,
            phase=fixed_phases,
        )[COL_SAMPLES].dropna().unique()
    )

    eligible_N_values = sorted(algorithm_N_values & fixed_N_values)

    if not eligible_N_values:
        print(
            f"Skipping fixed-cost vs {algorithm_label} algorithm-iteration graph: "
            f"requires C++ {algorithm_label} plus at least one of "
            "C++ SoA / C++ K-Means++."
        )
        return []

    if max_samples is None:
        max_N = max(eligible_N_values)
    elif max_samples in eligible_N_values:
        max_N = max_samples
    else:
        print(
            f"Skipping fixed-cost vs {algorithm_label} graph: "
            f"requested N={max_samples} is not present for both algorithm and fixed costs."
        )
        return []

    algorithm_cols = [
        COL_DIMENSIONS,
        COL_CLUSTERS,
        COL_VARIANT,
        COL_PARAMS,
        COL_TIME_PER_ALGORITHM_ITER,
    ]
    fixed_cols = [
        COL_DIMENSIONS,
        COL_CLUSTERS,
        COL_VARIANT,
        COL_PHASE,
        COL_TIME_S,
    ]

    df_algorithm = filter_bench(
        df_bench,
        phase=algorithm_phase,
        language=LANG_CPP,
        samples=max_N,
    )[algorithm_cols].copy()

    df_fixed = filter_bench(
        df_bench,
        phase=fixed_phases,
        language=LANG_CPP,
        samples=max_N,
    )[fixed_cols].copy()

    if df_algorithm.empty or df_fixed.empty:
        print(
            f"Skipping fixed-cost vs {algorithm_label} graph: missing algorithm or fixed-cost rows."
        )
        return []

    # Fixed phases are parameterization-independent. Collapse any accidental
    # duplicate Default-parameter rows before crossing them with algorithm params.
    df_fixed = (
        df_fixed
        .groupby(
            [COL_DIMENSIONS, COL_CLUSTERS, COL_VARIANT, COL_PHASE],
            observed=True,
            as_index=False,
        )[COL_TIME_S]
        .median()
    )

    df_plot = df_fixed.merge(
        df_algorithm,
        on=[COL_DIMENSIONS, COL_CLUSTERS, COL_VARIANT],
        validate="many_to_many",
    )

    if df_plot.empty:
        print(
            f"Skipping fixed-cost vs {algorithm_label} graph: fixed-cost variants "
            f"did not match any C++ {algorithm_label} variants."
        )
        return []

    df_plot[COL_EQUIVALENT_ALGORITHM_ITERS] = (
        df_plot[COL_TIME_S] / df_plot[COL_TIME_PER_ALGORITHM_ITER]
    )
    df_plot = _remove_unused_categories(
        df_plot,
        [COL_PHASE, COL_VARIANT, COL_PARAMS],
    )

    equivalent_iters_index_col = COL_EQUIVALENT_ALGORITHM_ITERS + "_index"
    time_per_iter_index_col = COL_TIME_PER_ALGORITHM_ITER + "_index"
    baseline_col = COL_TIME_PER_ALGORITHM_ITER + "_baseline_lowest_dim"

    baseline = (
        df_algorithm
        .sort_values([COL_VARIANT, COL_PARAMS, COL_CLUSTERS, COL_DIMENSIONS])
        .groupby(
            [COL_CLUSTERS, COL_VARIANT, COL_PARAMS],
            observed=True,
            as_index=False,
        )
        .first()[[COL_CLUSTERS, COL_VARIANT, COL_PARAMS, COL_TIME_PER_ALGORITHM_ITER]]
        .rename(columns={COL_TIME_PER_ALGORITHM_ITER: baseline_col})
    )

    df_plot_norm = df_plot.merge(
        baseline,
        on=[COL_CLUSTERS, COL_VARIANT, COL_PARAMS],
        validate="many_to_one",
    )

    df_plot_norm[equivalent_iters_index_col] = (
        df_plot_norm[COL_TIME_S] / df_plot_norm[baseline_col]
    )
    df_plot_norm[time_per_iter_index_col] = (
        df_plot_norm[COL_TIME_PER_ALGORITHM_ITER] / df_plot_norm[baseline_col]
    )
    df_plot_norm = df_plot_norm.sort_values(
        [COL_VARIANT, COL_PARAMS, COL_DIMENSIONS, COL_CLUSTERS, COL_PHASE]
    )
    df_plot_norm = _remove_unused_categories(
        df_plot_norm,
        [COL_PHASE, COL_VARIANT, COL_PARAMS],
    )

    figures = []

    for (variant, params), df_variant in df_plot_norm.groupby(
        [COL_VARIANT, COL_PARAMS],
        observed=True,
    ):
        if df_variant.empty:
            continue

        plot_keys = (
            df_variant[[COL_CLUSTERS]]
            .drop_duplicates()
            .sort_values([COL_CLUSTERS])
            .reset_index(drop=True)
        )
        fig, axes = create_subplot_grid(
            len(plot_keys),
            cols=2,
            row_height=5,
            fig_width=12,
        )
        axes_flat = axes.flatten()

        for i, row in plot_keys.iterrows():
            k = row[COL_CLUSTERS]
            ax = axes_flat[i]
            data_k = filter_bench(df_variant, clusters=k)

            sns.barplot(
                data=data_k,
                x=COL_DIMENSIONS,
                y=equivalent_iters_index_col,
                hue=COL_PHASE,
                ax=ax,
            )

            data_line = data_k.drop_duplicates(subset=[COL_DIMENSIONS]).copy()

            sns.pointplot(
                data=data_line,
                x=COL_DIMENSIONS,
                y=time_per_iter_index_col,
                ax=ax,
                color="black",
                markersize=6,
                errorbar=None,
            )

            ax.set_title(f"K = {k}", bbox=SMALL_MULTIPLE_TITLE_STYLE)

            if i % 2 == 0:
                ax.set_ylabel(f"Cost in baseline {algorithm_label} iterations")
            else:
                ax.set_ylabel("")

            ax.set_xlabel("D")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mtick.LogLocator(base=10, subs=(1, 5)))
            ax.yaxis.set_minor_locator(
                mtick.LogLocator(base=10, subs=(2, 3, 4, 6, 7, 8, 9))
            )
            ax.yaxis.set_minor_formatter(mtick.NullFormatter())
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:g}"))

            ax.grid(axis="y", which="major", linestyle="--", alpha=0.6)
            ax.grid(axis="y", which="minor", linestyle=":", alpha=0.25)

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()

                line_handle = Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    linestyle="-",
                    label=f"{algorithm_label} time / iter growth",
                )

                ax.legend(
                    handles=handles + [line_handle],
                    labels=labels + [f"{algorithm_label} time / iter growth"],
                    title=None,
                )
            elif ax.get_legend() is not None:
                ax.get_legend().remove()

        note = f"\n{fixed_cost_note}" if fixed_cost_note else ""
        fig.suptitle(
            f"Scaling of Fixed Costs vs {algorithm_label} Iteration Time\n"
            f"{_format_variant_params_title(variant, params)}; "
            f"normalized to the lowest D at N = {format_abbrev(max_N)}"
            f"{note}",
            fontsize=16,
            y=1,
        )

        plt.tight_layout()
        figures.append(fig)

    return figures


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

        ax.set_title(f"{COL_CLUSTERS}: {cluster}", bbox=SMALL_MULTIPLE_TITLE_STYLE)
        ax.set_xlabel(COL_SAMPLES)
        ax.set_ylabel(COL_DIMENSIONS if i % 2 == 0 else "")
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


def add_facet_suptitle(
    g,
    title,
    *,
    top_pad_inches=0.08,
    title_pad_inches=0.25,
    fontsize=16,
):
    fig = g.figure

    # The notebook has figure.autolayout=True, so disable it for this figure.
    fig.set_tight_layout(False)
    try:
        fig.set_layout_engine(None)
    except Exception:
        pass

    suptitle = fig.suptitle(
        title,
        x=0.5,
        y=1 - top_pad_inches / fig.get_figheight(),
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
    )

    # Temporarily exclude title so tight_layout does not double-count it.
    suptitle.set_in_layout(False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_bbox = suptitle.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )

    rect = list(getattr(g, "_tight_layout_rect", (0, 0, 1, 1)))

    rect[3] = min(
        rect[3],
        title_bbox.y0 - title_pad_inches / fig.get_figheight(),
    )
    rect[3] = max(0.05, rect[3])

    fig.tight_layout(rect=rect)

    # Important: include title again so notebook / bbox_inches='tight' actually renders it.
    suptitle.set_in_layout(True)

    # Prevent another automatic tight-layout pass from undoing the rect.
    fig.set_tight_layout(False)
    try:
        fig.set_layout_engine(None)
    except Exception:
        pass

    return suptitle