import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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
