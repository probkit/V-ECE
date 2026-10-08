"""Shared matplotlib style for the figures of the paper.

Figures are sized for the AISTATS two-column layout (3.25in columns, 6.75in text width) and
typeset with LaTeX in Computer Modern, the font of the paper, so that they can be included
at 100% scale.  When no LaTeX installation is found, matplotlib's Computer Modern mathtext
is used instead, which looks the same up to kerning.

The metrics keep one color and one marker shape across every figure (the shape is the
secondary encoding that survives grayscale printing); estimator names are mapped to the
names used in the paper by ``display_name``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMN_WIDTH = 3.25  # inches, one AISTATS column
TEXT_WIDTH = 6.75  # inches, full AISTATS text width

USE_TEX = shutil.which("latex") is not None
PERCENT = r"\%" if USE_TEX else "%"

# Metrics displayed per modality.  Binary predictions live on a segment, so the L1 and L2
# calibration errors coincide there and only the former is shown.
DISPLAYED_METRICS = {"binary": ("l1", "brier"), "multiclass": ("l1", "l2", "brier")}

# Metric used to decide whether an experiment is informative.
REFERENCE_METRIC = {"binary": "l1", "multiclass": "l2"}

METRIC_LABELS = {
    "binary": {
        "l1": r"$\mathrm{CE}_{|\cdot|}$",
        "l2": r"$\mathrm{CE}_{|\cdot|}$",
        "brier": r"$\mathrm{CE}_{(\cdot)^2}$",
    },
    "multiclass": {
        "l1": r"$\mathrm{CE}_{\|\cdot\|_1}$",
        "l2": r"$\mathrm{CE}_{\|\cdot\|_2}$",
        "brier": r"$\mathrm{CE}_{\|\cdot\|_2^2}$",
    },
}

# First three slots of a colorblind-safe categorical palette (validated all-pairs).
METRIC_COLORS = {"l1": "#2a78d6", "l2": "#1baf7a", "brier": "#eb6834"}
METRIC_MARKERS = {"l1": "o", "l2": "s", "brier": "D"}
METRIC_MARKER_SIZES = {"l1": 3.6, "l2": 3.3, "brier": 3.0}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
HIGHLIGHT = "#2a78d6"

# The warm-started CatBoost recalibrator scored by excess risk is the estimator of the paper.
OURS = "WS-CatBoost"
# Out-of-sample variational estimators are V-ECE with the recalibration function in
# parentheses; WS-CatBoost, the one we recommend, is shown as V-ECE (Default) in bold.
DISPLAY_NAMES = {
    "Uniform-CV": "V-ECE (Uniform)",
    "Quantile-CV": "V-ECE (Quantile)",
    "Isotonic-CV": "V-ECE (Isotonic)",
    "KDE-CV": "V-ECE (KDE)",
    "TS-CV": "V-ECE (TS)",
    "Quadratic-CV": "V-ECE (Quadratic)",
    "Beta-CV": "V-ECE (Beta)",
    "SMS-CV": "V-ECE (SMS)",
    OURS: "V-ECE (Default)",
}
# Recalibration functions of the first experiment.
RECALIBRATOR_NAMES = {"Hist-uniform": "Uniform", "Hist-quantile": "Quantile"}

# The three families of estimators of the semi-synthetic benchmark.  Estimator names are
# colored by family in the figures; the suffix of each name (-ECE, -IS, -CV) carries the
# same information, so the color is never the only cue.
ESTIMATOR_CATEGORIES = {
    "ece": ("Uniform-ECE", "Quantile-ECE", "Isotonic-ECE", "Debiased-ECE", "Smooth-ECE", "KDE-ECE"),
    "is": ("Uniform-IS", "Quantile-IS", "Isotonic-IS"),
    "cv": (
        "Uniform-CV", "Quantile-CV", "Isotonic-CV", "KDE-CV", "TS-CV", "Quadratic-CV", "Beta-CV",
        "SMS-CV",
        OURS,
    ),
}
# Pastel tag backgrounds from matplotlib's tab20 (light shades), chosen away from the
# blue / aqua / orange of the metric markers; names stay in black ink on top of them.
CATEGORY_COLORS = {"ece": "#ff9896", "is": "#dbdb8d", "cv": "#c5b0d5"}
TAG_STYLE = dict(boxstyle="round,pad=0.12,rounding_size=0.2", edgecolor="none")
CATEGORY_LABELS = {
    "ece": "plug-in",
    "is": "in-sample variational",
    "cv": "out-of-sample variational",
}


def category(name: str) -> str | None:
    """Family of an estimator (``ece``, ``is`` or ``cv``), or None for other names."""
    return next((key for key, names in ESTIMATOR_CATEGORIES.items() if name in names), None)


def display_name(name: str, estimator_figure: bool = True) -> str:
    """Name of an estimator (or, with ``estimator_figure=False``, of a recalibration
    function) as printed in the figures."""
    shown = (DISPLAY_NAMES if estimator_figure else RECALIBRATOR_NAMES).get(name, name)
    return bold(shown) if name == OURS else shown


def color_by_category(texts, names) -> None:
    """Write the tick labels (or titles) ``texts`` of the estimators ``names`` on a pastel
    tag whose color gives the family of the estimator."""
    for text, name in zip(texts, names):
        key = category(name)
        if key is not None:
            text.set_bbox(dict(TAG_STYLE, facecolor=CATEGORY_COLORS[key]))
    if texts and hasattr(texts[0], "axes") and texts[0].axes is not None:
        texts[0].axes.tick_params(axis="y", pad=4.0)  # room between the tags and the panel


def category_key(figure, ncol: int = 3) -> None:
    """Key of the name colors, on one line below the figure (kept by ``bbox_inches="tight"``)."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], linestyle="none") for _ in CATEGORY_LABELS]
    key = figure.legend(
        handles,
        list(CATEGORY_LABELS.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        handlelength=0.0,
        handletextpad=0.0,
        columnspacing=1.5,
        borderaxespad=0.0,
        borderpad=0.0,
        fontsize=7,
    )
    for text, color in zip(key.get_texts(), CATEGORY_COLORS.values()):
        text.set_bbox(dict(TAG_STYLE, facecolor=color))


def bold(text: str) -> str:
    return rf"\textbf{{{text}}}" if USE_TEX else text


def apply_style() -> None:
    """Set the rcParams shared by every figure."""
    params = {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "axes.edgecolor": SECONDARY_INK,
        "axes.labelcolor": INK,
        "axes.titlepad": 3.0,
        "xtick.color": SECONDARY_INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "xtick.major.pad": 1.5,
        "ytick.major.size": 0.0,
        "ytick.major.pad": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "legend.frameon": False,
        "legend.handletextpad": 0.2,
        "legend.columnspacing": 0.8,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    }
    if USE_TEX:
        params.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            }
        )
    else:
        params.update(
            {
                "font.family": "serif",
                "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True,
            }
        )
    plt.rcParams.update(params)


def style_dot_axis(axis, n_rows: int, xlim: tuple[float, float]) -> None:
    """Recessive chrome for a horizontal dot plot with ``n_rows`` categories."""
    axis.set_xlim(*xlim)
    axis.set_ylim(n_rows - 0.5, -0.5)  # first row on top
    axis.set_yticks(range(n_rows))
    axis.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def row_guides(axis, n_rows: int, highlight: int | None = None) -> None:
    """Hairline guides under each row, and a light band behind the highlighted row."""
    for row in range(n_rows):
        axis.axhline(row, color=GRID, linewidth=0.4, zorder=0.5)
    if highlight is not None:
        axis.axhspan(highlight - 0.5, highlight + 0.5, color=HIGHLIGHT, alpha=0.08, lw=0, zorder=0)


def add_axes_inches(figure, left: float, bottom: float, width: float, height: float, **kwargs):
    """``figure.add_axes`` with a rectangle given in inches from the bottom-left corner."""
    figure_width, figure_height = figure.get_size_inches()
    rectangle = [left / figure_width, bottom / figure_height, width / figure_width, height / figure_height]
    return figure.add_axes(rectangle, **kwargs)


def legend_above(axis, n_columns: int) -> None:
    """Legend flush right on the line above ``axis``."""
    axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=n_columns,
        handlelength=0.9,
        borderaxespad=0.0,
        borderpad=0.25,
    )


def panel_header(axis, title: str, n_legend_columns: int | None = None) -> None:
    """Title flush left above the panel and, optionally, the legend flush right on its line.

    Call ``align_titles(figure)`` once the figure is laid out, so that the title starts at
    the left edge of the tick labels rather than at the left edge of the plotting area.
    """
    axis.set_title(title, loc="left", fontsize=8, pad=4.0)
    if n_legend_columns:
        legend_above(axis, n_legend_columns)


def align_titles(figure) -> None:
    """Move every left title of ``figure`` to the left edge of the tick labels.

    Panels stacked on the same left edge share the leftmost label edge among them, so their
    titles start at the same position whatever the length of their own labels.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    titled = [
        axis
        for axis in figure.axes
        if axis._left_title.get_text() and any(l.get_text() for l in axis.get_yticklabels())
    ]
    label_left = {
        axis: min(
            label.get_window_extent(renderer).x0
            for label in axis.get_yticklabels()
            if label.get_text()
        )
        for axis in titled
    }
    for axis in titled:
        box = axis.get_window_extent(renderer)
        column = [a for a in titled if abs(a.get_window_extent(renderer).x0 - box.x0) < 1.0]
        left = min(label_left[a] for a in column)
        # A panel may reserve room further left for its title (``axis.title_left_inches``).
        if getattr(axis, "title_left_inches", None) is not None:
            left = min(left, axis.title_left_inches * figure.dpi)
        axis._left_title.set_x((left - box.x0) / box.width)


def dodge(n_metrics: int, spread: float | None = None) -> list[float]:
    """Vertical offsets (in row units) that separate the metrics of one row."""
    if n_metrics == 1:
        return [0.0]
    if spread is None:
        spread = 0.24 if n_metrics == 2 else 0.36
    step = spread / (n_metrics - 1)
    return [-spread / 2 + i * step for i in range(n_metrics)]


def save_figure(figure, path: Path) -> None:
    """Save ``path`` (PDF, for the paper) and a PNG preview next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(figure)
