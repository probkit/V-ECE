"""Score calibration error estimators against a semi-synthetic ground truth.

Reads the CSVs written by ``benchmark_ce_estimators.py``, where every row holds both the
true calibration error of an experiment and the value an estimator reported for it, and
scores each estimator by how far it lands from the truth.

Absolute errors cannot be averaged across experiments as they are: the true calibration
error spans two orders of magnitude over the benchmark (median ~0.012 for binary L1, ~4e-4
for the binary squared metric), so a raw MAE is decided by a handful of badly calibrated
models.  Dividing by the truth instead makes the near-zero experiments explode.  We
therefore normalise each experiment before averaging, in two ways that need no cutoff and
no tuning constant:

* **``norm_abs_error``** -- on each experiment, divide every estimator's absolute error by
  the largest absolute error any estimator made there, then average.  The worst estimator
  on an experiment scores 1, an estimator half as far off scores 0.5.  Bounded in [0, 1],
  scale-free, and unlike a rank it keeps the magnitude of the gaps.  This is the criterion
  the figure is ordered by.
* **``avg_rank``** -- the average of the per-experiment ranks by absolute error (rank 1 =
  closest).  Also scale-free, but it discards how large the gaps are.

Both are relative to the pool of estimators being compared: adding a very bad estimator
changes everyone's ``norm_abs_error``, and the binary and multiclass pools differ.  ``nmae``
is reported alongside as a pool-independent alternative -- ``sum_i |estimate_i - true_i| /
sum_i true_i``, which is exactly the mean of the per-experiment relative errors weighted by
the true calibration error.  ``median_ratio`` is the median of ``estimate / true``, the
share of the calibration error an estimator typically reports.

The main figure pairs the ranking with the direction of the error: the share of
experiments on which each estimator lands above rather than below the truth.  A second
figure plots ``estimate / true`` against the truth for a few estimators.

Outputs:

    {results_dir}/estimators/ce_estimator_accuracy.csv
    {figure_dir}/ce_estimator_accuracy.pdf               (and .png)
    {figure_dir}/ce_estimator_ratios.pdf                 (and .png; default paper_figures/)

Usage
-----
    python -m plotting.plot_ce_estimators
    python -m plotting.plot_ce_estimators --min_true_ce 0.01   # optional sensitivity check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmark_calibration_errors import METRICS, MODALITIES
from experiments.benchmark_ce_estimators import ESTIMATORS
from plotting.style import (
    DISPLAYED_METRICS,
    GRID,
    METRIC_COLORS,
    METRIC_LABELS,
    METRIC_MARKER_SIZES,
    METRIC_MARKERS,
    COLUMN_WIDTH,
    MUTED_INK,
    OURS,
    PERCENT,
    REFERENCE_METRIC,
    TEXT_WIDTH,
    add_axes_inches,
    align_titles,
    apply_style,
    display_name,
    dodge,
    legend_above,
    panel_header,
    category_key,
    color_by_category,
    row_guides,
    save_figure,
    style_dot_axis,
)

EXPERIMENT_KEYS = ["dataset", "model"]

# Estimators shown in the ratio figure: ours, the best competing variational estimator,
# and the plug-in / binned estimators most often reported.
RATIO_ESTIMATORS = {
    "binary": ("WS-CatBoost", "Quadratic-CV", "Quantile-ECE", "KDE-ECE"),
    "multiclass": ("WS-CatBoost", "SMS-CV", "TS-CV", "KDE-ECE"),
}
RATIO_LIMITS = (-0.5, 3.0)


def load_estimates(modality: str, results_dir: Path) -> pd.DataFrame:
    """Long dataframe with one row per experiment, label draw, metric and estimator.

    Columns ``dataset, model, draw, metric, estimator, estimate, true, time, test_size``.
    Each label draw is a separate semi-synthetic dataset, so every statistic is computed per
    draw and then averaged: this scores the estimates a user would actually obtain, rather
    than an average of estimates over draws.  Results written before the benchmark stored
    the individual draws fall back to their mean over draws, as a single draw 0.
    """
    estimates, missing = [], []
    for estimator in ESTIMATORS[modality]:
        path = results_dir / "estimators" / modality / f"{estimator}.csv"
        if not path.exists():
            # Report what is available rather than refusing: the benchmark writes one CSV
            # per estimator, so a partial run is a normal intermediate state.
            missing.append(estimator)
            continue
        results = pd.read_csv(path)
        for metric in METRICS:
            draws = sorted(
                int(column.rsplit("draw", 1)[1])
                for column in results.columns
                if column.startswith(f"{estimator}_{metric}_draw") and column[-1].isdigit()
            )
            columns = {d: f"{estimator}_{metric}_draw{d}" for d in draws} or {0: f"{estimator}_{metric}"}
            for draw, column in columns.items():
                estimates.append(
                    pd.DataFrame(
                        {
                            "dataset": results["dataset"],
                            "model": results["model"],
                            "draw": draw,
                            "metric": metric,
                            "estimator": estimator,
                            "estimate": results[column],
                            "true": results[f"true_{metric}"],
                            "time": results[f"{estimator}_time"],
                            "test_size": results["test_size"],
                        }
                    )
                )
    if missing:
        print(f"[warning] {modality}: no results yet for {', '.join(missing)}; skipping them.")
    if not estimates:
        return pd.DataFrame(
            columns=EXPERIMENT_KEYS + ["draw", "metric", "estimator", "estimate", "true"]
        )
    return pd.concat(estimates, ignore_index=True)


def drop_calibrated_experiments(
    estimates: pd.DataFrame, modality: str, min_true_ce: float
) -> pd.DataFrame:
    """Keep experiments whose true reference-metric calibration error reaches ``min_true_ce``."""
    reference = estimates[estimates["metric"] == REFERENCE_METRIC[modality]]
    true_ce = reference.groupby(EXPERIMENT_KEYS)["true"].first()
    informative = true_ce[true_ce >= min_true_ce].index
    return estimates.set_index(EXPERIMENT_KEYS).loc[informative].reset_index()


def summarize(estimates: pd.DataFrame) -> pd.DataFrame:
    """Per metric and estimator: how far from the truth it lands, and in which direction.

    Every (experiment, label draw) pair is one realization of what a user would observe: the
    absolute error and its normalisation by the worst estimator are computed per pair, and
    all statistics average over pairs (equivalently, over draws then experiments, since each
    experiment has the same number of draws).

    Estimators that define only some of the metrics (the debiased ECE is specific to the
    squared error, the smooth ECE to the L1 one) report NaN elsewhere; those
    metric/estimator pairs are dropped, and the per-experiment normalisations are taken among
    the estimators that do define the metric.
    """
    estimates = estimates.dropna(subset=["estimate"])
    estimates = estimates.assign(abs_error=(estimates["estimate"] - estimates["true"]).abs())
    per_experiment = estimates.groupby(EXPERIMENT_KEYS + ["draw", "metric"])["abs_error"]
    worst = per_experiment.transform("max")
    estimates = estimates.assign(
        rank_in_experiment=per_experiment.rank(),
        error_vs_worst=np.where(worst > 0, estimates["abs_error"] / worst, np.nan),
        ratio=estimates["estimate"] / estimates["true"],
        overestimates=estimates["estimate"] > estimates["true"],
        time_per_1k=1000.0 * estimates["time"] / estimates["test_size"],
    )

    by_estimator = estimates.groupby(["metric", "estimator"])
    summary = pd.DataFrame(
        {
            "norm_abs_error": by_estimator["error_vs_worst"].mean(),
            "avg_rank": by_estimator["rank_in_experiment"].mean(),
            # sum |estimate - true| / sum true, as a percentage.
            "nmae": 100.0 * by_estimator["abs_error"].sum() / by_estimator["true"].sum(),
            "mae": by_estimator["abs_error"].mean(),
            "median_ratio": by_estimator["ratio"].median(),
            "overestimation_rate": by_estimator["overestimates"].mean(),
            "time_per_1k": by_estimator["time_per_1k"].mean(),
            "n_experiments": by_estimator.apply(
                lambda g: len(g[EXPERIMENT_KEYS].drop_duplicates()), include_groups=False
            ),
            "n_draws": by_estimator["draw"].nunique(),
        }
    )
    summary["underestimation_rate"] = 1.0 - summary["overestimation_rate"]
    summary["rank"] = (
        summary.groupby("metric")["norm_abs_error"].rank(method="min").astype(int)
    )
    return summary.reset_index()


# Metric that orders the rows of the accuracy figure.  Estimators that do not define it
# (the smooth ECE only targets the L1 error) come last, ordered by their other metrics.
ACCURACY_ORDER_METRIC = "brier"


def estimator_order_by(values: pd.DataFrame, metric: str) -> list[str]:
    """Estimators sorted by ``values[metric]`` (best first), those undefined for it last."""
    keys = pd.DataFrame(
        {
            "undefined": values[metric].isna(),
            "key": values[metric].fillna(np.inf),
            "fallback": values.mean(axis=1),
        }
    )
    return list(keys.sort_values(["undefined", "key", "fallback"]).index)


def estimator_order(values: pd.DataFrame, ascending: bool = True) -> list[str]:
    """Estimators sorted by their mean over the metrics each of them defines (best first)."""
    return list(values.mean(axis=1).sort_values(ascending=ascending).index)


def dot_panel(axis, values: pd.DataFrame, modality: str, errors: pd.DataFrame | None = None):
    """One dot per (estimator, metric), rows in the order of ``values.index``."""
    metrics = list(values.columns)
    positions = np.arange(len(values))
    for offset, metric in zip(dodge(len(metrics)), metrics):
        x = values[metric].to_numpy(dtype=float)
        defined = np.isfinite(x)
        if errors is not None:
            axis.errorbar(
                x[defined],
                positions[defined] + offset,
                xerr=errors[metric].to_numpy(dtype=float)[defined],
                fmt="none",
                ecolor=METRIC_COLORS[metric],
                elinewidth=0.6,
                capsize=0,
                alpha=0.7,
                zorder=2,
            )
        axis.plot(
            x[defined],
            positions[defined] + offset,
            linestyle="none",
            marker=METRIC_MARKERS[metric],
            markersize=METRIC_MARKER_SIZES[metric],
            markerfacecolor=METRIC_COLORS[metric],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=METRIC_LABELS[modality][metric],
            zorder=3,
        )


# Horizontal layout of the two blocks (inches): the binary block on the left, the
# multiclass one on the right, each made of the estimator names, a main panel and a narrow
# side panel.  Rows keep a constant pitch so that dots have the same spacing everywhere.
BLOCKS = {
    "binary": {"left": 0.0, "labels": 1.05, "main": 1.35, "side": 0.75, "gap": 0.15},
    "multiclass": {"left": 3.45, "labels": 1.05, "main": 1.2, "side": 0.7, "gap": 0.2},
}
ROW_PITCH = {"binary": 0.145, "multiclass": 0.21}
PANEL_GAP, HEADER, FOOTER = 0.15, 0.24, 0.36


def two_panel_grid(modalities) -> tuple[plt.Figure, dict]:
    """A (main, side) pair of dot-plot axes per modality, blocks side by side, top-aligned."""
    n_rows = {modality: len(ESTIMATORS[modality]) for modality in modalities}
    body = max(n_rows[m] * ROW_PITCH[m] for m in modalities)
    height = HEADER + body + FOOTER
    figure = plt.figure(figsize=(TEXT_WIDTH, height))
    axes = {}
    for modality in modalities:
        block = BLOCKS[modality]
        panel_height = n_rows[modality] * ROW_PITCH[modality]
        bottom = height - HEADER - panel_height
        left = block["left"] + block["labels"]
        main = add_axes_inches(figure, left, bottom, block["main"], panel_height)
        side = add_axes_inches(
            figure, left + block["main"] + block.get("gap", PANEL_GAP), bottom, block["side"],
            panel_height,
            sharey=main,
        )
        main.title_left_inches = block["left"]
        axes[modality] = (main, side)
    return figure, axes


# Single-column layout of the accuracy figure: one block per modality, stacked vertically.
STACKED = {"labels": 1.05, "main": 1.35, "side": 0.7, "gap": 0.15}
STACKED_PITCH = {"binary": 0.13, "multiclass": 0.17}


def stacked_grid(modalities) -> tuple[plt.Figure, dict]:
    """A (main, side) pair of dot-plot axes per modality, one block above the other."""
    heights = {m: len(ESTIMATORS[m]) * STACKED_PITCH[m] for m in modalities}
    figure = plt.figure(
        figsize=(COLUMN_WIDTH, sum(heights.values()) + len(heights) * (HEADER + FOOTER))
    )
    axes, top = {}, figure.get_size_inches()[1]
    for modality in modalities:
        bottom = top - HEADER - heights[modality]
        left = STACKED["labels"]
        main = add_axes_inches(figure, left, bottom, STACKED["main"], heights[modality])
        side = add_axes_inches(
            figure, left + STACKED["main"] + STACKED["gap"], bottom, STACKED["side"],
            heights[modality], sharey=main,
        )
        axes[modality] = (main, side)
        top = bottom - FOOTER
    return figure, axes


def plot_accuracy(
    summaries: dict[str, pd.DataFrame], n_experiments: dict[str, int], output_path: Path
) -> None:
    """Normalised absolute error (left) and share of overestimates (right), one modality
    above the other in a single column.  Rows are sorted by the normalised absolute error
    on the squared calibration error (``ACCURACY_ORDER_METRIC``)."""
    apply_style()
    figure, axes = stacked_grid(list(summaries))

    for modality, summary in summaries.items():
        metrics = list(DISPLAYED_METRICS[modality])
        summary = summary[summary["metric"].isin(metrics)]
        pivot = lambda column: summary.pivot(
            index="estimator", columns="metric", values=column
        ).reindex(columns=metrics)
        error = pivot("norm_abs_error")
        order = estimator_order_by(error, ACCURACY_ORDER_METRIC)
        error, over = error.loc[order], 100.0 * pivot("overestimation_rate").loc[order]
        main, side = axes[modality]
        highlight = order.index(OURS) if OURS in order else None

        style_dot_axis(main, len(order), (0.0, 1.0))
        row_guides(main, len(order), highlight)
        dot_panel(main, error, modality)
        main.set_yticklabels([display_name(name) for name in order])
        color_by_category(main.get_yticklabels(), order)
        main.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
        main.set_xlabel("normalized absolute error")
        panel_header(main, f"{modality.capitalize()}: {n_experiments[modality]} classifiers")

        style_dot_axis(side, len(order), (-5, 105))
        row_guides(side, len(order), highlight)
        side.axvline(50, color=MUTED_INK, linewidth=0.6, zorder=0.6)
        dot_panel(side, over, modality)
        side.tick_params(axis="y", labelleft=False)
        side.set_xticks([0, 50, 100])
        side.set_xlabel(PERCENT + " above truth")
        legend_above(side, len(metrics))

    align_titles(figure)
    category_key(figure)
    save_figure(figure, output_path)


def plot_ratios(estimates: dict[str, pd.DataFrame], output_path: Path) -> None:
    """``estimate / true`` against ``true`` (log scale) for a few estimators.

    One row per modality, for the reference metric; each point is one experiment and label
    draw.  The line at 1 is a perfect estimate, points below it underestimate.  Ratios outside ``RATIO_LIMITS`` are drawn on the
    boundary with a triangle.
    """
    apply_style()
    n_columns = max(len(RATIO_ESTIMATORS[modality]) for modality in estimates)
    figure, axes = plt.subplots(
        len(estimates),
        n_columns,
        figsize=(TEXT_WIDTH, 1.55 * len(estimates) + 0.1),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    low, high = RATIO_LIMITS
    for row, (modality, table) in enumerate(estimates.items()):
        metric = REFERENCE_METRIC[modality]
        symbol = METRIC_LABELS[modality][metric].strip("$")
        table = table[table["metric"] == metric]
        for column, name in enumerate(RATIO_ESTIMATORS[modality]):
            axis = axes[row, column]
            rows = table[table["estimator"] == name]
            ratio = (rows["estimate"] / rows["true"]).to_numpy()
            truth = rows["true"].to_numpy()
            axis.axhline(1.0, color=MUTED_INK, linewidth=0.7, zorder=1)
            for mask, y, marker, size in (
                ((ratio >= low) & (ratio <= high), ratio, "o", 2.0),
                (ratio > high, np.full(len(ratio), high), "^", 2.8),
                (ratio < low, np.full(len(ratio), low), "v", 2.8),
            ):
                axis.plot(
                    truth[mask],
                    y[mask],
                    linestyle="none",
                    marker=marker,
                    markersize=size,
                    markerfacecolor=METRIC_COLORS[metric],
                    markeredgecolor="white",
                    markeredgewidth=0.25,
                    alpha=0.85,
                    zorder=3,
                )
            axis.set_xscale("log")
            axis.set_ylim(low - 0.12, high + 0.12)
            axis.set_yticks([0, 1, 2, 3])
            axis.grid(color=GRID, linewidth=0.4)
            axis.spines[["left"]].set_visible(True)
            axis.tick_params(axis="y", length=2.0)
            title = axis.set_title(display_name(name), fontsize=8)
            color_by_category([title], [name])
            axis.set_xlabel(f"true ${symbol}$ ({modality})", labelpad=1.0)
        estimate = symbol.replace(r"\mathrm{CE}", r"\widehat{\mathrm{CE}}")
        axes[row, 0].set_ylabel(rf"${estimate} \,/\, {symbol}$")
    save_figure(figure, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        default=None,
        help="Report a single modality. Omit to report every one that has results.",
    )
    parser.add_argument(
        "--min_true_ce",
        type=float,
        default=0.0,
        help="Optional sensitivity check: drop experiments whose true reference-metric "
        "calibration error is below this. The default keeps every experiment.",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("paper_figures"),
        help="Directory for the figures of the paper (PDF, with PNG previews).",
    )
    args = parser.parse_args()

    summaries, n_experiments, all_estimates = {}, {}, {}
    for modality in MODALITIES if args.modality is None else (args.modality,):
        estimates = load_estimates(modality, args.results_dir)
        if estimates.empty:
            print(f"{modality}: no results yet, skipping.")
            continue
        n_all = estimates[EXPERIMENT_KEYS].drop_duplicates().shape[0]
        estimates = drop_calibrated_experiments(estimates, modality, args.min_true_ce)
        n_kept = estimates[EXPERIMENT_KEYS].drop_duplicates().shape[0]
        print(
            f"{modality}: kept {n_kept}/{n_all} experiments with true "
            f"{REFERENCE_METRIC[modality]} calibration error >= {args.min_true_ce}"
        )
        summaries[modality] = summarize(estimates)
        n_experiments[modality] = n_kept
        all_estimates[modality] = estimates

    if not summaries:
        raise SystemExit("No results found. Run `python -m experiments.benchmark_ce_estimators` first.")
    summary = pd.concat(summaries, names=["modality"]).reset_index(level=0)
    output_dir = args.results_dir / "estimators"
    suffix = "" if args.modality is None else f"_{args.modality}"
    summary_path = output_dir / f"ce_estimator_accuracy{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    for modality, table in summaries.items():
        for metric in DISPLAYED_METRICS[modality]:
            rows = table[table["metric"] == metric].sort_values("rank")
            print(f"\n--- {modality}, {metric} (ranked by normalised absolute error) ---")
            print(
                rows[
                    [
                        "estimator",
                        "norm_abs_error",
                        "avg_rank",
                        "nmae",
                        "median_ratio",
                        "overestimation_rate",
                        "n_experiments",
                    ]
                ].to_string(index=False, float_format=lambda v: f"{v:9.3f}")
            )
    print(f"\nWrote {summary_path}")

    figure_path = args.figure_dir / f"ce_estimator_accuracy{suffix}.pdf"
    plot_accuracy(summaries, n_experiments, figure_path)
    ratio_path = args.figure_dir / f"ce_estimator_ratios{suffix}.pdf"
    plot_ratios(all_estimates, ratio_path)
    print(f"Wrote {figure_path} and {ratio_path} (and .png)")


if __name__ == "__main__":
    main()
