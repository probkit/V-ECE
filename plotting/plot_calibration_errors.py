"""Rank the recalibration models benchmarked by ``benchmark_calibration_errors.py``.

The variational estimator lower-bounds the true calibration error in expectation, so on a
given experiment the recalibration model producing the *largest* estimate is the one that
recovered the most calibration error.  We aggregate this over experiments with the average
rank (rank 1 = largest estimate; lower is better) and with the average and median
percentage of the largest estimate recovered (higher is better).

The script writes a summary table and a single-column figure of the average ranks:

    {results_dir}/calibration_error_ranks.csv
    {figure_dir}/calibration_error_ranks.pdf    (and .png; figure_dir defaults to paper_figures/)

Experiments on which every model agrees that ``f`` is (nearly) calibrated carry no signal
about the recalibration models, since all estimates are then noise around zero, so we drop
those whose largest estimate for the reference metric of the modality (``CE_|.|`` for
binary, ``CE_||.||_2`` for multiclass) falls below ``--min_ce`` (default ``1e-3``, as in
the paper).

Usage
-----
    python -m plotting.plot_calibration_errors
    python -m plotting.plot_calibration_errors --min_ce -1         # keep every experiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmark_calibration_errors import METRICS, MODALITIES, RECALIBRATORS
from plotting.style import (
    COLUMN_WIDTH,
    DISPLAYED_METRICS,
    METRIC_COLORS,
    METRIC_LABELS,
    METRIC_MARKER_SIZES,
    METRIC_MARKERS,
    MUTED_INK,
    OURS,
    REFERENCE_METRIC,
    align_titles,
    apply_style,
    display_name,
    dodge,
    panel_header,
    row_guides,
    save_figure,
    style_dot_axis,
)

# One experiment = one classifier whose calibration error we estimate.
EXPERIMENT_KEYS = ["dataset", "model"]


def load_estimates(modality: str, results_dir: Path) -> pd.DataFrame:
    """Calibration error estimates of every model, indexed by experiment and metric.

    Returns a long dataframe with columns ``dataset, model, metric, method, ce``.
    """
    estimates = []
    for method in RECALIBRATORS[modality]:
        path = results_dir / modality / f"{method}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing results for '{method}' ({modality}): {path}. "
                "Run `python -m experiments.benchmark_calibration_errors` first."
            )
        # `benchmark_calibration_errors.py` prefixes the metric columns with the method name.
        results = pd.read_csv(path)
        results = results[EXPERIMENT_KEYS + [f"{method}_{metric}" for metric in METRICS]]
        results.columns = EXPERIMENT_KEYS + list(METRICS)
        estimates.append(results.assign(method=method))

    return pd.concat(estimates).melt(
        id_vars=EXPERIMENT_KEYS + ["method"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="ce",
    )


def drop_calibrated_experiments(
    estimates: pd.DataFrame, modality: str, min_ce: float
) -> pd.DataFrame:
    """Drop experiments whose largest estimate for the reference metric is below ``min_ce``."""
    reference = estimates[estimates["metric"] == REFERENCE_METRIC[modality]]
    largest_ce = reference.groupby(EXPERIMENT_KEYS)["ce"].max()
    informative = largest_ce[largest_ce >= min_ce].index
    return estimates.set_index(EXPERIMENT_KEYS).loc[informative].reset_index()


def summarize(estimates: pd.DataFrame, min_ce: float) -> pd.DataFrame:
    """Average rank and percentage of the largest estimate, per method and metric.

    Both are computed within each (experiment, metric): rank 1 is the largest estimate, and
    the percentage is relative to that largest estimate.  Ranks are scale-free, but a ratio
    to a largest estimate that is itself close to zero is meaningless (and unbounded), so the
    percentage is only defined on the experiments whose largest estimate *for that metric*
    reaches ``min_ce``; ``n_experiments_pct`` reports how many of them there are.
    """
    per_experiment = estimates.groupby(EXPERIMENT_KEYS + ["metric"])["ce"]
    largest_ce = per_experiment.transform("max")
    estimates = estimates.assign(
        rank=per_experiment.rank(ascending=False),
        percentage_of_max=np.where(
            largest_ce >= max(min_ce, np.finfo(float).tiny),
            100.0 * estimates["ce"] / largest_ce,
            np.nan,
        ),
    )
    by_method = estimates.groupby(["metric", "method"])
    summary = by_method[["rank", "percentage_of_max"]].mean()
    summary.columns = ["avg_rank", "avg_pct_of_max_ce"]
    summary["median_pct_of_max_ce"] = by_method["percentage_of_max"].median()
    summary["n_experiments"] = by_method["rank"].size()
    summary["n_experiments_pct"] = by_method["percentage_of_max"].count()
    return summary.reset_index()


def plot_ranks(summaries: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Single-column dot plot of the average ranks, one panel per modality (lower is better)."""
    apply_style()
    heights = [len(RECALIBRATORS[modality]) + 1.2 for modality in summaries]
    figure, axes = plt.subplots(
        len(summaries),
        1,
        figsize=(COLUMN_WIDTH, 0.17 * sum(heights) + 0.45),
        gridspec_kw={"height_ratios": heights, "hspace": 0.62},
    )
    axes = np.atleast_1d(axes)

    for axis, (modality, summary) in zip(axes, summaries.items()):
        metrics = DISPLAYED_METRICS[modality]
        summary = summary[summary["metric"].isin(metrics)]
        n_experiments = summary["n_experiments"].iloc[0]
        ranks = summary.pivot(index="method", columns="metric", values="avg_rank")[list(metrics)]
        ranks = ranks.loc[ranks.mean(axis=1).sort_values().index]  # best method on top
        n_methods = len(ranks)

        style_dot_axis(axis, n_methods, (0.75, n_methods + 0.25))
        row_guides(axis, n_methods, highlight=list(ranks.index).index(OURS))
        # A method that is neither better nor worse than the others sits at this rank.
        axis.axvline((n_methods + 1) / 2, color=MUTED_INK, linewidth=0.6, zorder=0.6)
        for offset, metric in zip(dodge(len(metrics)), metrics):
            axis.plot(
                ranks[metric],
                np.arange(n_methods) + offset,
                linestyle="none",
                marker=METRIC_MARKERS[metric],
                markersize=METRIC_MARKER_SIZES[metric],
                markerfacecolor=METRIC_COLORS[metric],
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=METRIC_LABELS[modality][metric],
                zorder=3,
            )
        axis.set_yticklabels([display_name(name, estimator_figure=False) for name in ranks.index])
        axis.set_xticks(range(1, n_methods + 1))
        panel_header(axis, f"{modality.capitalize()}: {n_experiments} classifiers", len(metrics))

    axes[-1].set_xlabel("average rank of the estimate (1 = largest)")
    align_titles(figure)
    save_figure(figure, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory holding the benchmark results, also used for the outputs.",
    )
    parser.add_argument(
        "--min_ce",
        type=float,
        default=1e-3,
        help="Drop experiments whose largest reference-metric estimate is below this value.",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("paper_figures"),
        help="Directory for the figures of the paper (PDF, with PNG previews).",
    )
    args = parser.parse_args()

    summaries = {}
    for modality in MODALITIES:
        estimates = load_estimates(modality, args.results_dir)
        n_all = estimates[EXPERIMENT_KEYS].drop_duplicates().shape[0]
        estimates = drop_calibrated_experiments(estimates, modality, args.min_ce)
        n_kept = estimates[EXPERIMENT_KEYS].drop_duplicates().shape[0]
        print(
            f"{modality}: kept {n_kept}/{n_all} experiments with "
            f"{REFERENCE_METRIC[modality]} calibration error >= {args.min_ce}"
        )
        summaries[modality] = summarize(estimates, args.min_ce)

    summary = pd.concat(summaries, names=["modality"]).reset_index(level=0)
    summary_path = args.results_dir / "calibration_error_ranks.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{summary.to_string(index=False)}\n\nWrote {summary_path}")

    figure_path = args.figure_dir / "calibration_error_ranks.pdf"
    plot_ranks(summaries, figure_path)
    print(f"Wrote {figure_path} (and .png)")


if __name__ == "__main__":
    main()
