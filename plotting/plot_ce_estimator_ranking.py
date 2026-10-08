"""Which calibration error estimator ranks *models* correctly?

``plot_ce_estimators.py`` asks how close each estimator lands to the true calibration
error.  That is the right question when the number itself is reported, but most of the time
a calibration error is used comparatively: given several models trained on the same dataset,
which one is the best calibrated?  An estimator with a large but roughly constant bias is
useless for the first question and perfectly good for the second, so the two are scored
separately.

Every dataset in the benchmark carries the same eight learners (CatBoost, ExtraTrees,
LightGBM, LinearModel, NeuralNetFastAI, NeuralNetTorch, RandomForest, XGBoost).  For each
dataset we therefore rank those eight by the semi-synthetic true calibration error, rank
them again by what an estimator reports, and compare the two orderings.  Scores are averaged
over datasets, so every dataset weighs the same regardless of its scale.

Reported per metric and estimator:

* **``spearman``** -- rank correlation between the estimated and the true ordering of the
  eight models, computed on each label draw, averaged over the draws and then over
  datasets.  1 = the ordering is recovered exactly, 0 = no better
  than shuffling, negative = anti-correlated.  This is the criterion the figure is ordered by.
* **``pairwise_accuracy``** -- the share of the 28 model pairs per dataset the estimator
  orders the same way the truth does (a pair it ties counts as half), averaged over
  datasets.  The same information Kendall's
  tau carries, on a scale a reader can act on: 50% is a coin flip.
* **``top1_accuracy``** -- the share of datasets on which the estimator's best-calibrated
  model really is the best-calibrated one (shared among tied models).  Chance is 1/8 =
  12.5%.  This is the strictest
  and noisiest of the three, and the one that matters if the estimator is used to pick a
  single model.
* **``mean_sq_rank_error``** -- mean squared difference between the estimated and the true
  rank of a model.  Reported because it was asked for, and kept out of the figure because
  without ties it is an exact monotone transform of the Spearman correlation,
  ``(1 - spearman) * (n^2 - 1) / 6``, so it ranks the estimators identically.

``spearman_se`` is the standard error over datasets; differences smaller than a couple of
those are not worth reading into.

Outputs:

    {results_dir}/estimators/ce_estimator_model_ranking.csv
    {figure_dir}/ce_estimator_model_ranking.pdf   (and .png; default paper_figures/)

Usage
-----
    python -m plotting.plot_ce_estimator_ranking
    python -m plotting.plot_ce_estimator_ranking --modality binary
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.benchmark_calibration_errors import MODALITIES
from plotting.plot_ce_estimators import dot_panel, estimator_order, load_estimates, two_panel_grid
from plotting.style import (
    DISPLAYED_METRICS,
    OURS,
    PERCENT,
    align_titles,
    apply_style,
    display_name,
    legend_above,
    panel_header,
    category_key,
    color_by_category,
    row_guides,
    save_figure,
    style_dot_axis,
)

MIN_MODELS_PER_DATASET = 3  # a rank correlation over two models is a coin flip, not a score


def compare_orderings(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Agreement between the ordering of the models by ``estimate`` and by ``truth``."""
    n = len(estimate)
    # A pair tied by the estimate (e.g. two estimates clipped at 0) counts as half ordered,
    # as a coin flip would order it.
    concordant = [
        0.5
        if estimate[i] == estimate[j]
        else float(np.sign(estimate[i] - estimate[j]) == np.sign(truth[i] - truth[j]))
        for i, j in combinations(range(n), 2)
    ]
    # Several models tied for the smallest estimate share the credit of picking the best one.
    smallest = np.flatnonzero(estimate == np.min(estimate))
    estimated_rank = pd.Series(estimate).rank().to_numpy()
    true_rank = pd.Series(truth).rank().to_numpy()
    return {
        "spearman": float(spearmanr(estimate, truth).statistic),
        "pairwise_accuracy": float(np.mean(concordant)),
        # argmin: the best-calibrated model is the one with the smallest calibration error.
        "top1_accuracy": float(np.argmin(truth) in smallest) / len(smallest),
        "mean_sq_rank_error": float(np.mean((estimated_rank - true_rank) ** 2)),
    }


def score_datasets(estimates: pd.DataFrame) -> pd.DataFrame:
    """One row per (metric, estimator, dataset) holding the ordering scores for that dataset.

    The models are ordered within each label draw, which is what a user comparing models on
    one dataset would observe, and the scores are then averaged over the draws.
    """
    scores = []
    usable = estimates.dropna(subset=["estimate", "true"])
    if "draw" not in usable:
        usable = usable.assign(draw=0)
    for (metric, estimator, dataset, draw), group in usable.groupby(
        ["metric", "estimator", "dataset", "draw"]
    ):
        if len(group) < MIN_MODELS_PER_DATASET:
            continue
        scores.append(
            {
                "metric": metric,
                "estimator": estimator,
                "dataset": dataset,
                "draw": draw,
                "n_models": len(group),
                **compare_orderings(
                    group["estimate"].to_numpy(), group["true"].to_numpy()
                ),
            }
        )
    per_draw = pd.DataFrame(scores)
    return (
        per_draw.groupby(["metric", "estimator", "dataset"])
        .agg(
            n_models=("n_models", "first"),
            n_draws=("draw", "nunique"),
            spearman=("spearman", "mean"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            mean_sq_rank_error=("mean_sq_rank_error", "mean"),
        )
        .reset_index()
    )


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Average the per-dataset ordering scores, giving every dataset the same weight."""
    by_estimator = scores.groupby(["metric", "estimator"])
    summary = by_estimator[
        ["spearman", "pairwise_accuracy", "top1_accuracy", "mean_sq_rank_error"]
    ].mean()
    summary["spearman_se"] = by_estimator["spearman"].sem()
    summary["n_datasets"] = by_estimator["spearman"].count()
    summary["rank"] = (
        summary.groupby("metric")["spearman"].rank(method="min", ascending=False).astype(int)
    )
    return summary.reset_index()


def plot_ranking(
    summaries: dict[str, pd.DataFrame], n_datasets: dict[str, int], output_path: Path
) -> None:
    """Rank correlation (left, +/- one standard error) and pairwise accuracy (right)."""
    apply_style()
    figure, axes = two_panel_grid(list(summaries))

    for modality, summary in summaries.items():
        metrics = list(DISPLAYED_METRICS[modality])
        summary = summary[summary["metric"].isin(metrics)]
        pivot = lambda column: summary.pivot(
            index="estimator", columns="metric", values=column
        ).reindex(columns=metrics)
        correlation = pivot("spearman")
        order = estimator_order(correlation, ascending=False)  # best correlation on top
        correlation, error = correlation.loc[order], pivot("spearman_se").loc[order]
        accuracy = (100.0 * pivot("pairwise_accuracy")).loc[order]
        main, side = axes[modality]
        highlight = order.index(OURS) if OURS in order else None

        style_dot_axis(main, len(order), (0.0, 1.0))
        row_guides(main, len(order), highlight)
        dot_panel(main, correlation, modality, errors=error)
        main.set_yticklabels([display_name(name) for name in order])
        color_by_category(main.get_yticklabels(), order)
        main.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
        main.set_xlabel("Spearman correlation")
        panel_header(main, f"{modality.capitalize()}: {n_datasets[modality]} datasets")

        style_dot_axis(side, len(order), (50, 100))
        row_guides(side, len(order), highlight)
        dot_panel(side, accuracy, modality)
        side.tick_params(axis="y", labelleft=False)
        side.set_xticks([50, 75, 100])
        side.set_xlabel(PERCENT + " pairs ordered")
        legend_above(side, len(metrics))

    align_titles(figure)
    category_key(figure)
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
        "--figure_dir",
        type=Path,
        default=Path("paper_figures"),
        help="Directory for the figures of the paper (PDF, with PNG previews).",
    )
    args = parser.parse_args()

    summaries, n_datasets, all_scores = {}, {}, {}
    for modality in MODALITIES if args.modality is None else (args.modality,):
        estimates = load_estimates(modality, args.results_dir)
        if estimates.empty:
            print(f"{modality}: no results yet, skipping.")
            continue
        scores = score_datasets(estimates)
        summaries[modality] = summarize(scores)
        all_scores[modality] = scores
        n_datasets[modality] = scores["dataset"].nunique()
        print(f"{modality}: {n_datasets[modality]} datasets, {estimates['model'].nunique()} models each")

    if not summaries:
        raise SystemExit("No results found. Run `python -m experiments.benchmark_ce_estimators` first.")

    suffix = "" if args.modality is None else f"_{args.modality}"
    output_dir = args.results_dir / "estimators"
    summary = pd.concat(summaries, names=["modality"]).reset_index(level=0)
    summary_path = output_dir / f"ce_estimator_model_ranking{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    for modality, table in summaries.items():
        for metric in DISPLAYED_METRICS[modality]:
            rows = table[table["metric"] == metric].sort_values("rank")
            print(f"\n--- {modality}, {metric} (ranked by Spearman correlation) ---")
            print(
                rows[
                    [
                        "estimator",
                        "spearman",
                        "spearman_se",
                        "pairwise_accuracy",
                        "top1_accuracy",
                        "mean_sq_rank_error",
                        "n_datasets",
                    ]
                ].to_string(index=False, float_format=lambda v: f"{v:9.3f}")
            )
    print(f"\nWrote {summary_path}")

    figure_path = args.figure_dir / f"ce_estimator_model_ranking{suffix}.pdf"
    plot_ranking(summaries, n_datasets, figure_path)
    print(f"Wrote {figure_path} (and .png)")


if __name__ == "__main__":
    main()
