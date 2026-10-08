"""Write the LaTeX tables of the paper's appendix from the benchmark results.

Run after the three plotting scripts, which write the summaries this script reads:

    results/calibration_error_ranks.csv            (plotting/plot_calibration_errors.py)
    results/estimators/ce_estimator_accuracy.csv   (plotting/plot_ce_estimators.py)
    results/estimators/ce_estimator_model_ranking.csv (plotting/plot_ce_estimator_ranking.py)

Outputs one ``.tex`` file per table in ``paper_tables/`` (``--table_dir``):

* ``datasets.tex``                 -- the TabRepo datasets and their split sizes;
* ``recalibrators.tex``            -- first experiment: average rank, share of wins, runtime;
* ``estimators_{modality}.tex``    -- second experiment: accuracy of every estimator;
* ``ranking_{modality}.tex``       -- second experiment: ordering of the eight models.

In the accuracy tables a dagger marks the estimators whose per-classifier absolute errors
are *not* significantly different from those of V-ECE (two-sided Wilcoxon signed-rank
test, level 5%), and the best value of each column is set in bold.

Usage
-----
    python -m plotting.make_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.benchmark_calibration_errors import MODALITIES, RECALIBRATORS
from plotting.plot_calibration_errors import drop_calibrated_experiments as drop_uninformative
from plotting.plot_calibration_errors import load_estimates as load_recalibrator_estimates
from plotting.plot_ce_estimators import estimator_order, load_estimates
from plotting.style import DISPLAYED_METRICS, DISPLAY_NAMES, METRIC_LABELS, OURS, RECALIBRATOR_NAMES

SIGNIFICANCE_LEVEL = 0.05
EXPERIMENT_KEYS = ["dataset", "model"]


def name_cell(name: str, names: dict[str, str] = DISPLAY_NAMES) -> str:
    """Name of an estimator (or recalibration function) as in the figures; ours in bold."""
    shown = names.get(name, name)
    return rf"\textbf{{{shown}}}" if name == OURS else shown


def fmt(value: float, digits: int) -> str:
    """Fixed-point number, with a math minus sign and ``--`` for undefined values."""
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}".replace("-", "$-$")


def fmt_time(milliseconds: float) -> str:
    if milliseconds >= 100:
        return f"{milliseconds:.0f}"
    if milliseconds >= 1:
        return f"{milliseconds:.1f}"
    return f"{milliseconds:.2f}"


def bold_best(column: pd.Series, cells: pd.Series, best: str) -> pd.Series:
    """Bold the cells of ``column`` holding its best value (``best`` is ``min`` or ``max``)."""
    values = column.astype(float)
    target = values.min() if best == "min" else values.max()
    return cells.where(~np.isclose(values, target), r"\textbf{" + cells + "}")


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def dataset_table(data_dir: Path) -> list[str]:
    """Binary and multiclass datasets side by side, with their split sizes."""
    tables = {}
    for modality in MODALITIES:
        experiments = pd.read_csv(data_dir / f"experiments_{modality}.csv", index_col=0)
        columns = ["cal_size", "test_size"] + (["n_classes"] if modality == "multiclass" else [])
        tables[modality] = experiments.groupby("dataset")[columns].first().sort_index(
            key=lambda index: index.str.lower()
        )
    n_rows = max(len(table) for table in tables.values())
    escape = lambda name: name.replace("_", r"\_")
    lines = [
        r"\begin{tabular}{lrr @{\hskip 8mm} lrrr}",
        r"\toprule",
        r"\multicolumn{3}{c}{Binary} & \multicolumn{4}{c}{Multiclass} \\",
        r"\cmidrule(r{8mm}){1-3} \cmidrule(l){4-7}",
        r"Dataset & $n_{\mathrm{cal}}$ & $n$ & Dataset & $n_{\mathrm{cal}}$ & $n$ & $k$ \\",
        r"\midrule",
    ]
    for i in range(n_rows):
        cells = []
        for modality in MODALITIES:
            table = tables[modality]
            width = 3 if modality == "binary" else 4
            if i < len(table):
                name, row = table.index[i], table.iloc[i]
                cells += [r"\texttt{" + escape(name) + "}"] + [f"{int(v):,}" for v in row]
            else:
                cells += [""] * width
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


# ==========================================
# FIRST EXPERIMENT
# ==========================================


def recalibrator_table(results_dir: Path, min_ce: float) -> list[str]:
    summary = pd.read_csv(results_dir / "calibration_error_ranks.csv")
    lines = [
        r"\begin{tabular}{ll ccc ccc c}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Average rank $\downarrow$} "
        r"& \multicolumn{3}{c}{Largest estimate (\%) $\uparrow$} & Time \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"& Recalibrator & $L_1$ & $L_2$ & squared & $L_1$ & $L_2$ & squared & [ms/1k] \\",
        r"\midrule",
    ]
    for modality in MODALITIES:
        estimates = drop_uninformative(
            load_recalibrator_estimates(modality, results_dir), modality, min_ce
        )
        # A win is shared equally among the methods tied for the largest estimate (e.g. all
        # estimates clipped at 0), so that the shares of a metric sum to 100%.
        groups = estimates.groupby(EXPERIMENT_KEYS + ["metric"])["ce"]
        is_max = estimates["ce"] == groups.transform("max")
        credit = is_max / is_max.groupby([estimates[k] for k in EXPERIMENT_KEYS + ["metric"]]).transform("sum")
        wins = 100.0 * (
            estimates.assign(credit=credit).groupby(["metric", "method"])["credit"].sum().unstack()
            / groups.ngroup().groupby(estimates["metric"]).nunique().to_numpy()[:, None]
        )
        table = summary[summary["modality"] == modality].pivot(
            index="method", columns="metric", values="avg_rank"
        )
        order = list(table[list(DISPLAYED_METRICS[modality])].mean(axis=1).sort_values().index)
        columns = ["l1", "l2", "brier"]
        shown = [metric if metric in DISPLAYED_METRICS[modality] else None for metric in columns]
        rank_cells = {
            metric: bold_best(
                table.loc[order, metric], table.loc[order, metric].map(lambda v: f"{v:.2f}"), "min"
            )
            for metric in columns
            if metric in DISPLAYED_METRICS[modality]
        }
        win_table = wins.T.reindex(index=order)
        win_cells = {
            metric: bold_best(
                win_table[metric].fillna(0.0), win_table[metric].fillna(0.0).map(lambda v: f"{v:.0f}"), "max"
            )
            for metric in columns
            if metric in DISPLAYED_METRICS[modality]
        }
        times = {}
        for method in RECALIBRATORS[modality]:
            results = pd.read_csv(results_dir / modality / f"{method}.csv")
            times[method] = float(np.mean(1000.0 * 1000.0 * results[f"{method}_time"] / results["test_size"]))
        n = int(summary.loc[summary["modality"] == modality, "n_experiments"].iloc[0])
        for i, method in enumerate(order):
            label = (
                rf"\multirow{{{len(order)}}}{{*}}{{\rotatebox{{90}}{{{modality.capitalize()}}}}}"
                if i == 0
                else ""
            )
            name = name_cell(method, RECALIBRATOR_NAMES)
            cells = [rank_cells[m][method] if m else "--" for m in shown]
            cells += [win_cells[m][method] if m else "--" for m in shown]
            lines.append(f"{label} & {name} & " + " & ".join(cells) + f" & {fmt_time(times[method])} \\\\")
        lines.append(r"\midrule" if modality != MODALITIES[-1] else r"\bottomrule")
        print(f"recalibrators/{modality}: {n} classifiers")
    lines.append(r"\end{tabular}")
    return lines


# ==========================================
# SECOND EXPERIMENT
# ==========================================


def significance(estimates: pd.DataFrame, metric: str) -> dict[str, float]:
    """Wilcoxon p-value of |error| of every estimator against the one of V-ECE.

    Absolute errors are computed on each label draw and averaged over the draws of an
    experiment; the test then pairs the estimators by experiment.
    """
    rows = estimates[estimates["metric"] == metric].dropna(subset=["estimate"])
    errors = (rows["estimate"] - rows["true"]).abs()
    wide = rows.assign(error=errors).pivot_table(
        index=EXPERIMENT_KEYS, columns="estimator", values="error", aggfunc="mean"
    )
    return {
        name: float(wilcoxon(wide[name], wide[OURS]).pvalue) if name != OURS else np.nan
        for name in wide.columns
    }


def accuracy_table(results_dir: Path, modality: str) -> list[str]:
    summary = pd.read_csv(results_dir / "estimators" / "ce_estimator_accuracy.csv")
    summary = summary[summary["modality"] == modality]
    estimates = load_estimates(modality, results_dir)
    metrics = list(DISPLAYED_METRICS[modality])
    pivot = lambda column: summary.pivot(index="estimator", columns="metric", values=column)
    error = pivot("norm_abs_error")[metrics]
    order = estimator_order(error)

    stats = [
        ("norm_abs_error", 2, "min", r"NAE $\downarrow$"),
        ("nmae", 0, "min", r"NMAE $\downarrow$"),
        ("median_ratio", 2, None, r"ratio"),
        ("overestimation_rate", 0, None, r"over"),
    ]
    header_metrics = " & ".join(
        rf"\multicolumn{{{len(stats)}}}{{c}}{{{METRIC_LABELS[modality][m]}}}" for m in metrics
    )
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + i * len(stats)}-{1 + (i + 1) * len(stats)}}}"
        for i in range(len(metrics))
    )
    lines = [
        r"\begin{tabular}{l " + " ".join(["c" * len(stats)] * len(metrics)) + " r}",
        r"\toprule",
        f"& {header_metrics} & Time \\\\",
        rules,
        "Estimator & " + " & ".join([s[3] for s in stats] * len(metrics)) + r" & [ms/1k] \\",
        r"\midrule",
    ]
    columns = {}
    for metric in metrics:
        p_values = significance(estimates, metric)
        for column, digits, best, _ in stats:
            values = pivot(column)[metric].reindex(order)
            scale = 100.0 if column == "overestimation_rate" else 1.0
            cells = (values * scale).map(lambda v: fmt(v, digits))
            if best is not None:
                defined = values.notna()
                cells[defined] = bold_best(values[defined], cells[defined], best)
            if column == "norm_abs_error":
                tied = [
                    name
                    for name in order
                    if np.isfinite(values[name])
                    and name in p_values
                    and p_values[name] >= SIGNIFICANCE_LEVEL
                ]
                cells[tied] = cells[tied] + r"$^\dagger$"
            columns[(metric, column)] = cells
    times = summary.groupby("estimator")["time_per_1k"].max().reindex(order) * 1000.0
    for name in order:
        cells = [columns[(m, c)][name] for m in metrics for c, *_ in stats]
        lines.append(f"{name_cell(name)} & " + " & ".join(cells) + f" & {fmt_time(times[name])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def ranking_table(results_dir: Path, modality: str) -> list[str]:
    summary = pd.read_csv(results_dir / "estimators" / "ce_estimator_model_ranking.csv")
    summary = summary[summary["modality"] == modality]
    metrics = list(DISPLAYED_METRICS[modality])
    pivot = lambda column: summary.pivot(index="estimator", columns="metric", values=column)
    order = estimator_order(pivot("spearman")[metrics], ascending=False)
    header_metrics = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{METRIC_LABELS[modality][m]}}}" for m in metrics
    )
    rules = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(metrics)))
    lines = [
        r"\begin{tabular}{l " + " ".join(["ccc"] * len(metrics)) + "}",
        r"\toprule",
        f"& {header_metrics} \\\\",
        rules,
        "Estimator & "
        + " & ".join([r"Spearman $\uparrow$", r"pairs (\%)", r"top-1 (\%)"] * len(metrics))
        + r" \\",
        r"\midrule",
    ]
    cells = {}
    for metric in metrics:
        rho, se = pivot("spearman")[metric].reindex(order), pivot("spearman_se")[metric].reindex(order)
        rho_cells = pd.Series(
            [
                "--" if not np.isfinite(r) else f"{r:.2f}" + r"{\scriptsize$\pm$" + f"{s:.2f}" + "}"
                for r, s in zip(rho, se)
            ],
            index=order,
        )
        defined = rho.notna()
        rho_cells[defined] = bold_best(rho[defined], rho_cells[defined], "max")
        cells[(metric, "rho")] = rho_cells
        for column in ("pairwise_accuracy", "top1_accuracy"):
            values = 100.0 * pivot(column)[metric].reindex(order)
            formatted = values.map(lambda v: fmt(v, 0))
            formatted[values.notna()] = bold_best(values[values.notna()], formatted[values.notna()], "max")
            cells[(metric, column)] = formatted
    for name in order:
        row = [
            cells[(m, c)][name] for m in metrics for c in ("rho", "pairwise_accuracy", "top1_accuracy")
        ]
        lines.append(f"{name_cell(name)} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("paper_tables"),
        help="Directory for the LaTeX tables of the paper.",
    )
    parser.add_argument(
        "--min_ce",
        type=float,
        default=1e-3,
        help="Threshold of the first experiment; must match plot_calibration_errors.py.",
    )
    args = parser.parse_args()

    output_dir = args.table_dir
    write(output_dir / "datasets.tex", dataset_table(args.data_dir))
    write(output_dir / "recalibrators.tex", recalibrator_table(args.results_dir, args.min_ce))
    for modality in MODALITIES:
        write(output_dir / f"estimators_{modality}.tex", accuracy_table(args.results_dir, modality))
        write(output_dir / f"ranking_{modality}.tex", ranking_table(args.results_dir, modality))


if __name__ == "__main__":
    main()
