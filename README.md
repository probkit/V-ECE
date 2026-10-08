# V-ECE: Estimating General Expected Calibration Errors

Code for the experiments of the paper *V-ECE: Estimating General Expected Calibration
Errors*.

V-ECE estimates the calibration error `CE_d(f) = E[d(f(X), E[Y | f(X)])]` of a classifier
`f` as the excess risk of `f` over a recalibration `g ∘ f` of its own predictions. It uses
a prediction-dependent proper loss, so it works for any convex distance `d`: the L1 and L2
calibration errors, the squared (Brier) calibration error, over- and under-confidence
errors, and so on. It works in the binary and the multiclass setting. The recalibration
function `g` is fitted by cross-validation, so the estimate is a lower bound on the true
calibration error in expectation. Our default `g` is **WS-CatBoost**: CatBoost boosting a
residual on top of temperature-scaled logits.

## Using V-ECE

[`vece.py`](vece.py) is a self-contained implementation
(numpy, scipy, scikit-learn >= 1.8, catboost) that can be copied into any project:

```python
from vece import estimate_ce

# probs: (n, k) predicted probabilities (k = 2 for binary), y: (n,) labels in 0..k-1
ce_l1 = estimate_ce(probs, y, metric="l1")        # binary CE_|.| or multiclass CE_||.||_1
ce_l2 = estimate_ce(probs, y, metric="l2")        # multiclass CE_||.||_2
ce_sq = estimate_ce(probs, y, metric="brier")     # squared calibration error
```

`estimate_ce` never returns a negative value: the excess risk is averaged over all
out-of-fold predictions and only then clipped at 0. To evaluate several metrics with a
single fit, call `g = recalibrate_oof(probs, y)` once, then evaluate
`max(0, -proper_loss(g, y, probs, metric).mean())` for each metric. The fits run in a
multiprocessing pool, so call the estimator from within an `if __name__ == "__main__":`
block.

## Repository layout

| Path | Role |
| --- | --- |
| `vece.py` | **V-ECE**, self-contained: prediction-dependent proper losses (`proper_loss`), the WS-CatBoost recalibrator (`recalibrate_oof`) and `estimate_ce`. |
| `experiments/select_experiments.py` | Selects the TabRepo experiments used in the paper (`data/experiments_*.csv`). |
| `experiments/benchmark_calibration_errors.py` | **Experiment 1**: compares recalibration functions inside the variational estimator on real TabRepo predictions. |
| `experiments/make_ground_truth.py` | Builds the semi-synthetic ground truth `C = E[Y \| f(X)]` with TabICL (Experiment 2). |
| `experiments/benchmark_ce_estimators.py` | **Experiment 2**: scores V-ECE and 16 binary / 4 multiclass calibration error estimators against the ground truth. |
| `experiments/ece_kde.py` | Kernel utilities of KDE-ECE, vendored from [tpopordanoska/ece-kde](https://github.com/tpopordanoska/ece-kde) under the MIT License (notice kept in the file). |
| `plotting/plot_calibration_errors.py` | Experiment 1: average ranks (Figure 2 of the paper). |
| `plotting/plot_ce_estimators.py` | Experiment 2: accuracy (Figure 3) and estimate/truth ratios (appendix). |
| `plotting/plot_ce_estimator_ranking.py` | Experiment 2: how well each estimator ranks the 8 models of a dataset (appendix). |
| `plotting/make_tables.py` | LaTeX tables of the appendix. |
| `plotting/style.py` | Shared figure style. |
| `reproduce.sh` | Runs the whole pipeline. |
| `data/` | Input data (see below). |
| `results/` | Raw per-experiment results and summaries (CSV). |
| `paper_figures/` | Figures of the paper (PDF, with PNG previews). |
| `paper_tables/` | LaTeX tables of the appendix. |

All scripts are run from the repository root as modules, e.g.
`python -m experiments.benchmark_ce_estimators --help`.

## Installation

```bash
conda create -n vece python=3.12 && conda activate vece
pip install -r requirements.txt
```

We ran the benchmarks with numpy 2.3.3, scipy 1.17.1, pandas 3.0.3, h5py 3.16.0,
scikit-learn 1.8.0, catboost 1.2.10, torch 2.12.0, probmetrics 1.3.0, betacal 1.1.0 and
pytorch-minimize 0.1.0. We built the ground truth with tabicl 2.0.3. Only
`make_ground_truth.py` needs `tabicl`. scikit-learn >= 1.8 is required for
`CalibratedClassifierCV(method="temperature")`. The figures are typeset with LaTeX when a
`latex` executable is found; otherwise they fall back to matplotlib's Computer Modern
mathtext.

## Data

All experiments use the **TabRepo-binary** and **TabRepo-multiclass** benchmarks of
[CalArena](https://arxiv.org/abs/2605.30188) (Berta et al., NeurIPS 2026), which collect
predictions from [TabRepo](https://github.com/autogluon/tabrepo) (Salinas and Erickson,
2024). For each dataset they hold the first outer fold (`tabrepo_fold = 0`) and one
configuration of each of eight model families: CatBoost, ExtraTrees, LightGBM,
LinearModel, NeuralNetFastAI, NeuralNetTorch, RandomForest and XGBoost. These are
AutoGluon's bagged `*_BAG_L1` models. Every (dataset, model) pair is one *experiment*: a
classifier `f` whose calibration error we estimate.

| File | Content |
| --- | --- |
| `data/tabrepo-{binary,multiclass}.h5` | One group `{dataset}/{model}` per experiment with `probas_cal`, `labels_cal` (out-of-fold predictions on the training split, 90% of the data) and `probas_test`, `labels_test` (predictions on the test split, 10%). Binary probabilities are stored as the class-1 probability only. |
| `data/tabrepo-{binary,multiclass}-experiments.csv` | All 104 binary / 65 multiclass datasets of the extract, with split sizes and the TabRepo configuration of every model. |
| `data/experiments_{binary,multiclass}.csv` | The experiments kept by `experiments/select_experiments.py`: datasets with at least 8,000 calibration samples, i.e. 22 binary (176 experiments) and 21 multiclass (168 experiments, 3 to 26 classes). |
| `data/ground-truth-{binary,multiclass}.h5` | Output of `experiments/make_ground_truth.py`: one group `{dataset}/{model}` holding `probas_true`, the semi-synthetic `C` on the test split. |

`tabrepo-*.h5` and `tabrepo-*-experiments.csv` are too large for the git repository. They
are the CalArena files, available at
[huggingface.co/datasets/probkit/CalArena](https://huggingface.co/datasets/probkit/CalArena);
place them in `data/`. The ground-truth files `ground-truth-*.h5` (16 MB) are included, so
Experiment 2 can be rerun without TabICL; `experiments/make_ground_truth.py` rebuilds them
(it skips the experiments already present, so delete the files first to refit).

## Reproducing the experiments

`bash reproduce.sh` runs every step in order; the steps are:

```bash
python -m experiments.select_experiments            # data/experiments_*.csv
# Experiment 1: which recalibration function recovers the most calibration error?
python -m experiments.benchmark_calibration_errors  # results/{binary,multiclass}/*.csv (~20 min)
python -m plotting.plot_calibration_errors          # paper_figures/calibration_error_ranks.pdf
# Experiment 2: accuracy against a semi-synthetic ground truth
python -m experiments.make_ground_truth             # data/ground-truth-*.h5 (~4.5 h; resumable)
python -m experiments.benchmark_ce_estimators       # results/estimators/{binary,multiclass}/*.csv (~80 min)
python -m plotting.plot_ce_estimators               # paper_figures/ce_estimator_{accuracy,ratios}.pdf
python -m plotting.plot_ce_estimator_ranking        # paper_figures/ce_estimator_model_ranking.pdf
python -m plotting.make_tables                      # paper_tables/*.tex
```

The runtimes were measured on a 10-core Apple M2 Pro laptop. WS-CatBoost fits its models
in parallel on all physical cores. Every benchmark script accepts `--modality` and
`--method`/`--estimator` to run a subset, and writes one CSV per method. A partial rerun
therefore only overwrites the methods it runs. All randomness is seeded (`--seed`, default
0); rerunning the benchmark scripts reproduces the committed CSVs.

### Experiment 1: choosing the recalibration function

On each classifier's test split, with its real labels, every recalibration function `g`
is fitted by 5-fold cross-validation. It is scored with the variational estimator,
`-mean_i l_{f(X_i)}(g(f(X_i)), Y_i)`, for three metrics: `l1`, `l2` and `brier`. In the
binary case `l1 = l2 = CE_|.|` and `brier = CE_(.)^2`, both computed on the class-1
probability. Estimates are clipped at 0 after aggregating the folds. Since every estimate is
a lower bound in expectation, a larger estimate is better. `plot_calibration_errors.py` ranks the recalibration functions within each
classifier and averages the ranks. It drops the classifiers whose largest estimate of
`CE_|.|` (binary) or `CE_||.||_2` (multiclass) is below `--min_ce = 1e-3`, since every
estimate is noise around zero there. This keeps 149 binary and 150 multiclass classifiers.

| Binary | Multiclass |
| --- | --- |
| Hist-uniform, Hist-quantile (10 bins), Isotonic, Quadratic scaling, Beta calibration, WS-CatBoost | TS, SVS, SMS, WS-CatBoost |

### Experiment 2: semi-synthetic benchmark with known calibration error

`make_ground_truth.py` fits TabICL v2 on the *calibration* split of each classifier: the
features are the predictions `f(X)`, the targets the real labels, and the context is capped
at 10,000 random samples. It then predicts `C = g(f(X))` on the test split.
`benchmark_ce_estimators.py` redraws the test labels from `C`, so that `E[Y | f(X)] = C`
holds exactly and the true calibration error is `mean_i d(f(X_i), C_i)`. Estimators only
see the test predictions and the redrawn labels. The labels are drawn 3 times
independently; each draw is a separate run of every estimator, whose estimate is clipped
at 0 once aggregated over its folds (`{estimator}_{metric}_draw{r}` columns, unclipped
values in `*_unclipped`). The plotting scripts score each draw against the truth and
average the errors over draws, which is what a user observing a single dataset gets.

The estimators fall into three families, which the figures color differently:

| Family | Name | Recalibration `g` / construction | Estimate |
| --- | --- | --- | --- |
| (i) plug-in (ECE) | `Uniform-ECE`, `Quantile-ECE` | 15 equal-width / equal-mass bins | textbook binned ECE `sum_b w_b d(mean_b f, mean_b Y)` |
| | `Isotonic-ECE` | in-sample isotonic regression | plug-in `mean_i d(f(X_i), g(f(X_i)))` |
| | `Debiased-ECE` | 15 equal-width bins | binned squared ECE minus the variance of the bin frequencies (Kumar et al., 2019) |
| | `Smooth-ECE` | Gaussian kernel smoothing of the residuals, self-consistent bandwidth | smooth ECE (Błasiok and Nakkiran, 2024) |
| | `KDE-ECE` | leave-one-out Beta- (binary) or Dirichlet-kernel (multiclass) regression (Popordanoska et al., 2022) | plug-in |
| (ii) variational, in-sample (IS) | `Uniform-IS`, `Quantile-IS`, `Isotonic-IS` | in-sample histogram / isotonic regression (Isotonic-IS is CORP, Dimitriadis et al., 2021) | excess risk |
| (iii) variational, out-of-sample (CV) | `Uniform-CV`, `Quantile-CV`, `Isotonic-CV` | same, fitted out of fold (5 folds) | excess risk |
| | `TS-CV`, `Quadratic-CV` and `Beta-CV` (binary), `SMS-CV` (multiclass) | temperature scaling / quadratic scaling / Beta calibration / structured matrix scaling, 5 folds | excess risk |
| | `KDE-CV` | leave-one-out kernel regression of `KDE-ECE` | excess risk |
| | `WS-CatBoost` (= **V-ECE**) | WS-CatBoost, 5 outer folds | excess risk |

Family (iii) is our estimator with different recalibration functions. Only `KDE-ECE`,
`KDE-CV`, `TS-CV`, `SMS-CV` and `WS-CatBoost` apply to the
multiclass case. `plot_ce_estimators.py` scores estimators by the *normalized absolute
error*: the absolute error divided by the largest absolute error among the estimators on
that classifier, averaged over classifiers. It also reports the NMAE,
`sum |estimate - truth| / sum truth`, and the share of classifiers on which each estimator
overestimates. `plot_ce_estimator_ranking.py` compares, within each dataset, the ordering
of the eight models by estimated and by true calibration error.

## Citation

```bibtex
@article{berta2026vece,
  title = {{V-ECE}: Estimating General Expected Calibration Errors},
  author = {Berta, Eug{\`e}ne and Braun, Sacha and Bach, Francis and Jordan, Michael I. and Holzm{\"u}ller, David},
  journal = {arXiv preprint arXiv:2610.?????},
  year = {2026}
}
```

## License

MIT, see [`LICENSE`](LICENSE). `experiments/ece_kde.py` keeps its own MIT notice.
