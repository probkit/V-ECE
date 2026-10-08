"""Compare calibration error estimators against a semi-synthetic ground truth.

``make_ground_truth.py`` provides, for every experiment, a true calibrated distribution
``C = g(f(X))`` obtained by fitting TabICL on the calibration split.  Re-drawing the test
labels from ``C`` makes ``E[Y | f(X)] = C`` hold by construction, so the calibration error
of the classifier ``f`` on the test split is known exactly,

    CE_d(f) = mean_i d(f(X_i), C_i) ,                                            (truth)

and every estimator below is scored by how much of it it recovers from ``(f(X), Y)`` alone.

Two families of estimators are compared, over the same recalibration mechanisms ``g_hat``:

* **distance-based** (plug-in): plug ``g_hat(f(X))`` into (truth) in place of the unknown
  ``C``, i.e. ``mean_i d(f(X_i), g_hat(f(X_i)))``;
* **excess-risk based** (the variational estimator of the paper): the risk of ``f`` minus
  the risk of ``g_hat`` o ``f``, ``- mean_i l_{f(X_i)}(g_hat(f(X_i)), Y_i)``, computed with
  ``proper_loss`` from ``vece.py``.

Three binned estimators stand outside that grid because they compare bin *averages*: the
textbook binned ECE, ``sum_b w_b d(mean_b f, mean_b Y)``, with equal-width or equal-mass
bins (Naeini et al., 2015; Guo et al., 2017), and the debiased binned squared ECE of Kumar
et al. (2019).  Two further estimators are each defined for a single metric: the debiased
ECE (squared error only) and the smooth ECE of Blasiok and Nakkiran (2024), a
kernel-smoothed L1 calibration distance with a self-consistent bandwidth.  They report NaN
for the metrics they do not define.

Holding ``g_hat`` fixed and switching functional isolates what the two formulations
contribute, and each mechanism is also fitted either in-sample or out-of-fold, which
isolates what cross-validation contributes.  ``ESTIMATORS`` lists the resulting grid.

Outputs one CSV per estimator, as in ``benchmark_calibration_errors.py``:

    {results_dir}/estimators/{modality}/{estimator}.csv

Each row is one experiment and holds the true calibration errors (``true_{metric}``)
next to the estimate of every label draw (``{estimator}_{metric}_draw{r}``), the mean of
the draws (``{estimator}_{metric}``) and the runtime.  Every estimate is clipped at 0 once
aggregated over its folds, i.e. on each label draw; the ``*_unclipped`` columns hold the
same values without clipping.  The reporting scores each draw against the truth and
averages the errors over draws.  Only
the estimators that can return negative values are affected: the excess-risk estimators
and the debiased ECE.

Usage
-----
    python -m experiments.benchmark_ce_estimators                            # both modalities
    python -m experiments.benchmark_ce_estimators --modality binary --estimator Isotonic-IS
    python -m experiments.benchmark_ce_estimators --n_replicates 10          # damp the label noise further
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from tqdm import tqdm

from probmetrics.calibrators import (
    BetacalCalibrator,
    BinaryHistogramBinningCalibrator,
    BinaryLogisticCalibrator,
    SklearnCalibrator,
    SMSCalibrator,
    TemperatureScalingCalibrator,
)

from experiments.benchmark_calibration_errors import (
    METRICS,
    MODALITIES,
    as_probability_matrix,
    clip,
    mix_with_uniform,
    warm_started_catboost,
)
from experiments.ece_kde import get_bandwidth, get_kernel
from vece import proper_loss

warnings.filterwarnings("ignore", category=FutureWarning)

N_FOLDS = 5  # cross-validation folds of the out-of-fold estimators
N_BINS = 15  # bins of the histogram estimators

# Kernel bandwidth selection is O(n^2) per candidate bandwidth, so KDE-ECE selects it on a
# subsample and then applies it to the whole test split.
KDE_BANDWIDTH_SUBSAMPLE = 2_000


# ==========================================
# THE TWO FUNCTIONALS
# ==========================================


def mean_distance(probas: np.ndarray, target: np.ndarray, metric: str) -> float:
    """``mean_i d(f(X_i), target_i)`` with the distances that ``proper_loss`` induces.

    With ``target = C`` this is the true calibration error; with ``target = g_hat(f(X))`` it
    is the distance-based (plug-in) estimator.  Binary predictions are reduced to the
    class-1 probability, where ``proper_loss`` also puts them, so that ``l1`` and ``l2``
    coincide and ``brier`` is the squared gap.
    """
    difference = target - probas
    if probas.shape[1] == 2:
        difference = difference[:, 1:]
    if metric == "brier":
        return float(np.mean(np.sum(difference**2, axis=1)))
    order = {"l1": 1, "l2": 2}[metric]
    return float(np.mean(np.linalg.norm(difference, ord=order, axis=1)))


def excess_risk(
    probas: np.ndarray, labels: np.ndarray, recalibrated: np.ndarray, metric: str
) -> float:
    """``- mean_i l_{f(X_i)}(g_hat(f(X_i)), Y_i)``: the variational estimator of the paper."""
    return float(-np.mean(proper_loss(recalibrated, labels, probas, metric)))


# ==========================================
# RECALIBRATION MECHANISMS
# ==========================================
# A mechanism maps (probas, labels, seed) to recalibrated probabilities g_hat(f(X)) of the
# same shape.  `in_sample` and `out_of_fold` turn a calibrator into a mechanism; the kernel
# and CatBoost mechanisms hold their own resampling scheme.

Mechanism = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def histogram_calibrator(strategy: str) -> Callable[[], object]:
    """Histogram binning of the class-1 probability (Zadrozny and Elkan, 2001).

    ``strategy="uniform"`` gives equal-width bins and ``"quantile"`` equal-count bins; a bin
    left empty by the training fold falls back to the global outcome frequency.
    """
    return lambda: BinaryHistogramBinningCalibrator(n_bins=N_BINS, strategy=strategy)


def isotonic_calibrator() -> object:
    """Isotonic regression of the outcome on the class-1 probability.

    This is the pool-adjacent-violators fit underlying the CORP reliability diagrams of
    Dimitriadis, Gneiting and Jordan (2021).
    """
    return SklearnCalibrator(method="isotonic", cv="prefit")


def in_sample(calibrator_factory: Callable[[], object]) -> Mechanism:
    """Fit the calibrator on the whole test split and predict back on it."""

    def mechanism(probas: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
        inputs = mix_with_uniform(probas, n_train=len(probas))
        return calibrator_factory().fit(inputs, labels).predict_proba(inputs)

    return mechanism


def out_of_fold(calibrator_factory: Callable[[], object]) -> Mechanism:
    """Fit the calibrator on ``N_FOLDS - 1`` folds and predict on the held-out fold."""

    def mechanism(probas: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
        inputs = mix_with_uniform(probas, n_train=len(probas) * (N_FOLDS - 1) // N_FOLDS)
        recalibrated = np.empty_like(inputs)
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, val_idx in folds.split(inputs):
            calibrator = calibrator_factory().fit(inputs[train_idx], labels[train_idx])
            recalibrated[val_idx] = calibrator.predict_proba(inputs[val_idx])
        return recalibrated

    return mechanism


def kernel_smoothing(probas: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Leave-one-out kernel estimate of ``E[Y | f(X)]`` (Popordanoska et al., 2022).

    This is the recalibration mechanism inside KDE-ECE: a Nadaraya-Watson average of the
    outcomes, using the beta kernel on ``[0, 1]`` for binary predictions and the Dirichlet
    kernel on the simplex otherwise.  ``ece_kde.get_kernel`` puts ``-inf`` on the diagonal,
    which makes every estimate leave-one-out.
    """
    inputs = mix_with_uniform(probas, n_train=len(probas))  # the kernels take log(f)
    n_classes = inputs.shape[1]
    # `ece_kde` selects the kernel by the width of its input: one column for the binary
    # beta kernel, the full simplex vector for the Dirichlet kernel.
    features = torch.as_tensor(
        inputs[:, 1:] if n_classes == 2 else inputs, dtype=torch.float32
    )

    rng = np.random.RandomState(seed)
    subsample = rng.choice(
        len(features), min(KDE_BANDWIDTH_SUBSAMPLE, len(features)), replace=False
    )
    bandwidth = get_bandwidth(features[subsample], "cpu")

    log_kernel = get_kernel(features, bandwidth, "cpu")
    log_denominator = torch.logsumexp(log_kernel, dim=1)
    recalibrated = torch.stack(
        [
            torch.exp(
                torch.logsumexp(log_kernel[:, torch.as_tensor(labels == label)], dim=1)
                - log_denominator
            )
            for label in range(n_classes)
        ],
        dim=1,
    )
    return recalibrated.numpy().astype(float)


# ==========================================
# ESTIMATORS
# ==========================================
# An estimator maps (probas, labels, seed) to one value per metric.

Estimator = Callable[[np.ndarray, np.ndarray, int], Dict[str, float]]


def distance_based(mechanism: Mechanism) -> Estimator:
    """Classical ECE functional: the distance between ``f(X)`` and ``g_hat(f(X))``."""

    def estimate(probas: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
        recalibrated = mechanism(probas, labels, seed)
        return {metric: mean_distance(probas, recalibrated, metric) for metric in METRICS}

    return estimate


def excess_risk_based(mechanism: Mechanism) -> Estimator:
    """Variational functional: the excess risk of ``f`` over ``g_hat`` o ``f``."""

    def estimate(probas: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
        recalibrated = mechanism(probas, labels, seed)
        return {
            metric: excess_risk(probas, labels, recalibrated, metric) for metric in METRICS
        }

    return estimate


# The binned estimators compare bin averages rather than individual predictions, and two
# further estimators are tied to a single metric, so they are written directly rather than
# assembled from a mechanism and a functional.  Estimators return NaN for the metrics they
# do not define, which the reporting skips.


def bin_statistics(
    probas: np.ndarray, labels: np.ndarray, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occupancy, mean prediction and outcome frequency of the ``N_BINS`` bins of ``f``.

    Bins partition the class-1 probability, with equal-width (``"uniform"``) or equal-mass
    (``"quantile"``) edges computed on the test split itself; empty bins are dropped.
    """
    scores = probas[:, 1]
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, N_BINS + 1)
    else:
        edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, N_BINS + 1)))
    n_bins = max(len(edges) - 1, 1)
    indices = np.clip(np.digitize(scores, edges, right=False) - 1, 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins)
    occupied = counts > 0
    counts = counts[occupied]
    mean_scores = np.bincount(indices, weights=scores, minlength=n_bins)[occupied] / counts
    frequencies = np.bincount(indices, weights=labels, minlength=n_bins)[occupied] / counts
    return counts, mean_scores, frequencies


def binned_ece(strategy: str) -> Estimator:
    """Textbook binned ECE, ``sum_b (n_b / n) d(mean_b f, mean_b Y)`` (Naeini et al., 2015).

    Each bin compares its average prediction with its outcome frequency, so the L1 version
    is the usual ECE of Guo et al. (2017) and the squared version the binned squared ECE.
    """

    def estimate(probas: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
        counts, mean_scores, frequencies = bin_statistics(probas, labels, strategy)
        weights = counts / counts.sum()
        gaps = np.abs(mean_scores - frequencies)
        l1 = float(np.sum(weights * gaps))
        return {"l1": l1, "l2": l1, "brier": float(np.sum(weights * gaps**2))}

    return estimate


def debiased_binned_ece(
    probas: np.ndarray, labels: np.ndarray, seed: int
) -> Dict[str, float]:
    """Debiased binned estimator of ``CE_{(.)^2}`` (Kumar, Liang and Ma, 2019).

    The binned squared ECE charges the sampling noise of each bin's outcome frequency to the
    classifier: writing ``p_b`` for the frequency in bin ``b``, ``E[(mean_b f - p_b)^2]``
    equals ``(mean_b f - C_b)^2 + Var(p_b)``.  Subtracting the unbiased estimate
    ``p_b (1 - p_b) / (n_b - 1)`` of that variance removes the bias.  As in the reference
    implementation of Kumar et al., bins holding a single sample admit no variance estimate
    and contribute zero.  Bins are the equal-width bins of ``Uniform-ECE``.

    The argument is specific to the squared error, so the other two metrics are undefined.
    """
    counts, mean_scores, frequencies = bin_statistics(probas, labels, "uniform")
    estimable = counts > 1
    terms = (mean_scores - frequencies) ** 2 - frequencies * (1.0 - frequencies) / np.maximum(
        counts - 1, 1
    )
    value = float(np.sum(counts[estimable] / counts.sum() * terms[estimable]))
    return {"l1": np.nan, "l2": np.nan, "brier": value}


SMECE_GRID_SIZE = 1024  # grid over [0, 1] on which the smoothed residuals are integrated
SMECE_BISECTION_STEPS = 40


def _smooth_ece_at(residuals: np.ndarray, sigma: float, spacing: float, n: int) -> float:
    """``smECE_sigma``: the integrated absolute value of the kernel-smoothed residuals."""
    smoothed = gaussian_filter1d(residuals, sigma / spacing, mode="reflect")
    return float(np.sum(np.abs(smoothed)) / n)


def smooth_ece(probas: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
    """Smooth ECE of Blasiok and Nakkiran (2024), a consistent calibration distance.

    The signed residuals ``Y_i - f(X_i)`` are laid out on a grid over ``[0, 1]`` at the
    position of their prediction, smoothed with a Gaussian kernel reflected at the
    boundaries, and integrated in absolute value,

        smECE_sigma = (1 / n) * integral | sum_i K_sigma(t - f(X_i)) (Y_i - f(X_i)) | dt .

    Rather than fixing a bandwidth, the paper reports the value at the self-consistent
    ``sigma`` solving ``sigma = smECE_sigma``, which we locate by bisection: ``smECE_sigma``
    falls from ``mean |Y - f(X)|`` to ``|mean (Y - f(X))|`` as ``sigma`` grows, so
    ``sigma - smECE_sigma`` crosses zero exactly once.

    This estimates an L1 calibration error, which on binary predictions coincides with the
    L2 one, so both share the value and the squared metric is left undefined.
    """
    scores = probas[:, 1]
    spacing = 1.0 / (SMECE_GRID_SIZE - 1)
    positions = np.clip(np.round(scores / spacing).astype(int), 0, SMECE_GRID_SIZE - 1)
    residuals = np.bincount(
        positions, weights=labels - scores, minlength=SMECE_GRID_SIZE
    )

    low, high = spacing, 1.0
    for _ in range(SMECE_BISECTION_STEPS):
        middle = np.sqrt(low * high)
        if middle < _smooth_ece_at(residuals, middle, spacing, len(scores)):
            low = middle
        else:
            high = middle
    value = _smooth_ece_at(residuals, np.sqrt(low * high), spacing, len(scores))
    return {"l1": value, "l2": value, "brier": np.nan}


def _catboost_mechanism(probas: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    return warm_started_catboost(probas, labels, N_FOLDS, seed)


BINARY_ESTIMATORS: Dict[str, Estimator] = {
    # (i) Plug-in estimators (ECE).  Textbook binned ECE with equal-width and equal-count bins.
    "Uniform-ECE": binned_ece("uniform"),
    "Quantile-ECE": binned_ece("quantile"),
    # (ii) Variational estimators with an in-sample g_hat (IS) and (iii) out-of-sample (CV):
    # the same binning, scored by excess risk instead of distance.
    "Uniform-IS": excess_risk_based(in_sample(histogram_calibrator("uniform"))),
    "Quantile-IS": excess_risk_based(in_sample(histogram_calibrator("quantile"))),
    "Uniform-CV": excess_risk_based(out_of_fold(histogram_calibrator("uniform"))),
    "Quantile-CV": excess_risk_based(out_of_fold(histogram_calibrator("quantile"))),
    # Isotonic regression, in sample (the CORP decomposition) and out of fold.
    "Isotonic-IS": excess_risk_based(in_sample(isotonic_calibrator)),
    "Isotonic-CV": excess_risk_based(out_of_fold(isotonic_calibrator)),
    # Kernel smoothing, scored by distance (KDE-ECE) and by excess risk (KDE-CV).
    "KDE-ECE": distance_based(kernel_smoothing),
    "KDE-CV": excess_risk_based(kernel_smoothing),
    # Bias-corrected and kernel-smoothed distance estimators, each defined for one metric.
    "Debiased-ECE": debiased_binned_ece,
    "Smooth-ECE": smooth_ece,
    # Parametric recalibration, scored out of fold (quadratic scaling is the binary
    # counterpart of SMS, the best parametric method of the first experiment).
    "TS-CV": excess_risk_based(out_of_fold(lambda: TemperatureScalingCalibrator())),
    "Quadratic-CV": excess_risk_based(
        out_of_fold(lambda: BinaryLogisticCalibrator(type="quadratic"))
    ),
    "Beta-CV": excess_risk_based(out_of_fold(lambda: BetacalCalibrator())),
    # V-ECE: the best recalibration model of the first experiment, scored by excess risk.
    "WS-CatBoost": excess_risk_based(_catboost_mechanism),
    # Plug-in estimator with the isotonic (CORP) recalibration of Isotonic-IS.
    "Isotonic-ECE": distance_based(in_sample(isotonic_calibrator)),
}

MULTICLASS_ESTIMATORS: Dict[str, Estimator] = {
    "KDE-ECE": distance_based(kernel_smoothing),
    "KDE-CV": excess_risk_based(kernel_smoothing),
    "TS-CV": excess_risk_based(out_of_fold(lambda: TemperatureScalingCalibrator())),
    "SMS-CV": excess_risk_based(out_of_fold(lambda: SMSCalibrator())),
    "WS-CatBoost": excess_risk_based(_catboost_mechanism),
}

ESTIMATORS: Dict[str, Dict[str, Estimator]] = {
    "binary": BINARY_ESTIMATORS,
    "multiclass": MULTICLASS_ESTIMATORS,
}


# ==========================================
# BENCHMARK
# ==========================================


def draw_labels(probas_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one label per sample from ``C``, making ``E[Y | f(X)] = C`` hold exactly."""
    thresholds = np.cumsum(probas_true, axis=1)
    thresholds[:, -1] = 1.0  # guard against floating-point drift in the last column
    return (rng.random((len(probas_true), 1)) < thresholds).argmax(axis=1)


def benchmark_estimator(
    name: str,
    estimator: Estimator,
    experiments: pd.DataFrame,
    tabrepo: h5py.File,
    ground_truth: h5py.File,
    modality: str,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Score one estimator on every experiment of one modality."""
    rows = []
    for _, experiment in tqdm(
        experiments.iterrows(), total=len(experiments), desc=f"{modality}/{name}"
    ):
        dataset, model = experiment["dataset"], experiment["model"]
        probas = as_probability_matrix(tabrepo[f"{dataset}/{model}/probas_test"][:])
        probas_true = np.asarray(
            ground_truth[f"{dataset}/{model}/probas_true"][:], dtype=float
        )

        row = {"dataset": dataset, "model": model, "test_size": len(probas)}
        if modality == "multiclass":
            row["n_classes"] = probas.shape[1]
        row.update(
            {f"true_{metric}": mean_distance(probas, probas_true, metric) for metric in METRICS}
        )

        # Each label draw is a separate semi-synthetic dataset and a full run of the
        # estimator; the ground truth above does not depend on the draws.
        estimates = []
        start = time.perf_counter()
        for replicate in range(n_replicates):
            replicate_seed = seed + replicate
            labels = draw_labels(probas_true, np.random.default_rng(replicate_seed))
            estimates.append(estimator(probas, labels, replicate_seed))
        runtime = (time.perf_counter() - start) / n_replicates

        # The output of each run is clipped at 0 after aggregating its folds (``clip``), as
        # a user's single estimate would be.  Every draw is stored, so that the reporting
        # can score each draw against the truth and average the errors, rather than score
        # an average of estimates that no user would observe.  The means over draws are
        # kept for reference.  NaN, for a metric the estimator does not define, is left as is.
        for metric in METRICS:
            raw = [e[metric] for e in estimates]
            clipped = [value if np.isnan(value) else clip(value) for value in raw]
            for replicate, (value, raw_value) in enumerate(zip(clipped, raw)):
                row[f"{name}_{metric}_draw{replicate}"] = float(value)
                row[f"{name}_{metric}_draw{replicate}_unclipped"] = float(raw_value)
            row[f"{name}_{metric}"] = float(np.mean(clipped))
            row[f"{name}_{metric}_unclipped"] = float(np.mean(raw))
        row[f"{name}_time"] = runtime
        rows.append(row)

    return pd.DataFrame(rows)


def run(
    modality: str,
    names: Iterable[str],
    args: argparse.Namespace,
) -> None:
    """Score the requested estimators and write one CSV each."""
    data_dir = Path(args.data_dir)
    experiments = pd.read_csv(data_dir / f"experiments_{modality}.csv", index_col=0)
    tabrepo_file = data_dir / f"tabrepo-{modality}.h5"
    ground_truth_file = data_dir / f"ground-truth-{modality}.h5"
    if not ground_truth_file.exists():
        raise FileNotFoundError(
            f"Missing ground truth {ground_truth_file}. Run `python -m experiments.make_ground_truth` first."
        )
    output_dir = Path(args.results_dir) / "estimators" / modality
    output_dir.mkdir(parents=True, exist_ok=True)

    # `locking=False` lets the ground truth be scored while make_ground_truth.py is still
    # appending to it, which pairs with the partial-availability check just below.
    with h5py.File(tabrepo_file, "r") as tabrepo, h5py.File(
        ground_truth_file, "r", locking=False
    ) as truth:
        available = [
            key
            for key, experiment in experiments.iterrows()
            if f"{experiment['dataset']}/{experiment['model']}/probas_true" in truth
        ]
        if len(available) < len(experiments):
            print(
                f"[warning] ground truth available for {len(available)}/{len(experiments)} "
                f"{modality} experiments; scoring only those."
            )
            experiments = experiments.loc[available]

        for name in names:
            print(f"\n=== Scoring {modality} estimator {name} ===")
            results = benchmark_estimator(
                name=name,
                estimator=ESTIMATORS[modality][name],
                experiments=experiments,
                tabrepo=tabrepo,
                ground_truth=truth,
                modality=modality,
                n_replicates=args.n_replicates,
                seed=args.seed,
            )
            results.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modality", choices=MODALITIES, default=None, help="Omit to run both.")
    parser.add_argument(
        "--estimator",
        action="append",
        default=None,
        dest="estimators",
        help="Estimator to score (repeatable). Omit to score all of them.",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--n_replicates",
        type=int,
        default=3,
        help="Independent label draws per experiment; errors are averaged over them, "
        "which damps the sampling noise of a single synthetic outcome vector.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed of the label draws.")
    args = parser.parse_args()

    for modality in MODALITIES if args.modality is None else (args.modality,):
        available = ESTIMATORS[modality]
        names = list(available) if args.estimators is None else args.estimators
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(
                f"Unknown {modality} estimator(s) {unknown}. Available: {', '.join(available)}."
            )
        run(modality, names, args)


if __name__ == "__main__":
    # A `__main__` guard is required: WS-CatBoost fits its models in a multiprocessing pool.
    main()
