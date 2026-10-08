"""Benchmark recalibration models as *estimators* of calibration error.

We evaluate the variational estimator of the paper (Section 3 and Algorithm 1): for a
classifier ``f`` and a recalibration function ``g``, the calibration error induced by the
distance ``d`` is the excess risk of ``f`` over ``g`` o ``f``,

    CE_d(f) = E[l_f(f(X), Y)] - E[l_f(g o f(X), Y)] = - E[l_f(g o f(X), Y)] ,

where ``l_f`` is the (sample-dependent) proper loss induced by ``d`` and normalised so that
``l_f(f(X), Y) = 0`` -- see ``proper_loss`` in ``vece.py``.  Plugging in an
estimate ``g_hat`` fitted by cross-validation yields, in expectation, a *lower bound* on the
true calibration error.  Consequently **larger estimates are better**: they show that the
recalibration model recovered more of the calibration error of ``f``.

Protocol (one row of ``data/experiments_{binary,multiclass}.csv`` = one classifier ``f``)
-----------------------------------------------------------------------------------
Unlike a post-hoc calibration benchmark, we never touch the calibration split: calibration
error is a property of ``f`` on the test split alone.  For each experiment we therefore only
use ``(probas_test, labels_test)``, with the real test labels, and follow Algorithm 1 of the
paper: split it into ``n_folds`` folds, fit ``g_hat`` on the training folds, and evaluate
the loss above on the held-out fold.  Averaging the out-of-fold estimates is the same as
averaging the per-sample losses of the out-of-fold predictions, which is what we do.  The
histogram recalibrators use the default 10 bins of ``probmetrics``.

Outputs, one CSV per recalibration model:

    {results_dir}/{modality}/{method}.csv

with one row per experiment and the columns ``{method}_{metric}`` for each metric in
``METRICS`` (clipped at 0 after aggregating the folds, see ``clip``), the same estimates
before clipping as ``{method}_{metric}_unclipped``, and ``{method}_time`` (wall-clock seconds for the whole cross-validated
estimate; note that ``WS-CatBoost`` fits its models in parallel over all physical cores
whereas the ``probmetrics`` calibrators run single-threaded).

Usage
-----
    python -m experiments.benchmark_calibration_errors                      # both modalities, all methods
    python -m experiments.benchmark_calibration_errors --modality binary
    python -m experiments.benchmark_calibration_errors --modality multiclass --method SMS
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
from sklearn.model_selection import KFold
from tqdm import tqdm

from probmetrics.calibrators import (
    BetacalCalibrator,
    BinaryHistogramBinningCalibrator,
    BinaryLogisticCalibrator,
    SklearnCalibrator,
    SMSCalibrator,
    SVSCalibrator,
    TemperatureScalingCalibrator,
)
from vece import proper_loss, recalibrate_oof

# Silences a deprecation of the `penalty` argument raised inside probmetrics' logistic
# calibrators by recent scikit-learn versions.
warnings.filterwarnings("ignore", category=FutureWarning)

# Calibration errors we estimate, as implemented by `vece.proper_loss`:
# "brier" is the (proper) squared Euclidean calibration error, "l1" and "l2" are the
# non-proper L1 and L2 calibration errors.  In the binary case predictions live on a
# segment, so "l1" and "l2" both reduce to CE_|.| and take identical values.
METRICS = ("brier", "l1", "l2")

MODALITIES = ("binary", "multiclass")

# ==========================================
# RECALIBRATION MODELS
# ==========================================
# A recalibration model is a callable
#     (probas, labels, n_folds, seed) -> out-of-fold recalibrated probabilities.
# `cross_validated` builds one from a `probmetrics` calibrator; `WS-CatBoost` is the
# warm-started CatBoost recalibrator of `vece.py`, which runs its own
# (nested) cross-validation and is therefore plugged in directly.

Recalibrator = Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]


def as_probability_matrix(probas: np.ndarray) -> np.ndarray:
    """Return probabilities with shape ``(n, k)``.

    TabRepo stores binary predictions as the probability of class 1 only.
    """
    probas = np.asarray(probas, dtype=float)
    if probas.ndim == 1:
        return np.stack([1.0 - probas, probas], axis=1)
    return probas


def mix_with_uniform(probas: np.ndarray, n_train: int) -> np.ndarray:
    """Shrink predictions towards the uniform distribution so that ``log(0)`` cannot occur.

    Hard zeros are common in the TabRepo predictions (e.g. for forests) and break the
    logit-space calibrators.  We use the same ``1 / (n_train + 1)`` shrinkage as
    ``vece.recalibrate_oof``; being ``O(1/n)``, it is negligible compared
    with the calibration errors we estimate.  Only the *input* of the recalibration model
    is shrunk: the calibration error is always evaluated against the original ``f(X)``.
    """
    lam = 1.0 / (n_train + 1)
    return (1.0 - lam) * probas + lam / probas.shape[1]


def cross_validated(calibrator_factory: Callable[[], object]) -> Recalibrator:
    """Wrap a ``probmetrics`` calibrator factory into a cross-validated recalibrator."""

    def recalibrate(
        probas: np.ndarray, labels: np.ndarray, n_folds: int, seed: int
    ) -> np.ndarray:
        inputs = mix_with_uniform(probas, n_train=len(probas) * (n_folds - 1) // n_folds)
        recalibrated = np.empty_like(probas)
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, val_idx in folds.split(inputs):
            calibrator = calibrator_factory()
            calibrator.fit(inputs[train_idx], labels[train_idx])
            recalibrated[val_idx] = calibrator.predict_proba(inputs[val_idx])
        return recalibrated

    return recalibrate


def warm_started_catboost(
    probas: np.ndarray, labels: np.ndarray, n_folds: int, seed: int
) -> np.ndarray:
    """Recalibrator of ``vece.py``: CatBoost warm-started from scaled logits."""
    return recalibrate_oof(probas, labels, n_outer=n_folds, seed=seed)


BINARY_RECALIBRATORS: Dict[str, Recalibrator] = {
    "Hist-uniform": cross_validated(
        lambda: BinaryHistogramBinningCalibrator(strategy="uniform")
    ),
    "Hist-quantile": cross_validated(
        lambda: BinaryHistogramBinningCalibrator(strategy="quantile")
    ),
    "Isotonic": cross_validated(lambda: SklearnCalibrator(method="isotonic", cv="prefit")),
    "Quadratic": cross_validated(lambda: BinaryLogisticCalibrator(type="quadratic")),
    "Beta": cross_validated(lambda: BetacalCalibrator()),
    "WS-CatBoost": warm_started_catboost,
}

MULTICLASS_RECALIBRATORS: Dict[str, Recalibrator] = {
    "TS": cross_validated(lambda: TemperatureScalingCalibrator()),
    "SVS": cross_validated(lambda: SVSCalibrator()),
    "SMS": cross_validated(lambda: SMSCalibrator()),
    "WS-CatBoost": warm_started_catboost,
}

RECALIBRATORS: Dict[str, Dict[str, Recalibrator]] = {
    "binary": BINARY_RECALIBRATORS,
    "multiclass": MULTICLASS_RECALIBRATORS,
}


# ==========================================
# BENCHMARK
# ==========================================


def calibration_errors(
    probas: np.ndarray, labels: np.ndarray, recalibrated: np.ndarray
) -> Dict[str, float]:
    """Unclipped estimates ``CE_hat = - mean l_f(g_hat o f(X), Y)`` for each metric.

    The mean runs over the out-of-fold predictions of all folds at once, so this is the
    aggregate over folds of Algorithm 1; ``clip`` is applied to it afterwards.
    """
    return {
        metric: float(-np.mean(proper_loss(recalibrated, labels, probas, metric)))
        for metric in METRICS
    }


def clip(estimate: float) -> float:
    """Clip an estimate at 0, the smallest possible calibration error.

    Applied to the estimate aggregated over folds, never to the fold estimates: since the
    calibration error is nonnegative, clipping the aggregate can only reduce its error,
    whereas averaging clipped fold estimates would bias the estimate upwards.
    """
    return max(0.0, estimate)


def benchmark_recalibrator(
    name: str,
    recalibrator: Recalibrator,
    experiments: pd.DataFrame,
    h5: h5py.File,
    modality: str,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate the calibration error of every experiment with a single recalibration model."""
    rows = []
    for _, experiment in tqdm(
        experiments.iterrows(), total=len(experiments), desc=f"{modality}/{name}"
    ):
        dataset, model = experiment["dataset"], experiment["model"]
        group = h5[f"{dataset}/{model}"]
        probas = as_probability_matrix(group["probas_test"][:])
        labels = np.asarray(group["labels_test"][:]).astype(int)

        row = {
            "dataset": dataset,
            "model": model,
            "test_size": len(labels),
        }
        if modality == "multiclass":
            row["n_classes"] = probas.shape[1]

        start = time.perf_counter()
        recalibrated = recalibrator(probas, labels, n_folds, seed)
        runtime = time.perf_counter() - start

        estimates = calibration_errors(probas, labels, recalibrated)
        row.update({f"{name}_{metric}": clip(value) for metric, value in estimates.items()})
        row.update(
            {f"{name}_{metric}_unclipped": value for metric, value in estimates.items()}
        )
        row[f"{name}_time"] = runtime
        rows.append(row)

    return pd.DataFrame(rows)


def run(
    modality: str,
    methods: Iterable[str],
    data_dir: Path,
    results_dir: Path,
    n_folds: int,
    seed: int,
) -> None:
    """Run the benchmark for one modality and write one CSV per recalibration model."""
    experiments = pd.read_csv(data_dir / f"experiments_{modality}.csv", index_col=0)
    h5_file = data_dir / f"tabrepo-{modality}.h5"
    output_dir = results_dir / modality
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, "r") as h5:
        for name in methods:
            print(f"\n=== Estimating {modality} calibration errors with {name} ===")
            results = benchmark_recalibrator(
                name=name,
                recalibrator=RECALIBRATORS[modality][name],
                experiments=experiments,
                h5=h5,
                modality=modality,
                n_folds=n_folds,
                seed=seed,
            )
            results.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        default=None,
        help="Modality to benchmark. Omit to run both.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        dest="methods",
        help="Recalibration model to run (repeatable). Omit to run all of them.",
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data"), help="Directory holding the HDF5 files."
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results"), help="Directory for the CSV results."
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the cross-validation splits."
    )
    args = parser.parse_args()

    modalities = MODALITIES if args.modality is None else (args.modality,)
    for modality in modalities:
        available = RECALIBRATORS[modality]
        methods = list(available) if args.methods is None else args.methods
        unknown = [name for name in methods if name not in available]
        if unknown:
            raise ValueError(
                f"Unknown {modality} recalibration model(s) {unknown}. "
                f"Available: {', '.join(available)}."
            )
        run(
            modality=modality,
            methods=methods,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            n_folds=args.n_folds,
            seed=args.seed,
        )


if __name__ == "__main__":
    # A `__main__` guard is required: `WS-CatBoost` fits its models in a multiprocessing pool.
    main()
