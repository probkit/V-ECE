"""Build a semi-synthetic ground truth for the calibration error of the TabRepo classifiers.

A calibration error estimator can only be validated against a known truth, which real data
never provides.  We therefore turn each experiment into a semi-synthetic one:

1. Fit a tabular foundation model (TabICL v2) on the *calibration* split, using the
   predictions ``p_cal`` of the classifier as features and the outcomes ``y_cal`` as
   targets.  The fitted model ``g`` approximates ``E[Y | f(X)]``.  In the binary case the
   single feature is the class-1 probability; in the multiclass case it is the whole
   probability vector.
2. Apply it to the *test* split.  The resulting ``C = g(p_test)`` is declared to be the
   true calibrated distribution of the test predictions.

A caveat worth keeping in mind: ``C`` inherits the estimation noise of TabICL, and any such
noise inflates the calibration error of the semi-synthetic task relative to the real one,
since ``mean_i d(f_i, C_i)`` grows with noise (Jensen) while the held-out log-loss that
would reveal it barely moves.  Fitting two independent contexts and comparing their
predictions puts that noise at roughly 20-30% of the true calibration error on the
miscalibrated experiments.  This does not bias the benchmark, in which ``C`` is the truth
by construction, but it makes the semi-synthetic calibration map somewhat rougher than the
real one and hence slightly harder to recover for every estimator.  Averaging
several context draws (``--n_context_draws``) reduces the noise only marginally for twice
the compute, so the default is a single fit; when more than one draw is requested the
individual draws are also stored, as ``probas_true_draws``.

Because ``C`` is a function of ``f(X)`` alone, re-drawing the test labels from ``C``
(done by ``benchmark_ce_estimators.py``) gives a distribution whose calibration error is
known exactly: ``E[Y | f(X)] = C``, so ``CE_d(f) = mean d(f(X_i), C_i)``.  The calibration
split is never shown to the estimators, which see only ``(p_test, re-drawn labels)``.

This script only performs step 1-2 and caches ``C``; it is the expensive part and needs to
run once.  It requires the ``tabicl`` package, which the benchmark itself does not.

Outputs ``{out_dir}/ground-truth-{modality}.h5``, mirroring the layout of the TabRepo
files: one group ``{dataset}/{model}`` per experiment holding ``probas_true`` of shape
``(n_test, n_classes)``.  Re-running the script only fills in the missing groups, so an
interrupted run can be resumed.

Usage
-----
    python -m experiments.make_ground_truth                       # both modalities
    python -m experiments.make_ground_truth --modality binary --device mps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

MODALITIES = ("binary", "multiclass")

# TabICL is an in-context learner: its cost grows with the number of calibration samples
# put in context, and so does its memory.  The calibration splits hold up to 91k samples,
# far more than needed to pin down E[Y | f(X)] from a k-dimensional feature, so we cap the
# context at `--max_cal_size` samples drawn uniformly at random.
DEFAULT_MAX_CAL_SIZE = 10_000


def as_feature_matrix(probas: np.ndarray) -> np.ndarray:
    """Return predictions with shape ``(n, k)``; TabRepo stores binary ones as a vector."""
    probas = np.asarray(probas, dtype=np.float32)
    return probas.reshape(-1, 1) if probas.ndim == 1 else probas


def fit_one_draw(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    max_cal_size: int,
    device: str,
    n_estimators: int,
    seed: int,
) -> np.ndarray:
    """Fit TabICL on one context subsample and return ``g(p_test)``, shape ``(n, k)``."""
    from tabicl import TabICLClassifier

    rng = np.random.RandomState(seed)
    context = rng.choice(len(p_cal), min(max_cal_size, len(p_cal)), replace=False)

    model = TabICLClassifier(
        device=device, random_state=seed, n_estimators=n_estimators, allow_auto_download=True
    )
    model.fit(as_feature_matrix(p_cal)[context], np.asarray(y_cal)[context])
    probas = model.predict_proba(as_feature_matrix(p_test))

    # TabICL only returns columns for the classes it saw in context; restore the full width
    # so that rare classes keep their (near-zero) probability slot.
    n_classes = max(as_feature_matrix(p_test).shape[1], 2)
    if probas.shape[1] != n_classes:
        full = np.zeros((len(probas), n_classes), dtype=probas.dtype)
        full[:, np.asarray(model.classes_, dtype=int)] = probas
        probas = full

    return probas.astype(np.float32)


def fit_ground_truth(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    max_cal_size: int,
    device: str,
    n_estimators: int,
    n_context_draws: int,
    seed: int,
) -> np.ndarray:
    """Stack of ``n_context_draws`` fits of ``C = g(p_test)``, shape ``(n_draws, n, k)``."""
    draws = [
        fit_one_draw(
            p_cal, y_cal, p_test, max_cal_size, device, n_estimators, seed + draw
        )
        for draw in range(n_context_draws)
    ]
    return np.stack(draws)


def run(modality: str, args: argparse.Namespace) -> None:
    """Fit and cache the ground truth of every experiment of one modality."""
    experiments = pd.read_csv(Path(args.data_dir) / f"experiments_{modality}.csv", index_col=0)
    tabrepo_file = Path(args.data_dir) / f"tabrepo-{modality}.h5"
    output_file = Path(args.out_dir) / f"ground-truth-{modality}.h5"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(tabrepo_file, "r") as tabrepo, h5py.File(output_file, "a") as out:
        out.attrs["max_cal_size"] = args.max_cal_size
        out.attrs["n_estimators"] = args.n_estimators
        out.attrs["n_context_draws"] = args.n_context_draws
        out.attrs["seed"] = args.seed

        progress = tqdm(experiments.iterrows(), total=len(experiments), desc=modality)
        for _, experiment in progress:
            key = f"{experiment['dataset']}/{experiment['model']}"
            if f"{key}/probas_true" in out:
                continue  # already computed by an earlier run

            group = tabrepo[key]
            start = time.perf_counter()
            draws = fit_ground_truth(
                p_cal=group["probas_cal"][:],
                y_cal=group["labels_cal"][:],
                p_test=group["probas_test"][:],
                max_cal_size=args.max_cal_size,
                device=args.device,
                n_estimators=args.n_estimators,
                n_context_draws=args.n_context_draws,
                seed=args.seed,
            )
            if args.n_context_draws > 1:
                out.create_dataset(f"{key}/probas_true_draws", data=draws)
            dataset = out.create_dataset(f"{key}/probas_true", data=draws.mean(axis=0))
            dataset.attrs["cal_size"] = len(group["probas_cal"])
            dataset.attrs["n_context_draws"] = args.n_context_draws
            dataset.attrs["fit_seconds"] = time.perf_counter() - start
            out.flush()  # keep the file resumable after an interruption
            progress.set_postfix_str(f"{key} in {time.perf_counter() - start:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modality", choices=MODALITIES, default=None, help="Omit to run both.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="TabRepo HDF5 files.")
    parser.add_argument("--out_dir", type=Path, default=Path("data"), help="Where to cache C.")
    parser.add_argument(
        "--max_cal_size",
        type=int,
        default=DEFAULT_MAX_CAL_SIZE,
        help="Cap on the number of calibration samples put in TabICL's context.",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=8, help="Size of TabICL's internal ensemble."
    )
    parser.add_argument(
        "--n_context_draws",
        type=int,
        default=1,
        help="Independent context subsamples to average the ground truth over.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for TabICL (cpu, mps, cuda).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the context subsample.")
    args = parser.parse_args()

    for modality in MODALITIES if args.modality is None else (args.modality,):
        run(modality, args)


if __name__ == "__main__":
    main()
