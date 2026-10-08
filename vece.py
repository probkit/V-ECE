"""V-ECE: variational L_p calibration error with a warm-started CatBoost recalibrator (WS-CatBoost).

Dependencies: numpy, scipy, scikit-learn, catboost (joblib, shipped with scikit-learn, only for the physical core count).
estimate_ce(probs, y, metric): probs (n, k) predicted probabilities (binary: 2 columns), y (n,) labels in 0..k-1,
metric 'l1' (binary CE_|.|), 'l2' (multiclass CE_||.||_2) or 'brier'. Returns CE_hat = max(0, mean l(f,Y) - mean l(g,Y)):
the excess risk is aggregated over all out-of-fold predictions first and clipped at 0 only then, since the calibration
error is nonnegative (clipping can only reduce the error of the estimate; clipping each fold would bias it upwards).

Scheme: n_outer-fold CV; on each outer training set, temperature scaling is fitted and n_inner CatBoost models are
trained on the inner folds (warm start from the temperature-scaled logits, early stopping on the held-out inner fold).
The recalibrated prediction for the outer test fold is the average of the n_inner models. All fits run in one pool.
"""
import multiprocessing as mp

import numpy as np
from catboost import CatBoostClassifier, Pool
from joblib import cpu_count
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import KFold

CATBOOST_PARAMS = dict(iterations=100, early_stopping_rounds=20, learning_rate=0.2, bootstrap_type="Bernoulli",
                       subsample=0.5, border_count=32, thread_count=1, verbose=0, allow_writing_files=False)


def proper_loss(g, y, f, metric):
    """Sample-dependent proper loss l_f(g, Y) with l_f(f, Y) = 0; f = original predictions, g = recalibrated."""
    yo = np.eye(g.shape[1])[y]
    if g.shape[1] == 2:  # binary: l1 -> CE_|.|, brier -> CE_(.)^2 on the class-1 probability
        return (g[:, 1] - y) ** 2 - (f[:, 1] - y) ** 2 if metric == "brier" else np.sign(g[:, 1] - f[:, 1]) * (f[:, 1] - y)
    if metric == "brier":
        return np.sum((g - yo) ** 2 - (f - yo) ** 2, axis=1)
    p = {"l1": 1, "l2": 2}[metric]
    d = g - f
    norm = np.maximum(np.linalg.norm(d, ord=p, axis=1, keepdims=True), 1e-300)
    return np.sum(np.abs(d) ** (p - 1) * np.sign(d) / norm ** (p - 1) * (f - yo), axis=1)


class Passthrough(ClassifierMixin, BaseEstimator):
    """Classifier returning its input as predict_proba; lets CalibratedClassifierCV temperature-scale probabilities."""

    def fit(self, X, y):
        self.classes_ = np.arange(X.shape[1])
        return self

    def predict_proba(self, X):
        return X

    def predict(self, X):
        return X.argmax(axis=1)


def _pad_absent_classes(X, y, k):
    """Append one zero-weight row per class absent from y, returning (X, y, sample_weight).

    CatBoost and scikit-learn both infer the number of classes from the labels they are given, so a fold that
    happens to miss a rare class produces outputs narrower than the k columns the rest of the code expects. The
    padded rows declare the missing classes without contributing to any fit, since they carry zero weight."""
    missing = np.setdiff1d(np.arange(k), y)
    if len(missing) == 0:
        return X, np.asarray(y), None  # None, not ones: leaves the unpadded path untouched
    pad = np.zeros(len(missing), dtype=int)  # any row will do, the padded rows carry zero weight
    return (np.concatenate([X, X[pad]]), np.concatenate([y, missing]),
            np.concatenate([np.ones(len(y)), np.zeros(len(missing))]))


def _pool(features, labels, baseline, k):
    """CatBoost pool that always declares all k classes (see _pad_absent_classes)."""
    feats, labs, weight = _pad_absent_classes(features, labels, k)
    if weight is not None:
        baseline = np.concatenate([baseline, baseline[np.zeros(len(labs) - len(labels), dtype=int)]])
    return Pool(feats, label=labs, baseline=baseline, weight=weight)


def _fit_predict(P, Q, y, tr, va, te, params):
    """Fit one CatBoost on inner fold tr (validation va) and return its recalibrated probabilities for the test fold te.
    P: (mixed) input probabilities, Q: their temperature-scaled version (the warm start)."""
    k = P.shape[1]
    feats = lambda idx: P[idx, 1:] if k == 2 else P[idx]
    raw0 = lambda idx: np.log(Q[idx, 1] / Q[idx, 0]) if k == 2 else np.log(Q[idx])  # baseline logits
    pools = [_pool(feats(idx), y[idx], raw0(idx), k) for idx in (tr, va)]
    m = CatBoostClassifier(**params).fit(pools[0], eval_set=pools[1])
    raw = m.predict(feats(te), prediction_type="RawFormulaVal") + raw0(te)
    return np.stack([1 - expit(raw), expit(raw)], 1) if k == 2 else softmax(raw, axis=1)


def recalibrate_oof(probs, y, n_outer=5, n_inner=8, seed=0, n_jobs=None, **catboost_params):
    """Out-of-fold recalibrated probabilities g(f(X_i)), shape (n, k). n_jobs=None uses all physical cores."""
    probs, y = np.asarray(probs, float), np.asarray(y).astype(int)
    n, k = probs.shape
    lam = 1.0 / (n * (n_outer - 1) // n_outer + 1)  # mix with uniform so that log(0) cannot occur
    P = (1 - lam) * probs + lam / k
    params = {**CATBOOST_PARAMS, **catboost_params,
              "loss_function": "Logloss" if k == 2 else "MultiClass", **({"classes_count": k} if k > 2 else {})}
    jobs, g = [], np.zeros((n, k))
    for o, (tr, te) in enumerate(KFold(n_outer, shuffle=True, random_state=seed).split(P)):
        Xtr, ytr, wtr = _pad_absent_classes(P[tr], y[tr], k)  # keep all k classes visible to sklearn
        ts = CalibratedClassifierCV(FrozenEstimator(Passthrough().fit(P, y)),
                                    method="temperature").fit(Xtr, ytr, sample_weight=wtr)
        Q = np.clip(ts.predict_proba(P), 1e-12, 1)
        for i, (itr, iva) in enumerate(KFold(n_inner, shuffle=True, random_state=seed).split(tr)):
            jobs.append((P, Q, y, tr[itr], tr[iva], te, {**params, "random_state": seed * 1000 + o * n_inner + i}))
    with mp.Pool(n_jobs or cpu_count(only_physical_cores=True)) as pool:
        preds = pool.starmap(_fit_predict, jobs)
    for job, pred in zip(jobs, preds):
        g[job[5]] += pred / n_inner
    return g


def estimate_ce(probs, y, metric="l1", **kw):
    probs, y = np.asarray(probs, float), np.asarray(y).astype(int)
    g = recalibrate_oof(probs, y, **kw)
    # l_f(f, Y) = 0, so CE_hat = -mean l_f(g, Y); clipped after averaging over all out-of-fold predictions.
    return max(0.0, float(-np.mean(proper_loss(g, y, probs, metric))))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    p1 = rng.beta(0.5, 0.5, 3000)
    y = (rng.uniform(size=3000) < expit(0.4 * np.log(p1 / (1 - p1)) + 0.3)).astype(int)  # over-confident predictions
    f = np.stack([1 - p1, p1], 1)
    print("CE_l1 ~", estimate_ce(f, y, "l1"), " CE_brier ~", estimate_ce(f, y, "brier"))