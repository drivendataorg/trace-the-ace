"""Frozen global recalibrators for the submission, and wrappers that apply them.

Three monotone, ROC-AUC-preserving recalibrators, each fit on training or out-of-fold
data only and then frozen:

- ``GlobalRecenter``: one frozen bias in logit space; shifts the mean.
- ``ConfidenceShrink``: shrinks probabilities toward an anchor in logit space;
  tempers over-confidence (compresses spread) without moving the mean.
- ``PlattScaler``: two-parameter logistic map ``sigmoid(a*logit(p)+b)`` fit on
  out-of-fold predictions; suited to over-confident boosters.

Each is applied per sample at inference. No test-set statistic is computed at
inference time, so they comply with the competition rule against pooling across test
samples. ``CalibratedPipeline`` and ``SeedEnsembleCalibrated`` wrap a fitted model with
one of them behind the sklearn ``predict_proba`` contract.

The shipped model uses ``PlattScaler`` (both halves) and ``SeedEnsembleCalibrated`` (the
XGBoost half).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class GlobalRecenter:
    """Frozen recentring: one float ``bias_`` added in logit space.

    Args:
        bias: the logit-space shift.
    """

    def __init__(self, bias: float = 0.0):
        self.bias_ = float(bias)

    @classmethod
    def from_target_mean(cls, p_base, pi_test: float) -> "GlobalRecenter":
        """Solve for the bias that moves the mean prediction to a target rate.

        Solved on training predictions only, then frozen. The mean of
        ``sigmoid(logit(p) + bias)`` is monotone in the bias, so bisection on
        [-15, 15] (200 steps) converges.

        Args:
            p_base: 1-D array of training-set probabilities.
            pi_test: the target mean probability.

        Returns:
            A ``GlobalRecenter`` whose mean recentred probability over ``p_base`` equals
            ``pi_test``.
        """
        z = _logit(p_base)
        lo, hi = -15.0, 15.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _sigmoid(z + mid).mean() < pi_test:
                lo = mid
            else:
                hi = mid
        return cls(0.5 * (lo + hi))

    def transform(self, p):
        return np.clip(_sigmoid(_logit(p) + self.bias_), EPS, 1 - EPS)


class ConfidenceShrink:
    """Monotone, AUC-preserving shrink of probabilities toward ``anchor`` in logit space::

        p -> sigmoid( logit(anchor) + (logit(p) - logit(anchor)) * w )

    with ``0 < w <= 1``. ``w < 1`` reduces over-confidence (compresses the spread)
    while keeping the mean near the anchor, which suits a model that discriminates well
    but is too sure of itself. Strictly monotone in ``p`` for any ``w > 0``, so
    ROC-AUC is exactly preserved. One frozen parameter, applied per sample, no
    test-set statistic used. Unlike :class:`GlobalRecenter`, which shifts the mean,
    this squeezes the spread and keeps the centre.

    Args:
        w: shrink factor in logit space (1 = identity).
        anchor: the probability the predictions are shrunk toward.
    """

    def __init__(self, w: float = 1.0, anchor: float = 0.7025):
        self.w = float(w)
        self.anchor = float(anchor)

    def transform(self, p):
        a = _logit(self.anchor)
        return np.clip(_sigmoid(a + (_logit(p) - a) * self.w), EPS, 1 - EPS)


class PlattScaler:
    """Frozen two-parameter logistic recalibration in logit space::

        p -> sigmoid(a * logit(p) + b),   a > 0

    ``a, b`` are fit once on out-of-fold training predictions, then frozen. Strictly
    monotone in ``p`` for ``a > 0``, so applied as a single global map at inference it
    leaves ROC-AUC exactly unchanged; it can only fix calibration (rescale the spread via
    ``a``, shift the mean via ``b``).

    With only two parameters it cannot fit per-objective noise. ``a < 1`` shrinks the
    spread of an over-confident base. It holds two floats, so the pickle does not depend on
    library versions.

    Args:
        a: slope on ``logit(p)``.
        b: intercept.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def fit(cls, p_oof, y) -> "PlattScaler":
        """Fit ``a, b`` by 1-D logistic regression of ``y`` on ``logit(p_oof)``.

        Args:
            p_oof: 1-D array of out-of-fold training predictions (must be out-of-fold).
            y: their 0/1 labels.

        Returns:
            The fitted ``PlattScaler``.
        """
        from sklearn.linear_model import LogisticRegression

        z = _logit(p_oof).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(z, np.asarray(y))
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def transform(self, p):
        return np.clip(_sigmoid(self.a * _logit(p) + self.b), EPS, 1 - EPS)


class CalibratedPipeline:
    """Wrap a fitted estimator so ``predict_proba`` applies a frozen 1-D calibrator.

    Exposes the sklearn ``predict_proba`` contract that ``models/submission/main.py``
    calls (``model.predict_proba(X)[:, 1]``). Pickles cleanly: the base estimator plus a
    small calibrator.

    Args:
        base: a fitted estimator with ``predict_proba``.
        calibrator: any object exposing ``transform(p1d) -> p1d``
            (:class:`GlobalRecenter`, :class:`ConfidenceShrink`, :class:`PlattScaler`).
    """

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(base, "classes_", np.array([0, 1])))

    @property
    def recenter(self):  # alias of `calibrator`
        return self.calibrator

    def predict_proba(self, X):
        p = self.calibrator.transform(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SeedEnsembleCalibrated:
    """Average K seed-varied base estimators, then apply a frozen calibrator.

    Like :class:`CalibratedPipeline`, but the base probabilities are the mean over the
    seed ensemble (variance reduction). Same ``predict_proba(X)[:, 1]`` contract that
    ``models/submission/main.py`` calls. Pickles cleanly (K small XGBoost pipelines). The
    shipped XGBoost half is one of these: five XGBoost seeds and a frozen :class:`PlattScaler`.

    Args:
        bases: the fitted seed-varied estimators.
        calibrator: an object exposing ``transform(p1d) -> p1d``.
    """

    def __init__(self, bases, calibrator):
        self.bases = list(bases)
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(self.bases[0], "classes_", np.array([0, 1])))

    def predict_proba(self, X):
        p = np.mean([b.predict_proba(X)[:, 1] for b in self.bases], axis=0)
        p = self.calibrator.transform(p)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class IdentityScaler:
    """A calibrator that returns its input, for models scored on AUROC alone.

    Wrapping the seed ensemble in this keeps the ``predict_proba`` contract without
    claiming the probabilities are calibrated.
    """

    def transform(self, p):
        return np.asarray(p, float)

    def __repr__(self):
        return "IdentityScaler()"
