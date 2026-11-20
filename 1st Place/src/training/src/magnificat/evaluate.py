"""Cross-validation and metrics.

The primary view is GroupKFold on ``learning_objective_id``, because test objectives can be
new. A ``session``-grouped view shares all 398 objectives across folds, so a model can score
well on it by memorising objectives; it is an optimistic upper bound only.

Plain ``GroupKFold`` (no shuffle or seed) keeps the split deterministic on a given machine;
:mod:`magnificat.folds` holds the frozen partition. Whole objectives move together, so fold
base rates vary; :func:`fold_base_rates` reports them.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict


def group_oof(model, X, y, groups, n_splits: int = 5, n_jobs: int = 5):
    """Compute out-of-fold P(correct) under ``GroupKFold(groups)``.

    Every group's rows are confined to one fold, so the model never trains on a
    group it scores.

    Args:
        model: an unfitted sklearn estimator with ``predict_proba``.
        X: feature table, one row per response.
        y: 0/1 labels aligned to ``X``.
        groups: group key per row (e.g. ``learning_objective_id``).
        n_splits: number of folds.
        n_jobs: folds fitted in parallel.

    Returns:
        1-D array of out-of-fold probabilities in ``X``'s row order.
    """
    return cross_val_predict(
        model, X, y, cv=GroupKFold(n_splits=n_splits), groups=groups,
        method="predict_proba", n_jobs=n_jobs,
    )[:, 1]


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins.

    Args:
        y: 0/1 labels.
        p: predicted probabilities.
        n_bins: number of equal-width bins on [0, 1].

    Returns:
        The bin-weighted mean absolute gap between mean prediction and mean label.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def fold_base_rates(y, groups, n_splits: int = 5):
    """Positive rate of each ``GroupKFold`` test fold (shows how unstable LO-grouped folds are).

    Args:
        y: 0/1 labels.
        groups: group key per row.
        n_splits: number of folds.

    Returns:
        List of ``n_splits`` floats, the mean label of each test fold.
    """
    y = np.asarray(y, dtype=float)
    return [float(y[te].mean())
            for _, te in GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)]


# Score predictions with the standard metric set.
#
# Args:
#     y: 0/1 labels.
#     p: predicted probabilities.
#
# Returns:
#     Dict with logloss, auc, brier, ece, mean_pred, base_rate and n.
def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    return {
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "mean_pred": float(np.mean(p)),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def metrics_on_subset(y, p, mask) -> dict:
    """Score a subset of rows of full-data out-of-fold predictions.

    ``p`` must be computed on the full data (every fold trained on all groups); only the
    reporting is restricted to the subset. The subset base rate
    differs from the full set, so absolute log loss is not comparable across subsets;
    only differences between models scored on the same subset are meaningful.

    Args:
        y: 0/1 labels for all rows.
        p: out-of-fold probabilities for all rows.
        mask: boolean row mask selecting the subset.

    Returns:
        Dict with logloss, auc (NaN if the subset is single-class), ece, mean_pred,
        base_rate and n.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.asarray(mask, dtype=bool)
    ys, ps = y[m], p[m]
    out = {
        "logloss": float(log_loss(ys, ps, labels=[0.0, 1.0])),
        "auc": float("nan"),
        "ece": ece(ys, ps),
        "mean_pred": float(np.mean(ps)),
        "base_rate": float(ys.mean()),
        "n": int(m.sum()),
    }
    if len(np.unique(ys)) > 1:
        out["auc"] = float(roc_auc_score(ys, ps))
    return out


# Print a one-line metric summary and return the metrics.
#
# Args:
#     y: 0/1 labels.
#     p: predicted probabilities.
#     name: label printed at the start of the line.
#
# Returns:
#     The dict from metrics(y, p).
def report(y, p, name: str) -> dict:
    m = metrics(y, p)
    print(f"  [{name:>16}] logloss={m['logloss']:.4f}  AUC={m['auc']:.4f}  "
          f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}  "
          f"mean_pred={m['mean_pred']:.4f} (base {m['base_rate']:.4f})  n={m['n']}")
    return m
