"""Two optional XGBoost boosters: constrained interactions, and a ranking objective.

Used by ``src/training/scripts/train_model_e3l.py --variant icon|rank|rankndcg``.

Both resolve their configuration at fit time, because the matrix width is not known before
then: the ColumnTransformer emits ``[tfidf | onehot(lo_id) | passthrough(numeric)]`` and the
TF-IDF is refit per fold, so its width moves (1245 / 1245 / 1272 / 1268 / 1281 across the five
frozen folds). The numeric block is passthrough and last, so anything indexed from the end is
stable.

`icon`: interaction_constraints
-------------------------------
Two groups: the LO-sparse block `[0, n-n_dlg)` and the transcript block `[n-n_dlg, n)`. A tree
branch may then combine columns within a group but never across, which stops the booster from
spending splits on LO-identity × transcript interactions.

`rank`: a ranking objective that grows trees
--------------------------------------------
``E3_XGB_PARAMS`` carries `min_child_weight=126`, a leaf-mass floor set for logistic
hessians ≈ p(1-p) on 35,072 rows. A pairwise-ranking hessian is a per-pair lambda weight,
orders of magnitude smaller, so with those penalties no split clears the floor and XGBoost
returns a constant. `reg_alpha=10` and `gamma=0.02` add to it. Hence
this module scales the four sample-size-dependent penalties and asserts that the scores are not
constant.

`qid` groups by learning_objective_id, i.e. students are ranked within an objective, and it
nests inside the folds, so a query never spans train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier as _XGBClassifier
from xgboost import XGBRanker as _XGBRanker

#: The four sample-size-dependent penalties of ``E3_XGB_PARAMS``, all compared against sums
#: of hessians.
_HESSIAN_SCALED = ("min_child_weight", "reg_lambda", "reg_alpha", "gamma")


# Both classes are defined at module level, not inside their factory. joblib/loky pickles a
# class by module path, so a class created inside a function is unpicklable, and the fit fails
# with `Can't pickle <class '...IconXGB'>: it's not found as ...IconXGB` as soon as the trainer
# hands it to a worker. `xgb_pipeline.py`'s docstring describes the same issue for `densify`,
# which is why that function lives in the package rather than in a script.
class IconXGB(_XGBClassifier):
    """XGBClassifier with LO-sparse and transcript columns barred from interacting.

    Build it with :func:`make_icon_xgb`, which sets ``_n_dlg``, the number of trailing
    transcript (passthrough) columns.
    """

    def fit(self, X, y, **fk):
        """Set ``interaction_constraints`` from the matrix width, then fit.

        Args:
            X: the preprocessed feature matrix; its last ``_n_dlg`` columns are the
                transcript block.
            y: 0/1 labels.
            **fk: passed to ``XGBClassifier.fit``.

        Returns:
            self.

        Raises:
            ValueError: ``_n_dlg`` does not fit inside the matrix.
        """
        n = X.shape[1]
        if not 0 < self._n_dlg < n:
            raise ValueError(f"n_dlg={self._n_dlg} outside a {n}-column matrix")
        lo, dlg = list(range(0, n - self._n_dlg)), list(range(n - self._n_dlg, n))
        self.set_params(interaction_constraints=f"[{lo},{dlg}]".replace(" ", ""))
        return super().fit(X, y, **fk)


def make_icon_xgb(n_dlg: int, **kw):
    """Create an :class:`IconXGB`.

    Args:
        n_dlg: number of trailing transcript (passthrough) columns in the matrix.
        **kw: XGBClassifier parameters.

    Returns:
        An unfitted :class:`IconXGB`.
    """
    m = IconXGB(**kw)
    m._n_dlg = int(n_dlg)
    return m


class RankXGB(_XGBRanker):
    """A ranking booster exposing `predict_proba`, with the leaf-mass floor scaled down.

    ``predict_proba`` returns ``[1-s, s]`` with ``s = sigmoid(margin)``. The margin is a score,
    not a calibrated probability. That is fine: the frozen Platt map downstream is monotone,
    so AUROC is unaffected by it, and log loss comes from the calibration exactly as for the
    classifier. Build it with :func:`make_rank_xgb`.
    """

    def fit(self, X, y, qid=None, **fk):
        """Sort rows into contiguous queries, fit, and check that the scores are not constant.

        Args:
            X: the preprocessed feature matrix (DataFrame or array).
            y: 0/1 labels.
            qid: query id per row (learning_objective_id); strings are factorised to integer
                codes. Pass it as ``clf__qid`` to ``Pipeline.fit``.
            **fk: passed to ``XGBRanker.fit``.

        Returns:
            self.

        Raises:
            ValueError: ``qid`` is missing or has the wrong length.
            RuntimeError: the fitted booster scores the first 5,000 rows identically (it grew
                no tree).
        """
        if qid is None:
            raise ValueError("rank objective needs qid: pass clf__qid to Pipeline.fit")
        # qid must be integer codes. `learning_objective_id` is a string like 'aabhqcm', and
        # XGBoost's DMatrix casts qid to float -> `could not convert string to float`.
        # factorize() preserves the grouping under the stable argsort below, so the grouping
        # is identical either way.
        qid = np.asarray(qid)
        if qid.dtype.kind in "OUS":
            qid = pd.factorize(qid)[0]
        qid = np.asarray(qid)
        if len(qid) != len(y):
            raise ValueError(f"qid has {len(qid)} rows, y has {len(y)}")
        # XGBoost requires contiguous queries. The trainer stacks an ablated copy after the real
        # rows, so the same objective appears in 1+ncopy disjoint blocks; sorting is required.
        o = np.argsort(qid, kind="stable")
        Xs = X.iloc[o] if hasattr(X, "iloc") else X[o]
        super().fit(Xs, np.asarray(y)[o], qid=qid[o], **fk)
        # a constant score means the booster grew no tree
        head = Xs.iloc[:5000] if hasattr(Xs, "iloc") else Xs[:5000]
        if len(np.unique(np.round(self.predict(head), 9))) < 2:
            raise RuntimeError(
                "rank variant produced a constant score: the booster grew no tree. Lower "
                "--rank-scale further. A constant score is a failed run, not a null result.")
        return self

    def predict_proba(self, X):
        """Map ranking scores to ``[1-s, s]`` with ``s = sigmoid(score)``.

        Args:
            X: the preprocessed feature matrix.

        Returns:
            Array (n_rows, 2).
        """
        s = 1.0 / (1.0 + np.exp(-np.asarray(self.predict(X), dtype=float)))
        return np.column_stack([1.0 - s, s])


def make_rank_xgb(objective: str = "rank:pairwise", scale: float = 0.01, **kw):
    """Create a :class:`RankXGB` with the hessian-compared penalties scaled down.

    ``scale`` multiplies the four penalties in ``_HESSIAN_SCALED``: ``min_child_weight`` goes
    from 126 to 1.26 at the default. At 126 the booster grows no tree and returns a constant.

    Args:
        objective: XGBoost ranking objective.
        scale: multiplier for the four hessian-compared penalties.
        **kw: XGBoost parameters (e.g. ``E3_XGB_PARAMS``); ``eval_metric`` is dropped because
            log loss is not defined for a ranking objective.

    Returns:
        An unfitted :class:`RankXGB`.
    """
    cfg = dict(kw)
    for k in _HESSIAN_SCALED:
        if k in cfg:
            cfg[k] = float(cfg[k]) * scale
    cfg["objective"] = objective
    cfg.pop("eval_metric", None)          # logloss is not defined for a ranking objective
    return RankXGB(**cfg)
