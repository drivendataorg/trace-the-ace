"""The pipeline of the XGBoost half, kept in the vendored package so it unpickles.

The fitted pipeline contains a `FunctionTransformer` that holds a reference to
:func:`densify`. joblib pickles that reference by module path, so if the function lived in
a training script, the submission container (which only vendors `magnificat`) would raise
`ModuleNotFoundError` on load.

XGBoost: `xgboost 3.3.0` is in the official runtime (added in
tutoring-outcomes-runtime@8581996).

Dense input: the preprocessor emits a sparse matrix, and XGBoost reads an unstored CSR
entry as missing, which would collapse E2's 0-vs-NaN encoding (`confusion_rate = 0` means
"never signalled confusion", NaN means "could not measure"). The shipped booster was fitted
on dense input, so the pipeline casts to dense.
"""
from __future__ import annotations

import numpy as np

#: The booster parameters of the shipped XGBoost model, grown `lossguide` at 31 leaves.
#: `min_child_weight=126` is a hessian floor of 600 x p(1-p) at p~0.7, i.e. about 600 rows
#: per leaf.
E3_XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=0, grow_policy="lossguide",
    max_leaves=31, min_child_weight=126.0, reg_lambda=80.0, reg_alpha=10.0,
    colsample_bytree=0.35, subsample=0.6, gamma=0.02, tree_method="hist",
)


def densify(Z):
    """Convert a (sparse) matrix to dense float32.

    An unstored entry then stays 0 instead of becoming XGBoost 'missing'; NaN survives as
    NaN and is still routed as missing. The downcast happens before materialising: the
    TF-IDF block is float64, so calling `toarray()` first would peak at 2x the memory of
    the dense matrix it is about to throw away.

    Args:
        Z: scipy sparse matrix or array-like.

    Returns:
        Dense float32 numpy array.
    """
    if hasattr(Z, "toarray"):
        return Z.astype(np.float32).toarray()
    return np.asarray(Z, dtype=np.float32)


def build_e3_xgb_pipeline(dialogue_cols, n_jobs: int = -1, random_state: int = 42,
                          **overrides):
    """Build the XGBoost pipeline: preprocessor, dense cast, XGBClassifier.

    The preprocessor is the one of
    :func:`magnificat.baseline.build_meta_dialogue_gbm_pipeline` (TF-IDF over the objective
    + one-hot LO id + passthrough numerics), reused unchanged.

    Args:
        dialogue_cols: names of the numeric feature columns passed through.
        n_jobs: XGBoost threads.
        random_state: XGBoost seed.
        **overrides: XGBoost parameters that replace entries of ``E3_XGB_PARAMS``.

    Returns:
        An unfitted ``Pipeline`` with steps ``pre``, ``dense`` and ``clf``.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from xgboost import XGBClassifier

    from .baseline import build_meta_dialogue_gbm_pipeline

    pre = build_meta_dialogue_gbm_pipeline(dialogue_cols).named_steps["pre"]
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                        missing=np.nan, n_jobs=n_jobs, random_state=random_state,
                        verbosity=0, **dict(E3_XGB_PARAMS, **overrides))
    return Pipeline([("pre", pre),
                     ("dense", FunctionTransformer(densify, accept_sparse=True)),
                     ("clf", clf)])
