"""The booster for the XGBoost half, in the vendored package so it unpickles.

**Why this is not in `scripts/`.** The fitted pipeline contains a `FunctionTransformer`
that holds a reference to :func:`densify`. joblib pickles that reference by
*module path*, so if the function lived in a training script the submission
container, which only vendors `magnificat`, would raise `ModuleNotFoundError` on load.
Keeping it here is what makes the model shippable at all.

**XGBoost.** `xgboost 3.3.0` is in the official runtime (tutoring-outcomes-runtime@8581996).

**Why dense.** The preprocessor emits a sparse matrix, and XGBoost reads an unstored CSR
entry as *missing* rather than `0`, which would collapse E2's deliberate 0-vs-NaN encoding
(`confusion_rate = 0` = "never signalled confusion" vs NaN = "could not measure"). The
booster was trained on dense input, and inference densifies the same way.
"""
from __future__ import annotations

import numpy as np

#: Booster parameters: `lossguide` growth with at most 31 leaves per tree; `min_child_weight=126`
#: is the hessian of about 600 samples at p~0.7 (600 x p(1-p)).
E3_XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=0, grow_policy="lossguide",
    max_leaves=31, min_child_weight=126.0, reg_lambda=80.0, reg_alpha=10.0,
    colsample_bytree=0.35, subsample=0.6, gamma=0.02, tree_method="hist",
)


def densify(Z):
    """Sparse -> dense float32 so an unstored entry stays 0 instead of becoming
    XGBoost 'missing'. NaN survives as NaN and is still routed as missing.

    Downcast BEFORE materialising: the TF-IDF block is float64, so `toarray()` first
    would peak at 2x the memory of the dense matrix it is about to throw away."""
    if hasattr(Z, "toarray"):
        return Z.astype(np.float32).toarray()
    return np.asarray(Z, dtype=np.float32)


def build_e3_xgb_pipeline(dialogue_cols, n_jobs: int = -1, random_state: int = 42,
                          **overrides):
    """The shipped XGBoost pipeline: the SAME preprocessor as
    :func:`magnificat.baseline.build_meta_dialogue_gbm_pipeline` (TF-IDF over the
    objective + one-hot LO id + passthrough numerics), a dense cast, then XGBoost.
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
