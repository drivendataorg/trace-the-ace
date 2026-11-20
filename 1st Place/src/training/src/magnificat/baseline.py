"""Sklearn pipelines: transcript TF-IDF and/or learning-objective features fed to a
gradient-boosting or logistic-regression classifier.

Each builder returns one untrained sklearn ``Pipeline`` (or estimator), so it serialises
cleanly and the runtime predict path is just ``predict_proba``:

- ``build_pipeline``: TF-IDF -> SVD (dense) -> HistGradientBoosting.
- ``build_xgb_pipeline``: full sparse TF-IDF + structured columns -> XGBoost.
- ``build_lgbm_pipeline``: the same with LightGBM.
- ``build_meta_pipeline`` / ``build_meta_gbm_pipeline`` / ``build_meta_calibrated_pipeline``:
  learning-objective-only models (no transcript).
- ``build_meta_dialogue_gbm_pipeline``: the objective features plus hand-crafted
  transcript columns. The shipped XGBoost model reuses its preprocessor
  (:func:`magnificat.xgb_pipeline.build_e3_xgb_pipeline`).

Only that preprocessor is used by the shipped model; the other builders are optional.

The competition metric is log loss, so we want calibrated probabilities. We therefore do
not use ``scale_pos_weight`` (it would shift predictions off the ~70% base rate to
optimise a ranking objective, hurting log loss) and train XGBoost directly on
``binary:logistic`` / ``logloss``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import STRUCTURED_COLS

TEXT_COL = "transcript_text"


def _to_dense(x):
    # TruncatedSVD output is already dense; HGB needs dense arrays anyway.
    return x


def build_pipeline(
    max_features: int = 50_000,
    svd_components: int = 200,
    random_state: int = 42,
) -> Pipeline:
    """Build the untrained TF-IDF + SVD + HistGradientBoosting baseline.

    Text -> TF-IDF -> TruncatedSVD (LSA) to a dense block, concatenated with the
    structured numeric columns, then HistGradientBoosting.

    Args:
        max_features: TF-IDF vocabulary size.
        svd_components: number of SVD components kept from the TF-IDF matrix.
        random_state: seed for the SVD and the booster.

    Returns:
        An unfitted ``Pipeline`` that expects ``TEXT_COL`` and ``STRUCTURED_COLS`` in ``X``.
    """
    from sklearn.decomposition import TruncatedSVD

    text_branch = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
        )),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=random_state)),
    ])

    pre = ColumnTransformer([
        ("text", text_branch, TEXT_COL),
        ("num", FunctionTransformer(_to_dense, validate=False), STRUCTURED_COLS),
    ])

    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=400,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )

    return Pipeline([("pre", pre), ("clf", clf)])


def build_xgb_pipeline(
    max_features: int = 50_000,
    min_df: int = 3,
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.5,
    reg_lambda: float = 2.0,
    min_child_weight: float = 5.0,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Build an untrained sparse TF-IDF + structured features -> XGBoost pipeline.

    The TF-IDF matrix is fed to XGBoost sparse (no SVD), so the trees can split
    on individual n-grams. ``ColumnTransformer`` hstacks the sparse text block
    with the dense structured columns; the result stays sparse (CSR), which
    XGBoost consumes natively.

    Tuned for log loss: ``binary:logistic`` objective, ``logloss`` eval metric,
    no ``scale_pos_weight`` so predictions stay calibrated to the base rate.

    Args:
        max_features: TF-IDF vocabulary size.
        min_df: minimum document frequency of an n-gram.
        n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda,
            min_child_weight, n_jobs, random_state: passed to ``XGBClassifier``.

    Returns:
        An unfitted ``Pipeline`` that expects ``TEXT_COL`` and ``STRUCTURED_COLS`` in ``X``.
    """
    from xgboost import XGBClassifier

    text_branch = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
    )

    # sparse_threshold=1.0 -> always return a sparse hstack when any block is sparse
    pre = ColumnTransformer(
        [
            ("text", text_branch, TEXT_COL),
            ("num", FunctionTransformer(_to_dense, validate=False), STRUCTURED_COLS),
        ],
        sparse_threshold=1.0,
    )

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return Pipeline([("pre", pre), ("clf", clf)])


def build_meta_pipeline(
    C: float = 1.0,
    max_features: int = 5_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = False,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    """Build the metadata-only model: logistic regression on the learning objective alone.

    No transcript text: the prediction comes only from the learning objective the
    follow-up question targets. We combine
    - a one-hot of ``learning_objective_id`` (the per-objective rate, regularised by the
      logistic penalty and so smoothed toward the global rate), and
    - word TF-IDF of the ``learning_objective`` text (lets similar objectives share
      strength and handles ids unseen in training).

    Logistic regression keeps probabilities calibrated, which suits log loss.

    A small ``C`` shrinks toward the base rate. The pipeline has no post-hoc calibration
    step.

    Args:
        C: inverse L2 regularisation strength of the logistic regression.
        max_features: TF-IDF vocabulary size for the objective text.
        ngram_range: word n-gram range of the TF-IDF.
        sublinear_tf: use ``1 + log(tf)`` term frequencies.
        max_iter: solver iteration cap.
        random_state: seed for the solver.

    Returns:
        An unfitted ``Pipeline`` that expects ``learning_objective`` and
        ``learning_objective_id`` in ``X``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    pre = ColumnTransformer(
        [
            ("obj_text", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                         sublinear_tf=sublinear_tf),
             "learning_objective"),
            ("obj_id", OneHotEncoder(handle_unknown="ignore"),
             ["learning_objective_id"]),
        ],
        sparse_threshold=1.0,
    )
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_meta_gbm_pipeline(
    max_features: int = 5_000,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 50,
    reg_lambda: float = 5.0,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Build the metadata-only LightGBM model.

    Same transcript-free features as :func:`build_meta_pipeline` (one-hot of
    ``learning_objective_id`` + TF-IDF of the objective text), but a LightGBM
    tree booster instead of logistic regression.

    Regularised: few leaves, high ``min_child_samples``/``reg_lambda``.

    Args:
        max_features: TF-IDF vocabulary size for the objective text.
        n_estimators, learning_rate, num_leaves, min_child_samples, reg_lambda, n_jobs,
            random_state: passed to ``LGBMClassifier``.

    Returns:
        An unfitted ``Pipeline`` that expects ``learning_objective`` and
        ``learning_objective_id`` in ``X``.
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import OneHotEncoder

    pre = ColumnTransformer(
        [
            ("obj_text", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)),
             "learning_objective"),
            ("obj_id", OneHotEncoder(handle_unknown="ignore"),
             ["learning_objective_id"]),
        ],
        sparse_threshold=1.0,
    )
    clf = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_meta_dialogue_gbm_pipeline(
    dialogue_cols,
    max_features: int = 5_000,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 50,
    reg_lambda: float = 5.0,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Build the objective + hand-crafted transcript features -> LightGBM model.

    The metadata-only features of :func:`build_meta_gbm_pipeline` (one-hot
    ``learning_objective_id`` + TF-IDF of the objective text) plus the hand-crafted
    transcript ``dialogue_cols`` (passed through as dense numeric), in the same LightGBM
    booster with the same hyper-parameters. The shipped XGBoost model uses only the ``pre``
    step of this pipeline.

    Args:
        dialogue_cols: names of the numeric transcript feature columns.
        max_features: TF-IDF vocabulary size for the objective text.
        n_estimators, learning_rate, num_leaves, min_child_samples, reg_lambda, n_jobs,
            random_state: passed to ``LGBMClassifier``.

    Returns:
        An unfitted ``Pipeline``; ``X`` must carry ``learning_objective``,
        ``learning_objective_id`` and every column in ``dialogue_cols``.
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import OneHotEncoder

    pre = ColumnTransformer(
        [
            ("obj_text", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)),
             "learning_objective"),
            ("obj_id", OneHotEncoder(handle_unknown="ignore"),
             ["learning_objective_id"]),
            ("dlg", "passthrough", list(dialogue_cols)),
        ],
        sparse_threshold=1.0,
    )
    clf = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_meta_calibrated_pipeline(method: str = "isotonic", cv: int = 3, random_state: int = 42):
    """Wrap :func:`build_meta_pipeline` in ``CalibratedClassifierCV``.

    The logistic regression is already calibrated, so what this adds is the cross-fitted
    ``cv``-fold ensemble plus isotonic sharpening, not recalibration. ``ensemble=True``
    (the default) averages ``cv`` (classifier, calibrator) pairs.

    Pure sklearn, so it pickles cleanly under the runtime's sklearn and exposes the
    standard ``predict_proba``; ``models/submission/main.py`` serves it under
    model_kind ``meta``.

    Args:
        method: calibration method passed to ``CalibratedClassifierCV``.
        cv: number of cross-fitting folds.
        random_state: seed for the inner logistic regression.

    Returns:
        An unfitted ``CalibratedClassifierCV``.
    """
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(
        build_meta_pipeline(random_state=random_state), method=method, cv=cv)


def build_lgbm_pipeline(
    max_features: int = 20_000,
    min_df: int = 3,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    subsample: float = 0.8,
    colsample_bytree: float = 0.5,
    reg_lambda: float = 2.0,
    min_child_samples: int = 20,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Build an untrained sparse TF-IDF + structured features -> LightGBM pipeline.

    Same shape as :func:`build_xgb_pipeline`, but using LightGBM.

    Tuned for log loss: ``binary`` objective / ``binary_logloss`` metric, no
    ``class_weight`` / ``is_unbalance`` so predictions stay calibrated.

    Args:
        max_features: TF-IDF vocabulary size.
        min_df: minimum document frequency of an n-gram.
        n_estimators, learning_rate, num_leaves, subsample, colsample_bytree, reg_lambda,
            min_child_samples, n_jobs, random_state: passed to ``LGBMClassifier``.

    Returns:
        An unfitted ``Pipeline`` that expects ``TEXT_COL`` and ``STRUCTURED_COLS`` in ``X``.
    """
    from lightgbm import LGBMClassifier

    text_branch = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
    )

    pre = ColumnTransformer(
        [
            ("text", text_branch, TEXT_COL),
            ("num", FunctionTransformer(_to_dense, validate=False), STRUCTURED_COLS),
        ],
        sparse_threshold=1.0,
    )

    clf = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_samples=min_child_samples,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )

    return Pipeline([("pre", pre), ("clf", clf)])
