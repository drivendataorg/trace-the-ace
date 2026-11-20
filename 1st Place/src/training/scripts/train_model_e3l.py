#!/usr/bin/env python
"""Train the XGBoost half: a seed ensemble of XGBoost boosters with a frozen Platt map.

The flags select the feature blocks. reproduce.py step 05a trains the shipped XGBoost model with
it (run from build/workspace/):

    OMP_NUM_THREADS=1 python scripts/train_model_e3l.py --n-seeds 5 --n-jobs 12 --par 4 \
        --no-loknn --swap-student --extra-no-ablate --drop-cols ntodst_ \
        --extra-csv d29dst_=artifacts/doc29_features_distilled.csv \
        --extra-csv v2h=artifacts/v2fam_shrunk_122bC.csv \
        --out artifacts/baseline_v2shrunkD.joblib \
        --oof-out artifacts/v2shrunkD_oof.csv

That is: no objective-neighbourhood columns and no distilled NTO block, the 12 distilled
student-state rates in place of E2's student regex columns, and the interview-codebook
(``d29dst_``) and taxonomy-v2 (``v2ht_``/``v2hs_``) blocks added, 177 numeric columns in all.

The two LO-curriculum columns (``lo_nc_sim``, ``lo_year_group``) are always deleted
(``CURRIC_DROP``). The three objective-neighbourhood columns (``LOKNN_COLS``) are added unless
``--no-loknn`` is given.

Writes (defaults; the paths are flags):
    artifacts/baseline_e3l.joblib   SeedEnsembleCalibrated (pipelines + frozen Platt), --out
    artifacts/loknn_table.joblib    the frozen LoKnnTable, --table-out (not with --no-loknn)
    artifacts/e3l_oof.csv           ensemble OOF (response_id, y, p_raw, p_abl), --oof-out
    --fold-out PATH                 optional: one fold-twin model per fold (PATH stem + _k<fold>)
    --dump-x PATH                   optional: the design matrix and its column list, then exit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

import model_e2 as E2  # noqa: E402
from train_model_e2_robust import ablate_full_row  # noqa: E402
from magnificat import config, data, features, lo_knn  # noqa: E402
from magnificat.calibration import PlattScaler, SeedEnsembleCalibrated  # noqa: E402
from magnificat.evidence_features import EVIDENCE_COLS  # noqa: E402
from magnificat.folds import fold_sha, lo_folds  # noqa: E402
from magnificat.lo_knn import LOKNN_COLS  # noqa: E402
from magnificat.nto_features import NTODST_COLS  # noqa: E402
from magnificat.xgb_pipeline import build_e3_xgb_pipeline  # noqa: E402
from magnificat.vocab_order import dump_sorted  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: the two LO-curriculum columns deleted from every matrix
CURRIC_DROP = ["lo_nc_sim", "lo_year_group"]

#: E2's 14 student-side regex columns. `--swap-student` replaces them with the 12 distilled
#: student-state rates.
E2_STUDENT_REGEX = ["self_explanation_rate", "student_explains_rate", "confusion_rate",
                    "resolution_rate", "hedge_rate", "help_deep_rate", "help_shallow_rate",
                    "ack_only_rate", "substantive_turn_rate", "recovered", "ends_resolved",
                    "ends_confused", "last_confusion_pos", "math_doing_rate"]


def _lo_weights(lo, tr, kind):
    """Per-row sample weight from the objective's row count within this training fold.

    The weights are mean-normalised to 1, so the weight sum stays ~n.
    Enabled by ``--lo-weight``.

    Args:
        lo: learning_objective_id per row (all rows).
        tr: indices of the fold's training rows.
        kind: ``"sqrt"`` (1/sqrt(n)), ``"inv"`` (1/n), ``"cap200"`` or ``"cap50"``
            (min(1, cap/n)), where n is the row's objective count in ``tr``.

    Returns:
        Array of weights, one per row of ``tr``.
    """
    import pandas as _pd
    cnt = _pd.Series(lo[tr]).value_counts()
    n = _pd.Series(lo[tr]).map(cnt).to_numpy(float)
    w = {"sqrt": lambda: 1.0 / np.sqrt(n),
         "inv": lambda: 1.0 / n,
         "cap200": lambda: np.minimum(1.0, 200.0 / n),
         "cap50": lambda: np.minimum(1.0, 50.0 / n)}[kind]()
    return w / w.mean()


def _ablate_mult(Z, cols, rng):
    """Attenuate a whole extra block multiplicatively by U(0.2, 1.0) per row and column.

    Selected by ``--extra-ablate-mode mult``.

    Args:
        Z: feature frame, modified in place.
        cols: the block's columns.
        rng: numpy random generator.

    Returns:
        ``Z``, with ``cols`` attenuated.
    """
    V = Z[cols].to_numpy(dtype=float)
    Z[cols] = V * rng.uniform(0.2, 1.0, size=(len(Z), len(cols)))
    return Z


def _ablate_soft(Z, cols, rng):
    """Degrade a whole extra block toward its column mean (``--extra-ablate-mode soft``).

    Args:
        Z: feature frame, modified in place.
        cols: the block's columns; those ending in ``_any`` are presence flags.
        rng: numpy random generator.

    Returns:
        ``Z``. Each rate column becomes ``w * v + (1 - w) * column mean + noise * (1 - w)``
        with ``w ~ U(0.2, 1.0)`` per row and ``noise ~ N(0, 0.05)``; each flag is set to 0 with
        a probability drawn once from U(0.1, 0.5).
    """
    rate = [c for c in cols if not c.endswith("_any")]
    flags = [c for c in cols if c.endswith("_any")]
    if rate:
        V = Z[rate].to_numpy(dtype=float)
        mu = np.nanmean(V, axis=0)
        w = rng.uniform(0.2, 1.0, size=(len(Z), 1))
        noise = rng.normal(0.0, 0.05, size=V.shape)
        Z[rate] = w * V + (1.0 - w) * mu[None, :] + noise * (1.0 - w)
    if flags:
        F = Z[flags].to_numpy(dtype=float)
        p_drop = rng.uniform(0.1, 0.5)
        Z[flags] = np.where(rng.random(F.shape) > p_drop, F, 0.0)
    return Z


def _ablate(X, num, seed, nan_cols=(), soft_cols=(), soft_mode="soft"):
    """Build the ablated (simulated off-source) copy of a feature frame.

    The numeric columns go through ``ablate_full_row`` and the answer-log columns
    (``nan_cols``) are set to NaN wholesale rather than attenuated. ``soft_cols`` (an
    ``--extra-csv`` block under the ``soft`` and ``mult`` modes) are removed from the generic
    pass and get a whole-block treatment instead, so no column is degraded twice.

    ``soft_mode`` selects the semantics:
      ``soft``       whole block, pull toward the mean (:func:`_ablate_soft`)
      ``mult``       whole block, multiplicative (:func:`_ablate_mult`)
      ``rate-only``  ``soft_cols`` empty, so ``ablate_generic`` sees the block and hits only
                     its ``_rate`` columns multiplicatively (the shipped XGBoost model)

    Args:
        X: feature frame.
        num: numeric columns.
        seed: seed of the generic ablation (the block ablation uses ``seed + 777``).
        nan_cols: columns set to NaN in the copy.
        soft_cols: extra-block columns that get the whole-block treatment.
        soft_mode: ``"soft"`` or ``"mult"`` (``"rate-only"`` is expressed by empty ``soft_cols``).

    Returns:
        The ablated copy of ``X``.
    """
    soft = [c for c in soft_cols if c in X.columns]
    if soft:
        _s = set(soft)
        Z = ablate_full_row(X, [c for c in num if c not in _s], seed=seed)
        _fn = _ablate_mult if soft_mode == "mult" else _ablate_soft
        Z = _fn(Z, soft, np.random.default_rng(seed + 777))
    else:
        Z = ablate_full_row(X, num, seed=seed)
    hit = [c for c in nan_cols if c in Z.columns]
    if hit:
        Z[hit] = np.nan
    return Z


def _paraphrase_lo(Z, lo_para, seed, shuffle=False):
    """Replace the objective text of an ablated training copy with a paraphrase.

    Each row gets a seeded random paraphrase of its own objective or, with ``shuffle``, of a
    random other objective. ``learning_objective_id`` is unchanged and no rows are added. Never
    applied to the held-out ablated copy that feeds the Platt fit.
    Enabled by ``--lo-paraphrase-csv``.

    Args:
        Z: ablated training copy.
        lo_para: ``{learning_objective_id: [paraphrase, ...]}``, or None to do nothing.
        seed: seed (the generator uses ``seed + 4242``).
        shuffle: draw the paraphrase from a different objective.

    Returns:
        ``Z`` unchanged when ``lo_para`` is empty, otherwise a copy with new
        ``learning_objective`` text.
    """
    if not lo_para:
        return Z
    rng = np.random.default_rng(seed + 4242)
    ids = Z["learning_objective_id"].to_numpy()
    all_ids = np.asarray(sorted(lo_para))
    out = Z["learning_objective"].to_numpy(object).copy()
    for i, lid in enumerate(ids):
        src = lid
        if shuffle:
            src = all_ids[rng.integers(len(all_ids))]
            while src == lid:
                src = all_ids[rng.integers(len(all_ids))]
        cands = lo_para.get(src)
        if cands:
            out[i] = cands[rng.integers(len(cands))]
    Z = Z.copy()
    Z["learning_objective"] = out
    return Z


def _emb_by_id():
    """Load the BGE embeddings of the training objectives.

    Needed only for the objective-neighbourhood columns, so never with ``--no-loknn`` (the
    shipped XGBoost model); the two files are not part of this repository.

    Returns:
        ``{learning_objective_id: embedding}`` from ``artifacts/lo_text_emb_bge_large.npy``
        and its index ``artifacts/lo_text_emb_index.csv``.
    """
    emb = np.load(ROOT / "artifacts/lo_text_emb_bge_large.npy").astype(np.float64)
    idx = pd.read_csv(ROOT / "artifacts/lo_text_emb_index.csv").learning_objective_id.values
    if len(emb) != len(idx):
        raise SystemExit(f"embedding/index mismatch: {len(emb)} vs {len(idx)}")
    return {l: emb[i] for i, l in enumerate(idx)}



def _tfidf_svd(pipe, n_comp, min_df):
    """Rebuild the preprocessor's ``obj_text`` slot as TF-IDF -> TruncatedSVD.

    Enabled by ``--tfidf-svd``, which requires ``min_df >= 2``.
    Column order is preserved: the numeric block stays last and contiguous.

    Args:
        pipe: pipeline from ``build_e3_xgb_pipeline``, modified in place.
        n_comp: number of SVD components.
        min_df: ``min_df`` of the new TF-IDF.

    Returns:
        ``pipe``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline as _P
    pre = pipe.named_steps["pre"]
    name, _, cols = pre.transformers[0]
    assert name == "obj_text", f"expected obj_text first, got {name}"
    pre.transformers[0] = (name, _P([
        ("tfidf", TfidfVectorizer(max_features=5_000, ngram_range=(1, 2), min_df=min_df)),
        ("svd", TruncatedSVD(n_components=n_comp, random_state=42)),
    ]), cols)
    return pipe


def _build_pipe(num, n_jobs, seed, xgb, variant="none", rank_scale=0.01,
                tfsvd=0, tfsvd_min_df=2):
    """Build ``build_e3_xgb_pipeline`` with the classifier slot optionally swapped.

    ``variant="none"`` returns the shipped pipeline unchanged. The swap happens after
    construction rather than inside ``build_e3_xgb_pipeline`` because that function is also in
    the submission's ``magnificat`` copy.

    Args:
        num: numeric columns passed through to the booster.
        n_jobs: XGBoost threads.
        seed: booster ``random_state``.
        xgb: booster parameter overrides (``--xgb-json``), or None.
        variant: ``"none"``, ``"icon"``, ``"rank"`` or ``"rankndcg"``.
        rank_scale: multiplier on the regularisation penalties of the rank variants.
        tfsvd: number of SVD components for the objective TF-IDF (0 = off).
        tfsvd_min_df: ``min_df`` of the SVD option.

    Returns:
        An unfitted sklearn ``Pipeline``.
    """
    p = build_e3_xgb_pipeline(num, n_jobs=n_jobs, random_state=seed, **(xgb or {}))
    if tfsvd:
        p = _tfidf_svd(p, tfsvd, tfsvd_min_df)
    if variant == "none":
        return p
    from magnificat.xgb_pipeline import E3_XGB_PARAMS
    from magnificat import xgb_variants as xv
    base = dict(E3_XGB_PARAMS, **(xgb or {}))
    base.update(n_jobs=n_jobs, random_state=seed, missing=np.nan, verbosity=0)
    if variant == "icon":
        clf = xv.make_icon_xgb(len(num), objective="binary:logistic",
                               eval_metric="logloss", **base)
    elif variant in ("rank", "rankndcg"):
        obj = "rank:pairwise" if variant == "rank" else "rank:ndcg"
        clf = xv.make_rank_xgb(objective=obj, scale=rank_scale, **base)
    else:
        raise SystemExit(f"unknown --variant {variant!r}")
    p.steps[-1] = ("clf", clf)
    return p


def _one_fold(X, num, y, lo, folds, k, seed, nan_cols, ebi, n_jobs, xgb=None, ncopy=1,
              n_loknn=None, lo_weight=None, soft_cols=(), soft_mode="soft",
              variant="none", rank_scale=0.01, row_w=None, tfsvd=0, tfsvd_min_df=2,
              keep_model=False, lo_para=None, lo_para_shuffle=False):
    """Fit one (seed, fold) model and predict its held-out rows.

    With ``ebi`` (not with ``--no-loknn``) the kNN columns are rebuilt from this fold's
    training rows. The training set is the fold's rows plus ``ncopy`` ablated copies.

    ``keep_model`` also returns the fitted pipeline, for ``--fold-out`` (default False, the
    shipped behaviour).

    Args:
        X: feature frame (all rows, TtA rows first).
        num: numeric columns.
        y: labels.
        lo: learning_objective_id per row.
        folds: fold per row.
        k: held-out fold.
        seed: booster seed.
        nan_cols: columns set to NaN in the ablated copies.
        ebi: objective embeddings (``_emb_by_id()``), or None for no kNN columns (``--no-loknn``).
        n_jobs: booster threads.
        n_loknn: only the first ``n_loknn`` rows (the TtA rows) may enter the kNN table;
            None allows all.
        lo_weight: ``--lo-weight`` kind, or None.
        row_w: per-row weights (``--row-weight-csv``), or None.
        Other arguments: as in ``_ablate``, ``_build_pipe`` and ``_paraphrase_lo``.

    Returns:
        ``(seed, te, p_full, p_ablated, model)``: the held-out row indices, their predictions on
        the original and on an ablated copy (seed 999), and the pipeline or None.
    """
    tr = np.where(folds != k)[0]
    te = np.where(folds == k)[0]
    # loknn is a training-label statistic saved with the model, so by default only TtA rows
    # (the first `n_loknn`) may enter it; see --loknn-external.
    Xf = X.reset_index(drop=True)
    if ebi is not None:
        ltr = tr if n_loknn is None else tr[tr < n_loknn]
        tab = lo_knn.fit(lo[ltr], y[ltr], ebi)
        F = lo_knn.transform(lo, ebi, tab)       # exclude_self=True by default
        Xf = pd.concat([Xf, F], axis=1)
    # `ncopy` ablated copies (default 1 = the shipped behaviour).
    parts, ys = [Xf.iloc[tr]], [y[tr]]
    for j in range(ncopy):
        parts.append(_paraphrase_lo(
            _ablate(Xf.iloc[tr], num, 100 + k + 1000 * j, nan_cols, soft_cols, soft_mode),
            lo_para, 100 + k + 1000 * j, lo_para_shuffle))
        ys.append(y[tr])
    Xtr, ytr = pd.concat(parts, ignore_index=True), np.concatenate(ys)
    p = _build_pipe(num, n_jobs, seed, xgb, variant, rank_scale, tfsvd, tfsvd_min_df)
    _used_w = False
    if variant in ("rank", "rankndcg"):
        # qid must tile exactly as the rows do: the ablated copies mirror the originals
        # row for row, so the same objective appears in 1+ncopy disjoint blocks.
        p.fit(Xtr, ytr, clf__qid=np.tile(np.asarray(lo)[tr], 1 + ncopy))
    elif row_w is not None:
        _used_w = True
        # every ablated copy mirrors the originals row for row, so the vector tiles
        p.fit(Xtr, ytr, clf__sample_weight=np.tile(np.asarray(row_w)[tr], 1 + ncopy))
    elif lo_weight:
        _w = _lo_weights(lo, tr, lo_weight)
        # every ablated copy mirrors the originals row for row, so the weight vector tiles
        p.fit(Xtr, ytr, clf__sample_weight=np.tile(_w, 1 + ncopy))
    else:
        p.fit(Xtr, ytr)
    if row_w is not None and not _used_w:
        raise RuntimeError("row_w was supplied but never reached fit(): the run would train "
                           "unweighted and report a null result. Check the branches above "
                           "that pass clf__sample_weight.")
    return (seed, te,
            p.predict_proba(Xf.iloc[te])[:, 1],
            p.predict_proba(_ablate(Xf.iloc[te], num, 999, nan_cols, soft_cols,
                                    soft_mode))[:, 1],
            p if keep_model else None)


def _final_fit(Xtr, ytr, num, seed, n_jobs, xgb=None, variant="none", rank_scale=0.01,
               qid=None, tfsvd=0, tfsvd_min_df=2):   # Xtr/ytr already augmented
    """Fit one all-data booster on the augmented training set.

    Args:
        Xtr: training rows plus their ablated copies.
        ytr: labels of ``Xtr``.
        qid: query ids for the rank variants.
        Other arguments: as in ``_build_pipe``.

    Returns:
        The fitted pipeline.
    """
    p = _build_pipe(num, n_jobs, seed, xgb, variant, rank_scale, tfsvd, tfsvd_min_df)
    if variant in ("rank", "rankndcg"):
        p.fit(Xtr, ytr, clf__qid=qid)
    else:
        p.fit(Xtr, ytr)
    return p


def main() -> None:
    """Assemble the design matrix, fit the per-fold and all-data boosters, and save the model.

    Builds the E2 columns live (``model_e2.build_e2_live``), adds the evidence and distilled
    blocks selected by the flags, fits ``--n-seeds`` x 5 fold models on the frozen
    objective-disjoint folds to get the out-of-fold predictions, fits the frozen Platt map on
    the pooled full and ablated OOF (TtA rows only), then fits one booster per seed on all
    rows plus their ablated copies.

    Returns:
        None. Writes ``--oof-out``, ``--out``, ``--table-out`` unless ``--no-loknn`` (plus a
        ``.NEVER_SHIP.txt`` marker when non-commercial data entered training), the fold twins
        with ``--fold-out``, or only the design matrix with ``--dump-x``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--par", type=int, default=13)
    ap.add_argument("--ablation-seed", type=int, default=7)
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "baseline_e3l.joblib"))
    ap.add_argument("--table-out", default=str(config.ARTIFACTS_DIR / "loknn_table.joblib"))
    ap.add_argument("--oof-out", default=str(config.ARTIFACTS_DIR / "e3l_oof.csv"))
    ap.add_argument("--fold-out", default=None,
                    help="also write one model per fold (PATH stem + _k<fold>): a "
                         "SeedEnsembleCalibrated over the same seeds with the same frozen Platt "
                         "map, each trained on 4/5 of the objectives. Off by default; nothing "
                         "else changes.")
    ap.add_argument("--ablation-copies", type=int, default=1,
                    help="ablated training copies (1 = shipped)")
    ap.add_argument("--external", default=None,
                    help="comma-separated external-corpus feature pickles "
                         "(artifacts/extmat_*.pkl, not shipped) to pool into training")
    ap.add_argument("--loknn-external", action="store_true",
                    help="let external rows into the loknn neighbourhood fit. Off by default: "
                         "the loknn table is saved with the model and is a statistic of the "
                         "training labels, so non-commercial rows would end up in it. The "
                         "model is then marked NEVER_SHIP if any of them is non-commercial.")
    ap.add_argument("--no-loknn", action="store_true",
                    help="drop the 3 objective-neighbourhood columns (as the shipped XGBoost "
                         "model does)")
    ap.add_argument("--lo-paraphrase-csv", default=None,
                    help="CSV of objective paraphrases (learning_objective_id,k,text; not "
                         "shipped). On every ablated training copy the objective text is "
                         "replaced by a seeded random paraphrase of the same objective; no rows "
                         "are added.")
    ap.add_argument("--lo-paraphrase-shuffle", action="store_true",
                    help="with --lo-paraphrase-csv: use the paraphrase of a random other "
                         "objective")
    ap.add_argument("--lo-weight", default=None,
                    choices=["sqrt", "inv", "cap200", "cap50"],
                    help="per-row sample weight from the objective's in-fold row count (see "
                         "_lo_weights)")
    ap.add_argument("--extra-csv", action="append", default=None, metavar="PREFIX=PATH",
                    help="join a session- or response-keyed CSV and add every column whose "
                         "name starts with PREFIX. Repeatable; step 05a adds the d29dst_ and "
                         "v2h blocks this way.")
    ap.add_argument("--extra-ablate-mode", choices=["soft", "mult", "rate-only"], default="soft",
                    help="how --extra-csv columns are ablated in the training copies. `soft` "
                         "(default): the whole block is pulled toward the column mean plus "
                         "noise. `mult`: the whole block is multiplied by U(0.2, 1.0). "
                         "`rate-only` (the shipped XGBoost model's): "
                         "the block goes through ablate_generic, which matches only `_rate` "
                         "columns and two fixed lists, so e.g. 18 of the 34 d29dst_ columns "
                         "(`_late`, `_share`, `_contrast`, `_pos_mean`, `_any`) are left "
                         "unchanged.")
    ap.add_argument("--extra-no-ablate", action="store_true",
                    help="alias for --extra-ablate-mode rate-only (step 05a uses it)")
    ap.add_argument("--confirm", action="store_true",
                    help="add the earned-confirmation block (cfm_*, 11 cols, from "
                         "artifacts/confirm_features.csv, not shipped)")
    ap.add_argument("--swap-student", action="store_true",
                    help="replace E2's 14 student regex columns with the 12 distilled "
                         "student-state rates (net -2 cols). See E2_STUDENT_REGEX.")
    ap.add_argument("--nto-file", default="nto_dst_features.csv",
                    help="alternative distilled-move block; must carry the same NTODST_COLS")
    ap.add_argument("--evidence-file", default="evidence_features_full.csv",
                    help="alternative evidence block; must carry the same EVIDENCE_COLS. The "
                         "default reproduces the shipped block exactly. At training time the "
                         "`recdst_*` answer log keeps only turns the 122B annotator labelled "
                         "other than 'na' (47.2 per session; see distil_turn_correctness.py), "
                         "at inference every turn with p_ans >= 0.5 (~68 per session; "
                         "magnificat/evidence_features.py).")
    ap.add_argument("--nto-merge", choices=["none", "M1", "M2", "M3"], default="none",
                    help="collapse the four elicitation rates of the ntodst_ block (M1: into "
                         "one, M2: into two, M3: drop them)")
    ap.add_argument("--tfidf-svd", type=int, default=0,
                    help="compress the LO TF-IDF to N SVD components (0 = off, as in the "
                         "shipped model)")
    ap.add_argument("--tfidf-min-df", type=int, default=2,
                    help="min_df for the SVD option; must be >= 2")
    ap.add_argument("--row-weight-csv", default=None,
                    help="per-row sample weights from a response_id,w CSV")
    ap.add_argument("--dump-x", default=None,
                    help="write the assembled design matrix to this path and exit (no fit)")
    ap.add_argument("--drop-cols", default=None,
                    help="comma-separated column prefixes to delete from the design matrix, "
                         "e.g. 'd29dst_,studst_' (step 05a drops ntodst_)")
    ap.add_argument("--shuffle-block", default=None,
                    help="comma-separated column prefixes whose values are permuted across "
                         "sessions (width and marginals kept)")
    ap.add_argument("--shuffle-seed", type=int, default=20260813)
    ap.add_argument("--variant", choices=["none", "icon", "rank", "rankndcg"], default="none",
                    help="booster variants: `icon` forbids interactions between the objective "
                         "(sparse) columns and the transcript columns; `rank`/`rankndcg` use a "
                         "ranking objective with the hessian-based penalties scaled by "
                         "--rank-scale. The default `none` is the shipped booster.")
    ap.add_argument("--rank-scale", type=float, default=0.01,
                    help="multiplier on min_child_weight/reg_lambda/reg_alpha/gamma for the "
                         "rank variants (126 -> 1.26 at the default)")
    ap.add_argument("--xgb-json", default=None,
                    help='booster parameter overrides as JSON, e.g. '
                         '{"reg_lambda":20.0,"subsample":0.8,"n_estimators":900,'
                         '"learning_rate":0.02}. Default: none, i.e. the settings of the shipped '
                         'XGBoost model (lossguide, 31 leaves).')
    args = ap.parse_args()
    import json
    xgb = json.loads(args.xgb_json) if args.xgb_json else None
    print(f"booster overrides: {xgb or 'none (shipped lossguide-31 settings)'} | "
          f"ablation copies: {args.ablation_copies}", flush=True)
    lo_para = None
    if args.lo_paraphrase_csv:
        _pp = pd.read_csv(ROOT / args.lo_paraphrase_csv, dtype=str)
        _pp = _pp[_pp.text.notna() & (_pp.text.str.strip() != "")]
        lo_para = {k: v.tolist() for k, v in _pp.groupby("learning_objective_id")["text"]}
        _n = pd.Series({k: len(v) for k, v in lo_para.items()})
        print(f"LO PARAPHRASE: {len(lo_para)} objectives, {int(_n.sum())} paraphrases "
              f"(min {int(_n.min())}, mean {_n.mean():.2f}); shuffle={args.lo_paraphrase_shuffle}",
              flush=True)

    train = data.load_train()
    y = train["is_correct"].to_numpy(float)
    lo = train["learning_objective_id"].to_numpy()
    meta = features.build_meta_matrix(train)
    X, full_cols, e2_cols = E2.build_e2_live(train, meta)
    num = [c for c in e2_cols if c not in ("learning_objective", "learning_objective_id")]

    ev = train[["session_id"]].merge(
        pd.read_csv(ROOT / "artifacts" / args.evidence_file),
        on="session_id", how="left")
    if args.evidence_file != "evidence_features_full.csv":
        print(f"EVIDENCE FILE: {args.evidence_file} (not the shipped block)", flush=True)
    # The distilled-move block can be replaced with --nto-file.
    nd = train[["session_id"]].merge(
        pd.read_csv(ROOT / "artifacts" / args.nto_file),
        on="session_id", how="left")
    for nm, df_, cols in (("evidence", ev, EVIDENCE_COLS), ("ntodst", nd, NTODST_COLS)):
        missing = [c for c in cols if c not in df_.columns]
        if missing:
            raise SystemExit(f"{nm} file is missing {len(missing)} cols: {missing[:6]}")

    # --- optionally collapse the four elicitation families (--nto-merge) -----------------
    nto_cols = list(NTODST_COLS)
    if args.nto_merge != "none":
        _e = [f"ntodst_{f}_rate" for f in ("prompt_self_expl", "prompt_next_step",
                                           "probe_understand", "probe_prior")]
        if args.nto_merge == "M1":
            nd["ntodst_elicit_rate"] = nd[_e].sum(axis=1)
            _add = ["ntodst_elicit_rate"]
        elif args.nto_merge == "M2":
            nd["ntodst_prompt_rate"] = nd[_e[0:2]].sum(axis=1)
            nd["ntodst_probe_rate"] = nd[_e[2:4]].sum(axis=1)
            _add = ["ntodst_prompt_rate", "ntodst_probe_rate"]
        else:                                   # M3: drop, keep only the ratio
            _add = []
        nto_cols = [c for c in nto_cols if c not in _e] + _add
        print(f"NTO-MERGE[{args.nto_merge}]: -{len(_e)} elicit rates, +{len(_add)} merged "
              f"-> {len(nto_cols)} ntodst cols (was {len(NTODST_COLS)})", flush=True)

    X = pd.concat([X.reset_index(drop=True), ev[EVIDENCE_COLS].reset_index(drop=True),
                   nd[nto_cols].reset_index(drop=True)], axis=1)

    num = num + list(EVIDENCE_COLS) + list(nto_cols)

    # Columns from --extra-csv given whole-block ablation. Left empty under `rate-only` (the
    # shipped setting), where the block goes through `ablate_generic`, which matches only
    # `_rate` plus two hard-coded name lists. See _ablate's `soft_mode`.
    soft_cols: list[str] = []
    soft_mode = "rate-only" if args.extra_no_ablate else args.extra_ablate_mode

    for _spec in (args.extra_csv or []):
        _pref, _, _path = _spec.partition("=")
        _raw = pd.read_csv(ROOT / _path)
        # Blocks are keyed on either response_id (response-level) or session_id
        # (session-level), so detect the key rather than assume session_id.
        _key = "response_id" if "response_id" in _raw.columns else "session_id"
        _df = train[[_key]].merge(_raw, on=_key, how="left")
        _cols = [c for c in _df.columns if c.startswith(_pref)]
        if not _cols:
            raise SystemExit(f"--extra-csv {_spec}: no columns start with {_pref!r}")
        # Drop anything already present in the matrix (see also the final de-duplication
        # below).
        _dupe = [c for c in _cols if c in num]
        _cols = [c for c in _cols if c not in num]
        if _dupe:
            print(f"  (skipping {len(_dupe)} cols already in the matrix)", flush=True)
        if not _cols:
            continue
        X = pd.concat([X.reset_index(drop=True), _df[_cols].reset_index(drop=True)], axis=1)
        num = num + _cols
        _esc = sum(1 for c in _cols if not c.endswith("_rate"))
        if soft_mode == "rate-only":
            _how = (f"ablate_generic only (rate-only): {len(_cols) - _esc}/{len(_cols)} `_rate` "
                    f"cols attenuated, {_esc} left unchanged")
        else:
            soft_cols += _cols
            _sem = "pull to mean" if soft_mode == "soft" else "multiplicative"
            _how = (f"whole block ablated ({_sem}); the generic pass would have left {_esc} of "
                    f"these unchanged")
        print(f"EXTRA[{_key}]: +{len(_cols)} {_pref}* cols from {_path} -> {len(num)} numeric "
              f"({_df[_cols].notna().any(axis=1).mean():.1%} rows non-null) | {_how}", flush=True)

    if args.confirm:
        cfm = train[["session_id"]].merge(
            pd.read_csv(ROOT / "artifacts/confirm_features.csv"), on="session_id", how="left")
        c_cols = [c for c in cfm.columns if c.startswith("cfm_")]
        X = pd.concat([X.reset_index(drop=True), cfm[c_cols].reset_index(drop=True)], axis=1)
        num = num + c_cols
        print(f"CONFIRM: +{len(c_cols)} cfm cols -> {len(num)} numeric cols "
              f"({cfm[c_cols].notna().any(axis=1).mean():.1%} of rows non-null)", flush=True)

    if args.swap_student:
        stu = train[["session_id"]].merge(
            pd.read_csv(ROOT / "artifacts/stu_dst_features.csv"), on="session_id", how="left")
        s_rate = [c for c in stu.columns if c.startswith("studst_") and c.endswith("_rate")]
        missing = [c for c in E2_STUDENT_REGEX if c not in num]
        if missing:
            raise SystemExit(f"--swap-student: E2 regex cols absent from the matrix: {missing}")
        X = pd.concat([X.reset_index(drop=True), stu[s_rate].reset_index(drop=True)], axis=1)
        num = [c for c in num if c not in E2_STUDENT_REGEX] + s_rate
        print(f"SWAP: -{len(E2_STUDENT_REGEX)} E2 student regex, +{len(s_rate)} distilled "
              f"studst rates -> {len(num)} numeric cols", flush=True)
    dropped = [c for c in CURRIC_DROP if c in num]
    if len(dropped) != len(CURRIC_DROP):
        raise SystemExit(f"expected to drop {CURRIC_DROP}, only found {dropped}: the "
                         "curriculum block has changed; check it before using the model")
    # --no-loknn removes the 3 objective-neighbourhood columns.
    num = [c for c in num if c not in CURRIC_DROP] + (
        [] if args.no_loknn else list(LOKNN_COLS))
    # ---- drop or permute a block ---------------------------------------------------------
    # --drop-cols deletes a block (step 05a drops ntodst_); --shuffle-block
    # permutes it across sessions.
    for _pref in [q.strip() for q in (args.drop_cols or "").split(",") if q.strip()]:
        _hit = [c for c in num if c.startswith(_pref)]
        if not _hit:
            raise SystemExit(f"--drop-cols {_pref!r} matched nothing")
        num = [c for c in num if c not in _hit]
        print(f"DROP: -{len(_hit)} cols matching {_pref!r} -> {len(num)} numeric", flush=True)
    if args.shuffle_block:
        # Permute at the session level (a bijection on sessions), so the session-level
        # marginals are preserved exactly; the check below asserts this on the session-level
        # table.
        _rng = np.random.default_rng(args.shuffle_seed)
        _sid = train.session_id.to_numpy()
        _u = pd.unique(_sid)
        _donor = dict(zip(_u, _u[_rng.permutation(len(_u))]))
        for _pref in [q.strip() for q in args.shuffle_block.split(",") if q.strip()]:
            _hit = [c for c in num if c.startswith(_pref)]
            if not _hit:
                raise SystemExit(f"--shuffle-block {_pref!r} matched nothing")
            _bysess = X[_hit].groupby(_sid, sort=False).first()
            _perm = _bysess.reindex([_donor[u] for u in _bysess.index])
            _dm = float(np.nanmax(np.abs(np.nanmean(_bysess.to_numpy(float), 0)
                                         - np.nanmean(_perm.to_numpy(float), 0))))
            if _dm > 1e-12:
                raise SystemExit(f"permutation changed a session-level mean ({_dm:.2e}), so "
                                 "it is not a valid control")
            X.loc[:, _hit] = _bysess.reindex(
                pd.Series(_sid).map(_donor).to_numpy()).to_numpy()
            print(f"SHUFFLE: {len(_hit)} cols {_pref!r} permuted across {len(_u):,} sessions; "
                  f"session-level max|Δmean| = {_dm:.2e}", flush=True)
    nan_cols = [c for c in EVIDENCE_COLS if c.startswith("recdst_")]
    print(f"E3+Ndst+LoKNN: {X.shape[0]:,} responses x {len(num)} numeric cols "
          f"(−{len(CURRIC_DROP)} curriculum, +{0 if args.no_loknn else len(LOKNN_COLS)} loknn); "
          f"base rate {y.mean():.4f}", flush=True)

    # Final de-duplication, order-independent. Per-block dedup is not enough: --extra-csv runs
    # before --swap-student, so studst rates supplied by both are invisible to the earlier
    # check and would produce duplicate column names, and then a broadcast ValueError deep in
    # the fit.
    if len(set(num)) != len(num):
        _seen, _uniq = set(), []
        for c in num:
            if c not in _seen:
                _seen.add(c); _uniq.append(c)
        print(f"DEDUP: {len(num)} -> {len(_uniq)} numeric cols "
              f"({len(num) - len(_uniq)} duplicates removed)", flush=True)
        num = _uniq
    X = X.loc[:, ~X.columns.duplicated()]

    # --dump-x: write the assembled design matrix and its numeric column list, then stop.
    # Without the flag nothing below changes.
    if getattr(args, "dump_x", None):
        _dp = Path(args.dump_x)
        _dp.parent.mkdir(parents=True, exist_ok=True)
        X.to_pickle(_dp)
        (_dp.with_suffix(".cols.json")).write_text(json.dumps(list(num), indent=0))
        print(f"DUMP-X: {X.shape[0]:,} x {X.shape[1]} frame -> {_dp}; "
              f"{len(num)} numeric cols -> {_dp.with_suffix('.cols.json').name}", flush=True)
        return

    ebi = None if args.no_loknn else _emb_by_id()
    folds = lo_folds(lo)
    print(f"folds: frozen LO-disjoint sha={fold_sha(folds)} "
          f"sizes={np.bincount(folds).tolist()}", flush=True)

    # ---- mixed-corpus pooling (--external) ----
    # The TtA rows keep their frozen fold and stay first; each external row inherits the fold
    # of the TtA objective it was assigned to.
    n_tta = len(y)
    ext_corpora, ext_ids = [], []
    if args.external:
        import joblib as _jl
        lo2fold = pd.Series(folds, index=lo)
        lo2fold = lo2fold[~lo2fold.index.duplicated()]
        assert pd.Series(folds).groupby(lo).nunique().max() == 1, \
            "a learning objective spans >1 fold; the LO->fold map would be ill-defined"
        Xs, ys_, los, fs = [X.reset_index(drop=True)], [y], [lo], [folds]
        for spec in [q.strip() for q in args.external.split(",") if q.strip()]:
            d = _jl.load(ROOT / spec)
            need = [c for c in num if c not in LOKNN_COLS]
            miss = [c for c in need if c not in d["X"].columns]
            if miss:
                raise SystemExit(f"{spec}: missing {len(miss)} training columns, "
                                 f"e.g. {miss[:5]}")
            unseen = sorted(set(d["lo"]) - set(lo2fold.index))
            if unseen:
                raise SystemExit(f"{spec}: {len(unseen)} imputed objectives are not TtA "
                                 f"objectives, so they have no fold: {unseen[:3]}")
            ef = lo2fold.reindex(d["lo"]).to_numpy()
            Xe = d["X"].reindex(columns=X.columns)      # align, NaN for anything absent
            Xe["learning_objective"] = d["lo_text"]
            Xe["learning_objective_id"] = d["lo"]
            Xs.append(Xe.reset_index(drop=True)); ys_.append(d["y"])
            los.append(d["lo"]); fs.append(ef)
            ext_corpora.append(d["corpus"]); ext_ids.append(d["response_id"])
            print(f"  + {spec}: {len(d['y']):,} rows, corpus={d['corpus']} "
                  f"({d['licence']}), base {d['y'].mean():.4f}, "
                  f"per-fold {np.bincount(ef, minlength=5).tolist()}", flush=True)
        X = pd.concat(Xs, ignore_index=True)
        y = np.concatenate(ys_); lo = np.concatenate(los); folds = np.concatenate(fs)
        # the TtA slice must keep the frozen folds
        frozen = np.load(ROOT / "artifacts/folds_gkf5_lo.npy")
        assert np.array_equal(folds[:n_tta], frozen), "TtA fold slice was disturbed"
        assert fold_sha(folds[:n_tta]) == "59f0c807dad3ac08", "TtA fold sha changed"
        print(f"POOLED: {n_tta:,} TtA + {len(y) - n_tta:,} external = {len(y):,} rows; "
              f"TtA fold slice verified identical (sha 59f0c807dad3ac08)", flush=True)
        print(f"  pooled base rate {y.mean():.4f} (TtA alone {y[:n_tta].mean():.4f})",
              flush=True)

    if args.tfidf_svd:
        if args.tfidf_min_df < 2:
            raise SystemExit("--tfidf-svd requires --tfidf-min-df >= 2")
        print(f"TFIDF-SVD: {args.tfidf_svd} components, min_df={args.tfidf_min_df}", flush=True)

    row_w = None
    if args.row_weight_csv:
        _rw = pd.read_csv(args.row_weight_csv).set_index("response_id")["w"]
        row_w = train.response_id.map(_rw).to_numpy(float)
        if np.isnan(row_w).any():
            raise SystemExit(f"row-weight csv misses {int(np.isnan(row_w).sum())} responses")
        print(f"ROW WEIGHT: mean {row_w.mean():.3f} p5 {np.percentile(row_w,5):.3f} "
              f"p95 {np.percentile(row_w,95):.3f}", flush=True)

    seeds = [42 + 101 * s for s in range(args.n_seeds)]
    from joblib import Parallel, delayed
    ks = sorted(set(folds))
    tasks = [(k, sd) for sd in seeds for k in ks]
    per = max(1, args.n_jobs // max(1, min(args.par, len(tasks))))
    print(f"OOF for the Platt fit: {len(tasks)} (seed, fold) fits, "
          f"{min(args.par, len(tasks))} concurrent x {per} threads", flush=True)
    got = Parallel(n_jobs=min(args.par, len(tasks)), backend="loky", verbose=10)(
        delayed(_one_fold)(X, num, y, lo, folds, k, sd, nan_cols, ebi, per, xgb,
                           args.ablation_copies,
                           n_loknn=(len(y) if args.loknn_external else n_tta),
                           lo_weight=args.lo_weight, soft_cols=soft_cols, soft_mode=soft_mode,
                           variant=args.variant, rank_scale=args.rank_scale, row_w=row_w,
                           tfsvd=args.tfidf_svd, tfsvd_min_df=args.tfidf_min_df,
                           keep_model=bool(args.fold_out),
                           lo_para=lo_para, lo_para_shuffle=args.lo_paraphrase_shuffle)
        for k, sd in tasks)

    per_seed = {sd: (np.zeros(len(y)), np.zeros(len(y))) for sd in seeds}
    for sd, te, pf, pa, _pipe in got:
        per_seed[sd][0][te] = pf
        per_seed[sd][1][te] = pa
    # ---- everything below is scored on the TtA slice only ---------------------------
    # External rows (--external) never enter the metric or the Platt fit.
    T = slice(0, n_tta)
    yT = y[T]
    for sd in seeds:
        of = per_seed[sd][0][T]
        print(f"  seed {sd}: OOF LL={log_loss(yT, of):.4f} AUC={roc_auc_score(yT, of):.4f}"
              f"  [TtA slice]", flush=True)
    oof_full = np.mean([per_seed[sd][0] for sd in seeds], axis=0)[T]
    oof_abl = np.mean([per_seed[sd][1] for sd in seeds], axis=0)[T]
    platt = PlattScaler.fit(np.concatenate([oof_full, oof_abl]), np.concatenate([yT, yT]))
    cal = platt.transform(oof_full)
    y_metric = yT
    print(f"\nensemble OOF ({len(seeds)} seeds, ablation-augmented): "
          f"raw LL={log_loss(y_metric, oof_full):.4f} | "
          f"Platt LL={log_loss(y_metric, cal):.4f} "
          f"AUC={roc_auc_score(y_metric, cal) * 100:.2f}  [TtA slice, n={n_tta:,}]",
          flush=True)
    print(f"  frozen Platt: a={platt.a:.4f} b={platt.b:+.4f}", flush=True)
    pd.DataFrame({"response_id": train["response_id"].values, "y": y_metric,
                  "p_raw": oof_full, "p_abl": oof_abl}).to_csv(args.oof_out, index=False)
    print(f"  wrote {args.oof_out}", flush=True)

    # ---- the all-data table (not with --no-loknn) + the all-data boosters ----------
    # The table is fitted on all training rows; `exclude_self=True` keeps a training row from
    # seeing its own objective's rate.
    Xf = X.reset_index(drop=True)
    if ebi is not None:
        _lk = slice(0, len(y)) if args.loknn_external else T
        table = lo_knn.fit(lo[_lk], y[_lk], ebi)
        F = lo_knn.transform(lo, ebi, table)
        assert F[LOKNN_COLS[0]].between(0, 1).all(), "loknn_diff outside [0,1]"
        print(f"shipped LoKnnTable: {len(table.lo_ids)} objectives, prior={table.prior:.4f}, "
              f"emb {table.emb.shape}", flush=True)
        Xf = pd.concat([Xf, F], axis=1)

    print(f"Fitting {len(seeds)} final booster(s) on all data (ablation-augmented)...",
          flush=True)
    parts, ys = [Xf], [y]
    for j in range(args.ablation_copies):
        parts.append(_paraphrase_lo(
            _ablate(Xf, num, seed=args.ablation_seed + 1000 * j, nan_cols=nan_cols,
                    soft_cols=soft_cols, soft_mode=soft_mode),
            lo_para, args.ablation_seed + 1000 * j, args.lo_paraphrase_shuffle))
        ys.append(y)
    Xtr, ytr = pd.concat(parts, ignore_index=True), np.concatenate(ys)
    bases = Parallel(n_jobs=min(args.par, len(seeds)), backend="loky", verbose=10)(
        delayed(_final_fit)(Xtr, ytr, num, sd, max(1, args.n_jobs // len(seeds)), xgb,
                            variant=args.variant, rank_scale=args.rank_scale,
                            qid=np.tile(np.asarray(lo), 1 + args.ablation_copies),
                            tfsvd=args.tfidf_svd, tfsvd_min_df=args.tfidf_min_df)
        for sd in seeds)

    import joblib
    from magnificat import provenance as _pv
    _train_c = ["tta"] + sorted(set(ext_corpora))
    _loknn_c = _train_c if args.loknn_external else ["tta"]
    if ebi is not None:
        _tbl = lo_knn.LoKnnTable(lo_ids=table.lo_ids, emb=table.emb,
                                 shrunk=table.shrunk, count=table.count, prior=table.prior)
        _pv.stamp(_tbl, _train_c if args.loknn_external else ["tta"])
        joblib.dump(_tbl, args.table_out)
        print(f"saved -> {args.table_out}")
    model = SeedEnsembleCalibrated(bases, platt)
    _pv.stamp(model, _train_c, loknn_corpora=_loknn_c,
              note=f"n_tta={n_tta} n_ext={len(y) - n_tta}")
    _p = _pv.read(model)
    print(f"PROVENANCE {_p.describe()}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump_sorted(model, args.out)
    if _p.never_ship:
        Path(args.out).with_suffix(".NEVER_SHIP.txt").write_text(
            f"Trained on non-commercial data {list(_p.nc_corpora)} - RESEARCH ONLY, NEVER SHIP.\n"
            f"{_p.describe()}\n")
        print(f"  WARNING: wrote {Path(args.out).with_suffix('.NEVER_SHIP.txt').name}", flush=True)
    print(f"saved -> {args.out}  (E3+Ndst+LoKNN XGB lossguide31 x{len(seeds)} + Platt)")

    # ---- the fold twins (opt-in) ----------------------------------------------------
    # One SeedEnsembleCalibrated per fold, over the same seeds and the same frozen Platt as the
    # all-data model above, each trained on 4/5 of the objectives.
    # This runs last, after everything above is on disk.
    if args.fold_out:
        _by_fold: dict = {}
        for (k, sd), r in zip(tasks, got):
            if r[4] is None:
                raise RuntimeError("--fold-out was requested but a worker returned no model")
            _by_fold.setdefault(k, []).append(r[4])
        _base = Path(args.fold_out)
        _base.parent.mkdir(parents=True, exist_ok=True)
        # One file per fold, not a dict: each file loads as a single estimator, like the
        # all-data model, so the fold twins can be scored by the same code.
        for k, v in sorted(_by_fold.items()):
            _p = _base.with_name(f"{_base.stem}_k{k}{_base.suffix}")
            dump_sorted(SeedEnsembleCalibrated(v, platt), _p)
            print(f"  wrote {_p.name}: fold {k} model, {len(v)} seeds, trained on "
                  f"{int((folds != k).sum()):,} rows / "
                  f"{pd.Series(lo)[folds != k].nunique()} objectives", flush=True)
        print(f"  the fold models share the all-data model's frozen Platt a={platt.a:.4f} "
              f"b={platt.b:+.4f}",
              flush=True)

    print("Next: python src/run_training.py --steps 09a,09b stages and packs the submission "
          "(it takes the XGBoost model from artifacts/baseline_v2shrunkD.joblib)")


if __name__ == "__main__":
    main()
