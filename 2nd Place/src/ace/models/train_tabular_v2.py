#!/usr/bin/env python
r"""Train tabular_v2 (training stage 4): LightGBM on the base tabular features plus the
structured transcript features from build_tabular_v2_features, on the skill-cluster folds.

The objective target encoding is rebuilt inside every fold with the leave-own-objective-out kNN
rule; the shipped model is trained without it (--no-te-lo).

    python -m ace.models.train_tabular_v2 --data-dir data --eda-dir outputs/eda \
        --emb-dir outputs/embeddings --cache-dir cache --feature-dir outputs/features/tabular_v2 \
        --folds outputs/eda/cv_folds_skill.csv --out outputs/models/tabular_v2_bge_no_telo \
        --seeds 42,1337,2024 --no-te-lo
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ace.features import KNNBackoffTE, feature_columns, load_frame
from ace.tabular_v2_features import (
    TABULAR_V2_BGE_FEATURES,
    TABULAR_V2_RESPONSE_LEXICAL_FEATURES,
    TABULAR_V2_SESSION_FEATURES,
)
from ace.utils import RNG_SEED, logloss, md_table

KNN = dict(k=10, a_knn=10.0, a_prior=10.0)
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.02,
    num_leaves=31,
    min_data_in_leaf=120,
    feature_fraction=0.82,
    bagging_fraction=0.82,
    bagging_freq=1,
    lambda_l1=0.15,
    lambda_l2=2.0,
    max_bin=255,
    verbosity=-1,
)
NUM_ROUNDS = 5000
EARLY_STOP = 250
CLIP = (0.01, 0.99)


def _train_rate(lo_idx: np.ndarray, y: np.ndarray, n: int, prior: float):
    cnt = np.bincount(lo_idx, minlength=n).astype(float)
    sums = np.bincount(lo_idx, weights=y, minlength=n)
    rate = np.divide(sums, cnt, out=np.full(n, prior), where=cnt > 0)
    return cnt, rate


def _loo_knn_te(te: KNNBackoffTE, cnt: np.ndarray, rate: np.ndarray,
                prior: float) -> np.ndarray:
    """Difficulty estimate that excludes the objective's own labels."""
    w = te.nn_sim * cnt[te.nn_idx]
    r_knn = (w * rate[te.nn_idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
    return (te.a_knn * r_knn + te.a_prior * prior) / (te.a_knn + te.a_prior)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _cross_fit_platt(p: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(_logit(p[tr]).reshape(-1, 1), y[tr])
        out[va] = lr.predict_proba(_logit(p[va]).reshape(-1, 1))[:, 1]
    return np.clip(out, *CLIP)


def _load_training_frame(args) -> tuple[pd.DataFrame, list[str]]:
    df = load_frame(args.data_dir, args.eda_dir, args.emb_dir)

    # Replace load_frame's folds with the skill-cluster folds the LLM readers used.
    fold_df = pd.read_csv(Path(args.folds).expanduser(), usecols=["response_id", "fold"])
    df = df.drop(columns=["fold"]).merge(fold_df, on="response_id", validate="1:1")

    fdir = Path(args.feature_dir).expanduser()
    session = pd.read_csv(fdir / "session_features.csv")
    session_cols = [c for c in TABULAR_V2_SESSION_FEATURES if c in session.columns]
    df = df.merge(session[["session_id"] + session_cols], on="session_id",
                  how="left", validate="m:1")

    lexical = pd.read_csv(fdir / "response_lexical.csv")
    lexical_cols = [c for c in TABULAR_V2_RESPONSE_LEXICAL_FEATURES if c in lexical.columns]
    df = df.merge(lexical[["response_id"] + lexical_cols], on="response_id",
                  how="left", validate="1:1")

    bge_path = fdir / "response_student_bge.csv"
    bge_cols: list[str] = []
    if not args.no_student_bge and bge_path.exists():
        bge = pd.read_csv(bge_path)
        bge_cols = [c for c in TABULAR_V2_BGE_FEATURES if c in bge.columns]
        df = df.merge(bge[["response_id"] + bge_cols], on="response_id",
                      how="left", validate="1:1")

    # As at inference on unseen objectives: no objective frequency/category and no
    # sibling-row count. te_lo is fold-dependent and is filled in the training loop.
    drop = {"lo_count", "topic_emb"}
    if getattr(args, "no_te_lo", False):
        # te_lo is the objective target encoding. Under unseen-skill folds its kNN
        # neighbours are often held out themselves, so it is fold-fragile (one fold came
        # out anti-predictive); the shipped model drops it.
        drop.add("te_lo")
    base_cols = [c for c in feature_columns(include_session_count=False)
                 if c not in drop]
    # te_lo is fold-dependent and is overwritten inside every training fold.
    if "te_lo" not in df.columns:
        df["te_lo"] = np.nan
    cols = base_cols + session_cols + lexical_cols + bge_cols
    # keep the first occurrence if two feature sets share a name
    cols = list(dict.fromkeys(cols))

    numeric = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = numeric
    missing_all = [c for c in cols if c != "te_lo" and df[c].notna().sum() == 0]
    if missing_all:
        print(f"[tabv2] dropping all-NaN columns: {missing_all}")
        cols = [c for c in cols if c not in missing_all]
    return df, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--eda-dir", default="outputs/eda")
    ap.add_argument("--emb-dir", default="outputs/embeddings")
    ap.add_argument("--cache-dir", required=True, help="directory holding emb_objectives.npy")
    ap.add_argument("--feature-dir", default="outputs/features/tabular_v2")
    ap.add_argument("--folds", default="outputs/eda/cv_folds_skill.csv")
    ap.add_argument("--out", default="outputs/models/tabular_v2")
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--no-student-bge", action="store_true")
    ap.add_argument("--no-te-lo", action="store_true",
                    help="drop the objective target encoding te_lo (as the shipped model does)")
    ap.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    ap.add_argument("--early-stop", type=int, default=EARLY_STOP)
    ap.add_argument("--threads", type=int, default=0,
                    help="LightGBM threads; 0 lets LightGBM choose")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out).expanduser()
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    df, cols = _load_training_frame(args)
    y = df["is_correct"].to_numpy(dtype=float)
    folds = df["fold"].to_numpy(dtype=int)
    prior = float(y.mean())
    unique_folds = sorted(np.unique(folds))
    if unique_folds != list(range(len(unique_folds))):
        raise ValueError(f"unexpected fold ids: {unique_folds}")

    emb_obj = np.load(Path(args.cache_dir).expanduser() / "emb_objectives.npy")
    lo_ids = sorted(df["learning_objective_id"].astype(str).unique())
    if len(lo_ids) != emb_obj.shape[0]:
        raise ValueError(
            f"emb_objectives rows ({emb_obj.shape[0]}) do not match objective ids ({len(lo_ids)})"
        )
    te = KNNBackoffTE(emb_obj, lo_ids, **KNN)
    lo_idx = te.index_of(df["learning_objective_id"])

    print(f"[tabv2] rows={len(df):,}, features={len(cols)}, seeds={seeds}, folds={unique_folds}")
    print(f"[tabv2] structured session={sum(c in TABULAR_V2_SESSION_FEATURES for c in cols)}, "
          f"lexical={sum(c in TABULAR_V2_RESPONSE_LEXICAL_FEATURES for c in cols)}, "
          f"student-BGE={sum(c in TABULAR_V2_BGE_FEATURES for c in cols)}")

    oof_by_seed = {seed: np.full(len(df), np.nan) for seed in seeds}
    importance_rows: list[dict[str, float | int | str]] = []
    fold_rows: list[tuple] = []
    best_iters: dict[str, int] = {}

    for f in unique_folds:
        va = folds == f
        # Session purge exactly as in fit_final_v2: a validation session must not appear in
        # the training rows through another objective.
        va_sessions = set(df.loc[va, "session_id"].astype(str))
        tr = (folds != f) & ~df["session_id"].astype(str).isin(va_sessions).to_numpy()

        te.fit(lo_idx[tr], y[tr])
        cnt, rate = _train_rate(lo_idx[tr], y[tr], len(lo_ids), te.prior_)
        te_values = _loo_knn_te(te, cnt, rate, te.prior_)[lo_idx]
        fold_frame = df[cols].copy()
        fold_frame["te_lo"] = te_values
        fold_frame = fold_frame.replace([np.inf, -np.inf], np.nan)

        for seed in seeds:
            params = dict(LGB_PARAMS)
            params.update(seed=seed, feature_fraction_seed=seed + 11,
                          bagging_seed=seed + 23, data_random_seed=seed + 37)
            if args.threads > 0:
                params["num_threads"] = args.threads
            dtr = lgb.Dataset(fold_frame.loc[tr, cols], label=y[tr], free_raw_data=False)
            dva = lgb.Dataset(fold_frame.loc[va, cols], label=y[va], reference=dtr,
                              free_raw_data=False)
            bst = lgb.train(
                params,
                dtr,
                num_boost_round=args.num_rounds,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(args.early_stop, verbose=False)],
            )
            pred = bst.predict(fold_frame.loc[va, cols], num_iteration=bst.best_iteration)
            oof_by_seed[seed][va] = pred
            key = f"seed{seed}_fold{f}"
            best_iters[key] = int(bst.best_iteration)
            bst.save_model(str(model_dir / f"{key}.txt"), num_iteration=bst.best_iteration)
            gains = bst.feature_importance("gain")
            splits = bst.feature_importance("split")
            importance_rows.extend(
                {"seed": seed, "fold": f, "feature": c,
                 "gain": float(g), "split": int(s)}
                for c, g, s in zip(cols, gains, splits)
            )

        p_fold = np.mean([oof_by_seed[s][va] for s in seeds], axis=0)
        fold_rows.append((
            f, int(tr.sum()), int(va.sum()),
            f"{logloss(y[va], p_fold):.4f}",
            f"{roc_auc_score(y[va], p_fold):.4f}",
        ))
        print(f"  fold {f}: train={tr.sum():,} val={va.sum():,} "
              f"ll={logloss(y[va], p_fold):.4f} auc={roc_auc_score(y[va], p_fold):.4f}")

    for seed, pred in oof_by_seed.items():
        if np.isnan(pred).any():
            raise RuntimeError(f"seed {seed} has missing OOF predictions")

    oof_raw = np.mean([oof_by_seed[s] for s in seeds], axis=0)
    oof_cal = _cross_fit_platt(oof_raw, y, folds)
    ll_raw, auc_raw = logloss(y, oof_raw), roc_auc_score(y, oof_raw)
    ll_cal, auc_cal = logloss(y, oof_cal), roc_auc_score(y, oof_cal)
    print(f"[tabv2] pooled raw:   ll={ll_raw:.4f} auc={auc_raw:.4f}")
    print(f"[tabv2] pooled Platt: ll={ll_cal:.4f} auc={auc_cal:.4f}")

    oof_out = pd.DataFrame({
        "response_id": df["response_id"],
        "fold": folds,
        "is_correct": y,
        "oof_tabular_v2": oof_raw,
        "oof_tabular_v2_cal": oof_cal,
    })
    for seed in seeds:
        oof_out[f"oof_tabular_v2_s{seed}"] = oof_by_seed[seed]
    oof_out.to_csv(out / "oof.csv", index=False)

    imp_long = pd.DataFrame(importance_rows)
    imp_long.to_csv(out / "feature_importance_long.csv", index=False)
    imp = (imp_long.groupby("feature", as_index=False)[["gain", "split"]].mean()
           .sort_values("gain", ascending=False))
    imp.to_csv(out / "feature_importance.csv", index=False)

    cfg = {
        "version": 1,
        "model": "tabular_v2",
        "folds": str(Path(args.folds).expanduser()),
        "seeds": seeds,
        "features": cols,
        "knn": KNN,
        "params": LGB_PARAMS,
        "num_rounds": args.num_rounds,
        "early_stop": args.early_stop,
        "best_iters": best_iters,
        "oof_logloss_raw": ll_raw,
        "oof_auc_raw": auc_raw,
        "oof_logloss_crossfit_platt": ll_cal,
        "oof_auc_crossfit_platt": auc_cal,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    top = [
        (r.feature, f"{r.gain:,.1f}", f"{r.split:,.1f}")
        for r in imp.head(35).itertuples(index=False)
    ]
    report = [
        "# tabular_v2 — fixed v3 OOF report\n",
        f"Rows: {len(df):,}; features: {len(cols)}; seeds: {seeds}. "
        "Validation uses the exact `cv_folds_skill.csv` folds and purges validation sessions "
        "from each training fold.\n",
        "## Fold results\n",
        md_table(["fold", "train", "validation", "log loss", "AUC"], fold_rows) + "\n",
        "## Pooled results\n",
        md_table(
            ["prediction", "log loss", "AUC"],
            [
                ("raw seed ensemble", f"{ll_raw:.4f}", f"{auc_raw:.4f}"),
                ("cross-fit Platt", f"{ll_cal:.4f}", f"{auc_cal:.4f}"),
                ("prior", f"{logloss(y, np.full(len(y), prior)):.4f}", "0.5000"),
            ],
        ) + "\n",
        "## Mean feature importance\n",
        md_table(["feature", "gain", "splits"], top) + "\n",
        f"Runtime: {time.time() - t0:.0f}s.\n",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[tabv2] done -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
