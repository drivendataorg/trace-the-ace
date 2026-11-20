#!/usr/bin/env python
r"""Fit and export the base-blend inference assets (training stage 3), for unseen objectives.

The test set behaves as entirely unseen objectives, so the folds group by objective
(--group-threshold > 0: by skill cluster) and purge training rows whose session appears in the
validation fold. Training repeats over several fold seeds and ships the ensemble: fold metrics
are noisy, and seed-bagging stabilises both the model and the OOF estimate, which stays honest
per seed. The GBM's te_lo is the deployment encoding (leave-own-objective-out kNN difficulty);
the blend weight and a Platt recalibration are fitted on the ensemble OOF; te assets for the
runtime kNN difficulty of unknown objectives are exported with the TF-IDF, LR and GBM files.

    python -m ace.submission.fit_final_v2 --data-dir data --transcripts-dir data/train_transcripts \
        --eda-dir outputs/eda --emb-dir outputs/embeddings --cache-dir cache --staging outputs/staging \
        --seeds 42,1337,2024 --group-threshold 0.90 --bge-model models/hf/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ace.features import KNNBackoffTE, feature_columns, load_frame
from ace.transcripts import parse_transcript
from ace.utils import RNG_SEED, logloss, md_table

TFIDF_TRANSCRIPT = dict(ngram_range=(1, 2), min_df=5, max_features=300_000, sublinear_tf=True)
TFIDF_OBJECTIVE = dict(ngram_range=(1, 2), sublinear_tf=True)
OBJECTIVE_WEIGHT = 2.0
LR_PARAMS = dict(C=1.0, solver="liblinear", max_iter=2000)
CLIP = (0.01, 0.99)
KNN = dict(k=10, a_knn=10.0, a_prior=10.0)
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, seed=RNG_SEED)


def save_tfidf(vec, path: Path):
    terms = np.empty(len(vec.vocabulary_), dtype=object)
    for t, i in vec.vocabulary_.items():
        terms[i] = t
    np.savez_compressed(path, terms=terms, idf=vec.idf_.astype(np.float64))


def loo_knn_te(te: KNNBackoffTE, cnt: np.ndarray, rate: np.ndarray, prior: float) -> np.ndarray:
    """Zero-own-count kNN TE for every objective (what an unseen objective would get)."""
    w = te.nn_sim * cnt[te.nn_idx]
    r_knn = (w * rate[te.nn_idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
    return (te.a_knn * r_knn + te.a_prior * prior) / (te.a_knn + te.a_prior)


def train_rate(lo_idx, y, n, prior):
    cnt = np.bincount(lo_idx, minlength=n).astype(float)
    s = np.bincount(lo_idx, weights=y, minlength=n)
    rate = np.divide(s, cnt, out=np.full(n, prior), where=cnt > 0)
    return cnt, rate


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--eda-dir", default="outputs/eda")
    ap.add_argument("--emb-dir", default="outputs/embeddings")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--staging", default="outputs/submissions/staging_v2")
    ap.add_argument("--v1-staging", default="outputs/submissions/staging",
                    help="reuse the exported bge model dir from here if present")
    ap.add_argument("--bge-model", default=None,
                    help="export the bge model from this local snapshot (e.g. the pinned "
                         "models/hf/bge-base-en-v1.5 of run_training.sh stage 0) instead of "
                         "--v1-staging / the Hub; offline")
    ap.add_argument("--report-out", default="outputs/models/sub4_fit")
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--group-threshold", type=float, default=0.0,
                    help="if >0, fold by skill clusters (connected components of the "
                         "objective-similarity graph at cosine >= threshold) instead of "
                         "objective ids (removes near-twin leakage)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.bge_model and not (Path(args.bge_model).expanduser() / "modules.json").is_file():
        ap.error(f"--bge-model {args.bge_model}: not a sentence-transformers model dir "
                 f"(no modules.json); run training stage 0 first")

    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    eda = Path(args.eda_dir).expanduser()
    assets = Path(args.staging).expanduser() / "assets"
    gbm_dir = assets / "gbm"
    if gbm_dir.exists():
        shutil.rmtree(gbm_dir)  # remove stale single-seed boosters
    gbm_dir.mkdir(parents=True)
    rep_dir = Path(args.report_out).expanduser()
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame(args.data_dir, eda, args.emb_dir)
    y = df["is_correct"].to_numpy()
    prior = float(y.mean())
    n = len(df)

    # --------------------------------------------- shared, fold-independent pieces
    print("[fit] parsing train transcripts ...")
    tdir = Path(args.transcripts_dir).expanduser()
    sids = sorted(df["session_id"].unique())
    with Pool(args.workers) as pool:
        parsed = pool.map(parse_transcript, [str(tdir / f"{s}.csv") for s in sids], chunksize=64)
    text_by_sid = {sid: text for sid, _, text, _ in parsed}

    print("[fit] TF-IDF ...")
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tvec = TfidfVectorizer(**TFIDF_TRANSCRIPT)
    Xt_sess = tvec.fit_transform([text_by_sid[s] for s in sids])
    sid_ix = {s: i for i, s in enumerate(sids)}
    Xt = Xt_sess[[sid_ix[s] for s in df["session_id"]]]
    ovec = TfidfVectorizer(**TFIDF_OBJECTIVE)
    Xo = ovec.fit_transform(df["learning_objective"])
    X = hstack([Xt, Xo * OBJECTIVE_WEIGHT]).tocsr()
    save_tfidf(tvec, assets / "tfidf_transcript.npz")
    save_tfidf(ovec, assets / "tfidf_objective.npz")

    import lightgbm as lgb
    emb_obj = np.load(Path(args.cache_dir).expanduser() / "emb_objectives.npy")
    lo_ids = sorted(df["learning_objective_id"].unique())
    te = KNNBackoffTE(emb_obj, lo_ids, **KNN)
    lo_idx = te.index_of(df["learning_objective_id"])
    n_obj = len(lo_ids)
    cols = [c for c in feature_columns(include_session_count=False)
            if c not in ("lo_count", "topic_emb")]

    # ------------------------------------------------ grouping key (objective / skill cluster)
    if args.group_threshold > 0:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        S = emb_obj @ emb_obj.T
        n_comp, comp = connected_components(csr_matrix(S >= args.group_threshold),
                                            directed=False)
        sizes = np.bincount(comp)
        print(f"[cv] v3 skill clusters at sim>={args.group_threshold}: {n_comp} clusters "
              f"for {n_obj} objectives, largest {sizes.max()}, "
              f"{(sizes > 1).sum()} multi-objective clusters")
        groups = pd.Series(lo_idx).map(dict(enumerate(comp))).to_numpy()
        folds_file, cv_name = "cv_folds_skill.csv", \
            f"unseen-skill v3 (clusters at cosine>={args.group_threshold})"
    else:
        groups = df["learning_objective_id"].to_numpy()
        folds_file, cv_name = "cv_folds_objective.csv", "unseen-objective v2"

    # --------------------------------------------------------- per-seed training
    coefs, intercepts = [], []
    oof_lr = {s: np.full(n, np.nan) for s in seeds}
    oof_gbm = {s: np.full(n, np.nan) for s in seeds}
    fold_by_seed = {}
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold = np.full(n, -1)
        for f, (_, va) in enumerate(sgkf.split(df, y, groups=groups)):
            fold[va] = f
        fold_by_seed[seed] = fold
        if seed == seeds[0]:
            pd.DataFrame({"response_id": df["response_id"], "fold": fold}).to_csv(
                eda / folds_file, index=False)
            # design check: validation objectives' max cosine to their training folds; with
            # skill-cluster folds it should sit below the threshold (the "related" band)
            obj_fold = np.full(n_obj, -1)
            obj_fold[lo_idx] = fold  # last write wins; folds are constant per objective
            S = emb_obj @ emb_obj.T
            np.fill_diagonal(S, -1)
            max_sim = np.array([S[i, obj_fold != obj_fold[i]].max() for i in range(n_obj)])
            row_sim = max_sim[lo_idx]
            bands = [("twin >=0.95", row_sim >= 0.95),
                     ("sibling 0.90-0.95", (row_sim >= 0.90) & (row_sim < 0.95)),
                     ("related 0.85-0.90", (row_sim >= 0.85) & (row_sim < 0.90)),
                     ("novel <0.85", row_sim < 0.85)]
            print("[cv] validation rows by max sim to train folds: "
                  + ", ".join(f"{nm}: {m.mean():.0%}" for nm, m in bands))
        for f in range(5):
            va = fold == f
            va_sessions = set(df.loc[va, "session_id"])
            tr = (fold != f) & ~df["session_id"].isin(va_sessions).to_numpy()

            clf = LogisticRegression(**LR_PARAMS)
            clf.fit(X[tr], y[tr])
            oof_lr[seed][va] = clf.predict_proba(X[va])[:, 1]
            coefs.append(clf.coef_[0].astype(np.float32))
            intercepts.append(float(clf.intercept_[0]))

            te.fit(lo_idx[tr], y[tr])
            cnt_tr, rate_tr = train_rate(lo_idx[tr], y[tr], n_obj, te.prior_)
            d = df.copy()
            d["te_lo"] = loo_knn_te(te, cnt_tr, rate_tr, te.prior_)[lo_idx]
            Xg = d[cols]
            dtr = lgb.Dataset(Xg[tr], label=y[tr])
            dva = lgb.Dataset(Xg[va], label=y[va], reference=dtr)
            bst = lgb.train(LGB_PARAMS, dtr, 4000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
            oof_gbm[seed][va] = bst.predict(Xg[va], num_iteration=bst.best_iteration)
            bst.save_model(str(gbm_dir / f"seed{seed}_fold{f}.txt"),
                           num_iteration=bst.best_iteration)
        print(f"  [seed {seed}] lr ll={logloss(y, oof_lr[seed]):.4f} "
              f"auc={roc_auc_score(y, oof_lr[seed]):.4f} | "
              f"gbm ll={logloss(y, oof_gbm[seed]):.4f} "
              f"auc={roc_auc_score(y, oof_gbm[seed]):.4f}")
    np.savez_compressed(assets / "logreg.npz",
                        coefs=np.vstack(coefs), intercepts=np.array(intercepts))

    # ------------------------------------------- ensemble OOF -> blend + Platt
    lr_ens = np.mean([oof_lr[s] for s in seeds], axis=0)
    gbm_ens = np.mean([oof_gbm[s] for s in seeds], axis=0)
    ws = np.arange(0, 1.001, 0.05)
    lls = [logloss(y, w * gbm_ens + (1 - w) * lr_ens) for w in ws]
    w_best = float(ws[int(np.argmin(lls))])
    blend = np.clip(w_best * gbm_ens + (1 - w_best) * lr_ens, 1e-6, 1 - 1e-6)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    cal.fit(logit(blend).reshape(-1, 1), y)
    pa, pb = float(cal.coef_[0][0]), float(cal.intercept_[0])
    blend_cal = np.clip(1 / (1 + np.exp(-(pa * logit(blend) + pb))), *CLIP)
    ll_cal, auc_cal = logloss(y, blend_cal), roc_auc_score(y, blend_cal)
    per_seed_blend = {s: logloss(y, np.clip(w_best * oof_gbm[s] + (1 - w_best) * oof_lr[s],
                                            *CLIP)) for s in seeds}
    print(f"[blend] w_gbm={w_best:.2f}, platt a={pa:.3f} b={pb:.3f} -> "
          f"ensemble OOF ll={ll_cal:.4f} auc={auc_cal:.4f} "
          f"(per-seed blend ll: {', '.join(f'{s}:{v:.4f}' for s, v in per_seed_blend.items())})")

    # ------------------------------------------------- te assets (full train) + misc
    te.fit(lo_idx, y)
    cnt_all, rate_all = train_rate(lo_idx, y, n_obj, prior)
    te_table = loo_knn_te(te, cnt_all, rate_all, prior)
    np.savez_compressed(assets / "te_objective.npz",
                        lo_ids=np.array(lo_ids), te=te_table, cnt=cnt_all, rate=rate_all,
                        prior=np.array([prior]), emb=emb_obj.astype(np.float16))
    topics = pd.read_csv(Path(args.emb_dir).expanduser() / "objective_topics_emb.csv")
    (assets / "topic_map.json").write_text(json.dumps(
        dict(zip(topics["learning_objective_id"], topics["topic_emb"].astype(int).tolist()))))

    bge_dst = assets / "bge-base-en-v1.5"
    if not bge_dst.exists() and args.bge_model:
        src = Path(args.bge_model).expanduser()
        print(f"[fit] exporting bge model from {src} ...")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(str(src)).save(str(bge_dst))
    elif not bge_dst.exists():
        v1 = Path(args.v1_staging).expanduser() / "assets" / "bge-base-en-v1.5"
        if v1.exists():
            print("[fit] copying bge model from v1 staging ...")
            shutil.copytree(v1, bge_dst)
        else:
            print("[fit] exporting bge model ...")
            from sentence_transformers import SentenceTransformer
            SentenceTransformer("BAAI/bge-base-en-v1.5").save(str(bge_dst))

    cfg = dict(
        version=2, cv=cv_name, group_threshold=args.group_threshold,
        seeds=seeds, te_mode="loo_knn", knn=KNN,
        blend_w_gbm=w_best, clip=CLIP, platt=[pa, pb],
        gbm_features=cols, objective_weight=OBJECTIVE_WEIGHT,
        oof_logloss_blend=ll_cal, oof_auc_blend=auc_cal,
        oof_logloss_blend_per_seed=per_seed_blend,
        oof_logloss_lr=logloss(y, lr_ens), oof_logloss_gbm=logloss(y, gbm_ens),
        seed=RNG_SEED,
    )
    (assets / "config.json").write_text(json.dumps(cfg, indent=2))

    out = {"response_id": df["response_id"], "fold_obj": fold_by_seed[seeds[0]],
           "oof_lr_ens": lr_ens, "oof_gbm_ens": gbm_ens, "oof_blend_cal": blend_cal}
    for s in seeds:
        out[f"fold_s{s}"] = fold_by_seed[s]
        out[f"oof_lr_s{s}"] = oof_lr[s]
        out[f"oof_gbm_s{s}"] = oof_gbm[s]
    pd.DataFrame(out).to_csv(rep_dir / "oof.csv", index=False)

    R = ["# SUB-4 final fit — unseen-objective regime, seed ensemble\n",
         f"CV protocol v2, seeds {seeds}, te_mode loo_knn, {len(cols)} GBM features, "
         f"blend w_gbm={w_best:.2f}, Platt ({pa:.3f}, {pb:.3f}).\n",
         md_table(["model", "OOF ll", "OOF AUC"],
                  [("TF-IDF + LR (seed ens.)", f"{logloss(y, lr_ens):.4f}",
                    f"{roc_auc_score(y, lr_ens):.4f}"),
                   ("GBM loo_knn (seed ens.)", f"{logloss(y, gbm_ens):.4f}",
                    f"{roc_auc_score(y, gbm_ens):.4f}"),
                   ("blend + Platt (shipped)", f"**{ll_cal:.4f}**", f"**{auc_cal:.4f}**"),
                   ("prior", "0.6088", "0.5000")]) + "\n",
         f"\n_Runtime: {time.time()-t0:.0f}s._\n"]
    (rep_dir / "REPORT.md").write_text("\n".join(R), encoding="utf-8")
    print(f"[fit] staging ready: {assets}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()