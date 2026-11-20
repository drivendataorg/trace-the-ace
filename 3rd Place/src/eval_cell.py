"""A-cell (unseen objective, seen style) coefficient / judge / ensemble eval.

Same machinery as eval9b_holdout.py, but on the honest anchor for the real
test: the dblock-recipe checkpoints (9B ckpt-672 = preds_cross_9b.csv, 4B
ckpt-672 = preds_cross.csv) scored on obj_fold==0 & clus_fold!=0 rows.
Judge channel (TIM+MAS) trained on obj_fold!=0 & clus_fold!=0, as shipped.

KT sources: 9b, 4b, ens(9b+4b) (raw-logit mean; the only ensemble
available on this surface -- cross-model, single seed each).
Optionally --cell D scores the double-blocked cell (of==0 & cf==0) with the
per-checkpoint dblock preds (9B-672, 4B-672, Qwen3.8-504) instead.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kc_fulltest import (TimChannel, MasChannel, mastery_features, logit,
                         sigmoid, ll, auc, boot_by_session)
from eval9b_holdout import fit_tb, fit_shipped, shipped_p, nested_cv

from training_paths import DATA


def load_p(fname):
    return {r["response_id"]: float(r["p"])
            for r in csv.DictReader(open(DATA / fname))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["A", "D"], default="A")
    ap.add_argument("--old", default="pair_probs_qwen_all.csv")
    ap.add_argument("--new", default="pair_probs_sc.csv")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(DATA / "folds.csv")))
    y_all = np.array([float(r["y"]) for r in rows])
    sess_all = np.array([r["session_id"] for r in rows])
    rid_all = np.array([r["response_id"] for r in rows])
    of = np.array([int(r["obj_fold"]) for r in rows])
    cf = np.array([int(r["clus_fold"]) for r in rows])
    npz = np.load(DATA / "hardened.npz", allow_pickle=False)
    s_idx = {s: i for i, s in enumerate(npz["sessions"])}
    Fn = npz["Fn"]
    ri = np.array([s_idx[s] for s in sess_all])

    if args.cell == "A":
        srcs_p = {"9b": load_p("preds_cross_9b.csv"),
                  "4b": load_p("preds_cross.csv")}
        ev = (of == 0) & (cf != 0)
    else:
        srcs_p = {"9b": load_p("q9b/preds_dblock_672.csv"),
                  "4b": load_p("preds_4b_672.csv"),
                  "q38_504": load_p("q38/preds_dblock_504.csv")}
        ev = (of == 0) & (cf == 0)
    for k, d in srcs_p.items():
        ev &= np.isin(rid_all, list(d))
    tr = (of != 0) & (cf != 0)
    eidx = np.where(ev)[0]
    y = y_all[eidx]
    sess = sess_all[eidx]
    z0 = logit(y_all[tr].mean())
    n = len(y)
    const = ll(y, np.full(n, sigmoid(z0))).mean()
    print(f"cell {args.cell}: eval rows {n}  sessions {len(set(sess))}  "
          f"base rate {y.mean():.4f}  train rows {tr.sum()}  const LL {const:.5f}\n")

    kc = json.load(open(DATA / "kc_labels.json"))
    clus_kcs = {int(k.split("-")[1]): v for k, v in kc.items()
                if k.startswith("cluster-") and v}
    lo_kcs = {k: v for k, v in
              json.load(open(DATA / "lo_kcs_pred.json")).items() if v}
    assign = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "assign.csv")):
        assign[r["session_id"]][int(r["pair_idx"])] = int(r["cluster"])
    tc = TimChannel().fit(Fn[ri[tr]], y_all[tr])
    Xtr = tc._impute(Fn[ri[tr]])
    Xev = tc._impute(Fn[ri[eidx]])
    z_tim = tc.predict_logit(Fn[ri[eidx]])
    print(f"TIM only: dLL {ll(y, sigmoid(z0+0.8*(z_tim-z0))).mean()-const:+.5f}  AUC {auc(y, z_tim):.4f}")
    feats = {"none": None}
    for tag, fname in (("judge_old", args.old), ("judge_new", args.new)):
        probs = defaultdict(dict)
        for r in csv.DictReader(open(DATA / fname)):
            probs[r["session_id"]][int(r["pair_idx"])] = (
                float(r["p_true"]), float(r["p_false"]), float(r["p_na"]))
        M, _ = mastery_features(rows, probs, assign, clus_kcs, lo_kcs)
        bc = MasChannel(C=0.3).fit(np.hstack([Xtr, M[tr]]), y_all[tr])
        zf = bc.predict_logit(np.hstack([Xev, M[eidx]]))
        feats[tag] = zf
        print(f"TIM+MAS[{tag}] alone: dLL "
              f"{ll(y, sigmoid(z0+0.8*(zf-z0))).mean()-const:+.5f}  AUC {auc(y, zf):.4f}")
    print()

    srcs = {k: logit(np.array([d[r] for r in rid_all[eidx]]))
            for k, d in srcs_p.items()}
    keys = list(srcs)
    if len(keys) >= 2:
        srcs["ens_all"] = np.mean([srcs[k] for k in keys], axis=0)
        if "9b" in srcs and "4b" in srcs:
            srcs["ens_9b4b"] = (srcs["9b"] + srcs["4b"]) / 2
    for k, z in srcs.items():
        T, b = fit_tb(y, z)
        print(f"{k:9s} raw AUC {auc(y, z):.4f}  raw dLL {ll(y, sigmoid(z)).mean()-const:+.5f}"
              f"  T={T:.3f} b={b:+.3f}  calib dLL {ll(y, sigmoid(z/T+b)).mean()-const:+.5f}")
    print()

    results = {}
    hdr = (f"{'KT':9s} {'judge':9s} | {'in-sample':>9s}  T     b     w    lam "
           f"| {'nestedCV':>9s} {'stackCV':>9s} | vs 9b-alone(CV) [95% CI]")
    print(hdr)
    print("-" * len(hdr))
    order = [k for k in ("9b", "ens_9b4b", "ens_all", "q38_504", "4b") if k in srcs]
    for jk in ("none", "judge_old", "judge_new"):
        zf = feats[jk]
        for sk in order:
            zk = srcs[sk]
            T, b, w, lam = fit_shipped(y, zk, zf, z0)
            p_in = shipped_p(zk, zf, z0, T, b, w, lam)
            p_cv, p_st = nested_cv(y, zk, zf, z0, sess, reps=args.reps)
            results[(sk, jk)] = dict(T=T, b=b, w=w, lam=lam,
                                     d_in=ll(y, p_in).mean() - const,
                                     d_cv=ll(y, p_cv).mean() - const,
                                     d_st=ll(y, p_st).mean() - const,
                                     p_cv=p_cv)
    base = results[("9b", "none")]["p_cv"]
    for (sk, jk), r in results.items():
        d = ll(y, r["p_cv"]) - ll(y, base)
        lo, hi = boot_by_session(sess, lambda i: d[i].mean())
        flag = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "tie")
        print(f"{sk:9s} {jk:9s} | {r['d_in']:+.5f}  {r['T']:.2f} {r['b']:+.2f}"
              f"  {r['w']:.1f}  {r['lam']:.1f} | {r['d_cv']:+.5f} {r['d_st']:+.5f}"
              f" | {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] {flag}")
    print()

    def contrast(a, b_, label):
        if a not in results or b_ not in results:
            return
        d = ll(y, results[a]["p_cv"]) - ll(y, results[b_]["p_cv"])
        lo, hi = boot_by_session(sess, lambda i: d[i].mean())
        flag = ("FIRST BETTER" if hi < 0 else
                ("SECOND BETTER" if lo > 0 else "tie"))
        print(f"{label:46s} {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  {flag}")

    print("headline contrasts (nested-CV, negative = first better):")
    contrast(("9b", "judge_new"), ("9b", "none"), "9b+judge_new   vs  9b alone")
    contrast(("9b", "judge_old"), ("9b", "none"), "9b+judge_old   vs  9b alone")
    contrast(("9b", "judge_new"), ("9b", "judge_old"), "9b+judge_new   vs  9b+judge_old")
    contrast(("ens_9b4b", "none"), ("9b", "none"), "ens(9b+4b)     vs  9b alone")
    contrast(("ens_9b4b", "judge_new"), ("9b", "judge_new"), "ens(9b+4b)+jn  vs  9b+judge_new")
    contrast(("ens_all", "none"), ("9b", "none"), "ens_all        vs  9b alone")
    contrast(("q38_504", "none"), ("9b", "none"), "q38_504        vs  9b(672) alone")
    print()

    # what the currently SHIPPED constants score on this surface
    C = json.load(open(Path(__file__).resolve().parent.parent /
                       "models/assets/constants.json"))
    for jk in ("judge_old", "judge_new"):
        p_ship = shipped_p(srcs["9b"], feats[jk], z0, C["T"], C["b"], C["w_kt"], C["lam"])
        d = ll(y, p_ship) - ll(y, base)
        lo, hi = boot_by_session(sess, lambda i: d[i].mean())
        print(f"SHIPPED constants (T={C['T']} b={C['b']} w={C['w_kt']} lam={C['lam']}, {jk}): "
              f"dLL {ll(y, p_ship).mean()-const:+.5f}   vs 9b-alone(CV) {d.mean():+.5f} "
              f"[{lo:+.5f},{hi:+.5f}]")


if __name__ == "__main__":
    sys.exit(main())
