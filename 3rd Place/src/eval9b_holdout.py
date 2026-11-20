"""9B holdout evaluation: optimal KT/judge coefficients + is ensembling worth it.

Surface: the 5% session holdout of the full retrains (1,564 rows; sessions
never seen by any of the 4 seed adapters).  Judge channel = TIM+MAS features
(kc_fulltest) trained on all NON-holdout rows, i.e. exactly what ships.

Shipped form:  z' = z_kt/T + b ;  z = w*z' + (1-w)*z_feat ;
               p = sigmoid(z0 + lam*(z - z0))
For every KT source (s0, s1, s2, s3, ens2, ens3, ens4) x judge (none / old
Sonnet-only judge / new Sonnet+Codex judge) we report

  in-sample : (T,b) fit on holdout, (w,lam) grid-searched on holdout
  nested-CV : same search inside 5-fold session-grouped CV, repeated,
              scored on the held-out folds  (the honest number)
  stack     : free 3-parameter logistic stack a*z_kt + f*z_feat + c (CV'd),
              the unconstrained upper bound of the family

plus paired session-clustered bootstrap CIs of every config against
"s0 alone, calibrated" (the no-judge, no-ensemble baseline).

Usage: python kt/eval9b_holdout.py [--old pair_probs_qwen_all.csv]
                                   [--new pair_probs_sc.csv]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from kc_fulltest import (TimChannel, MasChannel, mastery_features, logit,
                         sigmoid, ll, auc, boot_by_session)

from training_paths import DATA
W_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
L_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def fit_tb(y, z):
    f = lambda tb: ll(y, sigmoid(z / tb[0] + tb[1])).mean()
    return minimize(f, [1.0, 0.0], method="Nelder-Mead").x


def fit_stack(y, zk, zf):
    if zf is None:
        f = lambda th: ll(y, sigmoid(th[0] * zk + th[1])).mean()
        th = minimize(f, [1.0, 0.0], method="Nelder-Mead").x
        return np.array([th[0], 0.0, th[1]])
    f = lambda th: ll(y, sigmoid(th[0] * zk + th[1] * zf + th[2])).mean()
    return minimize(f, [1.0, 0.5, 0.0], method="Nelder-Mead").x


def apply_stack(th, zk, zf):
    return sigmoid(th[0] * zk + (th[1] * zf if zf is not None else 0) + th[2])


def shipped_p(zk, zf, z0, T, b, w, lam):
    zc = zk / T + b
    z = zc if zf is None else w * zc + (1 - w) * zf
    return sigmoid(z0 + lam * (z - z0))


def fit_shipped(y, zk, zf, z0):
    """(T,b) from KT alone, then (w,lam) grid on the same rows."""
    T, b = fit_tb(y, zk)
    best = None
    ws = [1.0] if zf is None else W_GRID
    for w in ws:
        for lam in L_GRID:
            v = ll(y, shipped_p(zk, zf, z0, T, b, w, lam)).mean()
            if best is None or v < best[0]:
                best = (v, T, b, w, lam)
    return best[1:]


def group_folds(sess, k, rng):
    u = np.unique(sess)
    perm = rng.permutation(len(u))
    fold_of = {u[i]: j % k for j, i in enumerate(perm)}
    return np.array([fold_of[s] for s in sess])


def nested_cv(y, zk, zf, z0, sess, reps=10, k=5):
    """Returns (p_shipped_cv, p_stack_cv) averaged over repeats."""
    ps, pt = np.zeros(len(y)), np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        fold = group_folds(sess, k, rng)
        for f in range(k):
            tr, te = fold != f, fold == f
            zft = None if zf is None else zf[tr]
            zfe = None if zf is None else zf[te]
            T, b, w, lam = fit_shipped(y[tr], zk[tr], zft, z0)
            ps[te] += shipped_p(zk[te], zfe, z0, T, b, w, lam)
            th = fit_stack(y[tr], zk[tr], zft)
            pt[te] += apply_stack(th, zk[te], zfe)
    return ps / reps, pt / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="pair_probs_qwen_all.csv")
    ap.add_argument("--new", default="pair_probs_sc.csv")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    # ---------- rows / folds / TIM features ----------
    rows = list(csv.DictReader(open(DATA / "folds.csv")))
    y_all = np.array([float(r["y"]) for r in rows])
    sess_all = np.array([r["session_id"] for r in rows])
    rid_all = np.array([r["response_id"] for r in rows])
    npz = np.load(DATA / "hardened.npz", allow_pickle=False)
    s_idx = {s: i for i, s in enumerate(npz["sessions"])}
    Fn = npz["Fn"]
    ri = np.array([s_idx[s] for s in sess_all])

    # ---------- KT holdout preds (4 seeds) ----------
    seeds = ["", "_s1", "_s2", "_s3"]
    Z, y, hid = [], None, None
    for s in seeds:
        rr = list(csv.DictReader(open(DATA / f"preds_holdout_9b{s}.csv")))
        rr.sort(key=lambda r: r["response_id"])
        ids = np.array([r["response_id"] for r in rr])
        yy = np.array([float(r["y"]) for r in rr])
        if hid is None:
            hid, y = ids, yy
        assert np.array_equal(ids, hid) and np.array_equal(yy, y)
        Z.append(logit(np.array([float(r["p"]) for r in rr])))
    Z = np.array(Z)
    pos = {r: i for i, r in enumerate(rid_all)}
    hidx = np.array([pos[r] for r in hid])
    H = np.zeros(len(rows), bool)
    H[hidx] = True
    hold_sess = sess_all[hidx]
    assert not (set(hold_sess) & set(sess_all[~H])), "holdout leaks sessions"
    z0v = logit(y_all[~H].mean())
    z0 = z0v  # scalar; broadcasts in every fold
    n = len(y)
    print(f"holdout rows {n}  sessions {len(set(hold_sess))}  "
          f"base rate {y.mean():.4f}  train rows {(~H).sum()}")
    const = ll(y, np.full(n, sigmoid(z0))).mean()
    print(f"const LL {const:.5f}\n")

    # ---------- judge channels ----------
    kc = json.load(open(DATA / "kc_labels.json"))
    clus_kcs = {int(k.split("-")[1]): v for k, v in kc.items()
                if k.startswith("cluster-") and v}
    lo_kcs = {k: v for k, v in
              json.load(open(DATA / "lo_kcs_pred.json")).items() if v}
    assign = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "assign.csv")):
        assign[r["session_id"]][int(r["pair_idx"])] = int(r["cluster"])

    tc = TimChannel().fit(Fn[ri[~H]], y_all[~H])
    Xtr = tc._impute(Fn[ri[~H]])
    Xho = tc._impute(Fn[ri[hidx]])
    z_tim = tc.predict_logit(Fn[ri[hidx]])
    print(f"TIM only (no judge, no KT): dLL {ll(y, sigmoid(z0 + 0.8*(z_tim-z0))).mean()-const:+.5f}"
          f"  AUC {auc(y, z_tim):.4f}")

    feats = {"none": None}
    for tag, fname in (("judge_old", args.old), ("judge_new", args.new)):
        probs = defaultdict(dict)
        for r in csv.DictReader(open(DATA / fname)):
            probs[r["session_id"]][int(r["pair_idx"])] = (
                float(r["p_true"]), float(r["p_false"]), float(r["p_na"]))
        M, _ = mastery_features(rows, probs, assign, clus_kcs, lo_kcs)
        bc = MasChannel(C=0.3).fit(np.hstack([Xtr, M[~H]]), y_all[~H])
        zf = bc.predict_logit(np.hstack([Xho, M[hidx]]))
        feats[tag] = zf
        print(f"TIM+MAS[{tag}] alone: dLL "
              f"{ll(y, sigmoid(z0 + 0.8*(zf-z0))).mean()-const:+.5f}"
              f"  AUC {auc(y, zf):.4f}")
    print()

    # ---------- KT sources ----------
    srcs = {"s0": Z[0], "s1": Z[1], "s2": Z[2], "s3": Z[3],
            "ens2": Z[:2].mean(0), "ens3": Z[:3].mean(0),
            "ens4": Z.mean(0)}
    for k, z in srcs.items():
        T, b = fit_tb(y, z)
        print(f"{k:5s} raw AUC {auc(y, z):.4f}  T={T:.3f} b={b:+.3f}  "
              f"calib dLL {ll(y, sigmoid(z/T+b)).mean()-const:+.5f}  "
              f"distinct_p {len(np.unique(np.round(z, 4)))}")
    print()

    # ---------- baseline: s0 alone, nested-CV calibrated ----------
    results = {}
    hdr = (f"{'KT':5s} {'judge':9s} | {'in-sample':>9s}  T     b     w    lam "
           f"| {'nestedCV':>9s} {'stackCV':>9s} | vs s0-alone(CV) [95% CI]")
    print(hdr)
    print("-" * len(hdr))
    for jk in ("none", "judge_old", "judge_new"):
        zf = feats[jk]
        for sk in ("s0", "ens2", "ens3", "ens4", "s1", "s2", "s3"):
            zk = srcs[sk]
            T, b, w, lam = fit_shipped(y, zk, zf, z0)
            p_in = shipped_p(zk, zf, z0, T, b, w, lam)
            p_cv, p_st = nested_cv(y, zk, zf, z0, hold_sess, reps=args.reps)
            results[(sk, jk)] = dict(T=T, b=b, w=w, lam=lam,
                                     d_in=ll(y, p_in).mean() - const,
                                     d_cv=ll(y, p_cv).mean() - const,
                                     d_st=ll(y, p_st).mean() - const,
                                     p_cv=p_cv)
    base = results[("s0", "none")]["p_cv"]
    for (sk, jk), r in results.items():
        d = ll(y, r["p_cv"]) - ll(y, base)
        lo, hi = boot_by_session(hold_sess, lambda i: d[i].mean())
        flag = ("BETTER" if hi < 0 else ("WORSE" if lo > 0 else "tie"))
        print(f"{sk:5s} {jk:9s} | {r['d_in']:+.5f}  {r['T']:.2f} {r['b']:+.2f}"
              f"  {r['w']:.1f}  {r['lam']:.1f} | {r['d_cv']:+.5f} "
              f"{r['d_st']:+.5f} | {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] "
              f"{flag}")
    print()

    # ---------- headline pairwise contrasts (nested-CV probs) ----------
    def contrast(a, b_, label):
        d = ll(y, results[a]["p_cv"]) - ll(y, results[b_]["p_cv"])
        lo, hi = boot_by_session(hold_sess, lambda i: d[i].mean())
        flag = ("FIRST BETTER" if hi < 0 else
                ("SECOND BETTER" if lo > 0 else "tie"))
        print(f"{label:44s} {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  {flag}")

    print("headline contrasts (nested-CV, negative = first is better):")
    contrast(("s0", "judge_new"), ("s0", "none"), "s0+judge_new  vs  s0 alone")
    contrast(("s0", "judge_old"), ("s0", "none"), "s0+judge_old  vs  s0 alone")
    contrast(("s0", "judge_new"), ("s0", "judge_old"), "s0+judge_new  vs  s0+judge_old")
    contrast(("ens4", "none"), ("s0", "none"), "ens4 alone    vs  s0 alone")
    contrast(("ens2", "none"), ("s0", "none"), "ens2 alone    vs  s0 alone")
    contrast(("ens4", "judge_new"), ("s0", "judge_new"), "ens4+judge_new vs s0+judge_new")
    contrast(("ens4", "judge_new"), ("s0", "none"), "ens4+judge_new vs s0 alone")
    contrast(("ens4", "judge_new"), ("ens4", "none"), "ens4+judge_new vs ens4 alone")
    print()

    # ---------- transfer-aware shrink note for the winner ----------
    print("shipped-form coefficients (in-sample fit on holdout) for the "
          "main candidates:")
    for key in (("s0", "none"), ("s0", "judge_new"), ("ens4", "none"),
                ("ens4", "judge_new")):
        r = results[key]
        print(f"  {key[0]:5s} {key[1]:9s}: T={r['T']:.3f} b={r['b']:+.3f} "
              f"w={r['w']:.1f} lam={r['lam']:.1f}  (in-sample dLL {r['d_in']:+.5f},"
              f" nested-CV {r['d_cv']:+.5f})")
    print("\nnote: real-test transfer ~0.55 of offline dLL; recalib_transfer.py "
          "gives the robust extra shrink (s=0.625 -> T'~1.65 for ens4).")


if __name__ == "__main__":
    sys.exit(main())
