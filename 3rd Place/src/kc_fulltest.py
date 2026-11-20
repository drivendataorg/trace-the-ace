"""Powered double-blocked test of the KC-mastery channel (all 35k rows).

Inputs: data/pair_probs_enc.csv (encoder proxy labeler, all assessable
pairs), data/assign.csv, data/kc_labels.json, data/folds.csv,
data/hardened.npz. Mirrors the hardened_model double-blocked harness:
25 (obj_fold x clus_fold) train/eval combos, models trained fresh per combo,
eval rows concatenated, session-clustered bootstrap CIs.

Channels: TIM (27 invariant features, LR C=0.3 with median imputation),
MAS (6 mastery features), TIM+MAS (concatenated). The decision number is
dLL(TIM+MAS - TIM) with its CI: the KC channel ships only if the CI
excludes zero (pre-committed gate G2 analogue).

Usage: python kt/kc_fulltest.py [--probs data/pair_probs_enc.csv]
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training_paths import DATA
EPS = 1e-7
RNG = np.random.default_rng(0)
LAMBDA = 0.80


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def boot_by_session(sess, stat, n=2000):
    codes, inv = np.unique(sess, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(codes) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(codes))]
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        out.append(stat(idx))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def kc_levels(code):
    p = code.split(".")
    return code, ".".join(p[:3]), p[1]


class TimChannel:
    def fit(self, F, y):
        # A raw corpus may omit timestamps entirely; no training median then exists.
        observed = np.any(np.isfinite(F), axis=0)
        self.med = np.zeros(F.shape[1], dtype=float)
        self.med[observed] = np.nanmedian(F[:, observed], axis=0)
        X = self._impute(F)
        self.sc = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=0.3, max_iter=2000).fit(
            self.sc.transform(X), y)
        return self

    def _impute(self, F):
        X = F.copy()
        bad = np.isnan(X)
        X[bad] = np.take(self.med, np.nonzero(bad)[1])
        return X

    def predict_logit(self, F):
        return self.clf.decision_function(self.sc.transform(self._impute(F)))


class MasChannel:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(
            self.sc.transform(X), y)
        return self

    def predict_logit(self, X):
        return self.clf.decision_function(self.sc.transform(X))


def mastery_features(rows, probs, assign, clus_kcs, lo_kcs):
    """6 features per response row (same layout as kc_validity)."""
    # per-session evidence: (level-sets, correctness, weight)
    sess_ev = {}
    for s in {r["session_id"] for r in rows}:
        ev = []
        for i, (pt, pf, pn) in probs.get(s, {}).items():
            c = assign.get(s, {}).get(i)
            if c is None or c not in clus_kcs:
                continue
            lv = [set(), set(), set()]
            for code in clus_kcs[c]:
                for L, part in enumerate(kc_levels(code)):
                    lv[L].add(part)
            corr = pt / max(pt + pf, 1e-6)
            ev.append((lv, corr, 1.0 - pn))
        sess_ev[s] = ev
    X = np.zeros((len(rows), 6))
    lev_used = np.zeros(len(rows), int)
    for k, r in enumerate(rows):
        s, lo = r["session_id"], r["learning_objective_id"]
        ev = sess_ev[s]
        tsets = [set(), set(), set()]
        for code in lo_kcs.get(lo, []):
            for L, part in enumerate(kc_levels(code)):
                tsets[L].add(part)
        wa = sum(w for _, _, w in ev)
        overall = (sum(c * w for _, c, w in ev) / wa) if wa > 1e-6 else 0.5
        score, lev, wr = overall, 3, 0.0
        for L in range(3):
            if not tsets[L]:
                continue
            rel = [(c, w) for lv, c, w in ev if lv[L] & tsets[L]]
            wrel = sum(w for _, w in rel)
            if wrel > 1e-6:
                score = sum(c * w for c, w in rel) / wrel
                lev, wr = L, wrel
                break
        lev_used[k] = lev
        X[k] = [score, overall, lev, np.log1p(wr), np.log1p(wa),
                score - overall]
    return X, lev_used


def evaluate(tag, y, z, z0, sess, lam=1.0):
    p = sigmoid(z0 + lam * (z - z0))
    p0 = sigmoid(z0)
    d = ll(y, p) - ll(y, p0)
    lo, hi = boot_by_session(sess, lambda i: d[i].mean())
    print(f"{tag:24s} dLL {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  "
          f"AUC {auc(y, z):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=str(DATA / "pair_probs_enc.csv"))
    ap.add_argument("--restrict", action="store_true",
                    help="keep only rows whose session has labeled pairs "
                         "(for subset labeling runs)")
    ap.add_argument("--lo-pred", action="store_true",
                    help="use embedding-predicted LO KCs (lo_kcs_pred.json, "
                         "the shipped zero-leakage configuration) instead of "
                         "human annotation's LO labels")
    args = ap.parse_args()
    t0 = time.time()

    kc = json.load(open(DATA / "kc_labels.json"))
    clus_kcs = {int(k.split("-")[1]): v for k, v in kc.items()
                if k.startswith("cluster-") and v}
    lo_kcs = {k[3:]: v for k, v in kc.items() if k.startswith("lo-") and v}
    if args.lo_pred:
        lo_kcs = {k: v for k, v in
                  json.load(open(DATA / "lo_kcs_pred.json")).items() if v}
        print("using embedding-predicted LO KCs (shipped config)")

    probs = defaultdict(dict)
    for r in csv.DictReader(open(args.probs)):
        probs[r["session_id"]][int(r["pair_idx"])] = (
            float(r["p_true"]), float(r["p_false"]), float(r["p_na"]))
    assign = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "assign.csv")):
        assign[r["session_id"]][int(r["pair_idx"])] = int(r["cluster"])

    rows = list(csv.DictReader(open(DATA / "folds.csv")))
    if args.restrict:
        rows = [r for r in rows if r["session_id"] in probs]
        print(f"restricted to {len(rows)} rows with labeled sessions")
    y = np.array([float(r["y"]) for r in rows])
    sess = np.array([r["session_id"] for r in rows])
    of = np.array([int(r["obj_fold"]) for r in rows])
    cf = np.array([int(r["clus_fold"]) for r in rows])

    npz = np.load(DATA / "hardened.npz", allow_pickle=False)
    s_idx = {s: i for i, s in enumerate(npz["sessions"])}
    Fn = npz["Fn"]
    ri = np.array([s_idx[s] for s in sess])

    print(f"{len(rows)} rows; building mastery features...", flush=True)
    M, lev = mastery_features(rows, probs, assign, clus_kcs, lo_kcs)
    print("backoff level: " + "  ".join(
        f"L{L} {np.mean(lev == L):.1%}" for L in range(4)) +
        f"  [{time.time()-t0:.0f}s]", flush=True)

    acc = defaultdict(list)
    for i in range(5):
        for j in range(5):
            tr = (of != i) & (cf != j)
            ev = (of == i) & (cf == j)
            if ev.sum() == 0:
                continue
            z0 = logit(y[tr].mean())
            tc = TimChannel().fit(Fn[ri[tr]], y[tr])
            mc = MasChannel().fit(M[tr], y[tr])
            # TIM features contain NaNs -> impute for the combined model
            Xtr = tc._impute(Fn[ri[tr]])
            bc = MasChannel(C=0.3).fit(np.hstack([Xtr, M[tr]]), y[tr])
            Xev = tc._impute(Fn[ri[ev]])
            acc["y"].append(y[ev])
            acc["sess"].append(sess[ev])
            acc["z0"].append(np.full(int(ev.sum()), z0))
            acc["zt"].append(tc.predict_logit(Fn[ri[ev]]))
            acc["zm"].append(mc.predict_logit(M[ev]))
            acc["zb"].append(bc.predict_logit(np.hstack([Xev, M[ev]])))
    A = {k: np.concatenate(v) for k, v in acc.items()}
    z0 = A["z0"]
    print(f"\n== double-blocked, {len(A['y'])} eval rows ==", flush=True)
    evaluate("TIM alone", A["y"], A["zt"], z0, A["sess"])
    evaluate("MAS alone", A["y"], A["zm"], z0, A["sess"])
    evaluate("TIM+MAS joint", A["y"], A["zb"], z0, A["sess"])
    evaluate("logit avg (zt+zm)/2", A["y"], (A["zt"] + A["zm"]) / 2,
             z0, A["sess"])
    evaluate(f"TIM+MAS lam={LAMBDA}", A["y"], A["zb"], z0, A["sess"],
             lam=LAMBDA)

    # decision number: does adding MAS beat TIM alone?
    pt = sigmoid(z0 + LAMBDA * (A["zt"] - z0))
    pb = sigmoid(z0 + LAMBDA * (A["zb"] - z0))
    d = ll(A["y"], pb) - ll(A["y"], pt)
    lo, hi = boot_by_session(A["sess"], lambda i: d[i].mean())
    print(f"\ndLL(TIM+MAS - TIM) at lam={LAMBDA}: {d.mean():+.5f} "
          f"[{lo:+.5f},{hi:+.5f}]  "
          f"{'PASS' if hi < 0 else 'FAIL'} (need CI < 0)", flush=True)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
