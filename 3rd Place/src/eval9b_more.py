"""Follow-up 9B evals: which cell matches the real test, full-vs-dblock logit
scale, regret-minimizing T, and lattice (per-distinct-p) calibration.

Surfaces (dblock-recipe 9B ckpt-672 unless noted):
  A = of==0 & cf!=0  (unseen objective, seen style)   preds_cross_9b
  C = of!=0 & cf==0  (seen objective, unseen style)   preds_cross_9b
  D = of==0 & cf==0  (double-blocked)                 q9b/preds_dblock_672
  H = 5% session holdout, FULL retrain s0             preds_holdout_9b
Leaderboard: LL 0.5982 / AUROC 0.6442 with shipped constants
(T=1.10 b=+0.10 w=0.5 lam=0.8, old judge).
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from kc_fulltest import (TimChannel, MasChannel, mastery_features, logit,
                         sigmoid, ll, auc, boot_by_session)
from eval9b_holdout import fit_tb, shipped_p, group_folds

from training_paths import DATA
LB_AUC, LB_LL = 0.6442, 0.5982
C = json.load(open(DATA.parent / "models/assets/constants.json"))


def load_p(f):
    return {r["response_id"]: float(r["p"]) for r in csv.DictReader(open(DATA / f))}


def main():
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
    pos = {r: i for i, r in enumerate(rid_all)}

    cross = load_p("preds_cross_9b.csv")
    dbl = load_p("q9b/preds_dblock_672.csv")
    hold = load_p("preds_holdout_9b.csv")
    hmask = np.isin(rid_all, list(hold))

    kc = json.load(open(DATA / "kc_labels.json"))
    clus_kcs = {int(k.split("-")[1]): v for k, v in kc.items()
                if k.startswith("cluster-") and v}
    lo_kcs = {k: v for k, v in json.load(open(DATA / "lo_kcs_pred.json")).items() if v}
    assign = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "assign.csv")):
        assign[r["session_id"]][int(r["pair_idx"])] = int(r["cluster"])
    feats_M = {}
    for tag, fname in (("old", "pair_probs_qwen_all.csv"), ("new", "pair_probs_sc.csv")):
        probs = defaultdict(dict)
        for r in csv.DictReader(open(DATA / fname)):
            probs[r["session_id"]][int(r["pair_idx"])] = (
                float(r["p_true"]), float(r["p_false"]), float(r["p_na"]))
        feats_M[tag], _ = mastery_features(rows, probs, assign, clus_kcs, lo_kcs)

    def channel(trmask, evidx, tag):
        tc = TimChannel().fit(Fn[ri[trmask]], y_all[trmask])
        Xtr = tc._impute(Fn[ri[trmask]])
        Xev = tc._impute(Fn[ri[evidx]])
        bc = MasChannel(C=0.3).fit(np.hstack([Xtr, feats_M[tag][trmask]]), y_all[trmask])
        return bc.predict_logit(np.hstack([Xev, feats_M[tag][evidx]]))

    cells = {}
    dtr = (of != 0) & (cf != 0)
    for name, mask, src, trm in (
            ("A", (of == 0) & (cf != 0), cross, dtr),
            ("C", (of != 0) & (cf == 0), cross, dtr),
            ("D", (of == 0) & (cf == 0), dbl, dtr),
            ("H", hmask, hold, ~hmask)):
        m = mask & np.isin(rid_all, list(src))
        idx = np.where(m)[0]
        cells[name] = dict(idx=idx, y=y_all[idx], sess=sess_all[idx],
                           z=logit(np.array([src[r] for r in rid_all[idx]])),
                           z0=logit(y_all[trm].mean()),
                           zf_old=channel(trm, idx, "old"),
                           zf_new=channel(trm, idx, "new"))

    # ---------- 1. which cell looks like the leaderboard? ----------
    print("=== 1. cell fingerprints vs leaderboard (AUROC 0.6442, LL 0.5982) ===")
    print(f"{'cell':4s} {'n':>5s} {'base':>6s} {'AUC_kt':>7s} {'AUC_ship':>8s} "
          f"{'LL_ship':>7s} {'constLL':>7s} {'T_kt':>5s} {'T_ship':>6s}")
    for name, c in cells.items():
        p_ship = shipped_p(c["z"], c["zf_old"], c["z0"], C["T"], C["b"], C["w_kt"], C["lam"])
        T, b = fit_tb(c["y"], c["z"])
        Ts, bs = fit_tb(c["y"], logit(p_ship))
        const = ll(c["y"], np.full(len(c["y"]), c["y"].mean())).mean()
        print(f"{name:4s} {len(c['y']):5d} {c['y'].mean():6.3f} {auc(c['y'], c['z']):7.4f} "
              f"{auc(c['y'], p_ship):8.4f} {ll(c['y'], p_ship).mean():7.4f} {const:7.4f} "
              f"{T:5.2f} {Ts:6.2f}")
    print("  (T_ship = extra temperature the SHIPPED probabilities still want; "
          "1.0 = perfectly calibrated on that cell)\n")

    # ---------- 2. full vs dblock logit scale on shared rows ----------
    print("=== 2. full-retrain vs dblock-model logit scale on shared rows ===")
    shared = [r for r in hold if r in cross]
    zh = logit(np.array([hold[r] for r in shared]))
    zc = logit(np.array([cross[r] for r in shared]))
    yy = y_all[[pos[r] for r in shared]]
    cellA = np.array([of[pos[r]] == 0 for r in shared])
    for tag, m in (("all shared", np.ones(len(shared), bool)),
                   ("shared & unseen-obj(A) for dblock model", cellA),
                   ("shared & seen-obj(C) for dblock model", ~cellA)):
        if m.sum() < 30:
            continue
        sl = np.polyfit(zc[m], zh[m], 1)[0]
        print(f"  {tag:42s} n={m.sum():4d}  std(z_full)/std(z_dblock)={zh[m].std()/zc[m].std():.3f} "
              f"slope(full~dblock)={sl:.3f}  AUC full {auc(yy[m], zh[m]):.4f} dblock {auc(yy[m], zc[m]):.4f}")
    print("  NOTE: on shared rows the FULL model has seen these objectives (train), "
          "the dblock model has not for cell A -> the ratio bounds how much sharper the "
          "shipped model is on in-distribution vs unseen objectives.\n")

    # ---------- 3. regret-minimizing T over test compositions ----------
    print("=== 3. loss vs T on each cell (KT-only, b refit per T; dLL vs cell const) ===")
    Tgrid = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2]
    table = {}
    for name, c in cells.items():
        const = ll(c["y"], np.full(len(c["y"]), c["y"].mean())).mean()
        row = []
        for T in Tgrid:
            f = lambda b: ll(c["y"], sigmoid(c["z"] / T + b[0])).mean()
            b = minimize(f, [0.0], method="Nelder-Mead").x[0]
            row.append(ll(c["y"], sigmoid(c["z"] / T + b)).mean() - const)
        table[name] = np.array(row)
    print("T     " + " ".join(f"{T:7.2f}" for T in Tgrid))
    for name, row in table.items():
        print(f"{name:5s} " + " ".join(f"{v:+7.4f}" for v in row) + f"   best T={Tgrid[int(np.argmin(row))]}")
    print("regret (dLL - cell's own best):")
    reg = {n: r - r.min() for n, r in table.items()}
    for name, row in reg.items():
        print(f"{name:5s} " + " ".join(f"{v:+7.4f}" for v in row))
    mixes = {"A only": {"A": 1}, "A+C equal": {"A": .5, "C": .5}, "A+C+D equal": {"A": 1/3, "C": 1/3, "D": 1/3},
             "worst-case A/C/D": None}
    for label, wts in mixes.items():
        if wts is None:
            tot = np.max([reg[n] for n in ("A", "C", "D")], axis=0)
        else:
            tot = sum(w * reg[n] for n, w in wts.items())
        print(f"  {label:18s} min-regret T = {Tgrid[int(np.argmin(tot))]}  (regret {tot.min():+.4f})")
    print()

    # ---------- 4. lattice calibration vs T/b (CV) on A and D ----------
    print("=== 4. lattice (per-distinct-logit) calibration vs T/b, 10x5-fold session CV ===")
    for name in ("A", "C", "D", "H"):
        c = cells[name]
        y, z, sess = c["y"], c["z"], c["sess"]
        const = ll(y, np.full(len(y), y.mean())).mean()
        p_tb, p_iso, p_bin = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        reps = 10
        for r in range(reps):
            fold = group_folds(sess, 5, np.random.default_rng(200 + r))
            for f in range(5):
                tr, te = fold != f, fold == f
                T, b = fit_tb(y[tr], z[tr])
                p_tb[te] += sigmoid(z[te] / T + b)
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(z[tr], y[tr])
                p_iso[te] += iso.predict(z[te])
                # binned: mean y per distinct logit with Laplace smoothing toward T/b
                pt = sigmoid(z / T + b)
                key = np.round(z, 3)
                d = defaultdict(lambda: [0.0, 0.0])
                for k, yy_, pp in zip(key[tr], y[tr], pt[tr]):
                    d[k][0] += yy_; d[k][1] += 1
                a = 20.0
                p_bin[te] += np.array([(d[k][0] + a * pp) / (d[k][1] + a) if k in d else pp
                                       for k, pp in zip(key[te], pt[te])])
        p_tb /= reps; p_iso /= reps; p_bin /= reps
        d1 = ll(y, p_iso) - ll(y, p_tb); d2 = ll(y, p_bin) - ll(y, p_tb)
        lo1, hi1 = boot_by_session(sess, lambda i: d1[i].mean())
        lo2, hi2 = boot_by_session(sess, lambda i: d2[i].mean())
        print(f"{name}: distinct z {len(np.unique(np.round(z,3)))}  T/b dLL {ll(y,p_tb).mean()-const:+.5f}  "
              f"isotonic {ll(y,p_iso).mean()-const:+.5f} ({d1.mean():+.5f} [{lo1:+.5f},{hi1:+.5f}])  "
              f"binned+shrink {ll(y,p_bin).mean()-const:+.5f} ({d2.mean():+.5f} [{lo2:+.5f},{hi2:+.5f}])")
    print()

    # ---------- 5. judge weight stability across cells (in-sample) ----------
    print("=== 5. judge weight / lambda by cell (in-sample grid, T,b from KT alone) ===")
    from eval9b_holdout import fit_shipped
    for name, c in cells.items():
        const = ll(c["y"], np.full(len(c["y"]), c["y"].mean())).mean()
        for tag in ("old", "new"):
            zf = c["zf_" + tag]
            T, b, w, lam = fit_shipped(c["y"], c["z"], zf, c["z0"])
            p = shipped_p(c["z"], zf, c["z0"], T, b, w, lam)
            p1 = shipped_p(c["z"], None, c["z0"], T, b, 1.0, 1.0)
            print(f"{name} judge_{tag}: T={T:.2f} b={b:+.2f} w={w:.1f} lam={lam:.1f}  "
                  f"dLL {ll(c['y'],p).mean()-const:+.5f}  vs KT-alone {ll(c['y'],p).mean()-ll(c['y'],p1).mean():+.5f}")


if __name__ == "__main__":
    sys.exit(main())
