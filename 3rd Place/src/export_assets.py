"""Fit the TIM/mastery mapper and export inference-compatible array assets.

With --mas7, fits 27 imputed TIM plus 7 mastery features and exports the
34-feature logistic mapper, cluster centers/labels, CCSS embeddings, and
starting calibration settings to RUNS/features/. Calibration must be
refitted for newly trained models before deployment.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc_fulltest import TimChannel, mastery_features
from mas7 import mastery_features7
from lo2kc import embed

from training_paths import DATA
from training_paths import RUNS
OUT = RUNS / "features"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="pair_probs_qwen_all.csv",
                    help="judge pair probs used to build the mastery features")
    ap.add_argument("--mapper-only", action="store_true",
                    help="only (re)write mapper.npz; leave centers/ccss/constants")
    ap.add_argument("--mapper-out", default="mapper.npz")
    ap.add_argument("--mas7", action="store_true",
                    help="use the 7-feature mastery block (adds log1p(n_rel))")
    args = ap.parse_args()
    kc = json.load(open(DATA / "kc_labels.json"))
    clus_kcs = {int(k.split("-")[1]): v for k, v in kc.items()
                if k.startswith("cluster-") and v}
    lo_kcs = {k: v for k, v in
              json.load(open(DATA / "lo_kcs_pred.json")).items() if v}
    probs = defaultdict(dict)
    for r in csv.DictReader(open(DATA / args.probs)):
        probs[r["session_id"]][int(r["pair_idx"])] = (
            float(r["p_true"]), float(r["p_false"]), float(r["p_na"]))
    assign = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "assign.csv")):
        assign[r["session_id"]][int(r["pair_idx"])] = int(r["cluster"])

    rows = list(csv.DictReader(open(DATA / "folds.csv")))
    y = np.array([float(r["y"]) for r in rows])
    sess = np.array([r["session_id"] for r in rows])
    npz = np.load(DATA / "hardened.npz", allow_pickle=False)
    s_idx = {s: i for i, s in enumerate(npz["sessions"])}
    Fn = npz["Fn"]
    ri = np.array([s_idx[s] for s in sess])

    mf = mastery_features7 if args.mas7 else mastery_features
    M, lev = mf(rows, probs, assign, clus_kcs, lo_kcs)
    print(f"{len(rows)} rows, mastery L0 {np.mean(lev == 0):.1%}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tc = TimChannel().fit(Fn[ri], y)          # for median + tim-only fallback
    X = np.hstack([tc._impute(Fn[ri]), M])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X), y)
    print(f"joint mapper: train accuracy "
          f"{np.mean((clf.decision_function(sc.transform(X)) > 0) == y):.3f} "
          f"(sanity only)")

    np.savez(OUT / args.mapper_out,
             tim_median=tc.med,
             scaler_mean=sc.mean_, scaler_std=sc.scale_,
             coef=clf.coef_[0], intercept=clf.intercept_,
             t_mean=tc.sc.mean_, t_std=tc.sc.scale_,
             t_coef=tc.clf.coef_[0], t_intercept=tc.clf.intercept_)
    print(f"mapper written to {OUT / args.mapper_out} (probs={args.probs})")
    if args.mapper_only:
        return

    centers = np.load(DATA / "centers.npy")
    np.save(OUT / "centers.npy", centers.astype(np.float32))
    json.dump({str(k): v for k, v in clus_kcs.items()},
              open(OUT / "cluster_kcs.json", "w"))

    h = json.load(open(DATA / "ccss_hier.json"))
    std_re = re.compile(r"^(K|\d)\.([A-Z]+)\.([A-Z])\.(\d+)$")
    codes = sorted(k for k in h if std_re.match(k))
    from transformers import AutoModel, AutoTokenizer
    EMB = str(Path(__file__).resolve().parents[1] / "models/minilm")
    tok = AutoTokenizer.from_pretrained(EMB)
    model = AutoModel.from_pretrained(EMB).eval()
    E = embed([h[c] for c in codes], tok, model)
    np.savez(OUT / "ccss.npz", codes=np.array(codes),
             emb=E.astype(np.float32))
    print(f"ccss: {len(codes)} standards embedded")

    json.dump({
        "kt_adapters": ["s0"], "kt_cal": {"1": [1.6, 0.148]},
        "w_kt": 0.6, "lam": 1.3,
        "prior": float(y.mean()),
        "lo_topk": 4, "lo_rel": 0.85,
        "judge_ctx": 3, "judge_tcut": 600, "judge_scut": 400,
        "pair_tcut": 400, "pair_scut": 300,
        "kt_max_tr": 3400, "kt_max_len": 4096,
        "assessable_min_words": 8,
    }, open(OUT / "constants.json", "w"), indent=1)
    print("assets written to", OUT)


if __name__ == "__main__":
    sys.exit(main())
