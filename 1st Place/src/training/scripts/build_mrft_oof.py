#!/usr/bin/env python
"""Recover the `response_id` of each MRBench-tuned (`mrft`) 3-fold out-of-fold prediction.

`artifacts/mrft_oof_3fold.csv` is a bare `fold,y,p` table with no key to join the XGBoost
model's out-of-fold predictions on. The key can be recovered: a qual CSV's `Dialogue ID` is the
positional index into `traceace_full.pkl`, whose `index` column is the `response_id`. This
script rebuilds the join and checks it:

  1. Every recovered `response_id` exists in `train_labels.csv` with the same label as the
     qual CSV's `Correct` column, and no response appears in two folds. The label check also
     detects the qual-writer index bug (`batch_idx + sample_idx`, correct only at
     batch_size=1; the shipped dialogue-kt writes `len(all_labels) + sample_idx`). The bug
     cannot corrupt `Dialogue ID` (read from `meta_data`) but would corrupt the probability
     and label attached to it, and a scrambled file mismatches the labels at chance rate.
  2. The folds share no learning objective.
  3. The recovered (y, p) of each fold reproduce `artifacts/mrft_oof_3fold.csv` (same
     multiset of p), so the keyed table holds the data the calibrator was fitted on.

In:  external/dialogue-kt/data/annotated/traceace_full.pkl
     external/dialogue-kt/results/qual_lmkt_mrft_sa_8b_a1f{k}_incfirst_{k}.csv, k = 1, 2, 3
     data/train_labels.csv, data/train_features.csv, artifacts/mrft_oof_3fold.csv
Out: artifacts/mrft_oof_3fold_keyed.csv (response_id, fold, y, p)

Usage (reproduce.py step 07b, from build/workspace/; needs the qual files of a 06c re-run):
  python scripts/build_mrft_oof.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from magnificat.config import features_path, labels_path  # noqa: E402

DKT = ROOT / "external/dialogue-kt"
PKL = DKT / "data/annotated/traceace_full.pkl"
QUAL = {k: DKT / "results" / f"qual_lmkt_mrft_sa_8b_a1f{k}_incfirst_{k}.csv" for k in (1, 2, 3)}
BARE = ROOT / "artifacts/mrft_oof_3fold.csv"
OUT = ROOT / "artifacts/mrft_oof_3fold_keyed.csv"


def terminal_preds(qual_path: Path) -> pd.DataFrame:
    """Extract the terminal-turn prediction of each dialogue from a qual CSV.

    Takes the last row with a numeric `Prob` per `Dialogue ID`. The pooled
    `artifacts/mrft_oof_3fold.csv` keeps every row with a true/false `Correct` instead; on
    this data that is the same set of rows (non-target turns carry `na`), which check 3
    confirms.

    Args:
        qual_path: a dialogue-kt qual CSV (Dialogue ID, Prob, Correct, ...).

    Returns:
        DataFrame with `did` (Dialogue ID, int), `p` (probability) and `y` (0/1).
    """
    q = pd.read_csv(qual_path)
    q["p"] = pd.to_numeric(q["Prob"], errors="coerce")
    q = q[q["p"].notna()].copy()
    term = q.groupby("Dialogue ID", sort=False).tail(1)
    c = term["Correct"].astype(str).str.strip().str.lower()
    return pd.DataFrame(
        {"did": term["Dialogue ID"].astype(int).values,
         "p": term["p"].astype(float).values,
         "y": (c == "true").astype(int).values}
    )


def main() -> None:
    """Key the three folds' terminal predictions by response_id, check them and write them.

    Returns:
        None. Writes artifacts/mrft_oof_3fold_keyed.csv; exits non-zero when a check fails.
    """
    meta = pd.read_pickle(PKL)
    rid = meta["index"].values
    labels = pd.read_csv(labels_path()).set_index("response_id")["is_correct"]
    feats = pd.read_csv(features_path()).set_index("response_id")

    parts = []
    for k, path in QUAL.items():
        t = terminal_preds(path)
        t["response_id"] = rid[t["did"].values]
        t["fold"] = k
        parts.append(t)
        print(f"  fold {k}: {len(t):>6} responses from {path.name}")

    oof = pd.concat(parts, ignore_index=True)

    # --- check 1: the join is real (and the index bug did not scramble these files) ---
    missing = ~oof["response_id"].isin(labels.index)
    if missing.any():
        sys.exit(f"FAIL: {int(missing.sum())} recovered response_ids are not in train_labels.csv")
    y_true = labels.loc[oof["response_id"]].values.astype(int)
    agree = float((y_true == oof["y"].values).mean())
    print(f"\ncheck 1  label agreement vs train_labels.csv: {agree:.6f} "
          f"({int((y_true == oof['y'].values).sum())}/{len(oof)})")
    if agree < 1.0:
        sys.exit("FAIL: labels disagree, so the Dialogue ID mapping is not the identity "
                 "(or the qual-writer index bug described in this script's docstring "
                 "scrambled this run). Do not use.")

    dupes = int(oof["response_id"].duplicated().sum())
    print(f"check 1b duplicate response_ids across folds: {dupes}")
    if dupes:
        sys.exit("FAIL: a response appears in >1 fold -> folds are not disjoint")

    # --- check 2: folds are LO-disjoint ---
    los = {k: set(feats.loc[g["response_id"], "learning_objective_id"]) for k, g in oof.groupby("fold")}
    for a in (1, 2, 3):
        for b in (1, 2, 3):
            if a < b and los[a] & los[b]:
                sys.exit(f"FAIL: folds {a},{b} share {len(los[a] & los[b])} learning objectives")
    print(f"check 2  folds LO-disjoint: yes ({[len(los[k]) for k in (1, 2, 3)]} LOs per fold)")

    # --- check 3: reproduces the bare table the calibrator was fitted on ---
    bare = pd.read_csv(BARE)
    ok = True
    for k in (1, 2, 3):
        a = oof[oof["fold"] == k]
        b = bare[bare["fold"] == k]
        if len(a) != len(b):
            print(f"check 3  fold {k}: SIZE MISMATCH keyed={len(a)} bare={len(b)}")
            ok = False
            continue
        same_order = bool(np.allclose(a["p"].values, b["p"].values, atol=1e-9)
                          and (a["y"].values == b["y"].values).all())
        same_multiset = bool(np.allclose(np.sort(a["p"].values), np.sort(b["p"].values), atol=1e-9))
        print(f"check 3  fold {k}: n={len(a)}  identical in order={same_order}  "
              f"same p-multiset={same_multiset}")
        ok = ok and same_multiset

    print(f"\npooled: n={len(oof)}  base={oof['y'].mean():.4f}  "
          f"raw LL={log_loss(oof['y'], oof['p']):.4f}  AUROC={roc_auc_score(oof['y'], oof['p']):.4f}")
    if not ok:
        sys.exit("FAIL: keyed table does not reproduce artifacts/mrft_oof_3fold.csv")

    oof = oof.sort_values("response_id").reset_index(drop=True)
    oof[["response_id", "fold", "y", "p"]].to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
