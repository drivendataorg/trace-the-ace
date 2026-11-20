#!/usr/bin/env python
"""Train/serve parity check for the `d29dst_*` block (the interview-codebook features).

The inference code (`magnificat.doc29_features.build_d29_matrix`, the module main.py uses) must
reproduce the training-side block (`artifacts/doc29_features_distilled.csv`) on the same
sessions. Cross-validation cannot see a train/serve skew in this block, because every offline
experiment reads the distilled CSV; only a rebuild with the inference code shows it.

Only sessions without annotations are checked. `build_doc29_features.distilled_frame` builds
the training CSV as a mixture: the 4,000 annotated sessions get out-of-fold probabilities, the
other 18,821 the all-data fit. At inference there are no annotations, so the inference code
uses the all-data fit everywhere, and exact parity on the annotated 17.5% is impossible by
construction. A mismatch on the other 82.5%, where both sides use the same estimator, is a
real skew. `check_studst_parity.py` uses the same restriction for the same reason.

The two changes most likely to break parity (both documented in `doc29_features`):
  - adding a character cap (the training `build_text` applies none);
  - computing `position_frac` over the whole transcript instead of within role (14 of the 34
    columns threshold on it at 2/3).

In:  artifacts/doc29_features_distilled.csv, artifacts/doc29_turn_probs.pkl.gz,
     artifacts/doc29_clf.joblib, data/train_features.csv, data/train_transcripts/
Exit status 0 when every column is within --tol on the sampled sessions, 1 otherwise.

Usage (reproduce.py step 04c, from build/workspace/):
  PYTHONPATH=src python scripts/check_doc29_parity.py --n 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magnificat import data as mdata  # noqa: E402
from magnificat.doc29_features import D29DST_COLS, build_d29_matrix  # noqa: E402


def main() -> int:
    """Rebuild the block for sampled unannotated sessions and compare it column by column.

    Returns:
        0 when all D29DST_COLS columns match within --tol, 1 otherwise (prints a per-column
        table of max/mean absolute difference and correlation).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="sessions to check")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = mdata.load_train()
    ref = pd.read_csv(ROOT / "artifacts/doc29_features_distilled.csv")

    probs = pd.read_pickle(ROOT / "artifacts/doc29_turn_probs.pkl.gz", compression="gzip")
    annotated = set(probs.session_id.unique())
    print(f"excluding {len(annotated):,} OOF-scored (annotated) sessions of "
          f"{ref.session_id.nunique():,}", flush=True)

    have = set(ref.session_id) - annotated
    sids = (train[train.session_id.isin(have)]
            .drop_duplicates("session_id")
            .sample(n=args.n, random_state=args.seed).session_id.tolist())
    feats = train[train.session_id.isin(sids)].drop_duplicates("session_id")
    print(f"{len(feats)} sessions | tol={args.tol}", flush=True)

    clf = joblib.load(ROOT / "artifacts/doc29_clf.joblib")
    got = build_d29_matrix(feats, store=mdata.TranscriptStore(ROOT / "data/train_transcripts"),
                           clf=clf, show_progress=True)
    got = got.assign(session_id=feats.set_index("response_id")["session_id"].reindex(got.index))
    m = got.merge(ref[["session_id"] + D29DST_COLS], on="session_id", suffixes=("_twin", "_csv"))
    print(f"joined {len(m)} sessions\n")

    print(f"{'column':42s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'corr':>8s}  verdict")
    bad = []
    for c in D29DST_COLS:
        a, b = m[f"{c}_twin"].to_numpy(float), m[f"{c}_csv"].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 10:
            print(f"{c:42s} {'-':>10s} {'-':>10s} {'-':>8s}  TOO FEW ROWS")
            bad.append(c)
            continue
        d = np.abs(a[ok] - b[ok])
        r = (np.corrcoef(a[ok], b[ok])[0, 1]
             if np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else np.nan)
        v = "PASS" if d.max() <= args.tol else "FAIL"
        if v == "FAIL":
            bad.append(c)
        print(f"{c:42s} {d.max():10.6f} {d.mean():10.6f} {r:8.4f}  {v}")

    print()
    if bad:
        print(f"PARITY FAIL on {len(bad)}/{len(D29DST_COLS)} columns: {bad[:4]}")
        print("   Do not use the inference block until this passes. Most likely causes, in "
              "order: (1) a character cap added to the text builder (the training side applies "
              "none); (2) position_frac computed over the whole transcript rather than within "
              "role, which corrupts every `_late` column; (3) head order drifting from "
              "D29DST_COLS.")
        return 1
    print(f"PARITY PASS — all {len(D29DST_COLS)} columns within {args.tol} on "
          f"{len(m)} sessions. The inference code reproduces the training block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
