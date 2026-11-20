#!/usr/bin/env python
"""Train/serve parity check for the `studst_*` block (the student-state features).

The inference code (`magnificat.student_features.build_studst_matrix`, the module main.py uses)
must reproduce the training-side block (`artifacts/stu_dst_features.csv`) on the same sessions.
Cross-validation cannot see a train/serve skew in this block, because every offline experiment
reads the distilled CSV; only a rebuild with the inference code shows it.

The student head truncates in two stages (`turn_text` stores teacher[-220:] / student[:220],
then `_text` takes [-200:] / [:300]), so the effective student window is 220 characters, not
300. Inference code that applied [:300] directly would fail this check.

Passing means: same sessions, same columns, and per-column max |difference| within --tol, on
sessions without annotations (see the comment in `main`).

In:  artifacts/stu_dst_features.csv, artifacts/stu_turn_probs.pkl.gz,
     artifacts/stu_states_clf.joblib, data/train_features.csv, data/train_transcripts/
Exit status 0 on a pass, 1 on a column mismatch.

Usage (reproduce.py step 04c, from build/workspace/):
  PYTHONPATH=src python scripts/check_studst_parity.py --n 400
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
from magnificat.student_features import STUDST_COLS, build_studst_matrix  # noqa: E402


def main() -> int:
    """Rebuild the block for sampled unannotated sessions and compare it column by column.

    Returns:
        0 when all STUDST_COLS columns match within --tol, 1 otherwise. Exits with an error
        when no per-turn probability table is found.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="sessions to check")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = mdata.load_train()
    ref = pd.read_csv(ROOT / "artifacts/stu_dst_features.csv")
    # The training CSV is a mixture by construction: sessions with annotated turns (21%)
    # carry out-of-fold predictions, the rest the all-data fit (`distil_student_states.py`,
    # column `_labelled`). At inference there are no annotations, so the inference code uses
    # the all-data fit everywhere, and exact parity on labelled sessions is impossible. The
    # check is restricted to unlabelled sessions, where both sides use the same estimator; a
    # mismatch there is a real train/serve skew.
    # The second path below is a fallback name for the same table. Do not point the check at
    # stu_turn_probs_full.pkl.gz: every session is labelled there, so `unlabelled` is empty and
    # nothing is checked.
    _probs = ROOT / "artifacts/stu_turn_probs.pkl.gz"
    if not _probs.exists():
        _probs = ROOT / "artifacts/stu_turn_probs.pkl_32b_BACKUP.pkl.gz"
    if not _probs.exists():
        raise SystemExit(
            "no stu_turn_probs pickle found. This gate is the only check on the 12 shipped\n"
            "studst_ columns; refusing to report a pass it did not measure.")
    _t = pd.read_pickle(_probs, compression="gzip")
    _lab = _t.groupby("session_id")["_labelled"].max()
    unlabelled = set(_lab[_lab == 0].index)
    print(f"excluding {len(_lab) - len(unlabelled):,} OOF-scored (labelled) sessions of "
          f"{len(_lab):,}", flush=True)
    have = set(ref.session_id) & unlabelled
    sids = (train[train.session_id.isin(have)]
            .drop_duplicates("session_id")
            .sample(n=args.n, random_state=args.seed).session_id.tolist())
    feats = train[train.session_id.isin(sids)].drop_duplicates("session_id")
    print(f"{len(feats)} sessions | tol={args.tol}", flush=True)

    clf = joblib.load(ROOT / "artifacts/stu_states_clf.joblib")
    got = build_studst_matrix(feats, store=mdata.TranscriptStore(ROOT / "data/train_transcripts"),
                              clf=clf, show_progress=True)
    got = got.assign(session_id=feats.set_index("response_id")["session_id"].reindex(got.index))
    m = got.merge(ref[["session_id"] + STUDST_COLS], on="session_id",
                  suffixes=("_twin", "_csv"))
    print(f"joined {len(m)} sessions\n")

    print(f"{'column':34s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'corr':>8s}  verdict")
    bad = []
    for c in STUDST_COLS:
        a, b = m[f"{c}_twin"].to_numpy(float), m[f"{c}_csv"].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 10:
            print(f"{c:34s} {'-':>10s} {'-':>10s} {'-':>8s}  TOO FEW ROWS")
            bad.append(c)
            continue
        d = np.abs(a[ok] - b[ok])
        r = np.corrcoef(a[ok], b[ok])[0, 1] if np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else np.nan
        v = "PASS" if d.max() <= args.tol else "FAIL"
        if v == "FAIL":
            bad.append(c)
        print(f"{c:34s} {d.max():10.6f} {d.mean():10.6f} {r:8.4f}  {v}")

    print()
    if bad:
        print(f"PARITY FAIL on {len(bad)}/{len(STUDST_COLS)} columns: {bad[:4]}")
        print("   Do not use the inference block until this passes. The most likely cause is "
              "the truncation chain: the effective student window is 220 chars (turn_text "
              "stores [:220], _text takes [:300]), and the tutor window is 200.")
        return 1
    print(f"PARITY PASS — all {len(STUDST_COLS)} columns within {args.tol} on "
          f"{len(m)} sessions. The inference code reproduces the training block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
