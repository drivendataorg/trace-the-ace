#!/usr/bin/env python
"""Check that the inference code reproduces the training-side v2 family blocks.

The training-side families are built from `{prefix}_turn_probs_*.pkl.gz`, probabilities the
distillation already wrote. The inference code (`magnificat.v2_features.build_v2_family_matrix`)
re-derives them from the raw transcript through the classifier's vectoriser. The two paths
agree only if three things line up, and each can fail silently:

1. Text truncation. `turn_text.pkl.gz` truncates on write and `distil_v2_moves.build_text`
   truncates again; feeding the raw transcript with only the outer bound gives the vectoriser
   characters it never saw.
2. The turn axis. The training tables contain only turns with student text
   (`build_evidence_features.py` skips the rest), so inference code that iterates every turn
   `build_turns` returns shifts every position-dependent column (thirds, slopes, centroids, the
   LO window) while leaving the plain rates unchanged. A rate-only parity check would still
   pass; this script checks every family.
3. Duplicated constants. `_AXES_T`, `_AXES_S` and `LO_PAD_TURNS` are duplicated into
   `v2_features` because `scripts/` cannot be imported inside the runtime container.
   Duplicated constants drift, so they are compared here.

None of this is detectable in-domain: every local experiment reads the same already-truncated
table, so drifted inference code scores perfectly in cross-validation and is wrong only at
inference, i.e. on new data.

The exit status is the result: non-zero means do not score or pack.

In:  artifacts/v2t_clf_v2ext_122bC.joblib, artifacts/v2s_clf_v2stu_122bC.joblib,
     artifacts/v2fam_{family}_{tag}.csv, artifacts/v2t_features_v2ext_{tag}.meta.json,
     artifacts/v2_sessions_3072.txt, data/train_features.csv, data/train_transcripts/

Usage (reproduce.py step 04c, from build/workspace/):
  PYTHONPATH=src python scripts/check_v2_family_parity.py --families shrunk
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from magnificat import data, v2_features as v2f  # noqa: E402

#: float32 storage in the per-turn table puts the floor at ~1.2e-07; the tolerance lies well
#: above it.
TOL = 1e-4


def check_constants() -> list[str]:
    """Compare the axis and window constants duplicated into `magnificat.v2_features`.

    Returns:
        A list of mismatch descriptions; empty when the builders and the inference module
        agree.
    """
    import build_v2_families as B
    import build_v2_narrow as N  # noqa: F401  (imported so a syntax error here is caught too)

    bad = []
    if B.LO_PAD_TURNS != v2f.LO_PAD_TURNS:
        bad.append(f"LO_PAD_TURNS {B.LO_PAD_TURNS} != {v2f.LO_PAD_TURNS}")
    for name, a, b in (("AXES_T", B.AXES_T, v2f._AXES_T), ("AXES_S", B.AXES_S, v2f._AXES_S),
                       ("TUTOR_AXES", N.TUTOR_AXES, v2f._TUTOR_AXES),
                       ("STUDENT_AXES", N.STUDENT_AXES, v2f._STUDENT_AXES)):
        if {k: list(v) for k, v in a.items()} != {k: list(v) for k, v in b.items()}:
            bad.append(f"{name} differs between builder and inference module")
    return bad


def main() -> None:
    """Rebuild the selected families for sampled unlabelled sessions and compare them.

    Returns:
        None. Prints a per-family table of the maximum absolute difference; exits non-zero
        when constants differ, or when any difference exceeds TOL or NaN positions disagree.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutor-clf", default="artifacts/v2t_clf_v2ext_122bC.joblib")
    ap.add_argument("--student-clf", default="artifacts/v2s_clf_v2stu_122bC.joblib")
    ap.add_argument("--tag", default="122bC")
    ap.add_argument("--families", default="rate,shrunk,shrunkq,shrunkeb,shrunkk35,win12,win24,win48,instr,trim,bound,slope,disp,trans,ratio,qual,lowin,soft")
    ap.add_argument("--n", type=int, default=120, help="sessions to check")
    args = ap.parse_args()

    bad = check_constants()
    if bad:
        raise SystemExit("FAILED: the constants duplicated into magnificat.v2_features have "
                         "drifted:\n  " + "\n  ".join(bad))
    print("constants: builder and inference module agree", flush=True)

    tclf = joblib.load(ROOT / args.tutor_clf)
    sclf = joblib.load(ROOT / args.student_clf)
    fams = [f for f in args.families.split(",") if f]

    train = data.load_train()
    # Unlabelled sessions only. On a labelled session the training block carries the
    # classifier's out-of-fold value while inference necessarily uses the all-data fit, so
    # the two differ for a reason that is not drift. `distil_v2_moves`'s own self-check makes
    # the same choice.
    meta = pd.read_json(ROOT / f"artifacts/v2t_features_v2ext_{args.tag}.meta.json", typ="series")
    lab = set(pd.read_csv(ROOT / "artifacts/v2_sessions_3072.txt", header=None)[0].astype(str))
    print(f"excluding {len(lab):,} labelled sessions "
          f"(classifier trained on {meta['labelled_sessions']:,})", flush=True)

    pool = train[~train.session_id.isin(lab)].drop_duplicates("response_id")
    feats = pool.head(args.n * 3).drop_duplicates("session_id").head(args.n)
    feats = train[train.session_id.isin(set(feats.session_id))].copy()
    print(f"checking {feats.session_id.nunique()} sessions / {len(feats):,} responses",
          flush=True)

    got = v2f.build_v2_family_matrix(feats, tclf=tclf, sclf=sclf, families=fams)

    worst, rows = 0.0, []
    for fam in fams:
        # `soft` is emitted into the lowin file (same keyword hits, same response key), so it
        # is checked against that file rather than against one of its own.
        p = ROOT / f"artifacts/v2fam_{'lowin' if fam == 'soft' else fam}_{args.tag}.csv"
        if not p.exists():
            rows.append((fam, "-", "MISSING FILE"))
            continue
        ref = pd.read_csv(p)
        key = "response_id" if "response_id" in ref.columns else "session_id"
        cols_needed = ["response_id"] if key == "response_id" else ["response_id", key]
        want = (feats[cols_needed].merge(ref, on=key, how="left")
                .set_index("response_id"))
        cols = [c for c in want.columns if c in got.columns]
        if fam == "soft":
            cols = [c for c in cols if c.endswith("_soft_cov")]
        elif fam == "lowin":
            cols = [c for c in cols if not c.endswith("_soft_cov")]
        if not cols:
            rows.append((fam, "-", "NO SHARED COLUMNS"))
            continue
        a = got.loc[want.index, cols].to_numpy(float)
        b = want[cols].to_numpy(float)
        # NaN must agree as NaN: inference code that emits 0.0 where training emits NaN is a
        # real defect (the booster learned NaN as an in-distribution state), and abs() would
        # hide it.
        mism = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        d = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        worst = max(worst, d)
        rows.append((fam, f"{len(cols)} cols", f"max|Δ| {d:.3e}"
                     + (f"  WARNING: {mism} NaN mismatches" if mism else "")))
        if mism:
            worst = float("inf")

    print("\n| family | columns | parity |\n|---|---|---|")
    for f, c, m in rows:
        print(f"| `{f}` | {c} | {m} |")
    print(f"\nworst max|Δ| across all families = {worst:.3e} (tol {TOL:.0e})", flush=True)
    if worst > TOL:
        raise SystemExit(
            "PARITY FAILED: magnificat.v2_features does not reproduce what the training "
            "builder wrote, so a score computed with it on another corpus would measure "
            "train/serve skew rather than transfer, and the shipped block would not be the "
            "block that was validated. Fix the inference code before scoring or packing.")
    print("parity OK", flush=True)


if __name__ == "__main__":
    main()
