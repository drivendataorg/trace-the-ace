#!/usr/bin/env python
"""Check a predictions file from src/run_inference.py against our reference run, and score it.

    python src/tools/score_submission.py data/processed/submission.csv
    python src/tools/score_submission.py data/processed/submission.csv --labels /path/to/test_labels.csv
    python src/tools/score_submission.py data/processed/example_predictions.csv --max-diff 0.02 \
        --reference data/examples/new_data/expected_predictions_cpu.csv --reference-column probability

Without labels it compares your probabilities, row by row, with ``p_blend`` in
``models/predictions/test/shipped_components.csv``: the final probabilities of our H100 run of the
shipped submission on the 10,508 released test rows (that run scored log loss 0.5934 / AUROC 64.7).
It reports the response_id coverage, the largest absolute difference and the share of rows within
0.02. On another GPU, or on CPU, expect most rows to agree to about +-0.01 and about 0.1 % to move
by up to ~0.025: the LLM half can land one bf16 step away (README, Run inference;
reports/gpu_noise/report.md). The log loss should then come out at 0.5934 +- 0.0002.

With ``--max-diff D`` it also fails (exit 1) when any shared row differs from the reference by
more than D. ``make example`` uses it with D = 0.02 against our CPU run of the one-row example.

With ``--labels`` (the released ``test_labels.csv``, columns response_id, is_correct; not shipped
with this repository) it also prints log loss and AUROC (x 100) on the labelled rows. Probabilities are
clipped to [1e-6, 1 - 1e-6] first, as main.py does.

Exit status 1 when a probability is missing or out of range, when every probability is 0.5, when
the rows differ from the reference's, or when ``--max-diff`` is exceeded; otherwise 0.

For new data, pass ``--reference none`` (or a reference of your own with a ``p_blend`` column, or
another column via ``--reference-column``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REF = REPO / "models" / "predictions" / "test" / "shipped_components.csv"
EPS = 1e-6
TOL = 0.02


def _read(path: Path, cols: list[str], what: str) -> pd.DataFrame:
    """Read a CSV keyed by response_id and check its columns.

    Args:
        path: CSV file to read.
        cols: required columns; the first is ``response_id`` (kept as text), the rest are
            converted to numbers (unparseable values become NaN).
        what: name of the file used in error messages ("predictions", "reference", "labels").

    Returns:
        The table. Exits with an error message if a required column is missing or a
        response_id is repeated.
    """
    df = pd.read_csv(path, dtype={"response_id": str}, keep_default_na=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {what} {path} has no column(s) {missing} (columns: {list(df.columns)})")
    dup = df["response_id"].duplicated()
    if dup.any():
        sys.exit(f"ERROR: {what} {path} repeats {int(dup.sum())} response_id(s)")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    """Check a predictions file, compare it with a reference run and optionally score it.

    Returns:
        Nothing; prints the checks and scores and exits with status 1 on a missing or
        out-of-range probability, all probabilities equal to 0.5, rows that differ from the
        reference's, or a ``--max-diff`` violation, and 0 otherwise. Writes no files.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, help="CSV with response_id, probability")
    ap.add_argument("--labels", type=Path, default=None,
                    help="CSV with response_id, is_correct (the released test_labels.csv)")
    ap.add_argument("--reference", default=str(DEFAULT_REF),
                    help=f"reference predictions (default: {DEFAULT_REF.relative_to(REPO)}); 'none' to skip")
    ap.add_argument("--reference-column", default="p_blend")
    ap.add_argument("--max-diff", type=float, default=None,
                    help="exit 1 if any shared row differs from the reference by more than this")
    args = ap.parse_args()
    if args.max_diff is not None and args.reference.lower() == "none":
        ap.error("--max-diff needs a reference")
    if args.max_diff is not None and not (0 <= args.max_diff < float("inf")):
        ap.error("--max-diff must be a finite number >= 0")

    pred = _read(args.predictions, ["response_id", "probability"], "predictions")
    p = pred["probability"]
    n_bad = int(p.isna().sum() + ((p < 0) | (p > 1)).sum())
    print(f"predictions: {len(pred)} rows in {args.predictions}")
    if n_bad:
        print(f"  WARNING: {n_bad} probability value(s) are missing or outside [0, 1]")
    if len(pred) and (p == 0.5).all():
        print("  WARNING: every probability is exactly 0.5 (no row was matched to a prediction)")
        n_bad += 1
    status = 1 if n_bad else 0

    if args.reference.lower() != "none":
        ref_path = Path(args.reference)
        if not ref_path.is_file():
            sys.exit(f"ERROR: reference {ref_path} not found (pass --reference none to skip it)")
        ref = _read(ref_path, ["response_id", args.reference_column], "reference")
        ref = ref[["response_id", args.reference_column]].rename(columns={args.reference_column: "_ref"})
        both = pred.merge(ref, on="response_id", how="inner")
        only_pred = len(pred) - len(both)
        only_ref = len(ref) - len(both)
        print(f"coverage vs {ref_path.name}: {len(both)} shared response_ids; "
              f"{only_pred} only in the predictions, {only_ref} only in the reference")
        if only_pred or only_ref:
            print("  WARNING: the predictions do not cover the same rows as the reference "
                  "(for new data, pass --reference none)")
            status = 1
        if len(both):
            d = (both["probability"] - both["_ref"]).abs()
            print(f"vs {args.reference_column}: max |dp| = {d.max():.6f}, mean |dp| = {d.mean():.6f}, "
                  f"within {TOL}: {(d <= TOL).mean():.2%} of rows ({int((d <= TOL).sum())}/{len(d)}), "
                  f"identical: {int((d == 0).sum())}")
            if args.max_diff is not None:
                if d.isna().any() or d.max() > args.max_diff:
                    print(f"  FAIL: a row differs from the reference by more than {args.max_diff} "
                          "(or has no probability)")
                    status = 1
                else:
                    print(f"  OK: every row is within {args.max_diff} of the reference")
        elif args.max_diff is not None:
            print("  FAIL: no shared rows to compare (--max-diff)")
            status = 1

    if args.labels is not None:
        from sklearn.metrics import log_loss, roc_auc_score  # noqa: PLC0415

        lab = _read(args.labels, ["response_id", "is_correct"], "labels")
        j = pred.merge(lab[["response_id", "is_correct"]], on="response_id", how="left")
        has_label = j["is_correct"].notna()
        n_nolabel = int((~has_label).sum())              # id not in the labels, or a blank/non-numeric label
        n_noprob = int((has_label & j["probability"].isna()).sum())  # labelled, probability missing
        m = j[has_label & j["probability"].notna()]
        labelled_ids = set(lab.loc[lab["is_correct"].notna(), "response_id"])
        n_unpred = len(labelled_ids - set(pred["response_id"]))  # labelled ids absent from the predictions
        print(f"labels: {len(m)} of {len(pred)} predicted rows scored"
              + (f"; {n_nolabel} have no label" if n_nolabel else "")
              + (f"; {n_noprob} have a label but a missing probability" if n_noprob else ""))
        if n_unpred:
            print(f"  WARNING: {n_unpred} labelled response_id(s) are not in the predictions")
        if len(m):
            y = m["is_correct"].astype(float).astype(int).to_numpy()
            q = np.clip(m["probability"].to_numpy(dtype=float), EPS, 1 - EPS)
            ll = log_loss(y, q, labels=[0, 1])
            auc = roc_auc_score(y, q) if len(set(y)) == 2 else float("nan")
            print(f"log loss = {ll:.5f}   AUROC x 100 = {100 * auc:.2f}   (n = {len(m)}; our run: 0.5934 / 64.73 "
                  f"on the 10,508 released test rows)")
    sys.exit(status)


if __name__ == "__main__":
    main()
