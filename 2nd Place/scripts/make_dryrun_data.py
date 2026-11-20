#!/usr/bin/env python
"""Build a test-format data folder from a sample of training sessions, for a dry run.

The folder (test_features.csv, submission_format.csv, test_transcripts/, plus labels.csv) runs
through run_inference.sh, so the pipeline can be exercised and timed without the test set.
The score is optimistic (the full-fit models saw these sessions): a pipeline check only.

    python scripts/make_dryrun_data.py --data-dir data --out data_dryrun --sessions 300
    DATA_DIR=data_dryrun bash scripts/run_inference.sh
    python scripts/make_dryrun_data.py --score data_dryrun submission/submission.csv
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def build(a) -> None:
    data, out = Path(a.data_dir), Path(a.out)
    feats = pd.read_csv(data / "train_features.csv")
    labels = pd.read_csv(data / "train_labels.csv")
    rng = np.random.default_rng(a.seed)
    sids = rng.choice(np.sort(feats["session_id"].unique()), size=a.sessions, replace=False)
    sub = feats[feats["session_id"].isin(sids)].reset_index(drop=True)
    (out / "test_transcripts").mkdir(parents=True, exist_ok=True)
    for s in sids:
        shutil.copyfile(data / "train_transcripts" / f"{s}.csv", out / "test_transcripts" / f"{s}.csv")
    sub.to_csv(out / "test_features.csv", index=False)
    pd.DataFrame({"response_id": sub["response_id"], "probability": 0.5}).to_csv(
        out / "submission_format.csv", index=False)
    labels[labels["response_id"].isin(sub["response_id"])].to_csv(out / "labels.csv", index=False)
    print(f"{len(sids)} sessions, {len(sub)} rows -> {out}")


def score(folder: str, submission: str) -> None:
    from sklearn.metrics import log_loss, roc_auc_score
    y = pd.read_csv(Path(folder) / "labels.csv")
    p = pd.read_csv(submission).merge(y, on="response_id", validate="1:1")
    label = [c for c in y.columns if c != "response_id"][0]
    print(f"rows {len(p)}  log loss {log_loss(p[label], p['probability']):.4f}  "
          f"AUC {roc_auc_score(p[label], p['probability']):.4f}  (train sessions: optimistic)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data_dryrun")
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", nargs=2, metavar=("DRYRUN_DIR", "SUBMISSION_CSV"))
    args = ap.parse_args()
    score(*args.score) if args.score else build(args)
