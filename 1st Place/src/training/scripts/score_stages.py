#!/usr/bin/env python
"""Scorecard: the log loss and AUROC of the shipped model and its two halves, from the shipped files.

Every number comes from files in `artifacts/` of the pipeline's working directory (run from
build/workspace/) and, for the test set, from the released test labels plus our per-row test
predictions, so each line of the README's scorecard can be recomputed. The LLM half and the
XGBoost half are scored on the same rows:

  * oof3     - the 16,693 rows held out by the three LO-disjoint LLM fold models (each row is
               predicted by a model that never saw its learning objective)
  * clean    - the subset of oof3 whose session also never appears in the XGBoost model's
               training fold (7,738 rows; the hidden test is new objectives and new sessions)
  * gbm-all  - all 35,072 training rows, XGBoost only (its own 5-fold LO-disjoint OOF)
  * test     - the 10,508 released test rows (needs --test-labels and --test-components)

The LLM half is calibrated with the shipped recipe (Platt slope clamped at 0.85, intercept solved
so the mean equals the training base rate), fit on the oof3 rows; that reproduces the shipped
calibrator exactly.

Not a reproduce.py step; run it after ``python src/make_workspace.py`` (README, Scorecard):

    cd build/workspace
    python scripts/score_stages.py
    python scripts/score_stages.py --test-labels /path/to/test_labels.csv \
        --test-components ../../models/predictions/test/shipped_components.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from magnificat import data  # noqa: E402
from magnificat.calibration import PlattScaler  # noqa: E402
from magnificat.folds import clean_mask, lo_folds  # noqa: E402

W = 0.43  # assets/blend_weight.txt
A_MAX = 0.85


def platt_to_base_rate(p, y, q_train):
    """Fit the shipped calibration recipe on out-of-fold predictions.

    Args:
        p: raw out-of-fold probabilities.
        y: 0/1 labels of the same rows.
        q_train: training base rate.

    Returns:
        ``PlattScaler`` with a = min(fitted Platt slope, 0.85) and b solved so that the mean
        calibrated prediction equals ``q_train``.
    """
    a = min(PlattScaler.fit(p, y).a, A_MAX)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    b = brentq(lambda bb: (1 / (1 + np.exp(-(a * z + bb)))).mean() - q_train, -5, 5)
    return PlattScaler(a=a, b=b)


def score(y, p):
    """Return ``(log loss, AUROC x 100)`` of probabilities ``p`` (clipped to [1e-6, 1 - 1e-6])."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return log_loss(y, p), 100 * roc_auc_score(y, p)


def main() -> None:
    """Compute the scorecard and write it as a Markdown table.

    Returns:
        None. Prints the scorecard and writes it to ``--out`` (default
        ``figures/data/scorecard.md``); the released-test table is added only when
        ``--test-labels`` and ``--test-components`` are given.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-labels", type=Path, default=None)
    ap.add_argument("--test-components", type=Path, default=None,
                    help="per-row test predictions: response_id,p_llm,p_gbm,p_blend")
    ap.add_argument("--out", type=Path, default=ROOT / "figures/data/scorecard.md")
    args = ap.parse_args()

    train = data.load_train()
    q_train = float(train.is_correct.mean())

    s2 = pd.read_csv(ROOT / "artifacts/mrft_oof_3fold_keyed.csv")            # response_id,fold,y,p
    g = pd.read_csv(ROOT / "artifacts/v2shrunkD_oof.csv")                     # response_id,y,p_raw,p_abl
    gp = PlattScaler.fit(np.concatenate([g.p_raw, g.p_abl]), np.concatenate([g.y, g.y]))
    g["p_gbm"] = gp.transform(g.p_raw.to_numpy())                              # the trainer's own frozen map

    d = (s2.rename(columns={"p": "p_s2"})
           .merge(g[["response_id", "p_gbm"]], on="response_id"))
    y = d.y.to_numpy(int)
    cal2 = platt_to_base_rate(d.p_s2.to_numpy(), y, q_train)
    d["s2_cal"] = cal2.transform(d.p_s2.to_numpy())
    d["ens_s2"] = (1 - W) * d.s2_cal + W * d.p_gbm
    d["const"] = q_train

    keep = set(train.response_id.to_numpy()[clean_mask(train.session_id.to_numpy(),
                                                       lo_folds(train.learning_objective_id.to_numpy()))])
    views = {"oof3": d, "clean": d[d.response_id.isin(keep)]}

    rows = [("constant (training base rate %.4f)" % q_train, "const"),
            ("XGBoost half (shipped)", "p_gbm"),
            ("LLM half, raw", "p_s2"),
            ("LLM half, calibrated (shipped)", "s2_cal"),
            ("ensemble 0.57 LLM + 0.43 XGBoost (shipped model)", "ens_s2")]
    lines = [f"# Scorecard (log loss / AUROC)\n",
             f"LLM calibrator fit with the shipped recipe on the oof3 rows: "
             f"a={cal2.a:.4f} b={cal2.b:+.4f} (shipped: a=0.6876 b=+0.4565).\n",
             f"| stage | oof3 (n={len(views['oof3']):,}) | clean (n={len(views['clean']):,}) |",
             "|---|---:|---:|"]
    for label, col in rows:
        cells = []
        for v in views.values():
            yy = v.y.to_numpy(int)
            ll, auc = score(yy, v[col].to_numpy()) if col != "const" else (score(yy, np.full(len(yy), q_train))[0], 50.0)
            cells.append(f"{ll:.4f} / {auc:.2f}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    gll, gauc = score(g.y.to_numpy(int), g.p_gbm.to_numpy())
    lines += ["", f"XGBoost half on all {len(g):,} training rows (its own LO-disjoint 5-fold OOF): {gll:.4f} / {gauc:.2f}"]

    if args.test_labels and args.test_components:
        t = pd.read_csv(args.test_components).merge(pd.read_csv(args.test_labels), on="response_id")
        yt = t.is_correct.to_numpy(int)
        lines += ["", f"| released test set (n={len(t):,}) | log loss / AUROC |", "|---|---:|",
                  "| constant (training base rate) | %.4f / 50.00 |" % score(yt, np.full(len(yt), q_train))[0]]
        show = (("XGBoost half (shipped)", "p_gbm"), ("LLM half, calibrated (shipped)", "p_llm"),
                ("ensemble (shipped model)", "p_blend"))
        for label, col in show:
            ll, auc = score(yt, t[col].to_numpy())
            lines.append(f"| {label} | {ll:.4f} / {auc:.2f} |")

    out = "\n".join(lines) + "\n"
    print(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
