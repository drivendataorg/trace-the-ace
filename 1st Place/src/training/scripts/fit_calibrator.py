"""Fit and freeze a Platt calibrator for an LLMKT model from training data only.

Both parameters are derived from training data:

  a: the slope, ``a = min(out-of-fold fit of a, A_MAX)``. The out-of-fold fit uses training
     data only; A_MAX is a fixed over-confidence prior (a modelling assumption, not fitted), so
     the shipped map never sharpens.
  b: solved numerically so that the mean calibrated prediction over the pooled
     objective-disjoint out-of-fold predictions equals the training base rate
     (``data/train_labels*.csv``, 0.702469).

No test quantity enters the computation: re-run with the same training data against a
different test set, the script returns the same (a, b).

Usage (from build/workspace/). reproduce.py step 07a fits the shipped map, with single-threaded
BLAS in the KT_PYTHON environment:
    $KT_PYTHON scripts/fit_calibrator.py --model mrft \
        --oof-csv artifacts/mrft_oof_3fold.csv --out artifacts/llmkt_calibrator_mrft_sub0808.joblib
Without ``--oof-csv`` the script reads the per-fold result files listed in ``MODELS``
(``external/dialogue-kt/results/qual_lmkt_*.csv``, written by step 06c), which are not shipped:
    python scripts/fit_calibrator.py --model mrft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from magnificat.calibration import PlattScaler  # noqa: E402

A_MAX = 0.85  # fixed over-confidence prior: the shipped map never sharpens
RESULTS = ROOT / "external/dialogue-kt/results"

#: Per-fold result files of the shipped LLM half's three fold models (step 06c).
MODELS = {
    "mrft": [RESULTS / f"qual_lmkt_mrft_sa_8b_a1f{i}_incfirst_{i}.csv" for i in (1, 2, 3)],
}
OUT = {
    "mrft": ROOT / "artifacts/llmkt_calibrator_mrft.joblib",
}


def train_base_rate() -> float:
    """Read the training base rate, the only anchor of ``b``, from the delivered labels.

    Returns:
        The mean of ``is_correct`` in the first ``data/train_labels*.csv`` (sorted by name).
    """
    labels = next(iter(sorted((ROOT / "data").glob("train_labels*.csv"))))
    q = float(pd.read_csv(labels)["is_correct"].mean())
    print(f"train anchor: {labels.name} -> base rate {q:.6f}")
    return q


def load_oof(files) -> tuple[np.ndarray, np.ndarray]:
    """Pool the scored rows of dialogue-kt per-fold result files.

    Args:
        files: paths of ``qual_lmkt_*.csv`` files with ``Correct`` (True/False) and ``Prob``
            columns; rows with another label or a non-numeric probability are skipped.

    Returns:
        ``(y, p)``: 0/1 labels and predicted probabilities, concatenated over the files.
    """
    ys, ps = [], []
    for f in files:
        df = pd.read_csv(f)
        c = df["Correct"].astype(str).str.strip().str.lower()
        m = c.isin(["true", "false"]) & pd.to_numeric(df["Prob"], errors="coerce").notna()
        ys.append((c[m] == "true").astype(int).values)
        ps.append(pd.to_numeric(df.loc[m, "Prob"]).astype(float).values)
        print(f"  {f.name}: n={int(m.sum())}")
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    """Fit the clamped, base-rate-anchored Platt map and save it.

    Returns:
        None. Prints the out-of-fold scores before and after calibration and writes the
        ``PlattScaler`` with joblib to ``--out`` (default ``OUT[--model]``, under artifacts/).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--a-max", type=float, default=A_MAX)
    # --oof-csv reads the pooled OOF from a frozen `fold,y,p` CSV instead of the per-fold qual
    # files. The shipped map is fit on artifacts/mrft_oof_3fold.csv, the snapshot that
    # reproduces a=0.6876 / b=0.4565.
    ap.add_argument("--oof-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    q_train = train_base_rate()
    if args.oof_csv is not None:
        _oof = pd.read_csv(args.oof_csv)
        y, p = _oof["y"].astype(int).to_numpy(), _oof["p"].astype(float).to_numpy()
        print(f"  {args.oof_csv.name}: n={len(y)}")
    else:
        y, p = load_oof(MODELS[args.model])
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    print(f"\npooled OOF: n={len(y)}  base={y.mean():.4f}  mean_pred={p.mean():.4f}  "
          f"AUROC={roc_auc_score(y, p):.4f}  raw LL={log_loss(y, p):.4f}")

    fit = PlattScaler.fit(p, y)
    print(f"unconstrained OOF Platt: a={fit.a:.4f} b={fit.b:+.4f}  "
          f"(OOF LL {log_loss(y, fit.transform(p)):.4f})")

    a = min(fit.a, args.a_max)
    b = brentq(lambda bb: (1 / (1 + np.exp(-(a * z + bb)))).mean() - q_train, -5, 5)
    cal = PlattScaler(a=a, b=b)
    out_p = cal.transform(p)

    print(f"\nfrozen map: PlattScaler(a={a:.4f}, b={b:+.4f})   "
          f"[clamp {'active' if a < fit.a else 'inactive'}]")
    print(f"  OOF mean {out_p.mean():.6f} (= train base rate by construction)")
    print(f"  OOF LL   {log_loss(y, out_p):.4f}  (raw {log_loss(y, p):.4f})")
    print(f"  OOF AUROC {roc_auc_score(y, out_p):.4f} (unchanged: monotone map)")
    print(f"  range [{out_p.min():.3f}, {out_p.max():.3f}]")

    out = args.out or OUT[args.model]
    joblib.dump(cal, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
