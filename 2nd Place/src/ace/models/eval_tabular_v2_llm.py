#!/usr/bin/env python
r"""Evaluate tabular_v2 alongside the Qwen3-4B and Qwen3-8B OOF readers (evaluation only).

Every stack prediction is cross-fitted on the same skill-cluster folds, so the metrics are honest
second-level OOF estimates. Compares Q4 + Q8, base blend + Q4 + Q8 (with --base-oof),
tabular_v2 + Q4 + Q8, and base blend + tabular_v2 + Q4 + Q8.

    python -m ace.models.eval_tabular_v2_llm --tabular-oof outputs/models/tabular_v2_bge_no_telo/oof.csv \
        --q4-dir outputs/models/e13_qwen4 --q8-dir outputs/models/e13_qwen8b \
        --base-oof outputs/models/sub5_fit/oof.csv --out outputs/models/tabular_v2_llm_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ace.utils import logloss, md_table

CLIP = (0.01, 0.99)


def logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def read_llm_oof(directory: str | Path, output_name: str) -> pd.DataFrame:
    files = sorted(Path(directory).expanduser().glob("p_fold*.csv"))
    if not files:
        raise FileNotFoundError(f"no p_fold*.csv files found under {directory}")
    df = pd.concat([pd.read_csv(f, usecols=["response_id", "p_llm"]) for f in files],
                   ignore_index=True)
    if df["response_id"].duplicated().any():
        dup = df.loc[df["response_id"].duplicated(), "response_id"].iloc[0]
        raise ValueError(f"duplicate response_id in {directory}: {dup}")
    return df.rename(columns={"p_llm": output_name})


def cross_fit_stack(
    df: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    features: list[str],
    c_value: float,
) -> tuple[np.ndarray, list[dict]]:
    p = np.full(len(df), np.nan)
    coefs: list[dict] = []
    for fold in sorted(np.unique(folds)):
        tr, va = folds != fold, folds == fold
        Xtr = np.column_stack([logit(df.loc[tr, f]) for f in features])
        Xva = np.column_stack([logit(df.loc[va, f]) for f in features])
        lr = LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000)
        lr.fit(Xtr, y[tr])
        p[va] = lr.predict_proba(Xva)[:, 1]
        coefs.append({
            "fold": int(fold),
            "intercept": float(lr.intercept_[0]),
            "coef": {f: float(v) for f, v in zip(features, lr.coef_[0])},
        })
    return np.clip(p, *CLIP), coefs


def fit_full_stack(df: pd.DataFrame, y: np.ndarray, features: list[str], c_value: float) -> dict:
    X = np.column_stack([logit(df[f]) for f in features])
    lr = LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000)
    lr.fit(X, y)
    return {
        "features": features,
        "intercept": float(lr.intercept_[0]),
        "coef": [float(v) for v in lr.coef_[0]],
        "note": (
            "Fit on pooled OOF logits for analysis. Do not deploy without checking the "
            "full-fit-vs-fold logit scale for every reader, including tabular_v2."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabular-oof", required=True)
    ap.add_argument("--q4-dir", required=True)
    ap.add_argument("--q8-dir", required=True)
    ap.add_argument("--base-oof", default=None,
                    help="optional base-blend OOF, usually sub5_fit/oof.csv")
    ap.add_argument("--base-column", default="oof_blend_cal")
    ap.add_argument("--out", default="outputs/models/tabular_v2_llm_eval")
    ap.add_argument("--C", type=float, default=10.0,
                    help="L2 inverse regularization for the small correlated stack")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    tab = pd.read_csv(args.tabular_oof)
    required = {"response_id", "fold", "is_correct", "oof_tabular_v2"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"tabular OOF missing columns: {sorted(missing)}")
    df = tab[list(required)].rename(columns={"oof_tabular_v2": "p_tabv2"})
    df = df.merge(read_llm_oof(args.q4_dir, "p_q4"), on="response_id", validate="1:1")
    df = df.merge(read_llm_oof(args.q8_dir, "p_q8"), on="response_id", validate="1:1")

    have_base = args.base_oof is not None
    if have_base:
        base = pd.read_csv(args.base_oof, usecols=["response_id", args.base_column])
        base = base.rename(columns={args.base_column: "p_base"})
        df = df.merge(base, on="response_id", validate="1:1")

    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"missing values after OOF merge: {bad}")

    y = df["is_correct"].to_numpy(dtype=float)
    folds = df["fold"].to_numpy(dtype=int)
    variants: dict[str, list[str]] = {
        "q4_q8": ["p_q4", "p_q8"],
        "tabv2_q4_q8": ["p_tabv2", "p_q4", "p_q8"],
    }
    if have_base:
        variants.update({
            "base": ["p_base"],
            "base_tabv2": ["p_base", "p_tabv2"],
            "base_q4_q8": ["p_base", "p_q4", "p_q8"],
            "base_tabv2_q4_q8": ["p_base", "p_tabv2", "p_q4", "p_q8"],
        })

    predictions: dict[str, np.ndarray] = {}
    fold_coefs: dict[str, list[dict]] = {}
    pooled_rows = []
    per_fold_rows = []
    full_fits = {}
    for name, features in variants.items():
        p, coefs = cross_fit_stack(df, y, folds, features, args.C)
        predictions[name] = p
        fold_coefs[name] = coefs
        full_fits[name] = fit_full_stack(df, y, features, args.C)
        pooled_rows.append((name, f"{logloss(y, p):.4f}", f"{roc_auc_score(y, p):.4f}"))
        for fold in sorted(np.unique(folds)):
            m = folds == fold
            per_fold_rows.append((
                name, int(fold), int(m.sum()),
                f"{logloss(y[m], p[m]):.4f}",
                f"{roc_auc_score(y[m], p[m]):.4f}",
            ))

    # Reader-level scores and logit correlations make it easier to judge whether tabular_v2
    # adds a genuinely different signal even if the stack metric is unstable.
    readers = ["p_tabv2", "p_q4", "p_q8"] + (["p_base"] if have_base else [])
    reader_rows = [(r, f"{logloss(y, df[r]):.4f}", f"{roc_auc_score(y, df[r]):.4f}")
                   for r in readers]
    corr = pd.DataFrame({r: logit(df[r]) for r in readers}).corr()
    corr.to_csv(out / "logit_correlations.csv")

    pred_out = df[["response_id", "fold", "is_correct"]].copy()
    for name, p in predictions.items():
        pred_out[f"p_{name}"] = p
    pred_out.to_csv(out / "oof_stacks.csv", index=False)
    (out / "stack_fits.json").write_text(json.dumps({
        "C": args.C,
        "cross_fit_fold_coefficients": fold_coefs,
        "pooled_oof_full_fits": full_fits,
    }, indent=2), encoding="utf-8")

    # Sort pooled table by log loss for convenience.
    pooled_rows = sorted(pooled_rows, key=lambda x: float(x[1]))
    report = [
        "# tabular_v2 + Q4/Q8 stack evaluation\n",
        "Every stack row below is cross-fitted by the fixed v3 fold. No row is scored by a "
        "second-level stacker that was fitted on that row.\n",
        "## Individual OOF readers\n",
        md_table(["reader", "log loss", "AUC"], reader_rows) + "\n",
        "## Cross-fit stacks\n",
        md_table(["stack", "log loss", "AUC"], pooled_rows) + "\n",
        "## Per-fold stack metrics\n",
        md_table(["stack", "fold", "n", "log loss", "AUC"], per_fold_rows) + "\n",
        "## Logit correlations\n",
        md_table(["reader"] + readers, [[r] + [f"{corr.loc[r, c]:.4f}" for c in readers] for r in readers]) + "\n",
        "The coefficients in `stack_fits.json` are analytical OOF fits only. Before a code "
        "submission, measure and correct deployment scale differences between fold OOF and "
        "full-fit/ensemble predictions.\n",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\nCross-fit pooled stacks (LL / AUC):")
    for row in pooled_rows:
        print(f"  {row[0]:24s} {row[1]} / {row[2]}")
    print(f"\nDone -> {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
