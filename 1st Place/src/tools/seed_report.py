#!/usr/bin/env python
"""Score the seed replicates of the LLM half against the shipped adapter (reports/seeds/report.md).

Reads the ``score_llm_half.py`` outputs in ``models/predictions/test/seeds/`` (``<run>.csv`` with
``response_id, p_raw, z_raw, p_llm``), adds the shipped XGBoost half from
``models/predictions/test/shipped_components.csv`` and the released test labels, and writes a
markdown report with four tables:

1. each run through the shipped, frozen Platt map (what the shipped pipeline would output);
2. each run uncalibrated (raw probability) against the shipped adapter's raw probability;
3. equal-weight averages of the runs' predictions (nothing chosen on the test set);
4. a diagnostic only: the same intercept shift, fitted on the test labels, given to every run and to
   the shipped adapter. It bounds how much of each gap is calibration offset; it is not a score.

The shipped adapter's LLM half is ``shipped_components.csv`` ``p_llm`` (the reference run of
reports/gpu_noise/report.md); its raw probability is that column with the frozen Platt map
inverted, as in ``gpu_noise_report.py``. Every comparison is the paired difference run - shipped in log loss, with a
95 % interval from 2,000 bootstrap resamples of test sessions (all rows of a session together).

    python src/tools/seed_report.py --labels /path/to/test_labels.csv --features /path/to/test_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "models" / "submission" / "assets"
W = float((ASSETS / "blend_weight.txt").read_text().strip())   # XGBoost weight, 0.43
EPS = 1e-6
GROUPS = {   # equal-weight prediction averages; "shipped" is the shipped adapter
    "shipped + full seeds 222-225": ["shipped", "full_s222_l40s", "full_s223_b200", "full_s224_b200", "full_s225_b200"],
    "shipped + full seeds 223-225": ["shipped", "full_s223_b200", "full_s224_b200", "full_s225_b200"],
    "shipped + stage-2-only seeds 222-225": ["shipped", "stage2_only_s222_b200", "stage2_only_s223_b200",
                                             "stage2_only_s224_b200", "stage2_only_s225_b200"],
    "all nine": ["shipped", "full_s222_l40s", "full_s223_b200", "full_s224_b200", "full_s225_b200",
                 "stage2_only_s222_b200", "stage2_only_s223_b200", "stage2_only_s224_b200", "stage2_only_s225_b200"],
}


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def main() -> None:
    """Build the seed-replicate report from the scored runs, the shipped XGBoost half and labels.

    Returns:
        Nothing; writes the Markdown report to ``--out`` (default ``reports/seeds/report.md``)
        and prints it. Exits with an error if a run lacks predictions for some test rows.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=REPO / "models" / "predictions" / "test" / "seeds")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True, help="test features csv (for session_id)")
    ap.add_argument("--components", type=Path,
                    default=REPO / "models" / "predictions" / "test" / "shipped_components.csv")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "seeds" / "report.md")
    args = ap.parse_args()

    sys.dont_write_bytecode = True               # importing magnificat must not leave __pycache__ in assets/
    sys.path.insert(0, str(ASSETS))              # the calibrator pickle needs magnificat
    cal = joblib.load(ASSETS / "llmkt_calibrator.joblib")
    base = (pd.read_csv(args.components).merge(pd.read_csv(args.labels), on="response_id")
            .merge(pd.read_csv(args.features)[["response_id", "session_id"]], on="response_id"))
    y = base["is_correct"].to_numpy().astype(int)
    gbm = base["p_gbm"].to_numpy()
    raw = {"shipped": 1 / (1 + np.exp(-(logit(base["p_llm"].to_numpy()) - cal.b) / cal.a))}
    cal_p = {"shipped": base["p_llm"].to_numpy()}
    for f in sorted(args.runs.glob("*.csv")):
        r = base[["response_id"]].merge(pd.read_csv(f), on="response_id", how="left")
        if r["p_llm"].isna().any():
            raise SystemExit(f"{f.name}: missing rows")
        raw[f.stem], cal_p[f.stem] = r["p_raw"].to_numpy(), r["p_llm"].to_numpy()

    def ens(p):
        return clip((1 - W) * p + W * gbm)

    def rowll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    sess = base["session_id"].to_numpy()
    codes, inv = np.unique(sess, return_inverse=True)
    n_s = np.bincount(inv)
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(codes), len(codes)) for _ in range(2000)]

    def delta(p, q):   # paired ensemble-LL difference p - q with a session bootstrap
        d = np.bincount(inv, weights=rowll(p) - rowll(q))
        bs = [d[i].sum() / n_s[i].sum() for i in draws]
        return f"{d.sum() / n_s.sum():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]", d.sum() / n_s.sum()

    def sc(p):
        return f"{log_loss(y, clip(p), labels=[0, 1]):.5f} / {100 * roc_auc_score(y, p):.2f}"

    names = ["shipped"] + [k for k in cal_p if k != "shipped"]
    out = ["# Seed replicates of the LLM half: scores on the 10,508 released test rows", "",
           f"Generated by `src/tools/seed_report.py`. Ensemble = {1 - W:.2f} * LLM + {W:.2f} * shipped XGBoost half. "
           "Δ = run - shipped in ensemble log loss, 95 % session-bootstrap interval.", "",
           "## 1. Through the shipped, frozen Platt map", "",
           "| run | LLM LL / AUROC | ensemble LL / AUROC | Δ ensemble LL vs shipped |", "|---|---|---|---|"]
    dz = {}
    for k in names:
        txt, dz[k] = delta(ens(cal_p[k]), ens(cal_p["shipped"])) if k != "shipped" else ("–", 0.0)
        out.append(f"| {k} | {sc(cal_p[k])} | {sc(ens(cal_p[k]))} | {txt} |")
    out += ["", "## 2. Uncalibrated: raw probability against the shipped adapter's raw probability", "",
            f"Base rate of the test rows: {y.mean():.4f}.", "",
            "| run | raw LLM LL / AUROC | mean raw p | ensemble (raw LLM) LL / AUROC | Δ ensemble LL vs shipped raw |",
            "|---|---|---|---|---|"]
    dr = {}
    for k in names:
        txt, dr[k] = delta(ens(raw[k]), ens(raw["shipped"])) if k != "shipped" else ("–", 0.0)
        out.append(f"| {k} | {sc(raw[k])} | {raw[k].mean():.3f} | {sc(ens(raw[k]))} | {txt} |")
    out += ["", "Spread of Δ across seeds (mean, sd):", ""]
    for g, pre in (("full retrain, seeds 222-225", "full_"), ("full retrain, seeds 223-225", ("full_s223", "full_s224", "full_s225")),
                   ("stage-2 only, seeds 222-225", "stage2_only_")):
        ks = [k for k in names if k.startswith(pre)]
        a, b = np.array([dz[k] for k in ks]), np.array([dr[k] for k in ks])
        out.append(f"- {g}: frozen map {a.mean():+.4f} (sd {a.std(ddof=1):.4f}); raw {b.mean():+.4f} (sd {b.std(ddof=1):.4f})")
    out += ["", "## 3. Equal-weight prediction averages", "",
            "| average of | through frozen map: LLM / ensemble LL, AUROC | Δ vs shipped | raw: LLM / ensemble LL, AUROC | Δ vs shipped raw |",
            "|---|---|---|---|---|"]
    for g, ks in GROUPS.items():
        pc = np.mean([cal_p[k] for k in ks], 0)
        pr = np.mean([raw[k] for k in ks], 0)
        out.append(f"| {g} | {sc(pc)} ; {sc(ens(pc))} | {delta(ens(pc), ens(cal_p['shipped']))[0]} "
                   f"| {sc(pr)} ; {sc(ens(pr))} | {delta(ens(pr), ens(raw['shipped']))[0]} |")
    out += ["", "## 4. Diagnostic only (fitted on the test labels; not a score)", "",
            "Each run, and the shipped adapter, gets the one intercept shift c of logit(p_llm) that minimises the LLM half's "
            "test log loss; the shifted LLM half is then blended as usual.", "",
            "| run | shift c | ensemble LL after shift | Δ vs shipped (also shifted) |", "|---|---|---|---|"]
    shifted = {}
    for k in names:
        z = logit(cal_p[k])
        c = minimize_scalar(lambda v: log_loss(y, clip(1 / (1 + np.exp(-(z + v)))), labels=[0, 1]),
                            bounds=(-3, 3), method="bounded").x
        shifted[k] = (c, ens(1 / (1 + np.exp(-(z + c)))))
    for k in names:
        c, e = shifted[k]
        txt = delta(e, shifted["shipped"][1])[0] if k != "shipped" else "–"
        out.append(f"| {k} | {c:+.3f} | {log_loss(y, e, labels=[0, 1]):.5f} | {txt} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
