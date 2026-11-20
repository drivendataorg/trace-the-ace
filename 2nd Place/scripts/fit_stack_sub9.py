#!/usr/bin/env python
"""Re-derive the shipped level-2 stack (assets/llm_stack.json) from out-of-fold predictions.

Fit: unregularised logistic regression (C=1e6, lbfgs, max_iter=1000) on the pooled 5-fold OOF
logits of z_blend (sub5_fit/oof.csv), z_tabv2 (tabular_v2_bge_no_telo/oof.csv, which also holds
the labels), z_ft_4b and z_ft_8b (e13_qwen{4,8b}/p_fold*.csv); 35,072 rows. The shipped numbers
are the solution at scikit-learn's default tol=1e-4: a tighter tol moves them by ~2e-3, so the
tolerance is part of the recipe. The 4B/8B coefficients are then scaled by the fold/full-fit
logit-scale ratios (0.8567, 0.9821; --synth recomputes them from e13_qwen{4,8b}/synth/), and
Qwen3.5 enters only through the fixed blend z_ft_8b := 0.5*z_ft_8b + 0.5*z_ft_q35.

Input or write errors exit non-zero. --check REF: exit 1 unless the structure and the
coefficients (within --atol) match. --compare NEW REF: no refit; exit 1 if the structure
differs, coefficient differences are only reported.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

RECORDED_RATIOS = {"z_ft_4b": 0.8567, "z_ft_8b": 0.9821}
FEATURES = ["z_blend", "z_tabv2", "z_ft_4b", "z_ft_8b"]
STRUCTURE = ("features", "bundles", "clip", "manual_blend", "serialize", "max_input_tokens",
             "rowpass_budget")


def logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def fail(msg: str):
    raise SystemExit(f"!! fit_stack_sub9: {msg}")


def read_oof(path: Path, cols: list[str]) -> pd.DataFrame:
    """One OOF csv with the given columns; a clear error instead of a traceback."""
    if not path.is_file():
        fail(f"missing OOF file: {path}")
    try:
        df = pd.read_csv(path, usecols=cols)
    except ValueError as e:                        # missing columns
        fail(f"{path}: {e}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        fail(f"{path}: unreadable ({e})")
    if df.empty:
        fail(f"{path}: no rows")
    if df["response_id"].isna().any() or df["response_id"].duplicated().any():
        fail(f"{path}: missing or duplicate response_id")
    return df


def check_probs(df: pd.DataFrame, col: str, source) -> None:
    p = pd.to_numeric(df[col], errors="coerce")
    bad = p.isna() | ~np.isfinite(p) | (p < 0) | (p > 1)
    if bad.any():
        fail(f"{source}: {int(bad.sum())} NaN / non-finite / out-of-range values in "
             f"{col} (first response_id {df.loc[bad, 'response_id'].iloc[0]})")


def read_llm_oof(directory: Path, name: str) -> pd.DataFrame:
    parts = []
    for k in range(5):
        f = directory / f"p_fold{k}.csv"
        df = read_oof(f, ["response_id", "p_llm"])   # an empty fold fails here
        check_probs(df, "p_llm", f)
        parts.append(df)
    df = pd.concat(parts)
    if df["response_id"].duplicated().any():
        fail(f"{directory}: duplicate response_id across folds")
    return df.rename(columns={"p_llm": name})


def same_ids(ref: pd.DataFrame, ref_name, other: pd.DataFrame, other_name) -> None:
    a, b = set(ref["response_id"]), set(other["response_id"])
    if a != b:
        fail(f"response_ids of {other_name} do not match {ref_name}: "
             f"{len(a - b)} missing, {len(b - a)} extra")


def ratios_applied(stack: dict, src) -> dict:
    """The 4B/8B logit-scale ratios a stack json was built with (4 decimals)."""
    prov = (stack.get("provenance") or {}).get("ratios")
    if isinstance(prov, dict) and prov:
        return {k: round(float(v), 4) for k, v in sorted(prov.items())}
    note = str(stack.get("note", ""))                # the shipped json records them in its note
    m4 = re.search(r"c_4b = raw x ([0-9]+\.[0-9]+)", note)
    m8 = re.search(r"c_8b = raw x ([0-9]+\.[0-9]+)", note)
    if m4 and m8:
        return {"z_ft_4b": round(float(m4.group(1)), 4), "z_ft_8b": round(float(m8.group(1)), 4)}
    fail(f"{src}: cannot tell which logit-scale ratios were applied "
         f"(no provenance.ratios and no ratios in the note)")


def load_stack(path: Path) -> dict:
    try:
        stack = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"cannot read stack json {path}: {e}")
    if not isinstance(stack, dict):
        fail(f"{path}: not a JSON object")
    feats, coef, icpt = stack.get("features"), stack.get("coef"), stack.get("intercept")
    if not (isinstance(feats, list) and isinstance(coef, list) and len(coef) == len(feats)
            and all(isinstance(c, (int, float)) for c in coef)
            and isinstance(icpt, (int, float))):
        fail(f"{path}: invalid stack (needs features, coef of the same length, intercept)")
    return stack


def compare(new_path: Path, ref_path: Path, atol: float) -> int:
    """Structure must match (exit 1 otherwise); numbers are reported only (exit 0)."""
    new, ref = load_stack(new_path), load_stack(ref_path)
    bad = [f"{k}: NEW {new.get(k)!r} != REF {ref.get(k)!r}"
           for k in STRUCTURE if new.get(k) != ref.get(k)]
    rn, rr = ratios_applied(new, new_path), ratios_applied(ref, ref_path)
    if rn != rr:
        bad.append(f"logit-scale ratios: NEW {rn} != REF {rr}")
    if bad:
        print("STRUCTURE MISMATCH (the method changed):\n  " + "\n  ".join(bad))
        return 1
    print(f"structure OK: features, bundles, clip, manual_blend, ratios {rr}")
    rows = [("intercept", float(new["intercept"]), float(ref["intercept"]))]
    rows += [(f, float(c), float(r)) for f, c, r in zip(new["features"], new["coef"], ref["coef"])]
    print(f"{'parameter':<11}{'NEW':>22}{'REF':>22}{'|diff|':>12}")
    for name, a, b in rows:
        print(f"{name:<11}{a:>22.16f}{b:>22.16f}{abs(a - b):>12.2e}")
    d = max(abs(a - b) for _, a, b in rows)
    print(f"max |diff| {d:.2e}")
    if d <= atol:
        print(f"MATCH (max |diff| <= atol {atol:g})")
    else:
        print(f"DIFFERS (max |diff| {d:.2e} > atol {atol:g}; expected after a retrain, see README)")
    return 0


def synth_ratio(synth_dir: Path) -> float:
    """fold single-adapter logit std / full-fit logit std on identical synthetic rows."""
    full = read_oof(synth_dir / "p_full.csv", ["response_id", "p_llm"])
    stds = []
    for k in range(5):  # fold columns listed explicitly so p_full is never counted
        f = read_oof(synth_dir / f"p_fold{k}.csv", ["response_id", "p_llm"])
        m = full.merge(f, on="response_id", suffixes=("_full", "_fold"))
        stds.append(logit(m["p_llm_fold"]).std())
    return float(np.mean(stds) / logit(full["p_llm"]).std())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="outputs/models")
    ap.add_argument("--synth", action="store_true",
                    help="recompute the 4B/8B ratios from <models>/e13_qwen{4,8b}/synth")
    ap.add_argument("--w-q35", type=float, default=0.50)
    ap.add_argument("--out", default="outputs/models/stack_repro/llm_stack.json")
    ap.add_argument("--check", default=None, help="shipped llm_stack.json to compare against")
    ap.add_argument("--compare", nargs=2, metavar=("NEW", "REF"), type=Path,
                    help="compare two stack jsons (no refit): structure must match, "
                         "coefficient differences are reported")
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="tolerance on intercept/coef (ratios are recorded to 4 decimals)")
    a = ap.parse_args()
    if a.compare:
        if a.check:
            ap.error("--compare does not fit; it cannot be combined with --check")
        return compare(*a.compare, a.atol)
    m = Path(a.models)

    tab_f, base_f = m / "tabular_v2_bge_no_telo/oof.csv", m / "sub5_fit/oof.csv"
    tab = read_oof(tab_f, ["response_id", "is_correct", "oof_tabular_v2"])
    lab = pd.to_numeric(tab["is_correct"], errors="coerce")
    if lab.isna().any() or not lab.isin([0, 1]).all():
        fail(f"{tab_f}: is_correct must be 0/1 for every row "
             f"({int((lab.isna() | ~lab.isin([0, 1])).sum())} invalid)")
    check_probs(tab, "oof_tabular_v2", tab_f)
    base = read_oof(base_f, ["response_id", "oof_blend_cal"])
    check_probs(base, "oof_blend_cal", base_f)
    p4, p8 = read_llm_oof(m / "e13_qwen4", "p4"), read_llm_oof(m / "e13_qwen8b", "p8")
    for other, name in ((base, base_f), (p4, m / "e13_qwen4"), (p8, m / "e13_qwen8b")):
        same_ids(tab, tab_f, other, name)
    df = (tab.merge(base, on="response_id", validate="1:1")
             .merge(p4, on="response_id", validate="1:1")
             .merge(p8, on="response_id", validate="1:1"))
    if len(df) != 35072:
        fail(f"expected 35,072 aligned OOF rows, got {len(df):,}")

    X = np.column_stack([logit(df.oof_blend_cal), logit(df.oof_tabular_v2),
                         logit(df.p4), logit(df.p8)])
    y = df["is_correct"].to_numpy(int)
    lr = LogisticRegression(C=1e6, max_iter=1000, tol=1e-4, solver="lbfgs").fit(X, y)
    raw = dict(zip(FEATURES, map(float, lr.coef_[0])))
    intercept = float(lr.intercept_[0])

    if a.synth:
        ratios = {"z_ft_4b": synth_ratio(m / "e13_qwen4/synth"),
                  "z_ft_8b": synth_ratio(m / "e13_qwen8b/synth")}
    else:
        ratios = dict(RECORDED_RATIOS)
    coef = [raw[f] * ratios.get(f, 1.0) for f in FEATURES]

    stack = {
        "variant": "sub_q35_full_w050_sub9 (reproduced)",
        "features": FEATURES,
        "coef": coef,
        "intercept": intercept,
        "bundles": [{"feature": "z_ft_4b", "dir": "llm_4b"},
                    {"feature": "z_ft_8b", "dir": "llm_8b"},
                    {"feature": "z_ft_q35", "dir": "llm_q35"}],
        "clip": [0.01, 0.99],
        "serialize": 1,
        "max_input_tokens": 8192,
        "rowpass_budget": 50000,
        "manual_blend": [{"target": "z_ft_8b", "source": "z_ft_q35", "w": a.w_q35}],
        "provenance": {"raw_coef": raw, "ratios": ratios, "n_rows": int(len(df)),
                       "solver": "lbfgs C=1e6 tol=1e-4 max_iter=1000"},
    }
    out = Path(a.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stack, indent=2))
    except OSError as e:
        fail(f"cannot write {out}: {e}")
    print(f"raw  intercept {intercept:+.10f}  " + "  ".join(f"{f} {raw[f]:.10f}" for f in FEATURES))
    print(f"ratios {ratios}")
    print(f"wrote {out}")

    if a.check:
        ship = json.loads(Path(a.check).read_text())
        bad = []
        for k in ("features", "bundles", "clip", "serialize", "max_input_tokens",
                  "rowpass_budget", "manual_blend"):
            if stack[k] != ship.get(k):
                bad.append(f"{k}: {stack[k]!r} != {ship.get(k)!r}")
        d = max(abs(intercept - ship["intercept"]),
                *(abs(c - s) for c, s in zip(coef, ship["coef"])))
        print(f"max |delta| intercept/coef vs shipped: {d:.2e}")
        if d > a.atol:
            bad.append(f"coefficients differ by {d:.2e} > {a.atol}")
        if bad:
            print("MISMATCH:\n  " + "\n  ".join(bad))
            return 1
        print("OK: reproduced stack matches the shipped llm_stack.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
