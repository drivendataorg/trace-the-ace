#!/usr/bin/env python
r"""Full-fit vs fold-adapter logit scale on the synthetic holdout.

The stack was fitted on fold-adapter OOF logits but ships single full-fit adapters, and no real
fold can measure their logit scale because the full fit saw every real row. The synthetic
holdout (outputs/synth/pilot) was seen by none of the adapters. Its labels are the generator's
latent, so only relative logit statistics (mean, std, correlation) are compared.
Inputs: score_vllm outputs on the same rows (--full p_full.csv, --folds p_fold0..4.csv). If the
std ratio deviates by more than --tol, the report gives the rescale fold_std / full_std.

    python scripts/compare_synth_holdout.py --full <run>/synth/p_full.csv \
        --folds <run>/synth/p_fold{0,1,2,3,4}.csv [--stack llm_stack.json] [--meta meta.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="score_vllm CSV for the full-fit adapter")
    ap.add_argument("--folds", nargs="+", required=True,
                    help="score_vllm CSVs for the fold adapters (same synth rows)")
    ap.add_argument("--title", default=None,
                    help="model label for the report heading (default: the model dir "
                         "inferred from --full)")
    ap.add_argument("--stack", default=None,
                    help="optional stack JSON whose coef[1] is the reader coefficient to "
                         "rescale; without it only the logit scale is reported")
    ap.add_argument("--meta", default="outputs/synth/pilot/meta.jsonl",
                    help="synth meta.jsonl for the per-novelty-band breakdown")
    ap.add_argument("--out", default="outputs/models/sub7_8b_fit/synth_holdout.md")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="std-ratio deviation above which a c_ft rescale is advised")
    args = ap.parse_args()

    full = pd.read_csv(args.full).rename(columns={"p_llm": "p_full"})
    fold_dfs = []
    fold_cols = []
    for i, f in enumerate(args.folds):
        col = f"pfold_{i}"
        fold_dfs.append(pd.read_csv(f).rename(columns={"p_llm": col}))
        fold_cols.append(col)
    m = full
    for fd in fold_dfs:
        m = m.merge(fd, on="response_id", validate="1:1")
    n = len(m)
    if n == 0:
        raise SystemExit("no overlapping response_ids across the scored CSVs")

    # fold_cols is listed explicitly: a prefix match such as startswith("p_f") would also
    # catch "p_full" and pull the full-fit logits into the fold set.
    zf_each = np.column_stack([logit(m[c]) for c in fold_cols])
    z_full = logit(m["p_full"])
    z_foldmean = zf_each.mean(axis=1)  # the k-adapter test-time regime, per row

    full_std, fold_std = z_full.std(), z_foldmean.std()
    full_mu, fold_mu = z_full.mean(), z_foldmean.mean()
    corr = np.corrcoef(z_full, z_foldmean)[0, 1]
    ratio = full_std / fold_std
    # per-single-adapter std: the regime the OOF was fitted on (one adapter, not a mean),
    # which is also what the k=1 full-fit deployment uses.
    single_std = zf_each.std(axis=0).mean()

    # heading label: explicit --title, else the model dir two levels up from --full
    # (outputs/models/<MODEL>/synth/p_full.csv -> <MODEL>)
    title = args.title or Path(args.full).resolve().parent.parent.name
    lines = [f"# {title} — full-fit vs fold logit-scale check (synthetic holdout)\n",
             f"{n:,} synth rows scored by the full-fit adapter and {len(fold_cols)} "
             "fold adapters. RELATIVE comparison only (synth labels = generator "
             "latent, not truth).\n",
             "| quantity | full-fit | fold-mean | ratio |",
             "| --- | --- | --- | --- |",
             f"| logit mean | {full_mu:+.4f} | {fold_mu:+.4f} | — |",
             f"| logit std | {full_std:.4f} | {fold_std:.4f} | **{ratio:.3f}** |",
             f"| corr(full, fold-mean) | | | {corr:.4f} |",
             f"| mean per-single-adapter std | | {single_std:.4f} | "
             f"{full_std / single_std:.3f} |\n"]

    # The stack is used only when supplied, never defaulted: coef[1] is the reader
    # coefficient only for two-feature stacks, and a reader that enters indirectly
    # (Qwen3.5, through the z_ft_8b blend) has no coefficient of its own.
    c_ft = None
    if args.stack:
        stack_path = Path(args.stack)
        if not stack_path.exists():
            raise SystemExit(f"[synth-holdout] --stack {stack_path} not found")
        c_ft = json.loads(stack_path.read_text())["coef"][1]
    rescale = fold_std / full_std

    if abs(ratio - 1.0) > args.tol:
        head = (f"**Rescale advised.** full-fit logits are "
                f"{'wider' if ratio > 1 else 'narrower'} than the fold-mean by "
                f"{abs(ratio - 1) * 100:.1f}% (> {args.tol * 100:.0f}% tol). ")
        if c_ft is not None:
            lines += [head + f"c_ft={c_ft:.4f} was calibrated on fold logits; to keep "
                      f"the transcript-reader's spread in the blend, rescale:",
                      f"\n    c_ft: {c_ft:.4f} -> {c_ft * rescale:.4f}  "
                      f"(x {rescale:.3f})\n",
                      "then rebuild the zip. (Equivalent to Platt-matching the full-fit "
                      "logit onto the fold scale; intercept/c_blend unchanged.)\n"]
        else:
            lines += [head + "No stack was supplied, so this is a SCALE measurement "
                      "only:",
                      f"\n    full-fit logit scale multiplier = {rescale:.3f}\n",
                      "No stack coefficient was supplied or modified. Apply this "
                      "multiplier only to a coefficient you have confirmed multiplies "
                      "THIS reader's logit (equivalent to Platt-matching the full-fit "
                      "logit onto the fold scale).\n"]
    elif c_ft is not None:
        lines += [f"**No rescale needed.** full-fit vs fold-mean std ratio {ratio:.3f} "
                  f"is within {args.tol * 100:.0f}% — c_ft={c_ft:.4f} transfers as-is. "
                  "The k=1 full-fit deployment matches the single-adapter OOF regime the "
                  "stacker was calibrated on.\n"]
    else:
        lines += [f"**No rescale needed.** full-fit vs fold-mean std ratio {ratio:.3f} "
                  f"is within {args.tol * 100:.0f}%. No scale correction is indicated. "
                  "No stack coefficient was supplied or modified.\n"]

    # per-novelty-band shift (meta.jsonl: session_id -> novelty band vs real objectives)
    meta_path = Path(args.meta)
    if meta_path.exists() and "session_id" in m.columns:
        meta = {j["session_id"]: j for j in
                (json.loads(l) for l in meta_path.read_text().splitlines()) if l.strip()}
        band = m["session_id"].map(lambda s: (meta.get(s, {}) or {}).get("band", "?"))
        if band.notna().any() and (band != "?").any():
            lines += ["\n## by novelty band (logit std)\n",
                      "| band | n | full-fit | fold-mean | ratio |",
                      "| --- | --- | --- | --- | --- |"]
            for b in sorted(set(band) - {"?"}):
                mm = (band == b).values
                if mm.sum() < 50:
                    continue
                fs, ds = z_full[mm].std(), z_foldmean[mm].std()
                lines.append(f"| {b} | {mm.sum():,} | {fs:.4f} | {ds:.4f} | {fs / ds:.3f} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[synth-holdout] -> {out}")
    if abs(ratio - 1.0) > args.tol:
        if c_ft is not None:
            print(f"[synth-holdout] ACTION: rescale c_ft {c_ft:.4f} -> "
                  f"{c_ft * rescale:.4f} in llm_stack.json before building the final zip")
        else:
            # no coefficient to act on: report the measurement only
            print(f"[synth-holdout] SCALE: full-fit logit multiplier = {rescale:.3f}; "
                  f"no stack coefficient was supplied or modified")


if __name__ == "__main__":
    main()