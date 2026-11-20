#!/usr/bin/env python
"""Score the XGBoost half the way the runtime blends it, on `mrft`'s own OOF.

The shipped ensemble blends the calibrated LLM half (`mrft`, whose OOF covers 16,693
objective-disjoint rows) with the calibrated XGBoost half, so the blend is scored on those rows
after both Platt maps.

It takes XGBoost out-of-fold vectors (a trainer ``.csv``, or ``npz:key`` for an archive of
seed-ensembled, already Platt-calibrated OOF in training row order), joins them to `mrft`'s
keyed OOF by ``response_id``, and for each input:

  * scores the blend over a grid of fixed weights and reports LL/AUROC, including at the
    shipped weight ``SHIPPED_W`` (0.43);
  * bootstraps the log-loss-optimal weight w* over session-clustered resamples;
  * scores a single weight, and a weight that may differ by transcript length (or, separately,
    by whether the session carries more than one objective), both cross-fitted over `mrft`'s
    own folds;
  * with two or more inputs, scores 3-way blends over pairs of inputs. Step 08a
    passes one input.

Each half's OOF is out-of-fold for its own rows under its own objective-disjoint partition, and
the two partitions differ; row i is still out-of-fold for both models.

Out: figures/data/e3_blend_decide<tag>.md, artifacts/e3_blend_decide.csv, and with two or more
     inputs artifacts/e3_blend_decide<tag>_3way.csv

Usage (from build/workspace/; reproduce.py step 08a):
  OMP_NUM_THREADS=4 python scripts/e3_blend_decide.py --clean-only \
      --llm-calib llmkt_calibrator_mrft_sub0808.joblib \
      --cand v2shrunkD=artifacts/v2shrunkD_oof.csv \
      --tag _repro
An input (``--cand``) is ``name=path`` for a trainer CSV or ``name=path.npz:key`` for an ``.npz``
archive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from magnificat import data  # noqa: E402
from magnificat.calibration import PlattScaler  # noqa: E402

GRID = np.round(np.arange(0.0, 0.8001, 0.01), 4)
BOOT_GRID = np.round(np.arange(0.05, 0.8001, 0.02), 4)
N_BOOT = 400
SHIPPED_W = 0.43  # the shipped blend weight (assets/blend_weight.txt)
EPS = 1e-6


def logit(p):
    """Log-odds of ``p`` after clipping to [EPS, 1 - EPS]."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_candidate(spec: str, rid_order: np.ndarray) -> tuple[str, pd.Series]:
    """Load the calibrated XGBoost probabilities of one ``--cand`` input, keyed by response_id.

    A ``.csv`` input is the trainer's own OOF dump (``response_id,p_raw,p_abl,y``) and is
    calibrated here by the same frozen-Platt call the trainer makes (fit on the full and
    ablated predictions pooled). An ``.npz`` input holds seed-ensembled OOF, already
    calibrated and stored in training row order.

    Args:
        spec: ``name=path[:key]``, path relative to the workspace root; without ``name=`` the
            file stem is the name. ``key`` selects the array in an ``.npz`` file.
        rid_order: response ids in training row order (used for ``.npz`` inputs).

    Returns:
        ``(name, series)``: calibrated probabilities indexed by ``response_id``.
    """
    name, _, path = spec.partition("=")
    if not path:
        name, path = Path(spec).stem, spec
    if ":" in path and not path.endswith(".csv"):
        fn, _, key = path.partition(":")
    else:
        fn, key = path, None
    p = ROOT / fn
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        pl = PlattScaler.fit(np.concatenate([df.p_raw, df.p_abl]),
                             np.concatenate([df.y, df.y]))
        return name, pd.Series(pl.transform(df.p_raw.to_numpy()),
                               index=df.response_id.to_numpy())
    z = np.load(p, allow_pickle=False)
    if key not in z.files:
        raise SystemExit(f"{fn}: key {key!r} not found. available:\n  "
                         + "\n  ".join(k for k in z.files if "::" in k))
    v = z[key]
    if len(v) != len(rid_order):
        raise SystemExit(f"{fn}:{key} has {len(v)} rows, train has {len(rid_order)}")
    return name, pd.Series(v, index=rid_order)


def sweep(y, pl, pg, grid=GRID):
    """Score the blend ``(1 - w) * pl + w * pg`` at every weight of ``grid``.

    Args:
        y: 0/1 labels.
        pl: calibrated LLM probabilities.
        pg: calibrated XGBoost probabilities.
        grid: XGBoost weights to evaluate.

    Returns:
        DataFrame with columns ``w``, ``ll`` and ``auroc`` (x100), one row per weight.
    """
    rows = [dict(w=w, ll=log_loss(y, (1 - w) * pl + w * pg, labels=[0, 1]),
                 auroc=roc_auc_score(y, (1 - w) * pl + w * pg) * 100) for w in grid]
    return pd.DataFrame(rows)


def best_w(y, pl, pg, grid=BOOT_GRID) -> float:
    """Return the XGBoost weight in ``grid`` with the lowest blend log loss (args as ``sweep``)."""
    lls = [log_loss(y, (1 - w) * pl + w * pg, labels=[0, 1]) for w in grid]
    return float(grid[int(np.argmin(lls))])


W3_GRID = np.round(np.arange(0, 0.6001, 0.05), 4)


def best_wab(y, pl, pa, pb, grid=W3_GRID) -> tuple[float, float]:
    """Find the 3-way blend weights with the lowest log loss.

    Searches ``{wa, wb in grid, wa + wb <= 0.60}`` for the blend
    ``(1 - wa - wb) * pl + wa * pa + wb * pb``.

    Args:
        y: 0/1 labels.
        pl: calibrated LLM probabilities.
        pa: calibrated probabilities of XGBoost model a.
        pb: calibrated probabilities of XGBoost model b.
        grid: values tried for each weight.

    Returns:
        ``(wa, wb)``.
    """
    best = None
    for wa in grid:
        for wb in grid[grid <= 0.60 - wa + 1e-9]:
            ll = log_loss(y, (1 - wa - wb) * pl + wa * pa + wb * pb, labels=[0, 1])
            if best is None or ll < best[0]:
                best = (ll, float(wa), float(wb))
    return best[1], best[2]


def xfit3(y, pl, pa, pb, fold) -> tuple[np.ndarray, list[tuple]]:
    """Cross-fitted 3-way blend: (wa, wb) chosen on the other folds, applied to each fold.

    Args:
        y: 0/1 labels.
        pl: calibrated LLM probabilities.
        pa: calibrated probabilities of XGBoost model a.
        pb: calibrated probabilities of XGBoost model b.
        fold: `mrft`'s own objective-disjoint fold per row, so a held-out row shares no
            learning objective with the rows its weights were fitted on.

    Returns:
        ``(p, chosen)``: the cross-fitted blend probabilities, and ``(fold, wa, wb)`` per fold.
    """
    p = np.empty(len(y), float)
    chosen = []
    for k in sorted(np.unique(fold)):
        te = fold == k
        tr = ~te
        wa, wb = best_wab(y[tr], pl[tr], pa[tr], pb[tr])
        chosen.append((int(k), wa, wb))
        p[te] = ((1 - wa - wb) * pl + wa * pa + wb * pb)[te]
    return p, chosen


def main() -> None:
    """Score every ``--cand`` input in the calibrated blend with `mrft` and write the report.

    Returns:
        None. Prints the per-input results and writes
        ``figures/data/e3_blend_decide<tag>.md``, ``artifacts/e3_blend_decide.csv`` and, with
        two or more inputs, ``artifacts/e3_blend_decide<tag>_3way.csv``.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", required=True,
                    help="name=path[:npz_key]; repeatable")
    ap.add_argument("--llm-calib", default="llmkt_calibrator_mrft.joblib",
                    help="file name in artifacts/ of the LLM calibrator to apply (step 08a "
                         "passes the shipped llmkt_calibrator_mrft_sub0808.joblib: a=0.6876, "
                         "b=+0.4565)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--tag", default="",
                    help="suffix for figures/data/e3_blend_decide<TAG>.md and "
                         "artifacts/e3_blend_decide<TAG>_3way.csv")
    ap.add_argument("--clean-only", action="store_true",
                    help="restrict to the leakage-free rows (folds.clean_mask). On the pooled "
                         "OOF, 52%% of the rows were predicted by a model that saw their "
                         "transcript in training. The clean rows overlap mrft's own OOF on only "
                         "~7.7k rows, so expect wider intervals.")
    args = ap.parse_args()

    train = data.load_train()
    rid = train.response_id.to_numpy()

    mrft = pd.read_csv(ROOT / "artifacts/mrft_oof_3fold_keyed.csv")
    # Read the calibrator from artifacts/. The fallback path below does not exist in this
    # package.
    _cal = ROOT / "artifacts" / args.llm_calib
    if not _cal.exists():
        _cal = ROOT / "submission_src/assets/llmkt_calibrator.joblib"
    platt_llm = joblib.load(_cal)
    print(f"LLM calibrator: {_cal.name}", flush=True)
    print(f"mrft OOF: {len(mrft):,} rows / {mrft.fold.nunique()} folds; "
          f"shipped LLM Platt a={platt_llm.a:.4f} b={platt_llm.b:+.4f}", flush=True)

    cands = dict(load_candidate(c, rid) for c in args.cand)
    d = mrft.copy()
    for name, s in cands.items():
        d[name] = d.response_id.map(s)
    n0 = len(d)
    d = d.dropna(subset=list(cands)).reset_index(drop=True)
    if args.clean_only:
        from magnificat.folds import clean_mask, lo_folds
        _keep = set(train.response_id.to_numpy()[
            clean_mask(train.session_id.to_numpy(),
                       lo_folds(train.learning_objective_id.to_numpy()))])
        _before = len(d)
        d = d[d.response_id.isin(_keep)].reset_index(drop=True)
        print(f"clean rows only: {len(d):,} of {_before:,} rows kept "
              f"(base rate {d.y.mean():.4f})", flush=True)
    print(f"joined: {len(d):,}/{n0:,} mrft rows have every input", flush=True)

    y = d.y.to_numpy(int)
    p_llm = platt_llm.transform(d.p.to_numpy())
    print(f"LLM alone (calibrated): LL {log_loss(y, p_llm):.4f} / "
          f"AUROC {roc_auc_score(y, p_llm) * 100:.2f}\n", flush=True)

    # covariates for the varying-weight blends
    meta = train[["response_id", "session_id"]].copy()
    meta["n_resp"] = meta.groupby("session_id").response_id.transform("size")
    nlo = train.groupby("session_id").learning_objective_id.transform("nunique")
    meta["multi"] = (nlo > 1).to_numpy()
    tl = pd.read_pickle(ROOT / "artifacts/turn_text.pkl.gz", compression="gzip")
    ntur = tl.groupby("session_id").size().rename("n_turns")
    meta = meta.merge(ntur, on="session_id", how="left")
    m = d[["response_id"]].merge(meta, on="response_id", how="left")
    sess = m.session_id.to_numpy()
    long_q = float(np.nanquantile(m.n_turns, 0.75))
    Z = {"long transcript (top quartile turns)": (m.n_turns.to_numpy() >= long_q),
         "multi-objective session": m.multi.to_numpy(bool)}

    rows, per_w = [], {}
    for name in cands:
        pg = d[name].to_numpy()
        s = sweep(y, p_llm, pg)
        per_w[name] = s
        b = s.loc[s.ll.idxmin()]
        at_ship = s[np.isclose(s.w, SHIPPED_W)].iloc[0]
        inside = s[s.ll <= b.ll + 0.0005].w

        # session-clustered bootstrap of w*
        groups = pd.Series(np.arange(len(d))).groupby(sess).apply(np.array).values
        rng = np.random.default_rng(221)
        ws = []
        for _ in range(args.boot):
            take = np.concatenate(groups[rng.integers(0, len(groups), len(groups))])
            if y[take].min() == y[take].max():
                continue
            ws.append(best_w(y[take], p_llm[take], pg[take]))
        ws = np.asarray(ws)

        # covariate-dependent weight, cross-fitted over mrft's own folds
        varying = {}
        for zname, z in Z.items():
            pb = np.empty(len(y))
            for k in sorted(d.fold.unique()):
                te = (d.fold == k).to_numpy()
                tr = ~te
                for side in (True, False):
                    a = tr & (z == side)
                    b_ = te & (z == side)
                    if not a.any() or not b_.any() or y[a].min() == y[a].max():
                        w = best_w(y[tr], p_llm[tr], pg[tr]) if tr.any() else SHIPPED_W
                    else:
                        w = best_w(y[a], p_llm[a], pg[a])
                    pb[b_] = ((1 - w) * p_llm + w * pg)[b_]
            varying[zname] = (log_loss(y, pb, labels=[0, 1]),
                              roc_auc_score(y, pb) * 100)
        # a single weight, cross-fitted the same way
        pb1 = np.empty(len(y))
        for k in sorted(d.fold.unique()):
            te = (d.fold == k).to_numpy()
            w = best_w(y[~te], p_llm[~te], pg[~te])
            pb1[te] = ((1 - w) * p_llm + w * pg)[te]
        cf1 = (log_loss(y, pb1, labels=[0, 1]), roc_auc_score(y, pb1) * 100)

        rows.append(dict(cand=name, w_star=float(b.w), ll=float(b.ll),
                         auroc=float(b.auroc), ll_at_ship=float(at_ship.ll),
                         auroc_at_ship=float(at_ship.auroc),
                         plateau_lo=float(inside.min()), plateau_hi=float(inside.max()),
                         boot_lo=float(np.percentile(ws, 2.5)),
                         boot_hi=float(np.percentile(ws, 97.5)),
                         xfit1_ll=cf1[0], xfit1_auroc=cf1[1],
                         **{f"vary::{k}": v[0] for k, v in varying.items()},
                         **{f"varyauc::{k}": v[1] for k, v in varying.items()}))
        print(f"{name:28s} w*={b.w:.2f} LL {b.ll:.4f} AUROC {b.auroc:.2f} | "
              f"@{SHIPPED_W:.2f} {at_ship.ll:.4f}/{at_ship.auroc:.2f} | boot95 "
              f"[{np.percentile(ws, 2.5):.2f},{np.percentile(ws, 97.5):.2f}] | "
              f"xfit-1w {cf1[0]:.4f}/{cf1[1]:.2f}", flush=True)
        for zname, (ll, auc) in varying.items():
            print(f"{'':28s}   split by {zname}: {ll:.4f} / {auc:.2f} "
                  f"({ll - cf1[0]:+.5f} vs one weight)", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "artifacts/e3_blend_decide.csv", index=False)

    # ── 3-way: LLM + two XGBoost inputs (only with two or more inputs) ──────
    # Both the cross-fitted and the in-sample scores are reported.
    fold_v = d.fold.to_numpy()
    three = []
    names = list(cands)
    # the best cross-fitted single-weight (2-way) score over all inputs, reported beside the
    # 3-way blends
    best_2way = float(r.xfit1_ll.min()) if len(r) else np.nan
    best_2way_auc = float(r.loc[r.xfit1_ll.idxmin(), "xfit1_auroc"]) if len(r) else np.nan
    best_2way_cand = str(r.loc[r.xfit1_ll.idxmin(), "cand"]) if len(r) else ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb_ = d[a].to_numpy(), d[b].to_numpy()
            wa_is, wb_is = best_wab(y, p_llm, pa, pb_)
            p_is = (1 - wa_is - wb_is) * p_llm + wa_is * pa + wb_is * pb_
            p_cf, chosen = xfit3(y, p_llm, pa, pb_, fold_v)
            corr = float(np.corrcoef(logit(pa), logit(pb_))[0, 1])
            three.append(dict(
                a=a, b=b, corr=corr,
                ll=log_loss(y, p_cf, labels=[0, 1]),          # the cross-fitted LL
                auroc=roc_auc_score(y, p_cf) * 100,
                ll_insample=log_loss(y, p_is, labels=[0, 1]),
                auroc_insample=roc_auc_score(y, p_is) * 100,
                w_a=wa_is, w_b=wb_is,
                w_a_xfit="/".join(f"{w:.2f}" for _, w, _ in chosen),
                w_b_xfit="/".join(f"{w:.2f}" for _, _, w in chosen)))
            # cross-fitted minus in-sample LL
            three[-1]["optimism"] = three[-1]["ll"] - three[-1]["ll_insample"]
            three[-1]["vs_best_2way"] = three[-1]["ll"] - best_2way
    t = pd.DataFrame(three).sort_values("ll") if three else pd.DataFrame()
    if len(t):
        print(f"\n3-way blends (LLM + two GBMs), against the best cross-fitted 2-way blend "
              f"({best_2way_cand}) {best_2way:.4f} / {best_2way_auc:.2f}:")
        print(t[["a", "b", "corr", "ll", "auroc", "ll_insample", "optimism",
                 "vs_best_2way", "w_a", "w_b"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        _n_flip = int((t.ll_insample < best_2way).sum() - (t.ll < best_2way).sum())
        print(f"\n  in-sample optimism: mean {t.optimism.mean():+.4f}, max "
              f"{t.optimism.max():+.4f}. {int((t.ll_insample < best_2way).sum())} pairs beat "
              f"the best 2-way blend in-sample; {int((t.ll < best_2way).sum())} still do "
              f"cross-fitted ({_n_flip} were in-sample artefacts).", flush=True)

    md = ["# Blend report: the calibrated halves blended the way the runtime blends them\n\n",
          f"`mrft`'s own OOF ({len(d):,} LO-disjoint rows), **both halves Platt-mapped "
          "before they meet**, the transform `main.py` applies. "
          f"LLM alone here: **{log_loss(y, p_llm):.4f} / "
          f"{roc_auc_score(y, p_llm) * 100:.2f}**.\n\n",
          "`xfit-1w` cross-fits a single weight over `mrft`'s folds; the `split by` rows "
          "let that weight differ by a covariate, cross-fitted the same way.\n\n",
          "The `w*` / `LL` / `AUROC` columns below are in-sample: `w*` is read off the same rows "
          "it is scored on. The cross-fitted level is the `xfit-1w` number below.\n\n",
          f"| input | w* (in-sample) | LL (in-sample) | AUROC (in-sample) | LL @{SHIPPED_W:.2f} | "
          f"AUROC @{SHIPPED_W:.2f} | plateau | boot95 w* |\n",
          "|---|---|---|---|---|---|---|---|\n"]
    for _, x in r.iterrows():
        md.append(f"| {x.cand} | {x.w_star:.2f} | {x.ll:.4f} | {x.auroc:.2f} | "
                  f"{x.ll_at_ship:.4f} | {x.auroc_at_ship:.2f} | "
                  f"[{x.plateau_lo:.2f}, {x.plateau_hi:.2f}] | "
                  f"[{x.boot_lo:.2f}, {x.boot_hi:.2f}] |\n")
    vary_cols = [c for c in r.columns if c.startswith("vary::")]
    if vary_cols:
        md += ["\n## Does the mixing weight want to vary with the transcript?\n\n",
               "| input | one weight (x-fit) | " +
               " | ".join(c.split("::", 1)[1] for c in vary_cols) + " |\n",
               "|---|---|" + "---|" * len(vary_cols) + "\n"]
        for _, x in r.iterrows():
            md.append(f"| {x.cand} | {x.xfit1_ll:.4f} / {x.xfit1_auroc:.2f} | "
                      + " | ".join(f"{x[c]:.4f} / {x['varyauc::' + c.split('::', 1)[1]]:.2f}"
                                   for c in vary_cols) + " |\n")
    if len(t):
        md += ["\n## 3-way blends (LLM + two XGBoost inputs)\n\n",
               "Sorted on the cross-fitted column: `(w_a, w_b)` are fitted on the other folds of "
               "`mrft`'s LO-disjoint partition and applied to the held-out one.\n\n",
               f"Best cross-fitted 2-way blend (`{best_2way_cand}`): "
               f"**{best_2way:.4f} / {best_2way_auc:.2f}**.\n\n",
               "| XGBoost a | XGBoost b | logit corr(a,b) | **LL (x-fit)** | **AUROC (x-fit)** | "
               "vs best 2-way | LL (in-sample) | optimism | w_a/w_b (in-sample) | "
               "w_a per fold | w_b per fold |\n",
               "|---|---|---|---|---|---|---|---|---|---|---|\n"]
        for _, x in t.iterrows():
            md.append(f"| {x.a} | {x.b} | {x['corr']:.3f} | **{x.ll:.4f}** | "
                      f"**{x.auroc:.2f}** | {x.vs_best_2way:+.4f} | {x.ll_insample:.4f} | "
                      f"{x.optimism:+.4f} | {x.w_a:.2f}/{x.w_b:.2f} | "
                      f"{x.w_a_xfit} | {x.w_b_xfit} |\n")
    out_md = ROOT / f"figures/data/e3_blend_decide{args.tag}.md"
    out_md.write_text("".join(md))
    if len(t):
        t.to_csv(ROOT / f"artifacts/e3_blend_decide{args.tag}_3way.csv", index=False)
    print(f"\nwrote {out_md.relative_to(ROOT)} + artifacts/e3_blend_decide.csv")


if __name__ == "__main__":
    main()
