#!/usr/bin/env python
"""Score saved out-of-fold prediction sets on the pooled and the leakage-free (clean) rows.

``GroupKFold(learning_objective_id)`` splits 31.9% of sessions across folds (36.7% of sessions
carry more than one objective, but 1,084 of those land in a single fold; see
``magnificat.folds.clean_mask``), so 52.4% of responses are predicted by a model trained on
their own transcript. A clean row's session never entered its fold's training set, so its saved
OOF prediction is already leakage-free and can be scored without retraining.

The script reports pooled and clean scores side by side. Clean rows are a different population,
not a random subsample: base rate 0.634 vs 0.765 for the leaked half, 1.14 vs 2.80 responses per
session, 321 vs 359 objectives. A clean level is not comparable to a pooled level.

With two or more inputs it also bootstraps each input's difference to the reference on the
clean rows, resampling whole sessions and, separately, whole learning objectives; the verdict
uses the objective unit (86.6% of clean rows are singleton sessions, so a session bootstrap there
is effectively a row bootstrap). Step 05b passes one input.

An input is given as ``name=path[:key]``, as in ``e3_blend_decide.py``, whose
``load_candidate()`` is reused: a trainer ``.csv`` holds raw predictions and gets the frozen
Platt fit applied; an ``.npz`` array is already calibrated and stored in training row order.

Out: figures/data/rescore_clean<tag>.md, artifacts/rescore_clean<tag>.csv

Usage (from build/workspace/; reproduce.py step 05b):
    OMP_NUM_THREADS=4 python scripts/rescore_clean.py \
        --cand v2shrunkD=artifacts/v2shrunkD_oof.csv --tag _v2shrunkD
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from magnificat import data  # noqa: E402
from magnificat.folds import clean_mask, clean_sha, fold_sha, lo_folds  # noqa: E402
from e3_blend_decide import load_candidate  # noqa: E402


def load_any(spec, rid):
    """Load an input like ``load_candidate``, also accepting a ``response_id,y,oof`` CSV.

    Such a file holds a single raw vector, no ablated twin. It is Platt-calibrated on itself,
    the equivalent of the pooled full+ablated fit that trainer CSVs get. AUROC is unaffected
    either way; log loss is not.

    Args:
        spec: ``name=path[:key]``, path relative to the workspace root.
        rid: response ids in training row order (for ``.npz`` inputs).

    Returns:
        ``(name, series)``: calibrated probabilities indexed by ``response_id``.
    """
    from magnificat.calibration import PlattScaler
    try:
        return load_candidate(spec, rid)
    except AttributeError:
        name, _, path = spec.partition("=")
        df = pd.read_csv(ROOT / path)
        if "oof" not in df.columns:
            raise
        pl = PlattScaler.fit(df.oof.to_numpy(), df.y.to_numpy())
        return name, pd.Series(pl.transform(df.oof.to_numpy()),
                               index=df.response_id.to_numpy())


def metrics(y, p):
    """Return ``(log loss, AUROC x 100)``; log loss on probabilities clipped to [1e-6, 1 - 1e-6]."""
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]), roc_auc_score(y, p) * 100


def _ll(y, p):
    """Binary log loss, identical to `log_loss(y, clip(p), labels=[0, 1])` for y in {0, 1}.

    Reported levels still go through sklearn (`metrics()` above); this exists only for the
    bootstrap inner loop, which calls it ~8,000 times per compared model. sklearn re-runs
    LabelBinarizer and input validation on every call, which is ~50x the cost of the
    arithmetic. `_assert_ll_equivalence()` checks the equivalence on real resamples.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _assert_ll_equivalence(y, p, n=200, seed=0, tol=1e-12):
    """Fail loudly if the fast log loss ever drifts from sklearn's on real resamples."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        a = _ll(y[idx], p[idx])
        b = log_loss(y[idx], np.clip(p[idx], 1e-6, 1 - 1e-6), labels=[0, 1])
        worst = max(worst, abs(a - b))
    if worst > tol:
        raise SystemExit(f"fast log loss disagrees with sklearn by {worst:.3e} (> {tol:.0e})")
    return worst


def _clusters(unit):
    """Group row indices by ``unit``: the resampling blocks of a clustered bootstrap.

    The flat form is for speed: a session bootstrap has ~15.5k clusters, and rebuilding a
    resample with ``np.concatenate([groups[i] for i in pick])`` spends all its time in Python
    over 15.5k one-element arrays.

    Args:
        unit: cluster label per row.

    Returns:
        ``(order, starts, lens)``: ``order[starts[g]:starts[g] + lens[g]]`` are the rows of
        cluster ``g``.
    """
    unit = np.asarray(unit)
    order = np.argsort(unit, kind="stable")
    starts = np.r_[0, np.unique(unit[order], return_index=True)[1][1:]]
    lens = np.diff(np.r_[starts, len(order)])
    return order, starts, lens


def _resample(order, starts, lens, pick):
    """Concatenate the row indices of the clusters in ``pick`` (vectorised, no Python loop).

    Builds the gather offsets directly: each picked cluster contributes ``lens[g]`` consecutive
    positions beginning at ``starts[g]``. Same output as
    ``np.concatenate([order[starts[g]:starts[g]+lens[g]] for g in pick])``.
    """
    L = lens[pick]
    tot = int(L.sum())
    off = np.repeat(starts[pick] - np.r_[0, np.cumsum(L)[:-1]], L)
    return order[off + np.arange(tot)]


def boot_delta_both(y, pa, pb, unit, n=4000, seed=0, metrics=("ll", "auroc")):
    """Clustered paired bootstrap of metric(a) - metric(b) for several metrics at once.

    Both metrics are computed on the same resamples, so their intervals are consistent with each
    other. Resamples with a single class are skipped.

    Args:
        y: 0/1 labels.
        pa: probabilities of model a.
        pb: probabilities of model b (the reference).
        unit: cluster label per row; pass ``learning_objective_id`` for the interval the
            verdict uses (see the module docstring).
        n: number of resamples.
        seed: seed of the numpy generator.
        metrics: any of ``"ll"`` (log loss, negative is better) and ``"auroc"``.

    Returns:
        Dict ``{metric: (lo, hi)}``, the 95% percentile interval of the difference (AUROC x 100).
    """
    rng = np.random.default_rng(seed)
    order, starts, lens = _clusters(unit)
    ng = len(starts)
    want_ll, want_auc = "ll" in metrics, "auroc" in metrics
    dll, dau = [], []
    for _ in range(n):
        pick = rng.integers(0, ng, ng)
        idx = _resample(order, starts, lens, pick)
        if len(np.unique(y[idx])) < 2:
            continue
        if want_ll:
            dll.append(_ll(y[idx], pa[idx]) - _ll(y[idx], pb[idx]))
        if want_auc:
            dau.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    q = lambda v, s: (float(np.percentile(v, 2.5)) * s, float(np.percentile(v, 97.5)) * s)
    out = {}
    if want_ll:
        out["ll"] = q(dll, 1.0)
    if want_auc:
        out["auroc"] = q(dau, 100.0)
    return out


def boot_delta(y, pa, pb, unit, n=4000, seed=0, metric="auroc"):
    """Clustered paired bootstrap of metric(a) - metric(b), resampling whole ``unit`` blocks.

    ``metric="ll"`` returns the log-loss difference, where negative is better.

    Args:
        y, pa, pb, n, seed: as in ``boot_delta_both``.
        unit: cluster label per row, chosen by the caller. On the clean rows it must be
            ``learning_objective_id``: ``session_id`` there is effectively a row bootstrap
            (86.6% of clean rows are singleton sessions).
        metric: ``"auroc"`` or ``"ll"``.

    Returns:
        ``(lo, hi)``, the 95% interval of the difference.
    """
    return boot_delta_both(y, pa, pb, unit, n=n, seed=seed, metrics=(metric,))[metric]


def main() -> int:
    """Score every input on pooled and clean rows; bootstrap the others against the reference.

    Returns:
        0. Prints the comparison tables and writes ``figures/data/rescore_clean<tag>.md`` and
        ``artifacts/rescore_clean<tag>.csv``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", required=True, help="name=path[:key]")
    ap.add_argument("--ref", default=None,
                    help="name=path[:key] used as the delta reference; defaults to the "
                         "first --cand")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    train = data.load_train()
    rid = train["response_id"].to_numpy()
    y = train["is_correct"].to_numpy(float)
    sess = train["session_id"].to_numpy()
    lo_id = train["learning_objective_id"].to_numpy()
    folds = lo_folds(lo_id)
    C = clean_mask(sess, folds)
    print(f"folds sha={fold_sha(folds)}  clean sha={clean_sha(C)}", flush=True)
    print(f"pooled n={len(y):,} base={y.mean():.4f} | "
          f"clean n={int(C.sum()):,} base={y[C].mean():.4f} | "
          f"leaked n={int((~C).sum()):,} base={y[~C].mean():.4f}", flush=True)

    # Cluster sizes of the two bootstrap units; printed on every run.
    _sz = pd.Series(sess[C]).value_counts()
    _rw_sess = float((_sz.reindex(pd.Series(sess[C])).to_numpy()).mean())
    _lz = pd.Series(lo_id[C]).value_counts()
    _rw_lo = float((_lz.reindex(pd.Series(lo_id[C])).to_numpy()).mean())
    print(f"bootstrap units on clean rows: {len(_sz):,} sessions "
          f"(row-weighted mean size {_rw_sess:.3f}, {(_sz.reindex(pd.Series(sess[C])).to_numpy() == 1).mean():.1%} "
          f"of rows are singletons; max design effect {_rw_sess:.3f}x, i.e. a row bootstrap) "
          f"vs {len(_lz):,} objectives (row-weighted mean size {_rw_lo:.1f}).", flush=True)
    print("deciding unit: learning_objective_id (the test's objectives are new).\n",
          flush=True)

    specs = ([args.ref] if args.ref else []) + args.cand
    loaded = []
    for s in specs:
        name, ser = load_any(s, rid)
        p = ser.reindex(rid).to_numpy(float)
        if np.isnan(p).any():
            raise SystemExit(f"{name}: {int(np.isnan(p).sum())} rows failed to join on "
                             "response_id; this arm covers a different row set")
        loaded.append((name, p))
    ref_name, ref_p = loaded[0]

    rows = []
    nan2 = (np.nan, np.nan)
    for name, p in loaded:
        pl, al = metrics(y, p)
        pc, ac = metrics(y[C], p[C])
        s_ll = s_au = l_ll = l_au = nan2
        if name != ref_name:
            # Both units; the log-loss and AUROC intervals come from the same resamples.
            s = boot_delta_both(y[C], p[C], ref_p[C], sess[C], n=args.boot)
            l = boot_delta_both(y[C], p[C], ref_p[C], lo_id[C], n=args.boot)
            s_ll, s_au, l_ll, l_au = s["ll"], s["auroc"], l["ll"], l["auroc"]
        rows.append(dict(arm=name, pooled_ll=pl, pooled_auc=al,
                         clean_ll=pc, clean_auc=ac,
                         d_pooled_auc=al - metrics(y, ref_p)[1],
                         d_clean_auc=ac - metrics(y[C], ref_p[C])[1],
                         d_pooled_ll=pl - metrics(y, ref_p)[0],
                         d_clean_ll=pc - metrics(y[C], ref_p[C])[0],
                         # the unprefixed CI columns are the session unit, the lo_*
                         # columns the objective unit.
                         ci_lo=s_au[0], ci_hi=s_au[1],
                         ll_ci_lo=s_ll[0], ll_ci_hi=s_ll[1],
                         lo_ci_lo=l_au[0], lo_ci_hi=l_au[1],
                         lo_ll_ci_lo=l_ll[0], lo_ll_ci_hi=l_ll[1]))
    d = pd.DataFrame(rows)

    # leak_gap: the pooled minus the clean AUROC difference to the reference, i.e. the part of
    # the difference that comes from the leaked rows.
    d["leak_gap"] = d.d_pooled_auc - d.d_clean_auc

    # Verdict: from the objective-clustered interval of the clean log-loss difference (negative
    # is better); resolved only if the interval lies entirely on one side of zero.
    def _verdict(r):
        if np.isnan(r.lo_ll_ci_lo):
            return ""
        return ("RESOLVED BETTER" if r.lo_ll_ci_hi < 0 else
                "RESOLVED WORSE" if r.lo_ll_ci_lo > 0 else "null")

    print(f"{'arm':22s} | {'ΔcleanLL':>9s} {'95% CI (sess)':>20s} {'95% CI (LO) ← decides':>21s} | "
          f"{'Δclean AUC':>10s} {'CI (sess)':>16s} {'CI (LO)':>16s} | {'LEAKgap':>7s}")
    print("-" * 140)
    for r in d.itertuples():
        f4 = lambda a, b: "" if np.isnan(a) else f"[{a:+.5f}, {b:+.5f}]"
        f2 = lambda a, b: "" if np.isnan(a) else f"[{a:+.2f}, {b:+.2f}]"
        vd = _verdict(r)
        # flag only large positive gaps (pooled difference larger than the clean one)
        flag = " (large)" if r.leak_gap >= 0.30 and not np.isnan(r.ci_lo) else ""
        # flag a difference resolved by the session unit but not by the objective unit
        od = ""
        if not np.isnan(r.ll_ci_lo):
            s_res = (r.ll_ci_hi < 0) or (r.ll_ci_lo > 0)
            l_res = (r.lo_ll_ci_hi < 0) or (r.lo_ll_ci_lo > 0)
            od = ("  WARNING: resolved by the session unit only (over-confident)"
                  if s_res and not l_res else "")
        print(f"{r.arm:22s} | {r.d_clean_ll:+9.5f} {f4(r.ll_ci_lo, r.ll_ci_hi):>20s} "
              f"{f4(r.lo_ll_ci_lo, r.lo_ll_ci_hi):>21s} | "
              f"{r.d_clean_auc:+10.2f} {f2(r.ci_lo, r.ci_hi):>16s} "
              f"{f2(r.lo_ci_lo, r.lo_ci_hi):>16s} | {r.leak_gap:+7.2f}{flag}  {vd}{od}")
    print(f"\n{'arm':22s} | {'clean LL':>8s} {'clean AUC':>9s} | {'pooled LL':>9s} "
          f"{'pooled AUC':>10s}   (pooled includes the leaked rows; context only)")
    for r in d.itertuples():
        print(f"{r.arm:22s} | {r.clean_ll:8.4f} {r.clean_auc:9.2f} | "
              f"{r.pooled_ll:9.4f} {r.pooled_auc:10.2f}")
    print(f"\nreference = {ref_name}; Δ and CI are on clean rows, paired bootstrap n={args.boot}, "
          f"clustered on session and, separately, on learning_objective_id.")
    print("The LO column decides, because the test's objectives behave as new ones. On these "
          "rows the session column is effectively a row bootstrap (86.6% singleton sessions) "
          "and understates the interval.")
    print("Pooled and clean levels are not comparable: they are different populations.")

    out = ROOT / f"figures/data/rescore_clean{args.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# XGBoost half re-scored on the leakage-free rows\n\n")
        f.write(f"pooled n={len(y):,} (base {y.mean():.4f}) · clean n={int(C.sum()):,} "
                f"(base {y[C].mean():.4f}). Δ and CI on clean rows, paired bootstrap "
                f"n={args.boot}, reference `{ref_name}`.\n\n")
        f.write("Decision metric is **clean log loss** (negative Δ is better); AUROC is reported "
                "alongside, never alone.\n\n")
        f.write("🛑 **The deciding bootstrap unit is `learning_objective_id`.** Both units are "
                f"tabulated: {len(_sz):,} sessions (row-weighted mean size {_rw_sess:.3f} — "
                "86.6% of clean rows are singleton sessions *by construction*, so the "
                "\"session-clustered\" interval is a **row bootstrap** and understates the SE) "
                f"vs {len(_lz):,} objectives (row-weighted mean size {_rw_lo:.1f}). Rule 1: the "
                "hidden test behaves as if its objectives are new, so the objective is the "
                "resampling unit. Session columns are kept only so the existing log stays "
                "comparable.\n\n")
        f.write("| arm | pooled LL | pooled AUROC | clean LL | Δclean LL | 95% CI LL (session) | "
                "**95% CI LL (LO)** | clean AUROC | Δclean AUROC | 95% CI AUROC (session) | "
                "**95% CI AUROC (LO)** | leak gap | verdict (LO) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in d.itertuples():
            g4 = lambda a, b: "—" if np.isnan(a) else f"[{a:+.5f}, {b:+.5f}]"
            g2 = lambda a, b: "—" if np.isnan(a) else f"[{a:+.2f}, {b:+.2f}]"
            vd = "reference"
            if not np.isnan(r.lo_ll_ci_lo):
                vd = ("✅ resolved better" if r.lo_ll_ci_hi < 0 else
                      "🛑 resolved worse" if r.lo_ll_ci_lo > 0 else "null")
                if ((r.ll_ci_hi < 0 or r.ll_ci_lo > 0)
                        and not (r.lo_ll_ci_hi < 0 or r.lo_ll_ci_lo > 0)):
                    vd += " ⚠️ session-only"
            f.write(f"| {r.arm} | {r.pooled_ll:.4f} | {r.pooled_auc:.2f} | {r.clean_ll:.4f} | "
                    f"{r.d_clean_ll:+.5f} | {g4(r.ll_ci_lo, r.ll_ci_hi)} | "
                    f"**{g4(r.lo_ll_ci_lo, r.lo_ll_ci_hi)}** | {r.clean_auc:.2f} | "
                    f"{r.d_clean_auc:+.2f} | {g2(r.ci_lo, r.ci_hi)} | "
                    f"**{g2(r.lo_ci_lo, r.lo_ci_hi)}** | {r.leak_gap:+.2f} | {vd} |\n")
    d.to_csv(ROOT / f"artifacts/rescore_clean{args.tag}.csv", index=False)
    print(f"\n[done] {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
