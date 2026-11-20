#!/usr/bin/env python
"""Distil the Qwen3-32B NTO tutor-move annotator into a CPU classifier and build the
session-level `ntodst_*` block.

The codes come from a Qwen3-32B pass that cannot run in the 6-hour offline competition
runtime. This script applies the recipe of the correctness annotator
(distil_turn_correctness.py): TF-IDF (word + char) -> logistic regression, one binary head
per move family, with per-turn probabilities averaged to session rates. Pure CPU at
inference: no embeddings, no GPU.

What is coded: the NTO label for turn i describes the tutor's utterance in that turn, which
is `turn_text.tutor_q` (verified: `tutor_q(i+1) == tutor_reply(i)`). The head sees the
preceding student turn as context plus that utterance, both available at inference. 24.6%
of turns carry several codes (`|`-separated), so the task is multi-label: one independent
binary head per family, never a softmax.

Leakage controls (as in the correctness distillation):
  - Folds are session-disjoint: a session's turns are always scored by a model that never
    saw that session.
  - Every head's target is the annotation, never `is_correct`, so no fold can carry label
    information.
  - The block is built from out-of-fold predictions; the all-data fit is saved separately
    and never used for any number reported here.
  - Rates are means of soft probabilities, i.e. the expectation of the oracle's
    count / n_turns.

The shipped booster drops the 21 `ntodst_` columns (`--drop-cols ntodst_` in step 05a), but
the classifier ships and its columns are still computed at inference.

In:  artifacts/turn_text.pkl.gz, external/dialogue-kt/results/nto_moves_shard*.csv,
     artifacts/nto_rate_features.csv (oracle rates from step 03b, for the report)
Out: artifacts/nto_turn_probs.pkl.gz     (per-turn out-of-fold probabilities)
     artifacts/nto_dst_features.csv      (session-level `ntodst_*` block)
     artifacts/nto_moves_clf.joblib      (all-data fit, shipped in assets/)
     figures/data/nto_distil_validation.md
With --tag NAME every output gets the suffix _NAME instead.

Usage (reproduce.py step 03c, run from build/workspace/ with single-threaded BLAS,
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, which reproducing the
shipped classifier requires):
    python scripts/distil_nto_moves.py --overwrite-shipped
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_union  # noqa: E402
from magnificat.vocab_order import dump_sorted  # noqa: E402

from build_nto_rate_features import FAMILIES, SHARDS  # noqa: E402

N_FOLDS = 5
FIT_CAP = 250_000     # turns per fold fit (a random subsample above this)
CHUNK = 200_000
MIN_POS = 50          # a head with fewer positives in train falls back to the base rate


def vectorizer(a=None):
    """Build the TF-IDF feature union shared by all heads of one fold.

    Same shape as the correctness distillation: word n-grams for phrasing ("does that make
    sense", "so what if we"), char n-grams for the surface that word tokenisation destroys.
    Tutor utterances are shorter than that script's concatenated turn text, so the per-row
    n-gram count (which sets the sparse matrix size) is smaller and `min_df` can stay tight.

    Every default is the shipped value, so a run without tuning flags rebuilds the shipped
    classifier. char_wb is (4, 4), as in `distil_turn_correctness.vectorizer`; the
    student-state classifier uses (3, 5).

    Args:
        a: parsed command-line arguments whose attributes override the defaults, or None
            for the shipped defaults (distil_v2_moves.py calls it with None).

    Returns:
        An unfitted sklearn FeatureUnion of a word and a char_wb TfidfVectorizer (float32).
    """
    g = (lambda n, d: getattr(a, n, d) if a is not None else d)
    return make_union(
        TfidfVectorizer(ngram_range=tuple(g("word_ngram", (1, 2))),
                        min_df=g("word_min_df", 4),
                        max_features=g("word_max_features", 300_000),
                        sublinear_tf=g("sublinear_tf", True), dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(g("char_ngram", (4, 4))),
                        min_df=g("char_min_df", 10),
                        max_features=g("char_max_features", 150_000),
                        sublinear_tf=g("sublinear_tf", True), dtype=np.float32),
    )


def head(a=None):
    """Return an unfitted logistic-regression head for one move family.

    `C=1.0`, `class_weight=None` and `max_iter=400` are the shipped values; the
    command-line flags override them.

    Args:
        a: parsed command-line arguments overriding C / class_weight / max_iter, or None.

    Returns:
        An unfitted sklearn LogisticRegression.
    """
    g = (lambda n, d: getattr(a, n, d) if a is not None else d)
    return LogisticRegression(C=g("C", 1.0), max_iter=g("max_iter", 400),
                              class_weight=g("class_weight", None))


def _tag(p: Path, tag: str) -> Path:
    """Return `p` with `_<tag>` inserted before its extensions (unchanged if `tag` is empty),
    so an experimental run never overwrites a shipped file."""
    return p if not tag else p.with_name(f"{p.stem.split('.')[0]}_{tag}{''.join(p.suffixes)}")


def main() -> None:
    """Fit one head per NTO move family out of fold, write the per-turn table, the session
    block and the validation report, then fit and save the all-data classifier.

    Returns:
        None. Writes artifacts/nto_turn_probs.pkl.gz, artifacts/nto_dst_features.csv,
        figures/data/nto_distil_validation.md and (unless --no-ship)
        artifacts/nto_moves_clf.joblib, each with the --tag suffix when one is given.
        Exits without writing anything unless --tag or --overwrite-shipped is given.
    """
    import argparse
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test only, not a result")
    ap.add_argument("--families", type=int, default=None, help="smoke only")
    # Tuning flags. The defaults are the shipped configuration.
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", dest="class_weight", default=None,
                    choices=[None, "balanced"])
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=400)
    ap.add_argument("--word-ngram", dest="word_ngram", type=int, nargs=2, default=(1, 2))
    ap.add_argument("--word-min-df", dest="word_min_df", type=int, default=4)
    ap.add_argument("--word-max-features", dest="word_max_features", type=int,
                    default=300_000)
    ap.add_argument("--char-ngram", dest="char_ngram", type=int, nargs=2, default=(4, 4))
    ap.add_argument("--char-min-df", dest="char_min_df", type=int, default=10)
    ap.add_argument("--char-max-features", dest="char_max_features", type=int,
                    default=150_000)
    ap.add_argument("--no-sublinear-tf", dest="sublinear_tf", action="store_false",
                    default=True)
    ap.add_argument("--fit-cap", dest="fit_cap", type=int, default=FIT_CAP,
                    help="0 = use every row")
    ap.add_argument("--turn-text", dest="turn_text", default="turn_text.pkl.gz",
                    help="alternative per-turn table, e.g. a wider truncation window")
    ap.add_argument("--extra-turn-text", dest="extra_turn_text", default=None,
                    help="a second per-turn table with the same (session_id, turn) keys and "
                         "transformed text; every fit row whose transformed text differs is "
                         "added to the fold's fit pool with the same labels. Out-of-fold "
                         "predictions stay on the original text.")
    ap.add_argument("--tag", default="",
                    help="suffix for every output; set it for any experimental run")
    ap.add_argument("--no-ship", action="store_true",
                    help="skip the all-data fit (an experimental run only needs the "
                         "out-of-fold block)")
    ap.add_argument("--overwrite-shipped", action="store_true",
                    help="needed to write the untagged paths; the trained booster was "
                         "fitted on those files, so overwriting them invalidates it")
    args = ap.parse_args()
    fit_cap = args.fit_cap if args.fit_cap and args.fit_cap > 0 else None
    # An experimental run launched without --tag would overwrite the shipped outputs, and
    # nothing downstream would notice until a score came back wrong. Require an explicit
    # choice.
    if not args.tag and not args.overwrite_shipped:
        sys.exit("refusing to write the shipped artifacts: pass --tag <name> for an "
                 "experimental run, or --overwrite-shipped to rebuild the shipped files")

    print(f"CONFIG C={args.C} class_weight={args.class_weight} max_iter={args.max_iter} "
          f"word{tuple(args.word_ngram)}/min_df={args.word_min_df}/max={args.word_max_features} "
          f"char{tuple(args.char_ngram)}/min_df={args.char_min_df}/max={args.char_max_features} "
          f"sublinear={args.sublinear_tf} fit_cap={fit_cap} table={args.turn_text} "
          f"tag={args.tag or '(shipped paths)'}", flush=True)

    tt = pd.read_pickle(ROOT / "artifacts" / args.turn_text, compression="gzip")
    nto = pd.concat([pd.read_csv(p) for p in SHARDS], ignore_index=True) \
            .drop_duplicates(["session_id", "turn"])
    nto["codes"] = nto["codes"].fillna("")
    df = tt.merge(nto, on=["session_id", "turn"], how="inner")
    print(f"coded turns: {len(df):,} / {df.session_id.nunique():,} sessions", flush=True)

    # the coded utterance is the tutor's own turn; give the head the student turn that
    # preceded it as context (both are in the transcript at inference time)
    if args.extra_turn_text:
        _xt = pd.read_pickle(ROOT / "artifacts" / args.extra_turn_text, compression="gzip")
        _xt = _xt[["session_id", "turn", "tutor_q", "student"]].rename(
            columns={"tutor_q": "tutor_q_xt", "student": "student_xt"})
        n0 = len(df)
        df = df.merge(_xt, on=["session_id", "turn"], how="left", validate="one_to_one")
        assert len(df) == n0 and df.tutor_q_xt.notna().all(), \
            f"{args.extra_turn_text} does not cover the coded turns' (session_id, turn) keys"
    df = df.sort_values(["session_id", "turn"]).reset_index(drop=True)
    prev_student = df.groupby("session_id", sort=False)["student"].shift(1).fillna("")
    df["text"] = prev_student + " [T] " + df.tutor_q.fillna("")
    if args.extra_turn_text:
        prev_student_xt = df.groupby("session_id", sort=False)["student_xt"].shift(1).fillna("")
        df["text_xt"] = prev_student_xt + " [T] " + df.tutor_q_xt.fillna("")
        print(f"transformed text differs on {(df.text_xt != df.text).mean():.2%} of coded turns",
              flush=True)

    fams = sorted(FAMILIES)[:args.families] if args.families else sorted(FAMILIES)
    code_of = {c: f for f, cs in FAMILIES.items() for c in cs}
    hit = df.codes.str.split("|").apply(
        lambda cs: {code_of[c.strip()] for c in cs if c.strip() in code_of})
    Y = pd.DataFrame({f: hit.apply(lambda s, f=f: int(f in s)) for f in fams})
    print("family base rates:\n" + Y.mean().round(4).to_string(), flush=True)

    if args.limit:
        keep = np.sort(np.random.default_rng(0).choice(len(df), args.limit, replace=False))
        df, Y = df.iloc[keep].reset_index(drop=True), Y.iloc[keep].reset_index(drop=True)
        print(f"smoke test: {len(df):,} turns; not results", flush=True)

    text = df.text.values
    extra = df.text_xt.values if args.extra_turn_text else None
    groups = df.session_id.values
    oof = pd.DataFrame(index=df.index, columns=fams, dtype=float)

    rng = np.random.default_rng(221)
    shape0 = None
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(text, Y[fams[0]], groups)):
        t0 = time.time()
        fit_idx = (rng.choice(tr, fit_cap, replace=False)
                   if fit_cap and len(tr) > fit_cap else tr)
        # --extra-turn-text: append the typed twin of every fit row whose twin differs;
        # labels repeat by construction (same rows) and the folds stay session-disjoint
        # (same sessions)
        twin = (fit_idx[extra[fit_idx] != text[fit_idx]] if extra is not None
                else fit_idx[:0])
        fit_text = (np.concatenate([text[fit_idx], extra[twin]]) if len(twin)
                    else text[fit_idx])
        # vectorise once per fold and share the matrix across all heads
        vec = vectorizer(args)
        Xtr = vec.fit_transform(fit_text)
        # Print the fit matrix shape, to check a run against the shipped configuration.
        print(f"fold {k + 1}/{N_FOLDS}: vectorised {Xtr.shape[0]:,} rows x "
              f"{Xtr.shape[1]:,} features, nnz {Xtr.nnz:,} "
              f"in {time.time() - t0:.0f}s", flush=True)
        shape0 = shape0 or (Xtr.shape[0], Xtr.shape[1])
        Xte = vec.transform(text[te])
        nz = []
        for f in fams:
            ytr = (np.concatenate([Y[f].values[fit_idx], Y[f].values[twin]]) if len(twin)
                   else Y[f].values[fit_idx])
            if ytr.sum() < MIN_POS:
                oof.loc[te, f] = float(Y[f].values[tr].mean())
                print(f"    {f:22s} too rare ({ytr.sum()}) -> base rate", flush=True)
                continue
            m = head(args)
            m.fit(Xtr, ytr)
            if k == 0:
                nz.append(int((m.coef_ != 0).sum()))
                if not getattr(m, "n_iter_", [0])[0] < args.max_iter:
                    print(f"    WARNING: {f:22s} hit max_iter={args.max_iter} "
                          f"(n_iter={m.n_iter_[0]}), not converged", flush=True)
            oof.loc[te, f] = np.concatenate(
                [m.predict_proba(Xte[i:i + CHUNK])[:, 1] for i in range(0, Xte.shape[0], CHUNK)])
        if nz:
            print(f"  fold 1 non-zero coefficients: mean {np.mean(nz):,.0f} / "
                  f"{Xtr.shape[1]:,}", flush=True)
        print(f"  fold {k + 1} done in {time.time() - t0:.0f}s", flush=True)
        del vec, Xtr, Xte
    print(f"fold 1 fit matrix: rows={shape0[0]:,} features={shape0[1]:,}", flush=True)

    aucs = {f: (roc_auc_score(Y[f], oof[f]) if Y[f].nunique() > 1 and oof[f].nunique() > 1
                else float("nan")) for f in fams}
    print("\nper-family OOF AUC:", flush=True)
    for f in fams:
        print(f"  {f:22s} AUC {aucs[f]:.4f}   base {Y[f].mean():.4f}", flush=True)

    # ------------------------------------------------- persist the per-turn OOF
    # The per-turn probabilities are reusable (step 03d reads them); session rates are one
    # aggregation of them.
    turns = df[["session_id", "turn", "pos", "n_turns"]].copy()
    for f in fams:
        turns[f] = oof[f].astype(np.float32).values
    p_turn = _tag(ROOT / "artifacts/nto_turn_probs.pkl.gz", args.tag)
    turns.to_pickle(p_turn, compression="gzip")
    print(f"wrote {p_turn.name} ({len(turns):,} turns x {len(fams)} families)", flush=True)

    # ---------------------------------------------------------- session-level block
    # mean of soft probabilities == expectation of the oracle's hits/n_turns
    S = oof.assign(session_id=df.session_id.values).groupby("session_id").mean()
    S.columns = [f"ntodst_{c}_rate" for c in S.columns]
    if {"ntodst_prompt_self_expl_rate", "ntodst_explain_conceptual_rate"} <= set(S.columns):
        S["ntodst_elicit_to_tell"] = (
            (S.ntodst_prompt_self_expl_rate + S.ntodst_prompt_next_step_rate
             + S.ntodst_probe_understand_rate + S.ntodst_probe_prior_rate)
            / (S.ntodst_explain_conceptual_rate + S.ntodst_explain_procedural_rate
               + S.ntodst_give_answer_rate + 1e-3))
        S["ntodst_feedback_positive_ratio"] = (
            S.ntodst_fb_correct_rate
            / (S.ntodst_fb_correct_rate + S.ntodst_fb_incorrect_rate + 1e-3))
    out = S.reset_index()
    p_dst = _tag(ROOT / "artifacts/nto_dst_features.csv", args.tag)
    out.to_csv(p_dst, index=False)
    print(f"\nwrote {p_dst.name} "
          f"({len(out):,} sessions x {out.shape[1] - 1} features)", flush=True)
    # For a tagged run, report the mean absolute difference to the shipped block.
    ship = ROOT / "artifacts/nto_dst_features.csv"
    if args.tag and ship.exists():
        base = pd.read_csv(ship).set_index("session_id")
        cols = [c for c in base.columns if c in out.columns]
        d = (out.set_index("session_id")[cols] - base[cols].reindex(out.session_id)).abs()
        print(f"vs shipped block: mean |Δ| {d.mean().mean():.5f}   largest "
              f"{d.mean().idxmax()} {d.mean().max():.5f}", flush=True)

    # session-level correlation of each distilled rate with the oracle rate it imitates
    orc = pd.read_csv(ROOT / "artifacts/nto_rate_features.csv").set_index("session_id")
    cors = {}
    for f in fams:
        a, b = f"ntodst_{f}_rate", f"nto_{f}_rate"
        if b in orc.columns:
            j = S[[a]].join(orc[[b]], how="inner").dropna()
            cors[f] = float(j[a].corr(j[b])) if len(j) > 100 else float("nan")

    md = ["# Distilled NTO tutor-move annotator (LLM codes -> TF-IDF + logistic regression)\n\n",
          "Pure CPU, no GPU at inference. Session-disjoint 5-fold OOF over "
          f"{len(df):,} coded turns; every target is the **annotation**, never "
          "`is_correct`. Rates are means of soft probabilities.\n\n",
          "| family | base rate | OOF AUC | corr(distilled, oracle) session rate |\n",
          "|---|---|---|---|\n"]
    for f in fams:
        md.append(f"| {f} | {Y[f].mean():.4f} | {aucs[f]:.4f} | "
                  f"{cors.get(f, float('nan')):.3f} |\n")
    md.append(f"\nMean OOF AUC {np.nanmean(list(aucs.values())):.4f}; mean session-rate "
              f"correlation with the oracle {np.nanmean(list(cors.values())):.3f}.\n")
    p_md = _tag(ROOT / "figures/data/nto_distil_validation.md", args.tag)
    p_md.write_text("".join(md))
    print(f"wrote {p_md}", flush=True)
    print(f"MEAN OOF AUC {np.nanmean(list(aucs.values())):.4f}", flush=True)

    # ------------------------------------------------------------- shipped classifier
    # A run that only needs the out-of-fold block can skip it (--no-ship) and save ~1/6 of
    # the runtime.
    if args.no_ship:
        print("--no-ship: skipping the all-data fit", flush=True)
        return
    fit_idx = (rng.choice(len(text), min(fit_cap, len(text)), replace=False)
               if fit_cap else np.arange(len(text)))
    twin = (fit_idx[extra[fit_idx] != text[fit_idx]] if extra is not None else fit_idx[:0])
    fit_text = np.concatenate([text[fit_idx], extra[twin]]) if len(twin) else text[fit_idx]
    vec = vectorizer(args)
    Xall = vec.fit_transform(fit_text)
    heads = {}
    for f in fams:
        ytr = (np.concatenate([Y[f].values[fit_idx], Y[f].values[twin]]) if len(twin)
               else Y[f].values[fit_idx])
        if ytr.sum() < MIN_POS:
            continue
        m = head(args)
        m.fit(Xall, ytr)
        heads[f] = m
    p_clf = _tag(ROOT / "artifacts/nto_moves_clf.joblib", args.tag)
    dump_sorted({"vec": vec, "heads": heads}, p_clf, compress=3)
    print(f"wrote {p_clf.name} ({p_clf.stat().st_size / 1e6:.1f} MB, all-data fit, "
          f"{len(heads)} heads)", flush=True)


if __name__ == "__main__":
    main()
