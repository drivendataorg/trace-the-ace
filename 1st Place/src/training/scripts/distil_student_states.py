#!/usr/bin/env python
"""Distil the Qwen3-32B student-state annotator into a CPU classifier and build the
session-level `studst_*` block.

The recipe of distil_turn_correctness.py and distil_nto_moves.py: TF-IDF (word + char) ->
one binary logistic head per code, session-disjoint folds, soft probabilities aggregated to
session level. The 12 `studst_*_rate` columns replace Model E2's student-side regex columns
in the shipped booster (`--swap-student` in train_model_e3l.py).

Two things differ from the NTO distillation, both because the labels cover a sample:

1. The annotation covers ~21% of sessions (annotate_student_states.py in
   external/dialogue-kt/; the classifier needs enough turns to fit, not the whole corpus).
   Labelled sessions get out-of-fold predictions from a model that never saw them;
   unlabelled sessions get the all-data fit. Both match the inference condition (a session
   the annotator has not seen). Using the all-data fit on a labelled session would be
   optimistic, so it is not done. distil_turn_correctness.py makes the same split between
   its feature block and its shipped classifier.
2. Coverage is reported per session, and sessions with no student turns get NaN rather
   than 0: a rate of 0 claims the student never did the thing, which for `EXPLAINS_WHY`
   is a strong and probably false claim.

No fold can leak `is_correct`: every target is the annotation.

In:  external/dialogue-kt/results/student_states_shard*.csv, artifacts/turn_text.pkl.gz,
     artifacts/turn_correctness_probs.pkl.gz and artifacts/nto_turn_probs.pkl.gz (steps
     03a and 03c, for the contingency and uptake columns; skipped if missing),
     artifacts/dialogue_features.csv and artifacts/response_turn_index.csv (report only)
Out: artifacts/stu_turn_probs.pkl.gz     (per-turn out-of-fold / all-data probabilities)
     artifacts/stu_dst_features.csv      (session-level `studst_*` block)
     artifacts/stu_states_clf.joblib     (all-data fit, shipped in assets/)
     figures/data/stu_distil_validation.md
With STUDST_TAG=NAME in the environment every output gets the suffix _NAME instead.

Usage (reproduce.py step 03d, after 03a and 03c; run from build/workspace/ with
single-threaded BLAS, OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, which
reproducing the shipped classifier requires):
    python scripts/distil_student_states.py --overwrite-shipped
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external/dialogue-kt"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_union  # noqa: E402
from magnificat.vocab_order import dump_sorted  # noqa: E402

from annotate_student_states import ALL_CODES  # noqa: E402

SHARDS = sorted((ROOT / "external/dialogue-kt/results").glob("student_states_shard*.csv"))
N_FOLDS = 5
#: Output suffix and fit cap, both set from the environment so an exploratory rebuild never
#: touches the shipped files. STUDST_TAG="" (default) writes the shipped paths;
#: STUDST_TAG=full writes stu_*_full.* next to them.
TAG = os.environ.get("STUDST_TAG", "")
SUF = f"_{TAG}" if TAG else ""
#: Rows the per-code heads are fitted on (STUDST_FIT_CAP; 0 = uncapped).
FIT_CAP = int(os.environ.get("STUDST_FIT_CAP", "250000")) or 10**9
CHUNK = 200_000
MIN_POS = 50

#: Codes that also get early / late / trend columns.
TEMPORAL = ["EXPLAINS_WHY", "CONFUSED", "INSIGHT", "MINIMAL"]


#: Head configuration. Unset (the default) rebuilds the shipped classifier; `HEAD=wide_bal`
#: selects a wider vectoriser and `class_weight="balanced"`.
HEAD = os.environ.get("HEAD", "incumbent")


def vectorizer():
    """Build the TF-IDF feature union for the student-state heads.

    Student turns are shorter than the tutor turns the NTO classifier reads, and much
    shorter than the correctness classifier's three-field text, so word n-grams alone
    would be sparse; the char band carries the terse ones ("idk", "cos", "6?"). HEAD=wide_bal
    selects the wider, non-shipped configuration.

    Returns:
        An unfitted sklearn FeatureUnion of a word and a char_wb TfidfVectorizer (float32).
    """
    if HEAD == "wide_bal":
        return make_union(
            TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=600_000,
                            sublinear_tf=True, dtype=np.float32),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4,
                            max_features=400_000, sublinear_tf=True, dtype=np.float32),
        )
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000,
                        sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=8,
                        max_features=200_000, sublinear_tf=True, dtype=np.float32),
    )


def _head_clf():
    """Return an unfitted per-code logistic-regression head.

    `class_weight="balanced"` only under HEAD=wide_bal, so the shipped classifier is
    rebuilt unchanged when HEAD is unset.
    """
    if HEAD == "wide_bal":
        return LogisticRegression(C=1.0, max_iter=400, class_weight="balanced")
    return LogisticRegression(C=1.0, max_iter=400)


def _text(df):
    """Return the head's input text per row: the last 200 characters of the tutor's
    preceding turn (what was asked), " [S] ", and the first 300 characters of the student's
    turn. Both are in the transcript at inference time."""
    return (df.tutor_q.fillna("").str[-200:] + " [S] " + df.student.fillna("").str[:300]).values


def _predict(vec, heads, text, base):
    """Score texts with the all-data classifier, CHUNK rows at a time.

    Args:
        vec: fitted vectoriser.
        heads: code -> fitted LogisticRegression, or None for a code too rare to fit.
        text: input texts (see `_text`).
        base: code -> base rate, used for the codes whose head is None.

    Returns:
        dict code -> probability array, shape (len(text),).
    """
    out = {}
    X = None
    for i in range(0, len(text), CHUNK):
        Xi = vec.transform(text[i:i + CHUNK])
        X = Xi if X is None else X
        for c, m in heads.items():
            p = np.full(Xi.shape[0], base[c]) if m is None else m.predict_proba(Xi)[:, 1]
            out.setdefault(c, []).append(p)
    return {c: np.concatenate(v) for c, v in out.items()}


def build_session_block(turns, codes):
    """Aggregate per-turn state probabilities to the session-level `studst_*` block.

    Separate from `main` so that `--reuse-turns` can re-run only this part from the saved
    per-turn probabilities, without the ~30 min of fitting. Only the 12 per-code
    `studst_*_rate` columns enter the shipped booster (`--swap-student`).

    Args:
        turns: per-turn table with session_id, turn, pos, n_turns and one probability
            column per code.
        codes: the code columns of `turns` to aggregate.

    Returns:
        DataFrame indexed by session_id: per-code rates, early/late/trend for TEMPORAL
        codes, ratio columns, last-k/recency stats for EXPLAINS_WHY and CONFUSED, and,
        when the step 03a/03c per-turn tables exist, correctness-contingent and
        tutor-prompt uptake columns.
    """
    rel = turns.pos / turns.n_turns.clip(lower=1)
    # ── session-level block ──────────────────────────────────────────────────
    g = turns.groupby("session_id", sort=False)
    S = g[codes].mean()
    S.columns = [f"studst_{c.lower()}_rate" for c in codes]
    for c in TEMPORAL:
        if c not in codes:
            continue
        e = turns[rel <= 0.3].groupby("session_id")[c].mean()
        l = turns[rel >= 0.7].groupby("session_id")[c].mean()
        S[f"studst_{c.lower()}_early"] = e
        S[f"studst_{c.lower()}_late"] = l
        S[f"studst_{c.lower()}_trend"] = l - e
    # the reasoning ladder and the help-seeking depth, as ratios of the per-code rates
    def col(c):
        return S[f"studst_{c.lower()}_rate"]
    if {"EXPLAINS_WHY", "SHOWS_METHOD", "ANSWER_ONLY"} <= set(codes):
        S["studst_reasoning_depth"] = col("EXPLAINS_WHY") / (
            col("EXPLAINS_WHY") + col("SHOWS_METHOD") + col("ANSWER_ONLY") + 1e-3)
    if {"ASKS_CONCEPTUAL", "ASKS_VERIFICATION"} <= set(codes):
        S["studst_help_depth"] = col("ASKS_CONCEPTUAL") / (
            col("ASKS_CONCEPTUAL") + col("ASKS_VERIFICATION") + 1e-3)
    if {"INSIGHT", "CONFUSED"} <= set(codes):
        S["studst_repair_ratio"] = col("INSIGHT") / (col("INSIGHT") + col("CONFUSED") + 1e-3)

    # ── recency / last-k, for EXPLAINS_WHY and CONFUSED ─────────────────────
    # `record_block` (the answer log's statistics) computes these from a sequence of soft
    # probabilities.
    from magnificat.evidence_features import RECORD_STATS, record_block
    # Not "trend": the TEMPORAL loop above already emits `_trend` for these codes, and
    # `DataFrame.join` raises on a duplicate name.
    keep_stats = ["last_1", "last_3", "recency_acc", "half_delta"]
    for c in ("EXPLAINS_WHY", "CONFUSED"):
        if c not in codes:
            continue
        rows_ = []
        for sid, gg in turns.groupby("session_id", sort=False):
            rb = record_block(gg[c].to_numpy(float), gg.pos.to_numpy(float),
                              float(gg.n_turns.iloc[0]) or len(gg), "x")
            rows_.append({"session_id": sid,
                          **{f"studst_{c.lower()}_{s}": rb[f"x_{s}"] for s in keep_stats}})
        add = pd.DataFrame(rows_).set_index("session_id")
        dup = [c for c in add.columns if c in S.columns]
        assert not dup, f"record-block columns collide with existing: {dup}"
        S = S.join(add)

    # ── does the student move up the reasoning ladder across the session ─────
    if {"EXPLAINS_WHY", "ANSWER_ONLY"} <= set(codes):
        lad = turns.assign(_l=turns["EXPLAINS_WHY"] - turns["ANSWER_ONLY"])
        e = lad[rel <= 0.3].groupby("session_id")._l.mean()
        l = lad[rel >= 0.7].groupby("session_id")._l.mean()
        S["studst_ladder_trend"] = l - e

    # ── contingency: student state x whether the student is answering correctly ────
    # For EXPLAINS_WHY, CONFUSED and HEDGED: the state's rate on answered turns graded right
    # vs wrong (p_ans >= 0.5; p_cor >= 0.5 vs < 0.5).
    # `stu_turn_probs` and `turn_correctness_probs` share the (session_id, turn) index
    # exactly, so the join is lossless.
    try:
        tc = pd.read_pickle(ROOT / "artifacts/turn_correctness_probs.pkl.gz",
                            compression="gzip")[["session_id", "turn", "p_ans", "p_cor"]]
        m = turns.merge(tc, on=["session_id", "turn"], how="left")
        ok = (m.p_ans >= 0.5) & m.p_cor.notna()
        for c in ("EXPLAINS_WHY", "CONFUSED", "HEDGED"):
            if c not in codes:
                continue
            right = m[ok & (m.p_cor >= 0.5)].groupby("session_id")[c].mean()
            wrong = m[ok & (m.p_cor < 0.5)].groupby("session_id")[c].mean()
            S[f"studst_{c.lower()}_when_right"] = right
            S[f"studst_{c.lower()}_when_wrong"] = wrong
            # hand the tree the contrast directly (evidence_features does the same for ctg_)
            S[f"studst_{c.lower()}_contrast"] = right - wrong
        print("  + correctness-contingent student states", flush=True)
    except Exception as e:
        print(f"  (contingency skipped: {e})", flush=True)

    # ── uptake: does a tutor prompt produce the student behaviour ──
    # P(student EXPLAINS_WHY at t+1 | tutor prompted self-explanation at t) minus the
    # session's own rate, and the same for probe_understand -> INSIGHT.
    try:
        nt = pd.read_pickle(ROOT / "artifacts/nto_turn_probs.pkl.gz",
                            compression="gzip")[["session_id", "turn", "prompt_self_expl",
                                                 "probe_understand"]]
        m = turns.merge(nt, on=["session_id", "turn"], how="left").sort_values(
            ["session_id", "turn"])
        for mv, code, tag in (("prompt_self_expl", "EXPLAINS_WHY", "self_expl"),
                              ("probe_understand", "INSIGHT", "probe")):
            if code not in codes:
                continue
            nxt = m.groupby("session_id", sort=False)[code].shift(-1)
            w = m[mv].to_numpy(float)
            d = pd.DataFrame({"session_id": m.session_id.values, "w": w,
                              "nxt": nxt.to_numpy(float)}).dropna()
            num_ = d.assign(z=d.w * d.nxt).groupby("session_id").z.sum()
            den_ = d.groupby("session_id").w.sum()
            S[f"studst_uptake_{tag}"] = (num_ / den_.replace(0, np.nan)) - col(code)
        print("  + tutor-prompt uptake", flush=True)
    except Exception as e:
        print(f"  (uptake skipped: {e})", flush=True)

    return S


def main() -> None:
    """Fit one head per student-state code out of fold, score the whole corpus, and write
    the per-turn table, the session block, the classifier and the validation report.

    Returns:
        None. Writes artifacts/stu_states_clf.joblib, artifacts/stu_turn_probs.pkl.gz,
        artifacts/stu_dst_features.csv and figures/data/stu_distil_validation.md (with
        the STUDST_TAG suffix when set). With --reuse-turns it only re-aggregates the
        existing per-turn table into stu_dst_features. Exits without writing anything
        unless STUDST_TAG or --overwrite-shipped is given.
    """
    import argparse
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test only, not a result")
    ap.add_argument("--codes", type=int, default=None, help="smoke only")
    ap.add_argument("--reuse-turns", action="store_true",
                    help="skip loading the annotations and the distillation, and only "
                         "aggregate the existing artifacts/stu_turn_probs.pkl.gz into the "
                         "session block. The classifier and the per-turn probabilities are "
                         "written first and survive a crash in the aggregation, so this saves "
                         "about 30 min of fitting.")
    ap.add_argument("--shards", default=None,
                    help="comma list of shard files; default = every "
                         "student_states_shard*.csv. Name them explicitly to exclude a "
                         "shard that is still being appended to by a running job.")
    ap.add_argument("--overwrite-shipped", action="store_true",
                    help="rebuild the shipped artifacts in place. Without it, and without "
                         "STUDST_TAG, this script refuses to run, because the shipped "
                         "booster's 12 studst_ columns were fitted on these files.")
    args = ap.parse_args()

    # Like the sibling distillers (distil_nto_moves.py, distil_turn_correctness.py), refuse
    # to write the shipped paths without an explicit opt-in: an untagged re-run would
    # overwrite artifacts/stu_turn_probs.pkl.gz, the only input of check_studst_parity.py
    # (the parity gate on the 12 shipped studst_ columns).
    if not TAG and not args.overwrite_shipped:
        sys.exit("refusing to write the shipped artifacts (stu_states_clf.joblib, "
                 "stu_turn_probs.pkl.gz, stu_dst_features.csv): set STUDST_TAG=<name> for an "
                 "experimental run, or pass --overwrite-shipped to rebuild the shipped files")

    if args.reuse_turns:
        # the per-turn probabilities and the classifier are written before the session
        # block, so they survive an aggregation crash; re-run just the aggregation.
        tp = ROOT / f"artifacts/stu_turn_probs{SUF}.pkl.gz"
        turns = pd.read_pickle(tp, compression="gzip")
        codes = [c for c in ALL_CODES if c in turns.columns]
        print(f"reusing {tp.name}: {len(turns):,} turns x {len(codes)} codes "
              f"({int(turns._labelled.sum()):,} from labelled sessions)", flush=True)
        S = build_session_block(turns, codes)
        out = S.reset_index()
        out.to_csv(ROOT / f"artifacts/stu_dst_features{SUF}.csv", index=False)
        print(f"wrote artifacts/stu_dst_features.csv ({len(out):,} sessions x "
              f"{out.shape[1] - 1} features)", flush=True)
        return

    if not SHARDS:
        raise SystemExit("no student_states_shard*.csv found; run the annotation first")
    tt = pd.read_pickle(ROOT / "artifacts/turn_text.pkl.gz", compression="gzip")
    shards = ([ROOT / "external/dialogue-kt/results" / s for s in args.shards.split(",")]
              if args.shards else SHARDS)
    # `on_bad_lines="skip"`: a shard being appended to by a live annotation job can end in
    # a half-written row, and losing one turn of 165k is immaterial next to crashing.
    lab = pd.concat([pd.read_csv(p, on_bad_lines="skip") for p in shards],
                    ignore_index=True).dropna(subset=["session_id", "turn"]) \
            .drop_duplicates(["session_id", "turn"])
    lab["turn"] = lab["turn"].astype(int)
    print(f"shards: {[p.name for p in shards]}", flush=True)
    lab["codes"] = lab["codes"].fillna("")
    df = tt.merge(lab, on=["session_id", "turn"], how="inner")
    df = df.sort_values(["session_id", "turn"]).reset_index(drop=True)
    print(f"labelled turns: {len(df):,} / {df.session_id.nunique():,} sessions "
          f"({df.session_id.nunique() / tt.session_id.nunique():.1%} of the corpus)",
          flush=True)

    codes = ALL_CODES[:args.codes] if args.codes else ALL_CODES
    hit = df.codes.str.split("|").apply(lambda cs: {c.strip() for c in cs if c.strip()})
    Y = pd.DataFrame({c: hit.apply(lambda s, c=c: int(c in s)) for c in codes})
    print("code base rates:\n" + Y.mean().round(4).to_string(), flush=True)

    if args.limit:
        keep = np.sort(np.random.default_rng(0).choice(len(df), args.limit, replace=False))
        df, Y = df.iloc[keep].reset_index(drop=True), Y.iloc[keep].reset_index(drop=True)
        print(f"smoke test: {len(df):,} turns; not results", flush=True)

    text = _text(df)
    groups = df.session_id.values
    oof = pd.DataFrame(index=df.index, columns=codes, dtype=float)

    rng = np.random.default_rng(221)
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(text, Y[codes[0]], groups)):
        t0 = time.time()
        fit_idx = rng.choice(tr, FIT_CAP, replace=False) if len(tr) > FIT_CAP else tr
        vec = vectorizer()
        Xtr = vec.fit_transform(text[fit_idx])
        Xte = vec.transform(text[te])
        print(f"fold {k + 1}/{N_FOLDS}: {Xtr.shape[0]:,} x {Xtr.shape[1]:,} in "
              f"{time.time() - t0:.0f}s", flush=True)
        for c in codes:
            ytr = Y[c].values[fit_idx]
            if ytr.sum() < MIN_POS:
                oof.loc[te, c] = float(Y[c].values[tr].mean())
                print(f"    {c:20s} too rare ({int(ytr.sum())}) -> base rate", flush=True)
                continue
            m = _head_clf()
            m.fit(Xtr, ytr)
            oof.loc[te, c] = np.concatenate(
                [m.predict_proba(Xte[i:i + CHUNK])[:, 1]
                 for i in range(0, Xte.shape[0], CHUNK)])
        print(f"  fold {k + 1} done in {time.time() - t0:.0f}s", flush=True)
        del vec, Xtr, Xte

    aucs = {c: (roc_auc_score(Y[c], oof[c]) if Y[c].nunique() > 1 and oof[c].nunique() > 1
                else float("nan")) for c in codes}
    print("\nper-code OOF AUC:", flush=True)
    for c in codes:
        print(f"  {c:20s} AUC {aucs[c]:.4f}   base {Y[c].mean():.4f}", flush=True)

    # ── the all-data classifier: ships, and scores every unlabelled session ────────
    t0 = time.time()
    fit_idx = (rng.choice(len(df), FIT_CAP, replace=False) if len(df) > FIT_CAP
               else np.arange(len(df)))
    vec = vectorizer()
    Xall = vec.fit_transform(text[fit_idx])
    heads, base = {}, {}
    for c in codes:
        yc = Y[c].values[fit_idx]
        base[c] = float(Y[c].mean())
        heads[c] = None
        if yc.sum() >= MIN_POS:
            heads[c] = _head_clf().fit(Xall, yc)
    dump_sorted({"vec": vec, "heads": heads, "base": base},
                ROOT / f"artifacts/stu_states_clf{SUF}.joblib", compress=3)
    size = (ROOT / f"artifacts/stu_states_clf{SUF}.joblib").stat().st_size / 1e6
    print(f"\nall-data classifier fitted in {time.time() - t0:.0f}s -> "
          f"artifacts/stu_states_clf.joblib ({size:.1f} MB)", flush=True)

    # ── per-turn probabilities for the whole corpus ──────────────────────────
    # Labelled sessions keep their OOF value; everything else gets the all-data
    # classifier. Both are "a session this annotator has not seen"; using the all-data
    # fit on a labelled session would be optimistic.
    lab_keys = set(zip(df.session_id, df.turn))
    tt = tt.sort_values(["session_id", "turn"]).reset_index(drop=True)
    is_lab = pd.Series(list(zip(tt.session_id, tt.turn))).isin(lab_keys).to_numpy()
    print(f"scoring {int((~is_lab).sum()):,} unlabelled turns with the classifier...", flush=True)
    P = pd.DataFrame(index=tt.index, columns=codes, dtype=np.float32)
    pred = _predict(vec, heads, _text(tt[~is_lab]), base)
    for c in codes:
        P.loc[~is_lab, c] = pred[c].astype(np.float32)
    key = pd.MultiIndex.from_arrays([df.session_id, df.turn])
    src = oof.set_index(key)
    tgt = pd.MultiIndex.from_arrays([tt.session_id[is_lab], tt.turn[is_lab]])
    for c in codes:
        P.loc[is_lab, c] = src[c].reindex(tgt).to_numpy(dtype=np.float32)

    turns = tt[["session_id", "turn", "pos", "n_turns"]].copy()
    for c in codes:
        turns[c] = P[c].to_numpy(np.float32)
    turns["_labelled"] = is_lab
    turns.to_pickle(ROOT / f"artifacts/stu_turn_probs{SUF}.pkl.gz", compression="gzip")
    print(f"wrote artifacts/stu_turn_probs.pkl.gz ({len(turns):,} turns x {len(codes)})",
          flush=True)

    S = build_session_block(turns, codes)
    out = S.reset_index()
    out.to_csv(ROOT / f"artifacts/stu_dst_features{SUF}.csv", index=False)
    print(f"wrote artifacts/stu_dst_features.csv ({len(out):,} sessions x "
          f"{out.shape[1] - 1} features)", flush=True)

    # ── validation report: correlation of each distilled rate with its E2 regex twin ──
    md = ["# Distilling the student-state annotator\n\n",
          f"{len(df):,} labelled turns over {df.session_id.nunique():,} sessions "
          f"({df.session_id.nunique() / turns.session_id.nunique():.1%} of the corpus); "
          f"child = TF-IDF (word 1-2 + char 3-5) -> one logistic head per code, "
          f"{N_FOLDS} session-disjoint folds; {size:.1f} MB, pure CPU.\n\n",
          "| code | base rate | OOF AUC |\n|---|---|---|\n"]
    for c in codes:
        md.append(f"| {c} | {Y[c].mean():.4f} | {aucs[c]:.4f} |\n")
    try:
        e2 = pd.read_csv(ROOT / "artifacts/dialogue_features.csv")
        pairs = [("studst_explains_why_rate", "self_explanation_rate"),
                 ("studst_confused_rate", "confusion_rate"),
                 ("studst_insight_rate", "resolution_rate"),
                 ("studst_hedged_rate", "hedge_rate"),
                 ("studst_asks_conceptual_rate", "help_deep_rate"),
                 ("studst_asks_verification_rate", "help_shallow_rate"),
                 ("studst_minimal_rate", "ack_only_rate")]
        # dialogue_features.csv is keyed by response, not session: map responses to
        # sessions and average per session before the join.
        idx = pd.read_csv(ROOT / "artifacts/response_turn_index.csv")[
            ["response_id", "session_id"]].drop_duplicates("response_id")
        e2 = e2.merge(idx, on="response_id", how="inner")
        e2 = e2.groupby("session_id").mean(numeric_only=True).reset_index()
        j = out.merge(e2, on="session_id", how="inner")
        if j.empty:
            raise ValueError("regex-twin join produced no rows")
        md += ["\n## Distilled rate vs the E2 regex twin it re-measures\n\n",
               "Pearson correlation of each distilled rate with the regex column of the same "
               "code.\n\n",
               "| distilled | E2 regex twin | Pearson r |\n|---|---|---|\n"]
        for a, b in pairs:
            if a in j.columns and b in j.columns:
                md.append(f"| {a} | {b} | {j[a].corr(j[b]):.3f} |\n")
    except Exception as e:  # the comparison is optional; the outputs above are written
        md.append(f"\n(regex-twin comparison unavailable: {e})\n")
    (ROOT / f"figures/data/stu_distil_validation{SUF}.md").write_text("".join(md))
    print("wrote figures/data/stu_distil_validation.md")


if __name__ == "__main__":
    main()
