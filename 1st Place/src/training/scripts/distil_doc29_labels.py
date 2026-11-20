#!/usr/bin/env python
"""Distil the interview-codebook annotator (Qwen2.5-14B-Instruct, step 02d) into CPU heads.

The codebook is described in src/training/docs/29_interviews_and_labels.md. Same recipe as
distil_nto_moves.py and distil_turn_correctness.py: TF-IDF (word + char) over the coded
utterance plus the preceding opposite-role turn -> one binary logistic head per label
value. Session-level rates average the per-turn noise away (~100 turns per session), so a
head with mediocre per-turn agreement can still give a usable session-level feature.

Fourteen heads over two passes. The default value of each field is the complement and gets
no head, so the rates are directly interpretable:

    tutor    NORMALIZING_DIFFICULTY, RELEASE_HANDOFF, TRANSFER_PROBE,
             SIMPLIFYING_TO_SUBPROBLEM, MOTIVATING_RELEVANCE,
             SUMMARIZING_PROGRESS, OFFERING_CHOICE          (NONE = complement)
    student  reasoning:  ANSWER_ONLY, REASONING_SHOWN       (NOT_ANSWERING = complement)
             confidence: hedged, unmarked, assertive        (na = complement)
             affect:     INSIGHT_DELIGHT, FRUSTRATION       (NONE = complement)

Leakage controls (as in the two other distillations):
  - Folds are session-disjoint: a session's turns are always scored by a model that never
    saw that session.
  - Every head's target is the annotation, never `is_correct`, so no fold can carry label
    information.
  - The per-turn out-of-fold probabilities are what is saved; the all-data fit is saved
    separately and only used to score sessions the annotator never saw.
  - Rates are means of soft probabilities, i.e. the expectation of the oracle's count / n.

The session-level `d29dst_` block is built from both outputs by
`build_doc29_features.py --mode distilled` (step 04a).

In:  artifacts/doc29_records_14b/doc29_raw_records_14b.parquet
     data/train_transcripts/
Out: artifacts/doc29_turn_probs.pkl.gz   per-turn out-of-fold probabilities
     artifacts/doc29_clf.joblib          all-data fit, shipped in assets/
     figures/data/doc29_distil_validation.md

Usage (reproduce.py step 03e, run from build/workspace/ with single-threaded BLAS,
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, which reproducing the
shipped classifier requires):
    python scripts/distil_doc29_labels.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_union  # noqa: E402

from magnificat import data  # noqa: E402
from magnificat.vocab_order import dump_sorted  # noqa: E402

N_FOLDS = 5
FIT_CAP = 250_000     # turns per fold fit; larger training sets are subsampled to this
CHUNK = 200_000
MIN_POS = 50          # a head with fewer positives in train falls back to the base rate

#: (pass, field, value) per head. The default value of each field is the complement and
#: gets no head.
HEADS = [
    ("doc29_tutor", "code", "NORMALIZING_DIFFICULTY"),
    ("doc29_tutor", "code", "RELEASE_HANDOFF"),
    ("doc29_tutor", "code", "TRANSFER_PROBE"),
    ("doc29_tutor", "code", "SIMPLIFYING_TO_SUBPROBLEM"),
    ("doc29_tutor", "code", "MOTIVATING_RELEVANCE"),
    ("doc29_tutor", "code", "SUMMARIZING_PROGRESS"),
    ("doc29_tutor", "code", "OFFERING_CHOICE"),
    ("doc29_student", "reasoning", "ANSWER_ONLY"),
    ("doc29_student", "reasoning", "REASONING_SHOWN"),
    ("doc29_student", "confidence", "hedged"),
    ("doc29_student", "confidence", "unmarked"),
    ("doc29_student", "confidence", "assertive"),
    ("doc29_student", "affect", "INSIGHT_DELIGHT"),
    ("doc29_student", "affect", "FRUSTRATION"),
]


def head_name(pass_name: str, field: str, value: str) -> str:
    """Column name of a head: "tut_" or "stu_" plus the lower-cased label value."""
    side = "tut" if pass_name == "doc29_tutor" else "stu"
    return f"{side}_{value.lower()}"


def vectorizer():
    """Build the TF-IDF feature union shared by the heads of one pass.

    Same shape as `distil_nto_moves.vectorizer`: word n-grams for phrasing ("now you try",
    "I think it's"), char n-grams for the surface that word tokenisation destroys. The input
    is a single utterance plus one turn of context, so the per-row n-gram count stays small
    and `min_df` can stay tight.

    Returns:
        An unfitted sklearn FeatureUnion of a word and a char_wb TfidfVectorizer (float32).
    """
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_features=300_000,
                        sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=10,
                        max_features=150_000, sublinear_tf=True, dtype=np.float32),
    )


#: `TranscriptStore.__init__` globs all 22,821 files, and build_doc29_features.py calls
#: `build_text` once per batch per pass (~100 times), so stores are cached by path.
_STORES: dict[str, "data.TranscriptStore"] = {}


def _store(transcripts: Path):
    key = str(transcripts)
    if key not in _STORES:
        _STORES[key] = data.TranscriptStore(transcripts)
    return _STORES[key]


def build_text(records: pd.DataFrame, transcripts: Path, role: str,
               marker: str) -> pd.DataFrame:
    """Attach each coded utterance's own text plus the preceding opposite-role turn.

    Both come from the transcript alone, so the same input can be built at inference
    without any annotation.

    Args:
        records: annotation records of one pass, with session_id and utterance_id columns.
        transcripts: folder of per-session transcript CSVs.
        role: role of the coded utterances, "tutor" or "student".
        marker: tag placed between context and utterance ("T" or "S").

    Returns:
        `records` rows whose utterance was found in its transcript, plus a `text` column
        "<previous other-role turn> [marker] <utterance>"; an empty DataFrame if none was.
        Sessions whose transcript cannot be loaded are skipped.
    """
    store = _store(transcripts)
    other = "student" if role == "tutor" else "tutor"
    out = []
    for sid, grp in records.groupby("session_id", sort=False):
        try:
            df = store.load(sid)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.sort_values("utterance_id").reset_index(drop=True)
        df["content"] = df["content"].fillna("").astype(str)
        # most recent preceding utterance by the OTHER role, per utterance
        prev, last = [], ""
        for r, c in zip(df["role"].astype(str), df["content"]):
            prev.append(last)
            if r == other:
                last = c
        df["_prev"] = prev
        m = df.set_index("utterance_id")
        own = grp["utterance_id"].map(m["content"])
        ctx = grp["utterance_id"].map(m["_prev"])
        g = grp.copy()
        g["text"] = ctx.fillna("") + f" [{marker}] " + own.fillna("")
        g["_has_text"] = own.notna()
        out.append(g)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    return res[res["_has_text"]].drop(columns=["_has_text"]).reset_index(drop=True)


def distil_pass(records: pd.DataFrame, transcripts: Path, pass_name: str,
                heads_spec: list[tuple[str, str, str]], limit: int | None):
    """Distil one annotation pass: out-of-fold heads, then an all-data fit.

    Args:
        records: all annotation records (both passes; filtered on annotation_type here).
        transcripts: folder of per-session transcript CSVs.
        pass_name: "doc29_tutor" or "doc29_student".
        heads_spec: the HEADS entries of this pass.
        limit: smoke-test subsample size in turns, or None for all.

    Returns:
        (turns, bundle, aucs, base): per-turn out-of-fold probabilities (session_id,
        utterance_id, position_frac, one column per head, annotation_type); the all-data
        fit as {"vec": vectoriser, "heads": {name: LogisticRegression}} (heads with fewer
        than MIN_POS positives are left out); OOF AUC per head; base rate per head.
    """
    role = "tutor" if pass_name == "doc29_tutor" else "student"
    marker = "T" if role == "tutor" else "S"
    df = records[records["annotation_type"] == pass_name]
    if df.empty:
        raise SystemExit(f"no records for pass {pass_name}")
    t0 = time.time()
    df = build_text(df, transcripts, role, marker)
    print(f"[{pass_name}] {len(df):,} coded turns / {df.session_id.nunique():,} sessions "
          f"(text built in {time.time()-t0:.0f}s)", flush=True)

    names = [head_name(*h) for h in heads_spec]
    Y = pd.DataFrame({head_name(p, f, v): (df[f] == v).astype(int).values
                      for p, f, v in heads_spec}, index=df.index)
    print("  base rates:\n" + Y.mean().round(4).to_string(), flush=True)

    if limit and len(df) > limit:
        keep = np.sort(np.random.default_rng(0).choice(len(df), limit, replace=False))
        df, Y = df.iloc[keep].reset_index(drop=True), Y.iloc[keep].reset_index(drop=True)
        print(f"  smoke test: {len(df):,} turns; not results", flush=True)

    text = df["text"].values
    groups = df["session_id"].values
    oof = pd.DataFrame(index=df.index, columns=names, dtype=float)

    rng = np.random.default_rng(221)
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(text, Y[names[0]], groups)):
        t0 = time.time()
        fit_idx = rng.choice(tr, FIT_CAP, replace=False) if len(tr) > FIT_CAP else tr
        # vectorise once per fold and share the matrix across every head
        vec = vectorizer()
        Xtr = vec.fit_transform(text[fit_idx])
        Xte = vec.transform(text[te])
        print(f"  fold {k+1}/{N_FOLDS}: {Xtr.shape[0]:,} x {Xtr.shape[1]:,} "
              f"in {time.time()-t0:.0f}s", flush=True)
        for h in names:
            ytr = Y[h].values[fit_idx]
            if ytr.sum() < MIN_POS:
                oof.loc[te, h] = float(Y[h].values[tr].mean())
                print(f"    {h:26s} too rare ({int(ytr.sum())}) -> base rate", flush=True)
                continue
            m = LogisticRegression(C=1.0, max_iter=400)
            m.fit(Xtr, ytr)
            oof.loc[te, h] = np.concatenate(
                [m.predict_proba(Xte[i:i+CHUNK])[:, 1]
                 for i in range(0, Xte.shape[0], CHUNK)])
        del vec, Xtr, Xte

    aucs = {h: (roc_auc_score(Y[h], oof[h]) if Y[h].nunique() > 1 and oof[h].nunique() > 1
                else float("nan")) for h in names}
    print(f"\n[{pass_name}] per-head OOF AUC:", flush=True)
    for h in names:
        print(f"  {h:26s} AUC {aucs[h]:.4f}   base {Y[h].mean():.4f}", flush=True)

    # ------------------------------------------------------------ shipped classifier
    fit_idx = rng.choice(len(text), min(FIT_CAP, len(text)), replace=False)
    vec = vectorizer()
    Xall = vec.fit_transform(text[fit_idx])
    fitted = {}
    for h in names:
        ytr = Y[h].values[fit_idx]
        if ytr.sum() < MIN_POS:
            continue
        m = LogisticRegression(C=1.0, max_iter=400)
        m.fit(Xall, ytr)
        fitted[h] = m

    turns = df[["session_id", "utterance_id", "position_frac"]].copy()
    for h in names:
        turns[h] = oof[h].astype(np.float32).values
    turns["annotation_type"] = pass_name
    return turns, {"vec": vec, "heads": fitted}, aucs, Y.mean().to_dict()


def main() -> None:
    """Distil the tutor and the student pass and write the combined outputs.

    Returns:
        None. Writes --out-probs (default artifacts/doc29_turn_probs.pkl.gz), --out-clf
        (default artifacts/doc29_clf.joblib, a dict pass_name -> {"vec", "heads"}) and
        figures/data/doc29_distil_validation.md.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=ROOT / "artifacts/doc29_records_14b/doc29_raw_records_14b.parquet")
    # In the workspace the training transcripts are under data/.
    ap.add_argument("--transcripts", type=Path,
                    default=ROOT / "data/train_transcripts")
    ap.add_argument("--out-probs", type=Path, default=ROOT / "artifacts/doc29_turn_probs.pkl.gz")
    ap.add_argument("--out-clf", type=Path, default=ROOT / "artifacts/doc29_clf.joblib")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only, not a result")
    args = ap.parse_args()

    recs = pd.read_parquet(args.records)
    print(f"records: {len(recs):,} over {recs.session_id.nunique():,} sessions", flush=True)

    all_turns, bundle, all_aucs, all_base = [], {}, {}, {}
    for pass_name in ("doc29_tutor", "doc29_student"):
        spec = [h for h in HEADS if h[0] == pass_name]
        turns, clf, aucs, base = distil_pass(recs, args.transcripts, pass_name, spec,
                                             args.limit)
        all_turns.append(turns)
        bundle[pass_name] = clf
        all_aucs.update(aucs)
        all_base.update(base)

    probs = pd.concat(all_turns, ignore_index=True)
    probs.to_pickle(args.out_probs, compression="gzip")
    print(f"\nwrote {args.out_probs} ({len(probs):,} turns)", flush=True)

    dump_sorted(bundle, args.out_clf, compress=3)
    mb = args.out_clf.stat().st_size / 1e6
    n_heads = sum(len(v["heads"]) for v in bundle.values())
    print(f"wrote {args.out_clf} ({mb:.1f} MB, all-data fit, {n_heads} heads)", flush=True)

    md = ["# Distilled docs/29 annotator (LLM labels -> TF-IDF + logistic regression)\n\n",
          "Makes the interview-derived labels shippable: pure CPU, no GPU at inference. ",
          "Session-disjoint 5-fold OOF; every target is the **annotation**, never ",
          "`is_correct`. Rates are means of soft probabilities.\n\n",
          "| head | base rate | OOF AUC |\n|---|---|---|\n"]
    for h, a in all_aucs.items():
        md.append(f"| {h} | {all_base[h]:.4f} | {a:.4f} |\n")
    md.append(f"\nMean OOF AUC {np.nanmean(list(all_aucs.values())):.4f} over "
              f"{len(all_aucs)} heads.\n")
    (ROOT / "figures/data").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures/data/doc29_distil_validation.md").write_text("".join(md))
    print("wrote figures/data/doc29_distil_validation.md", flush=True)


if __name__ == "__main__":
    main()
