#!/usr/bin/env python
"""Distil the Qwen3.5-122B per-turn correctness annotator into a CPU classifier and build
the `recdst_*` performance-record block from it.

The `recdst_*` block is an explicit right/wrong tally of the student's answers. The labels
come from a Qwen3.5-122B annotation pass, which cannot run inside the 6-hour offline
competition runtime, so the shipped model produces the tally with a cheaper classifier.

This script distils the annotator: TF-IDF (word + char) over the turn's text -> logistic
regression, with two heads because the annotator makes two decisions:
  p_ans  P(the turn contains an answerable attempt)   ('na' vs true/false)
  p_cor  P(correct | answerable)
The validation report gives the out-of-fold AUC of both heads and the agreement of the
thresholded correctness head with the 122B labels.

Leakage controls:
  - Folds are session-disjoint, so a session's turns are always scored by a model that
    never saw that session. The heads' target is the annotation, never `is_correct`, so
    cross-fold contamination is the only way information could leak.
  - The answer log is built from the out-of-fold predictions, not the in-sample ones.
  - A final all-data fit is saved separately. It is the classifier the submission ships
    and is never used for the numbers reported here.

The `recdst_*` block is built from the soft p_cor probabilities, with no threshold on p_cor.

In:  artifacts/turn_text.pkl.gz, artifacts/evidence_features.csv,
     external/dialogue-kt/results/traceace_turn_labels_qwen35_122b.csv
Out: artifacts/turn_correctness_probs.pkl.gz (per-turn out-of-fold p_ans, p_cor)
     artifacts/evidence_features_full.csv    (evidence_features.csv + the recdst_ block)
     artifacts/turn_correctness_clf.joblib   (all-data fit, shipped)
     figures/data/evidence_distil_validation.md
With --tag NAME the three artifacts/ outputs get the suffix _NAME instead.

Usage (reproduce.py step 03a, run from build/workspace/ with single-threaded BLAS,
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, which reproducing the
shipped classifier requires):
    python scripts/distil_turn_correctness.py --overwrite-shipped
"""
from __future__ import annotations

import sys
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
from sklearn.pipeline import make_pipeline, make_union  # noqa: E402
from magnificat.vocab_order import dump_sorted  # noqa: E402

from build_evidence_features import record_block, LABELS_122B  # noqa: E402

N_FOLDS = 5
FIT_CAP = 350_000      # turns per fold fit (a random subsample above this)
CHUNK = 150_000        # rows per predict call


_A = None  # set from main(); the zero-arg helpers below read the command-line flags from it


def vectorizer():
    """Build the TF-IDF feature union both heads are fitted on.

    Word n-grams catch the phrasing ("i think", "not sure", "carry the one"); char n-grams
    catch the arithmetic surface (digit patterns, operators) that word tokenisation
    destroys. char_wb is (4, 4); at ~600 characters per document, the n-gram count per row,
    not the vocabulary, sets the size of the sparse matrix. The defaults are the shipped
    values; command-line flags (in `_A`) override them.

    Returns:
        An unfitted sklearn FeatureUnion of a word and a char_wb TfidfVectorizer (float32).
    """
    g = (lambda n, d: getattr(_A, n, d) if _A is not None else d)
    return make_union(
        TfidfVectorizer(ngram_range=tuple(g("word_ngram", (1, 2))),
                        min_df=g("word_min_df", 4),
                        max_features=g("word_max_features", 300_000),
                        sublinear_tf=g("sublinear_tf", True), dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(g("char_ngram", (4, 4))),
                        min_df=g("char_min_df", 10),
                        max_features=g("char_max_features", 200_000),
                        sublinear_tf=g("sublinear_tf", True), dtype=np.float32),
    )


def clf():
    """Return an unfitted TF-IDF + logistic-regression pipeline for one binary head."""
    g = (lambda n, d: getattr(_A, n, d) if _A is not None else d)
    return make_pipeline(vectorizer(),
                         LogisticRegression(C=g("C", 1.0), max_iter=g("max_iter", 1000),
                                            class_weight=g("class_weight", None)))


def predict_chunked(m, text):
    """Positive-class probabilities of a fitted pipeline, predicted CHUNK rows at a time."""
    return np.concatenate([m.predict_proba(text[i:i + CHUNK])[:, 1]
                           for i in range(0, len(text), CHUNK)])


def _tag(rel: str) -> Path:
    """Return ROOT / rel, with the --tag suffix inserted before the extensions if one is set.

    An experimental run must never overwrite the shipped files, which the trained booster's
    `recdst_*` columns were built from.
    """
    q = ROOT / rel
    t = getattr(_A, "tag", "") or ""
    return q if not t else q.with_name(f"{q.stem.split('.')[0]}_{t}{''.join(q.suffixes)}")


def _with_twins(text, y, idx, extra_text):
    """Fit rows plus their typed-register twins (--extra-turn-text).

    `extra_text` is row-aligned with `text`, so a twin inherits its row's label and session
    and the folds stay session-disjoint. Prediction always uses the raw text (inference sees
    raw transcripts); the twins are fit-time augmentation only.

    Args:
        text: per-turn input texts, shape (n,).
        y: binary targets, shape (n,).
        idx: indices of the rows to fit on.
        extra_text: transformed texts row-aligned with `text`, or None for no augmentation.

    Returns:
        (fit_text, fit_y): the rows at `idx`, followed by the twin of every row in `idx`
        whose twin text differs from the original.
    """
    if extra_text is None:
        return text[idx], y[idx]
    twin = idx[extra_text[idx] != text[idx]]
    return (np.concatenate([text[idx], extra_text[twin]]),
            np.concatenate([y[idx], y[twin]]))


def oof_head(text, y, groups, name, rng=np.random.default_rng(221), extra_text=None):
    """Session-disjoint out-of-fold predictions of one binary head.

    Args:
        text: per-turn input texts, shape (n,).
        y: binary targets (annotation labels), shape (n,).
        groups: session_id per row; folds are GroupKFold(N_FOLDS) on it.
        name: head name used in the progress lines.
        rng: generator for subsampling each fold's fit rows to --fit-cap.
        extra_text: optional typed-register twins (see `_with_twins`).

    Returns:
        (p, auc): out-of-fold probabilities, shape (n,), and their ROC AUC against `y`.
    """
    p = np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(text, y, groups)):
        cap = getattr(_A, 'fit_cap', FIT_CAP) or None
        if cap and len(tr) > cap:
            tr = rng.choice(tr, cap, replace=False)
        ft, fy = _with_twins(text, y, tr, extra_text)
        m = clf()
        m.fit(ft, fy)
        p[te] = predict_chunked(m, text[te])
        del m
        print(f"    {name} fold {k + 1}/{N_FOLDS} done (fit n={len(ft):,}"
              f"{'' if extra_text is None else f', {len(ft) - len(tr):,} transformed copies'})",
              flush=True)
    auc = roc_auc_score(y, p)
    print(f"  {name}: OOF AUC {auc:.4f}  (n={len(y):,}, base {y.mean():.4f})", flush=True)
    return p, auc


def main() -> None:
    """Fit both heads out of fold, write the per-turn table and the answer-log block, then
    fit and save the all-data classifier.

    Returns:
        None. Writes artifacts/turn_correctness_probs.pkl.gz,
        artifacts/evidence_features_full.csv and artifacts/turn_correctness_clf.joblib
        (with the --tag suffix when one is given) and
        figures/data/evidence_distil_validation.md, and prints the same report. Exits
        without writing anything unless --tag or --overwrite-shipped is given.
    """
    import argparse
    import joblib

    global _A
    # Same flags as distil_nto_moves.py. Every default below is the shipped value, so a run
    # without tuning flags rebuilds the shipped classifier. `--fit-cap 0` uses every row.
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", dest="class_weight", default=None,
                    choices=[None, "balanced"])
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=1000)
    ap.add_argument("--word-ngram", dest="word_ngram", type=int, nargs=2, default=(1, 2))
    ap.add_argument("--word-min-df", dest="word_min_df", type=int, default=4)
    ap.add_argument("--word-max-features", dest="word_max_features", type=int, default=300_000)
    ap.add_argument("--char-ngram", dest="char_ngram", type=int, nargs=2, default=(4, 4))
    ap.add_argument("--char-min-df", dest="char_min_df", type=int, default=10)
    ap.add_argument("--char-max-features", dest="char_max_features", type=int, default=200_000)
    ap.add_argument("--no-sublinear-tf", dest="sublinear_tf", action="store_false", default=True)
    ap.add_argument("--fit-cap", dest="fit_cap", type=int, default=FIT_CAP,
                    help="0 = every row")
    ap.add_argument("--turn-text", dest="turn_text", default="turn_text.pkl.gz",
                    help="alternative per-turn table (same convention as distil_nto_moves)")
    ap.add_argument("--extra-turn-text", dest="extra_turn_text", default=None,
                    help="a second per-turn table with the same (session_id, turn) keys and "
                         "transformed text; every fit row whose transformed text differs is "
                         "added to the fit pool with the same label. Out-of-fold prediction, "
                         "the answer log and the probs pickle stay on the original text.")
    ap.add_argument("--tag", default="",
                    help="suffix for every output; set it for any experimental run")
    ap.add_argument("--overwrite-shipped", action="store_true")
    _A = ap.parse_args()
    if not _A.tag and not _A.overwrite_shipped:
        raise SystemExit("refusing to write the shipped artifacts: pass --tag <name> for an "
                         "experimental run, or --overwrite-shipped to rebuild the shipped "
                         "files (the trained booster reads evidence_features_full.csv)")
    print(f"CONFIG C={_A.C} cw={_A.class_weight} word{tuple(_A.word_ngram)}/"
          f"min_df={_A.word_min_df} char{tuple(_A.char_ngram)}/min_df={_A.char_min_df} "
          f"sublinear={_A.sublinear_tf} fit_cap={_A.fit_cap or 'ALL'} "
          f"tag={_A.tag or '(shipped paths)'}", flush=True)

    tt = pd.read_pickle(ROOT / "artifacts" / _A.turn_text, compression="gzip")
    print(f"turn text: {len(tt):,} student turns / {tt.session_id.nunique():,} sessions "
          f"({_A.turn_text})", flush=True)
    if _A.extra_turn_text:
        _xt = pd.read_pickle(ROOT / "artifacts" / _A.extra_turn_text, compression="gzip")
        _xt = _xt[["session_id", "turn", "tutor_q", "student", "tutor_reply"]].rename(
            columns={c: c + "_xt" for c in ("tutor_q", "student", "tutor_reply")})
        n0 = len(tt)
        tt = tt.merge(_xt, on=["session_id", "turn"], how="left", validate="one_to_one")
        assert len(tt) == n0 and tt.tutor_q_xt.notna().all(), \
            f"{_A.extra_turn_text} does not cover the base table's (session_id, turn) keys"
        print(f"  extra table {_A.extra_turn_text}: joined 1:1", flush=True)

    lab = pd.read_csv(LABELS_122B).drop_duplicates(["session_id", "turn"])
    tt = tt.merge(lab, on=["session_id", "turn"], how="left")
    tt["label"] = tt["label"].fillna("missing")
    print(f"  label join: {(tt.label != 'missing').mean():.1%} of turns annotated",
          flush=True)

    # the annotator sees the question it is grading against, the answer, and the
    # tutor's reaction; all three are in the transcript at inference time too
    tt["text"] = (tt.tutor_q.fillna("") + " [A] " + tt.student.fillna("")
                  + " [R] " + tt.tutor_reply.fillna(""))
    if _A.extra_turn_text:
        tt["text_xt"] = (tt.tutor_q_xt.fillna("") + " [A] " + tt.student_xt.fillna("")
                         + " [R] " + tt.tutor_reply_xt.fillna(""))
        _diff = (tt.text_xt != tt.text).mean()
        print(f"  transformed text differs on {_diff:.2%} of turns", flush=True)

    known = tt[tt.label != "missing"].reset_index(drop=True)
    txt = known.text.values
    grp = known.session_id.values
    xt = known.text_xt.values if _A.extra_turn_text else None

    print("\nhead 1/2: answerability (na vs true/false)", flush=True)
    y_ans = (known.label != "na").to_numpy(int)
    p_ans, auc_ans = oof_head(txt, y_ans, grp, "p_ans", extra_text=xt)

    print("\nhead 2/2: correctness (true vs false, answerable only)", flush=True)
    ans = known[known.label != "na"].reset_index(drop=True)
    y_cor = (ans.label == "true").to_numpy(int)
    xt_ans = ans.text_xt.values if _A.extra_turn_text else None
    p_cor_ans, auc_cor = oof_head(ans.text.values, y_cor, ans.session_id.values, "p_cor",
                                  extra_text=xt_ans)

    # scatter p_cor back onto every annotated turn; unannotated turns (0.1%) drop out
    known["p_ans"] = p_ans
    known["p_cor"] = np.nan
    known.loc[known.label != "na", "p_cor"] = p_cor_ans

    # ------------------------------------------------- persist the per-turn OOF
    # The per-turn probabilities are also read by step 03d (distil_student_states.py); the
    # answer log below is one aggregation of them.
    known[["session_id", "turn", "pos", "n_turns", "p_ans", "p_cor"]].to_pickle(
        _tag("artifacts/turn_correctness_probs.pkl.gz"), compression="gzip")
    print(f"wrote artifacts/turn_correctness_probs.pkl.gz ({len(known):,} turns)",
          flush=True)

    # ------------------------------------------------------------ the answer log
    use = known[(known.p_ans >= 0.5) & known.p_cor.notna()].sort_values(["session_id", "pos"])
    print(f"\nsoft answer log: {len(use):,} turns kept (p_ans>=0.5) over "
          f"{use.session_id.nunique():,} sessions", flush=True)

    rows = []
    for sid, g in use.groupby("session_id", sort=False):
        rows.append({"session_id": sid} | record_block(
            g.p_cor.tolist(), g.pos.tolist(), int(g.n_turns.iloc[0]), "recdst"))
    rec = pd.DataFrame(rows)

    ev = pd.read_csv(ROOT / "artifacts/evidence_features.csv")
    full = ev.merge(rec, on="session_id", how="left")
    dst = _tag("artifacts/evidence_features_full.csv")
    full.to_csv(dst, index=False)
    print(f"wrote {dst}  ({full.shape[0]:,} sessions x {full.shape[1] - 1} features)")

    # -------------------------------------------------- ship artifact + report
    rng = np.random.default_rng(7)
    cap = getattr(_A, 'fit_cap', FIT_CAP) or len(txt)
    sa = rng.choice(len(txt), min(cap, len(txt)), replace=False)
    sc = rng.choice(len(ans), min(cap, len(ans)), replace=False)
    fa, ya = _with_twins(txt, y_ans, sa, xt)
    fc, yc = _with_twins(ans.text.values, y_cor, sc, xt_ans if _A.extra_turn_text else None)
    m_ans, m_cor = clf(), clf()
    m_ans.fit(fa, ya)
    m_cor.fit(fc, yc)
    dump_sorted({"p_ans": m_ans, "p_cor": m_cor},
                _tag("artifacts/turn_correctness_clf.joblib"), compress=3)
    size = (_tag("artifacts/turn_correctness_clf.joblib")).stat().st_size / 1e6
    print(f"wrote artifacts/turn_correctness_clf.joblib ({size:.1f} MB, all-data fit)")

    # agreement (accuracy, Cohen's kappa) of the thresholded correctness head with the 122B
    # labels
    acc = float(((p_cor_ans >= 0.5).astype(int) == y_cor).mean())
    pe = ((p_cor_ans >= 0.5).mean() * y_cor.mean()
          + (1 - (p_cor_ans >= 0.5).mean()) * (1 - y_cor.mean()))
    kappa = (acc - pe) / (1 - pe)
    txt_out = [
        "# Distilled per-turn correctness annotator (122B -> TF-IDF + logistic regression)\n\n",
        "Session-disjoint 5-fold OOF. Targets are the **122B annotation**, never `is_correct`,\n",
        "so no target information can cross folds.\n\n",
        "| head | n | base rate | OOF AUC |\n|---|---|---|---|\n",
        f"| answerability (na vs true/false) | {len(y_ans):,} | {y_ans.mean():.3f} | **{auc_ans:.4f}** |\n",
        f"| correctness (true vs false) | {len(y_cor):,} | {y_cor.mean():.3f} | **{auc_cor:.4f}** |\n\n",
        f"Thresholded at 0.5 the correctness head agrees with the 122B labels "
        f"**{acc:.3f}** of the time, Cohen's kappa **{kappa:.3f}**. The distilled head covers "
        f"**{len(use) / max(1, len(tt)):.1%}** of all student turns.\n\n",
        f"Ship artifact: `turn_correctness_clf.joblib`, {size:.1f} MB, pure CPU.\n",
    ]
    (ROOT / "figures/data/evidence_distil_validation.md").write_text("".join(txt_out))
    print("".join(txt_out))


if __name__ == "__main__":
    main()
