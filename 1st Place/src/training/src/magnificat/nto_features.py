"""Distilled NTO tutor-move rates, computed from transcripts at inference.

The training-side block (`artifacts/nto_dst_features.csv`) is built from the
session-disjoint out-of-fold predictions of `scripts/distil_nto_moves.py`. This module
is the inference twin: it applies the shipped all-data fit
(`artifacts/nto_moves_clf.joblib`, 18.4 MB, CPU only) to unseen test sessions. The
asymmetry is intended and follows the `recdst_*` block in
:mod:`magnificat.evidence_features`: at inference the classifier has not seen the test
sessions either, so out-of-fold quality is what the booster was trained on.

Column order is fixed by :data:`NTODST_COLS`. Training and inference must agree, because
the booster reads columns by position, not by name.

Each session is processed independently (no pooling across test samples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence_features import build_turns

#: move families, in the order `distil_nto_moves.py` wrote them (sorted)
NTO_FAMILIES = ["clarify", "encourage", "explain_conceptual", "explain_procedural",
                "fb_correct", "fb_incorrect", "fb_neutral", "give_answer", "hint",
                "praise", "probe_prior", "probe_understand", "prompt_next_step",
                "prompt_self_correct", "prompt_self_expl", "rapport", "revoicing",
                "strategize", "technical"]
#: the two ratios E2 already builds from regex, rebuilt on the distilled coding
NTODST_COLS = ([f"ntodst_{f}_rate" for f in NTO_FAMILIES]
               + ["ntodst_elicit_to_tell", "ntodst_feedback_positive_ratio"])

_ELICIT = ["prompt_self_expl", "prompt_next_step", "probe_understand", "probe_prior"]
_TELL = ["explain_conceptual", "explain_procedural", "give_answer"]


def move_text(turns, i: int) -> str:
    """Build the input text the distilled heads were trained on for one tutor turn.

    The text is the tutor utterance being coded, with the preceding student turn as
    context. It must match `distil_nto_moves.py` (`prev_student + " [T] " + tutor_q`),
    or the vectoriser sees a different distribution than it was fit on.

    Args:
        turns: list of turn dicts from :func:`magnificat.evidence_features.build_turns`,
            each with ``"teacher"`` and ``"student"`` text.
        i: index of the tutor turn to code.

    Returns:
        The string ``"<previous student text> [T] <tutor text>"``, truncated as in training.
    """
    # The classifier was fitted on `artifacts/turn_text.pkl.gz`, whose fields were
    # truncated when that table was built (in `scripts/build_evidence_features.py`):
    # teacher[-220:], student[:220]. Mirror the truncation: passing the full text would
    # show the vectoriser longer strings than anything it was fit on. CV cannot detect such
    # a skew, because the training side reads `nto_dst_features.csv`, which is
    # built from the truncated text; it would show only at inference.
    # The training context is groupby(session_id).shift(1) over coded turns, and turn 0 is
    # never coded (min turn == 1 in all 22,811 sessions). A student-first transcript makes
    # turns[0] a bare student block with no teacher, so taking turns[i-1] by position would
    # give the vectoriser an opening utterance that training never saw (17.9% of sessions).
    prev = (turns[i - 1]["student"] if i > 0 and turns[i - 1]["teacher"] else "")[:220]
    return f"{prev} [T] {turns[i]['teacher'][-220:]}"


def _rates(probs: np.ndarray) -> dict:
    """Aggregate per-turn move probabilities into session-level rates.

    A session rate is the mean of the soft probabilities, which is the expectation of
    the oracle's hits/n_turns definition. Soft rather than thresholded: a threshold would
    discard most of what a 19-way multi-label annotator expresses.

    Args:
        probs: array of shape (n_tutor_turns, len(NTO_FAMILIES)) with per-turn
            probabilities, columns in :data:`NTO_FAMILIES` order.

    Returns:
        Dict mapping each name in :data:`NTODST_COLS` to a float.
    """
    m = probs.mean(axis=0)
    out = {f"ntodst_{f}_rate": float(m[j]) for j, f in enumerate(NTO_FAMILIES)}
    idx = {f: j for j, f in enumerate(NTO_FAMILIES)}
    el = sum(m[idx[f]] for f in _ELICIT)
    te = sum(m[idx[f]] for f in _TELL)
    out["ntodst_elicit_to_tell"] = float(el / (te + 1e-3))
    fc, fi = m[idx["fb_correct"]], m[idx["fb_incorrect"]]
    out["ntodst_feedback_positive_ratio"] = float(fc / (fc + fi + 1e-3))
    return out


def build_nto_matrix(feats: pd.DataFrame, store=None, clf=None,
                     batch: int = 200_000, show_progress: bool = False) -> pd.DataFrame:
    """Build the :data:`NTODST_COLS` block from transcripts, keyed by ``response_id``.

    Turn texts are vectorised in one batched pass over all sessions rather than per
    session: a full test set is ~2.4M turns and 19 heads, so per-session vectorising
    would dominate the 6 h budget for no benefit.

    Args:
        feats: rows to score, with ``response_id`` and ``session_id`` columns.
        store: transcript store with an ``iter(session_ids)`` method yielding
            ``(session_id, transcript DataFrame)``; defaults to
            :func:`magnificat.data.default_store`.
        clf: ``{"vec": vectoriser, "heads": {family: LogisticRegression}}``. If None,
            every column is NaN, which the booster also meets whenever a transcript
            yields no tutor turns.
        batch: number of turn texts vectorised per chunk.
        show_progress: show a tqdm progress bar when tqdm is installed.

    Returns:
        Float DataFrame indexed by ``response_id`` with the :data:`NTODST_COLS` columns
        in order; sessions without tutor turns get NaN.
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="nto")
        except ImportError:
            pass

    texts, owner = [], []
    seen = []
    for sid, tdf in it:
        seen.append(sid)
        try:
            turns, _ = build_turns(tdf)
        except Exception:
            turns = []
        for i, t in enumerate(turns):
            if t["teacher"].strip():
                texts.append(move_text(turns, i))
                owner.append(sid)

    sess: dict[str, dict] = {}
    if clf is not None and texts:
        arr = np.asarray(texts, dtype=object)
        P = np.zeros((len(arr), len(NTO_FAMILIES)), dtype=np.float32)
        for start in range(0, len(arr), batch):
            chunk = arr[start:start + batch]
            Xv = clf["vec"].transform(chunk)
            for j, f in enumerate(NTO_FAMILIES):
                head = clf["heads"].get(f)
                # a family too rare to fit in training falls back to its base rate, as
                # the distillation did, rather than inventing a prediction
                P[start:start + len(chunk), j] = (
                    head.predict_proba(Xv)[:, 1] if head is not None else 0.0)
        own = np.asarray(owner)
        for sid in dict.fromkeys(owner):
            sess[sid] = _rates(P[own == sid])

    blank = {c: np.nan for c in NTODST_COLS}
    out = pd.DataFrame(
        [sess.get(s, blank) for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in NTODST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[NTODST_COLS].astype(float)
