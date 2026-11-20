"""Distilled student-state rates (``studst_*``) computed live from transcripts.

The inference twin of ``src/training/scripts/distil_student_states.py``. The model ships
these 12 columns in place of E2's 14 student-side regexes.
The other distilled blocks have twins too: :mod:`magnificat.evidence_features`,
:mod:`magnificat.nto_features`.

The main risk is the truncation chain, which is subtler here than in ``nto_features``.
The head was fitted on ``artifacts/turn_text.pkl.gz``, whose fields were already truncated
when that table was built (``src/training/scripts/build_evidence_features.py``:
``teacher[-220:]``, ``student[:220]``). ``distil_student_states._text`` then truncates again
(``tutor_q[-200:]``, ``student[:300]``). So the text the vectoriser actually saw is:

    tutor   = teacher[-220:][-200:]   ==  teacher[-200:]
    student = student[:220][:300]     ==  student[:220]     <-- not student[:300]

Feeding ``student[:300]`` at inference would hand the vectoriser up to 80 characters it
never saw in training, and no cross-validation could detect it, because the training side
reads the distilled CSV, which is itself built from the truncated text. Both
stages are applied below, and ``src/training/scripts/check_studst_parity.py`` checks the
result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence_features import build_turns

#: Codes in the order `distil_student_states.py` writes them (`annotate_student_states.ALL_CODES`).
STUDENT_CODES = ["EXPLAINS_WHY", "SHOWS_METHOD", "ANSWER_ONLY", "CONFUSED", "HEDGED",
                 "INSIGHT", "ASKS_CONCEPTUAL", "ASKS_VERIFICATION", "SELF_CORRECTS",
                 "GUESSES", "MINIMAL", "OFF_TASK"]

#: The 12 shipped columns, in the order the training CSV emits them. The order is fixed: the
#: booster indexes columns positionally through the `dlg` passthrough transformer.
STUDST_COLS = [f"studst_{c.lower()}_rate" for c in STUDENT_CODES]


def state_text(turns, i: int) -> str:
    """Return the text the distilled head reads for turn ``i``.

    Mirrors `distil_student_states._text` applied to `turn_text.pkl.gz`'s already truncated
    fields (see the module docstring). Do not simplify the double slice: the outer bound is
    what the head was fitted on, the inner bound is what the table stored, and the effective
    student window is 220 characters, not 300.

    Args:
        turns: the turn list from :func:`magnificat.evidence_features.build_turns`.
        i: index of the turn.

    Returns:
        ``"<tutor text> [S] <student text>"``.
    """
    t = turns[i]
    tutor = t["teacher"][-220:][-200:]
    student = t["student"][:220][:300]
    return f"{tutor} [S] {student}"


def _rates(probs: np.ndarray) -> dict:
    """Average the soft per-turn probabilities into session rates.

    Matches `distil_student_states.py`'s `groupby("session_id").mean()` on the training side.
    Soft, not thresholded, the same choice as `nto_features._rates`.

    Args:
        probs: array (n_turns, 12) of per-turn probabilities in ``STUDENT_CODES`` order.

    Returns:
        Dict mapping each ``STUDST_COLS`` name to its session mean.
    """
    m = probs.mean(axis=0)
    return {col: float(m[j]) for j, col in enumerate(STUDST_COLS)}


def build_studst_matrix(feats: pd.DataFrame, store=None, clf=None,
                        batch: int = 200_000, show_progress: bool = False) -> pd.DataFrame:
    """Build :data:`STUDST_COLS` live from transcripts, keyed by ``response_id``.

    A head is ``None`` when its code was too rare to fit; the distillation fell back to the
    code's base rate, and so does this (any other prediction would differ from the
    training-side block). Only turns with a non-empty student block are coded, as in
    ``turn_text.pkl.gz``.

    Args:
        feats: rows to score, with ``response_id`` and ``session_id``.
        store: a ``TranscriptStore``; defaults to the training store.
        clf: the classifier dict dumped by `distil_student_states.py`:
            ``{"vec": ..., "heads": {CODE: LogisticRegression|None}, "base": {CODE: float}}``.
            ``None`` yields all-NaN, a state the booster already meets whenever a transcript
            has no student turns.
        batch: number of turn texts vectorised at a time.
        show_progress: show a tqdm bar if tqdm is installed.

    Returns:
        DataFrame indexed by ``response_id`` with the 12 ``STUDST_COLS`` (float, NaN for a
        session with no coded turn).
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="studst")
        except ImportError:
            pass

    texts, owner = [], []
    for sid, tdf in it:
        try:
            turns, _ = build_turns(tdf)
        except Exception:
            turns = []
        for i, t in enumerate(turns):
            # `turn_text.pkl.gz` keeps a row only where the student block is non-empty
            # (build_evidence_features.py). Coding a turn with no student utterance
            # would feed the head a class of input it never saw.
            if t["student"].strip():
                texts.append(state_text(turns, i))
                owner.append(sid)

    sess: dict[str, dict] = {}
    if clf is not None and texts:
        arr = np.asarray(texts, dtype=object)
        P = np.zeros((len(arr), len(STUDENT_CODES)), dtype=np.float32)
        for start in range(0, len(arr), batch):
            chunk = arr[start:start + batch]
            Xv = clf["vec"].transform(chunk)
            for j, code in enumerate(STUDENT_CODES):
                head = clf["heads"].get(code)
                P[start:start + len(chunk), j] = (
                    head.predict_proba(Xv)[:, 1] if head is not None
                    else float(clf.get("base", {}).get(code, 0.0)))
        own = np.asarray(owner)
        for sid in dict.fromkeys(owner):
            sess[sid] = _rates(P[own == sid])

    blank = {c: np.nan for c in STUDST_COLS}
    out = pd.DataFrame(
        [sess.get(s, blank) for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in STUDST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[STUDST_COLS].astype(float)
