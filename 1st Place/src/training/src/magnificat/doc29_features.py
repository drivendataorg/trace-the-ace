"""Distilled interview-codebook rates (``d29dst_*``), computed from transcripts at inference.

The codebook comes from three tutor interviews (src/training/docs/29_interviews_and_labels.md).
This module is the inference twin of ``scripts/build_doc29_features.py --mode distilled``.
The annotation prompts are in :mod:`magnificat.doc29_prompts`. The other distilled blocks
have the same kind of twin: :mod:`magnificat.evidence_features`, :mod:`magnificat.nto_features` and
:mod:`magnificat.student_features`.

The 34 columns are 7 tutor moves (rate and late-third share), student reasoning, confidence
and affect, and four derived quantities. All are session-level and are broadcast to every
response row of that session.

Two things must mirror the training side exactly, and one cannot:

1. No truncation. Unlike ``student_features``, whose head was fitted on twice-truncated text
   (the 220/200-character chain that a plain ``[:300]`` would break),
   ``distil_doc29_labels.build_text`` applies no character cap: the full preceding
   opposite-role turn plus the full own turn. Adding a cap here would create a train/serve
   skew.

2. ``position_frac`` is within-role: ``i / max(len(role_turns) - 1, 1)``, not the position in
   the whole transcript. Both the annotator (``sandpiper_live_infer.build_prompts_whole``,
   over role-filtered turns) and the all-data scorer
   (``build_doc29_features.distilled_frame``) use this, so the two halves of the training
   CSV agree (verified). Fourteen of the 34 columns threshold on it at 2/3, so an
   error here corrupts the whole ``_late`` family without any error being raised.

3. Exact parity on the annotated sessions is impossible by construction. The training CSV is a
   mixture: the 4,000 annotated sessions carry out-of-fold probabilities, the other 18,821
   carry the all-data fit. At inference there are no annotations, so this module uses the
   all-data head everywhere, and ``scripts/check_doc29_parity.py`` therefore restricts the
   comparison to the unannotated sessions, as ``check_studst_parity.py`` does for ``studst``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Tutor move codes, in the order `distil_doc29_labels.HEADS` declares them. The order is fixed:
#: it sets the column order below, and the booster reads columns by position.
TUTOR_CODES = ["NORMALIZING_DIFFICULTY", "RELEASE_HANDOFF", "TRANSFER_PROBE",
               "SIMPLIFYING_TO_SUBPROBLEM", "MOTIVATING_RELEVANCE", "SUMMARIZING_PROGRESS",
               "OFFERING_CHOICE"]

#: Student head names as `distil_doc29_labels.head_name` emits them.
STU_HEADS = ["stu_answer_only", "stu_reasoning_shown", "stu_hedged", "stu_unmarked",
             "stu_assertive", "stu_insight_delight", "stu_frustration"]

LATE = 2.0 / 3.0          # "final third" boundary on position_frac

#: The 34 shipped columns in the order `artifacts/doc29_features_distilled.csv` emits them.
D29DST_COLS = (
    [f"d29dst_{c}_{s}" for c in TUTOR_CODES for s in ("rate", "late")]
    + ["d29dst_any_move_rate"]
    + [f"d29dst_{n}_{s}"
       for n in ("answer_only", "reasoning_shown", "insight_delight", "frustration",
                 "hedged", "unmarked", "assertive")
       for s in ("rate", "late")]
    + ["d29dst_answering_rate", "d29dst_reasoning_share", "d29dst_confidence_contrast",
       "d29dst_delight_pos_mean", "d29dst_delight_any"]
)


def _prev_other_role(df: pd.DataFrame, other: str) -> list[str]:
    """Return, for each row, the most recent strictly preceding utterance by the other role.

    Mirrors `distil_doc29_labels.build_text`: append first, then update, so a turn never
    sees itself.

    Args:
        df: one session's utterances in transcript order, with ``role`` and ``content``.
        other: the role whose utterances provide the context (``"tutor"`` or ``"student"``).

    Returns:
        List of strings, one per row of ``df`` (empty string before the first such utterance).
    """
    prev, last = [], ""
    for r, c in zip(df["role"].astype(str), df["content"]):
        prev.append(last)
        if r == other:
            last = c
    return prev


def _session_turns(tdf: pd.DataFrame, role: str, marker: str):
    """Build the classifier inputs and within-role positions for one session's turns of a role.

    Args:
        tdf: one session's transcript (``utterance_id``, ``role``, ``content``).
        role: ``"tutor"`` or ``"student"``.
        marker: tag placed between context and turn text (``"T"`` or ``"S"``).

    Returns:
        Tuple ``(texts, pos)``: the strings ``"<previous other-role text> [marker] <turn>"``
        and the within-role position fractions in [0, 1], both in transcript order.
    """
    df = tdf.sort_values("utterance_id").reset_index(drop=True)
    df = df.assign(content=df["content"].fillna("").astype(str))
    other = "student" if role == "tutor" else "tutor"
    df["_prev"] = _prev_other_role(df, other)
    sub = df[df["role"].astype(str) == role]
    n = max(len(sub) - 1, 1)
    texts = (sub["_prev"] + f" [{marker}] " + sub["content"]).tolist()
    pos = [i / n for i in range(len(sub))]
    return texts, pos


def _agg(P: np.ndarray, names: list[str], denom: np.ndarray, late: np.ndarray,
         prefix: str) -> dict:
    """Compute a weighted rate and its late-third share per head (as `build_doc29_features._agg`).

    Args:
        P: per-turn probabilities, shape (n_turns, len(names)).
        names: column stem for each head.
        denom: per-turn weights, shape (n_turns,); ones, or the answering probability.
        late: 1.0 for turns in the final third of the session, else 0.0.
        prefix: column prefix, e.g. ``"d29dst_"``.

    Returns:
        Dict with ``{prefix}{name}_rate`` and ``{prefix}{name}_late`` per head (NaN when the
        weights sum to zero).
    """
    out = {}
    d = float(denom.sum())
    dl = float((denom * late).sum())
    for j, name in enumerate(names):
        v = P[:, j]
        out[f"{prefix}{name}_rate"] = float((v * denom).sum() / d) if d > 0 else np.nan
        out[f"{prefix}{name}_late"] = float((v * denom * late).sum() / dl) if dl > 0 else np.nan
    return out


def _session_row(Pt, Ps, post, poss, prefix: str) -> dict:
    """Compile one session's 34 columns, mirroring `build_doc29_features.compile_block`.

    Args:
        Pt: tutor-turn probabilities, shape (n_tutor_turns, len(TUTOR_CODES)), or None.
        Ps: student-turn probabilities, shape (n_student_turns, len(STU_HEADS)), or None.
        post: within-role position fractions of the tutor turns.
        poss: within-role position fractions of the student turns.
        prefix: column prefix, e.g. ``"d29dst_"``.

    Returns:
        Dict of column name to float; columns of a role with no turns are absent.
    """
    r: dict = {}
    if Pt is not None and len(Pt):
        one = np.ones(len(Pt))
        late = (np.nan_to_num(post, nan=0.0) >= LATE).astype(float)
        r.update(_agg(Pt, TUTOR_CODES, one, late, prefix))
        # coverage: share of tutor turns performing any of the interview-codebook moves
        r[f"{prefix}any_move_rate"] = float(np.clip(Pt.sum(axis=1), 0, 1).mean())
    if Ps is not None and len(Ps):
        one = np.ones(len(Ps))
        late = (np.nan_to_num(poss, nan=0.0) >= LATE).astype(float)
        ans_only, reas = Ps[:, 0], Ps[:, 1]
        hed, unm, ass = Ps[:, 2], Ps[:, 3], Ps[:, 4]
        del_, fru = Ps[:, 5], Ps[:, 6]
        answering = np.clip(ans_only + reas, 0, 1)      # complement = NOT_ANSWERING
        r.update(_agg(np.stack([ans_only, reas, del_, fru], axis=1),
                      ["answer_only", "reasoning_shown", "insight_delight", "frustration"],
                      one, late, prefix))
        # confidence is defined on answering turns only (see build_doc29_features' docstring)
        r.update(_agg(np.stack([hed, unm, ass], axis=1),
                      ["hedged", "unmarked", "assertive"], answering, late, prefix))
        r[f"{prefix}answering_rate"] = float(answering.mean())
        tot = float((ans_only + reas).sum())
        r[f"{prefix}reasoning_share"] = float(reas.sum() / tot) if tot > 0 else np.nan
        hs, as_ = float(hed.sum()), float(ass.sum())
        r[f"{prefix}confidence_contrast"] = ((as_ - hs) / (as_ + hs)
                                             if (as_ + hs) > 0 else np.nan)
        w = float(del_.sum())
        r[f"{prefix}delight_pos_mean"] = (
            float((del_ * np.nan_to_num(poss, nan=0.5)).sum() / w) if w > 0 else np.nan)
        r[f"{prefix}delight_any"] = float(min(w, 1.0))
    return r


def _score(texts, clf, pass_name: str, head_names: list[str], batch: int) -> np.ndarray:
    """Score turn texts with one pass (tutor or student) of the distilled classifier.

    A head absent from the bundle fell back during distillation (fewer than MIN_POS positives)
    and gets 0.0, matching the all-data path of `build_doc29_features.distilled_frame`, which
    is the path inference takes. The out-of-fold path fills the same head with its base rate instead, so a
    fallen-back head would be discontinuous at the 4,000-session boundary of the training CSV.
    No head fell back in the distillation run (all 14 fitted), so this is latent, not active.

    Args:
        texts: turn input strings from :func:`_session_turns`.
        clf: the distilled bundle (see :func:`build_d29_matrix`).
        pass_name: ``"doc29_tutor"`` or ``"doc29_student"``.
        head_names: heads to score, in output column order.
        batch: number of texts vectorised per chunk.

    Returns:
        float32 array of shape (len(texts), len(head_names)).
    """
    heads = clf[pass_name]["heads"]
    vec = clf[pass_name]["vec"]
    P = np.zeros((len(texts), len(head_names)), dtype=np.float32)
    arr = np.asarray(texts, dtype=object)
    for start in range(0, len(arr), batch):
        chunk = arr[start:start + batch]
        X = vec.transform(chunk)
        for j, h in enumerate(head_names):
            m = heads.get(h)
            if m is not None:
                P[start:start + len(chunk), j] = m.predict_proba(X)[:, 1]
    return P


def build_d29_matrix(feats: pd.DataFrame, store=None, clf=None,
                     batch: int = 100_000, show_progress: bool = False) -> pd.DataFrame:
    """Build the :data:`D29DST_COLS` block from transcripts, keyed by ``response_id``.

    Args:
        feats: rows to score, with ``response_id`` and ``session_id`` columns.
        store: transcript store with an ``iter(session_ids)`` method yielding
            ``(session_id, transcript DataFrame)``; defaults to
            :func:`magnificat.data.default_store`.
        clf: the bundle written by `distil_doc29_labels.py`,
            ``{"doc29_tutor": {"vec":…, "heads":{…}}, "doc29_student": {…}}``. ``None`` gives
            all-NaN, the state the booster already meets for a session with no turns.
        batch: number of turn texts vectorised per chunk.
        show_progress: show a tqdm progress bar when tqdm is installed.

    Returns:
        Float DataFrame indexed by ``response_id`` with the :data:`D29DST_COLS` columns in order.
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="d29dst")
        except ImportError:
            pass

    t_txt, t_pos, t_own = [], [], []
    s_txt, s_pos, s_own = [], [], []
    for sid, tdf in it:
        try:
            tt, tp = _session_turns(tdf, "tutor", "T")
            st, sp = _session_turns(tdf, "student", "S")
        except Exception:
            continue
        t_txt += tt; t_pos += tp; t_own += [sid] * len(tt)
        s_txt += st; s_pos += sp; s_own += [sid] * len(st)

    sess: dict[str, dict] = {}
    if clf is not None:
        Pt = _score(t_txt, clf, "doc29_tutor", [f"tut_{c.lower()}" for c in TUTOR_CODES], batch) \
            if t_txt else np.zeros((0, len(TUTOR_CODES)), dtype=np.float32)
        Ps = _score(s_txt, clf, "doc29_student", STU_HEADS, batch) \
            if s_txt else np.zeros((0, len(STU_HEADS)), dtype=np.float32)
        to = np.asarray(t_own); so = np.asarray(s_own)
        tpos = np.asarray(t_pos, dtype=float); spos = np.asarray(s_pos, dtype=float)
        for sid in dict.fromkeys(list(t_own) + list(s_own)):
            mt, ms = to == sid, so == sid
            sess[sid] = _session_row(Pt[mt], Ps[ms], tpos[mt], spos[ms], "d29dst_")

    blank = {c: np.nan for c in D29DST_COLS}
    out = pd.DataFrame(
        [{**blank, **sess.get(s, {})} for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in D29DST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[D29DST_COLS].astype(float)
