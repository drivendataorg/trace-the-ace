"""Distilled taxonomy-v2 move rates (``v2t_*`` / ``v2s_*``), computed from transcripts at inference.

This is the inference twin of ``scripts/distil_v2_moves.py``: it rebuilds the v2 blocks from
transcripts with the saved classifiers, so they can be computed on the test set or on any other
corpus's transcripts without the training-side session CSVs.

Two roles, one shape. A tutor classifier codes the tutor slot of a turn with the preceding
student turn as context; a student classifier codes the student slot with the preceding tutor
utterance as context. ``distil_v2_moves`` stores both as
``{"vec", "heads", "base", "families", "role", "prefix"}``, so the column names and their order
come from the saved artefact rather than from a constant here. A v2 run with a different code
inventory therefore cannot be misaligned against a booster trained on another.

Each session is processed independently (no pooling across test samples). The
``elicit_to_tell`` cap is a fixed constant for the same reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence_features import build_turns

#: Must equal `distil_v2_moves.ELICIT_TELL_CAP`. Duplicated rather than imported because
#: `scripts/` is not importable from the runtime container.
ELICIT_TELL_CAP = 10.0

#: The two derived ratios, mirroring `distil_v2_moves.RATIO_SPEC["v2"]`.
_V2_ELICIT = ["PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP",
              "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
              "PROMPTING_SELF_CORRECTION", "RELEASE_HANDOFF"]
_V2_TELL = ["EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
            "WORKED_DEMONSTRATION"]


# ── Truncation: two bounds, and the inner one is not visible from the training side ─────────
#
# Every field of `turn_text.pkl.gz` was truncated when that table was built
# (`build_evidence_features.py`: `teacher[-220:]`, `student[:220]`; verified, max length 220
# on all three text columns). `distil_v2_moves.build_text` then truncates again. So the text
# the vectoriser was fitted on is the composition:
#
#     tutor pass:   prev student = student[:220][:220]  == student[:220]
#                   tutor utt    = teacher[-220:][-220:] == teacher[-220:]
#     student pass: tutor ctx    = teacher[-220:][-200:] == teacher[-200:]
#                   student utt  = student[:220][:300]   == student[:220]   <-- not [:300]
#
# The last line is the pitfall. Reading the raw transcript and applying only the outer `[:300]`
# gives the vectoriser up to 80 characters it never saw in training. No in-domain check can
# detect this, because the training side reads the already-truncated table, so the skew would
# appear only at inference. `student_features.state_text` carries the same warning, and
# `scripts/check_v2_family_parity.py` checks this module against the training-side blocks.
#
# Keep the double slices below as they are: the outer bound is what the head was fitted on,
# the inner bound is what the table stored.


#: the shipped bounds, used when a classifier carries no `trunc` field
DEFAULT_TRUNC = {"table": 220, "tutor_ctx": -200, "tutor_utt": -220,
                 "student_ctx": 220, "student_utt": 300}


def tutor_text(turns, i: int, trunc: dict | None = None) -> str:
    """Build the tutor-pass input text, as `distil_v2_moves.build_text(role="tutor")` on the table.

    Args:
        turns: list of turn dicts from :func:`magnificat.evidence_features.build_turns`.
        i: index of the turn whose tutor slot is coded.
        trunc: truncation bounds of the classifier; :data:`DEFAULT_TRUNC` if None.

    Returns:
        ``"<previous student text> [T] <tutor text>"`` with the training truncation applied.
    """
    tc = trunc or DEFAULT_TRUNC
    tb = tc["table"]
    prev = (turns[i - 1]["student"] if i > 0 and turns[i - 1]["teacher"] else "")
    return f"{prev[:tb][:tb]} [T] {turns[i]['teacher'][-tb:][tc['tutor_utt']:]}"


def student_text(turns, i: int, trunc: dict | None = None) -> str:
    """Build the student-pass input text, as `distil_v2_moves.build_text(role="student")`.

    Args:
        turns: list of turn dicts from :func:`magnificat.evidence_features.build_turns`.
        i: index of the turn whose student slot is coded.
        trunc: truncation bounds of the classifier; :data:`DEFAULT_TRUNC` if None.

    Returns:
        ``"<tutor text of the same turn> [S] <student text>"`` with the training truncation.
    """
    tc = trunc or DEFAULT_TRUNC
    tb = tc["table"]
    t = turns[i]
    return (f"{t['teacher'][-tb:][tc['tutor_ctx']:]} [S] "
            f"{t['student'][:tb][:tc['student_utt']]}")


def block_cols(clf: dict) -> list[str]:
    """Return the column names of a classifier's block, in the order `distil_v2_moves` wrote them.

    Args:
        clf: a saved `distil_v2_moves` classifier (uses ``prefix``, ``families``, ``role``).

    Returns:
        ``{prefix}_{family}_rate`` per family, plus the two ratio columns for the tutor role.
    """
    pre, fams = clf["prefix"], clf["families"]
    cols = [f"{pre}_{f.lower()}_rate" for f in fams]
    if clf.get("role") == "tutor":
        cols += [f"{pre}_elicit_to_tell", f"{pre}_feedback_positive_ratio"]
    return cols


def _rates(P: np.ndarray, clf: dict) -> dict:
    """Aggregate per-turn probabilities into session rates (and tutor ratios).

    A session rate is the mean of the soft probabilities, the expectation of the oracle's
    hits/n_turns; this makes the block length-normalised.

    Args:
        P: per-turn probabilities, shape (n_turns, len(clf["families"])).
        clf: the saved classifier (uses ``prefix``, ``families``, ``role``).

    Returns:
        Dict of column name to float, as named by :func:`block_cols`.
    """
    pre, fams = clf["prefix"], clf["families"]
    m = P.mean(axis=0)
    out = {f"{pre}_{f.lower()}_rate": float(m[j]) for j, f in enumerate(fams)}
    if clf.get("role") != "tutor":
        return out
    idx = {f: j for j, f in enumerate(fams)}
    el = sum(m[idx[f]] for f in _V2_ELICIT if f in idx)
    te = sum(m[idx[f]] for f in _V2_TELL if f in idx)
    # clip: the ratio is unbounded and its tail is set by the 1e-3 guard. On a corpus where
    # tutors explain less, the denominator shrinks and the ratio runs past every split
    # threshold learned on TtA. An XGBoost model is invariant to monotone transforms, so log1p
    # would change nothing; bounding is what transfers.
    out[f"{pre}_elicit_to_tell"] = float(min(el / (te + 1e-3), ELICIT_TELL_CAP))
    fc = m[idx["FEEDBACK_CORRECT"]] if "FEEDBACK_CORRECT" in idx else 0.0
    fi = m[idx["FEEDBACK_INCORRECT"]] if "FEEDBACK_INCORRECT" in idx else 0.0
    out[f"{pre}_feedback_positive_ratio"] = float(fc / (fc + fi + 1e-3))
    return out


def build_v2_matrix(feats: pd.DataFrame, store=None, clf=None,
                    batch: int = 200_000, show_progress: bool = False) -> pd.DataFrame:
    """Build one v2 block (tutor or student) from transcripts, keyed by ``response_id``.

    Args:
        feats: rows to score, with ``response_id`` and ``session_id`` columns.
        store: transcript store with an ``iter(session_ids)`` method yielding
            ``(session_id, transcript DataFrame)``; defaults to
            :func:`magnificat.data.default_store`.
        clf: a saved `distil_v2_moves` classifier. If None, an empty frame (no columns) is
            returned; sessions without turns of the classifier's role get NaN, a state the
            booster also meets in training.
        batch: number of turn texts vectorised per chunk.
        show_progress: show a tqdm progress bar when tqdm is installed.

    Returns:
        Float DataFrame indexed by ``response_id`` with the :func:`block_cols` columns.
    """
    from . import data as _data

    if clf is None:
        return pd.DataFrame(index=pd.Index(feats["response_id"], name="response_id"))
    cols = block_cols(clf)
    role = clf.get("role", "tutor")
    slot = "teacher" if role == "tutor" else "student"
    _mk = tutor_text if role == "tutor" else student_text
    _tc = clf.get("trunc") or DEFAULT_TRUNC
    mk = lambda turns, i: _mk(turns, i, _tc)  # noqa: E731

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc=clf["prefix"])
        except ImportError:
            pass

    texts, owner = [], []
    for sid, tdf in it:
        try:
            turns, _ = build_turns(tdf)
        except Exception:
            turns = []
        for i, t in enumerate(turns):
            if t[slot].strip():
                texts.append(mk(turns, i))
                owner.append(sid)

    sess: dict[str, dict] = {}
    if texts:
        arr = np.asarray(texts, dtype=object)
        fams = clf["families"]
        P = np.zeros((len(arr), len(fams)), dtype=np.float32)
        for start in range(0, len(arr), batch):
            chunk = arr[start:start + batch]
            Xv = clf["vec"].transform(chunk)
            for j, f in enumerate(fams):
                head = clf["heads"].get(f)
                # a family too rare to fit falls back to its training base rate, as the
                # distillation did, rather than inventing a prediction
                P[start:start + len(chunk), j] = (
                    head.predict_proba(Xv)[:, 1] if head is not None
                    else np.float32(clf["base"].get(f, 0.0)))
        own = np.asarray(owner)
        for sid in dict.fromkeys(owner):
            sess[sid] = _rates(P[own == sid], clf)

    blank = {c: np.nan for c in cols}
    out = pd.DataFrame(
        [sess.get(s, blank) for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].astype(float)


# =======================================================================================
# Aggregation families of the same per-turn probabilities  (see `magnificat.v2_agg`)
# =======================================================================================
#
# `build_v2_matrix` above collapses the per-turn probabilities with one `P.mean(axis=0)`.
# The families in :mod:`magnificat.v2_agg` aggregate them in other ways, and this section is
# their inference twin. The shipped XGBoost model uses the `shrunk` family (`v2ht_` /
# `v2hs_`), selected by `assets/v2_spec.txt` (`narrow=none`, `source=shrunk`).
#
# The turn axis is not "all turns", and an error there does not show in-domain.
# The training-side families are built from `{prefix}_turn_probs_*.pkl.gz`, which is derived
# from `turn_text.pkl.gz`, which `build_evidence_features.py` writes with `if not
# t["student"].strip(): continue`. So the training axis is the turns that carry student text,
# and the tutor pass is a further subset of that (turns which also carry a tutor utterance).
# Using "every turn `build_turns` returns" as the axis here would shift every
# position-dependent column (thirds, slopes, centroids, the LO window) while leaving the plain
# rates untouched, so the parity check would pass on the rates and the block would still be
# wrong. `_family_axis` below is the one place that rule is expressed.


def _family_axis(turns: list) -> list[int]:
    """Return the indices of the turns the training tables contain (turns with student text).

    Args:
        turns: list of turn dicts from :func:`magnificat.evidence_features.build_turns`.

    Returns:
        List of turn indices, in order.
    """
    return [i for i, t in enumerate(turns) if t["student"].strip()]


def _predict(clf: dict, texts: list, batch: int = 200_000):
    """Score turn texts with a saved classifier, in the classifier's own family order.

    Args:
        clf: a saved `distil_v2_moves` classifier.
        texts: input strings from :func:`tutor_text` or :func:`student_text`.
        batch: number of texts vectorised per chunk.

    Returns:
        float32 array of shape (len(texts), len(clf["families"])); a family without a fitted
        head gets its training base rate.
    """
    fams = clf["families"]
    P = np.zeros((len(texts), len(fams)), dtype=np.float32)
    if not texts:
        return P
    arr = np.asarray(texts, dtype=object)
    for start in range(0, len(arr), batch):
        Xv = clf["vec"].transform(arr[start:start + batch])
        for j, f in enumerate(fams):
            h = clf["heads"].get(f)
            P[start:start + Xv.shape[0], j] = (
                h.predict_proba(Xv)[:, 1] if h is not None
                else np.float32(clf["base"].get(f, 0.0)))
    return P


def family_cols(families: list[str], tclf: dict, sclf: dict) -> list[str]:
    """Return the column names for a set of aggregation families, in emission order.

    The names come from the classifiers' own family lists rather than from a constant, for the
    reason given in the module docstring: a v2 run with a different code inventory must not be
    misaligned against a booster fitted on another. The names are found by running
    :func:`_families_for` on a dummy 3-turn session.

    Args:
        families: family names (e.g. ``"rate"``, ``"instr"``, ``"bound"``).
        tclf: the saved tutor classifier.
        sclf: the saved student classifier.

    Returns:
        List of column names.
    """
    from . import v2_agg as A
    T = np.zeros((3, len(tclf["families"])), np.float32)
    S = np.zeros((3, len(sclf["families"])), np.float32)
    m = np.ones(3, bool)
    return list(_families_for(T, S, m, m, tclf, sclf, None, families))


def _families_for(T, S, tmask, smask, tclf, sclf, win, families) -> dict:
    """Compute one session's requested aggregation families.

    Mirrors `scripts/build_v2_families.session_families` and must stay in step with it;
    `scripts/check_v2_family_parity.py` checks that it does.

    Args:
        T: tutor probabilities on the family axis, shape (n_axis, len(tclf["families"])); rows
            without a tutor utterance are zero.
        S: student probabilities on the family axis, shape (n_axis, len(sclf["families"])).
        tmask: boolean mask of axis turns that carry a tutor utterance.
        smask: boolean mask of axis turns used for the student rates.
        tclf: the saved tutor classifier.
        sclf: the saved student classifier.
        win: LO window mask, or a ``(hit_mask, window_mask)`` tuple, for the ``lowin`` and
            ``soft`` families; None otherwise.
        families: the family names to compute.

    Returns:
        Dict of column name to float.
    """
    from . import v2_agg as A
    tc, sc = tclf["families"], sclf["families"]
    out: dict = {}
    wT = A.instructional_weight(T, tc, A.TUTOR_RESID)
    wS = A.instructional_weight(S, sc, A.STUDENT_RESID)

    def emit(pre, vals, cols):
        out.update({f"{pre}_{c.lower()}_rate": float(v) for c, v in zip(cols, vals)})

    def ratios(pre, vals, cols):
        ix = {c: j for j, c in enumerate(cols)}
        el = sum(vals[ix[c]] for c in A.ELICIT if c in ix)
        te = sum(vals[ix[c]] for c in A.TELL if c in ix)
        out[f"{pre}_elicit_to_tell"] = A._ratio(el, te)
        fc = vals[ix["FEEDBACK_CORRECT"]] if "FEEDBACK_CORRECT" in ix else 0.0
        fi = vals[ix["FEEDBACK_INCORRECT"]] if "FEEDBACK_INCORRECT" in ix else 0.0
        out[f"{pre}_feedback_positive_ratio"] = float(fc / (fc + fi + A.EPS))

    rT = A.rates(T, tc, mask=tmask)
    rS = A.rates(S, sc, mask=smask)

    if "rate" in families:
        # The plain-mean aggregation, emitted from the same pass. Identical to
        # `build_v2_matrix`'s output (asserted by `check_v2_family_parity.py`), but computed
        # here so a narrow block costs one walk of the transcripts instead of three.
        emit("v2t", rT, tc)
        ratios("v2t", rT, tc)
        emit("v2s", rS, sc)
    if "shrunk" in families:
        sT = A.shrunk_rates(T, tc, mask=tmask)
        sS = A.shrunk_rates(S, sc, mask=smask)
        emit("v2ht", sT, tc)
        ratios("v2ht", sT, tc)
        emit("v2hs", sS, sc)
    for _k, _pre in ((12, "v2wa"), (24, "v2wb"), (48, "v2wc")):
        if f"win{_k}" in families:
            _wT = A.window_rates(T, tc, mask=tmask, k=_k)
            _wS = A.window_rates(S, sc, mask=smask, k=_k)
            emit(f"{_pre}t", _wT, tc)
            ratios(f"{_pre}t", _wT, tc)
            emit(f"{_pre}s", _wS, sc)
    if "shrunkeb" in families:
        _e = A.eb_shrunk_rates(T, tc, mask=tmask)
        emit("v2et", _e, tc); ratios("v2et", _e, tc)
        emit("v2es", A.eb_shrunk_rates(S, sc, mask=smask), sc)
    if "shrunkk35" in families:
        _g = A.shrunk_rates(T, tc, mask=tmask, kappa=35.0)
        emit("v2gt", _g, tc); ratios("v2gt", _g, tc)
        emit("v2gs", A.shrunk_rates(S, sc, mask=smask, kappa=35.0), sc)
    if "shrunkq" in families:
        sT2 = A.shrunk_rates(T, tc, mask=tmask); sS2 = A.shrunk_rates(S, sc, mask=smask)
        emit("v2qt", sT2, tc)
        _ix = {c: j for j, c in enumerate(tc)}
        _el = sum(sT2[_ix[c]] for c in A.ELICIT if c in _ix)
        _te = sum(sT2[_ix[c]] for c in A.TELL if c in _ix)
        out["v2qt_elicit_share"] = float(_el / (_el + _te + A.EPS))
        _fc = sT2[_ix["FEEDBACK_CORRECT"]] if "FEEDBACK_CORRECT" in _ix else 0.0
        _fi = sT2[_ix["FEEDBACK_INCORRECT"]] if "FEEDBACK_INCORRECT" in _ix else 0.0
        out["v2qt_feedback_positive_ratio"] = float(_fc / (_fc + _fi + A.EPS))
        emit("v2qs", sS2, sc)
    if "instr" in families:
        iT = A.rates(T, tc, weights=wT, mask=tmask)
        iS = A.rates(S, sc, weights=wS, mask=smask)
        for j, c in enumerate(tc):
            if c in A.TUTOR_RESID:
                iT[j] = rT[j]
        for j, c in enumerate(sc):
            if c in A.STUDENT_RESID:
                iS[j] = rS[j]
        emit("v2ti", iT, tc)
        ratios("v2ti", iT, tc)
        emit("v2si", iS, sc)
    if "trim" in families:
        lo, hi = A.instructional_span(wT)
        span = np.zeros(len(wT), bool)
        span[lo:hi + 1] = True
        tT = A.rates(T, tc, mask=tmask & span)
        emit("v2tt", tT, tc)
        ratios("v2tt", tT, tc)
        emit("v2st", A.rates(S, sc, mask=smask & span), sc)
    if "bound" in families:
        # `_code_mass` is dropped: it is mean_t sum_c P[t,c] = sum of the rates, an exact
        # linear combination of columns the block already carries.
        out.update({k: v for k, v in A.boundary_features(wT, wS, T, tc, "v2b").items()
                    if not k.endswith("_code_mass")})
        out["v2b_stu_instr_share"] = float(wS.mean()) if len(wS) else np.nan
    if "slope" in families:
        out.update(A.slope_features(T, tc, _AXES_T, "v2sl"))
        out.update(A.slope_features(S, sc, _AXES_S, "v2sl_s"))
    if "disp" in families:
        out.update(A.dispersion_features(T, tc, _AXES_T, "v2d"))
    if "trans" in families:
        out.update(A.transition_features(T, tc, S, sc, "v2x"))
    if "ratio" in families:
        out.update(A.ratio_features(T, tc, "v2r"))
    if "qual" in families:
        # the two `_entropy` columns of `quality_features` are not emitted.
        out.update({k: v for k, v in A.quality_features(T, tc, S, sc, tmask, "v2q").items()
                    if not k.endswith("_entropy")})
    if "lowin" in families or "soft" in families:
        h, w = win if isinstance(win, tuple) else (None, win)
        if "lowin" in families:
            out.update(A.lo_window_features(T, tc, S, sc, w, "v2w"))
        if "soft" in families:
            out.update(A.lo_soft_coverage(h, T.shape[0], "v2w"))
    return out


def build_v2_family_matrix(feats: pd.DataFrame, store=None, tclf: dict | None = None,
                           sclf: dict | None = None, families=("instr",),
                           show_progress: bool = False) -> pd.DataFrame:
    """Build alternative v2 aggregations from transcripts, keyed by ``response_id``.

    ``lowin`` (and ``soft``) are the only families whose values depend on the row's objective;
    every other family is a session property broadcast across that session's responses. Both
    kinds are returned on the same response-keyed frame, so the caller does not need to know
    which is which.

    Args:
        feats: rows to score, with ``response_id`` and ``session_id`` columns, and
            ``learning_objective`` when ``lowin`` or ``soft`` is requested.
        store: transcript store with an ``iter(session_ids)`` method; defaults to
            :func:`magnificat.data.default_store`.
        tclf: the saved tutor classifier.
        sclf: the saved student classifier. If either is None an empty frame is returned.
        families: aggregation families to compute (see :func:`_families_for`).
        show_progress: show a tqdm progress bar when tqdm is installed.

    Returns:
        Float DataFrame indexed by ``response_id`` with the :func:`family_cols` columns.
    """
    from . import data as _data
    from .evidence_features import build_turns
    from .histpos_features import _lo_words

    families = list(families)
    if tclf is None or sclf is None:
        return pd.DataFrame(index=pd.Index(feats["response_id"], name="response_id"))
    cols = family_cols(families, tclf, sclf)
    need_lo = bool({"lowin", "soft"} & set(families))

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="v2fam")
        except ImportError:
            pass

    _tt = tclf.get("trunc") or DEFAULT_TRUNC
    _st = sclf.get("trunc") or DEFAULT_TRUNC
    lo_by_sess: dict = {}
    if need_lo:
        for s, l in zip(feats["session_id"], feats["learning_objective"]):
            lo_by_sess.setdefault(s, []).append(l)

    sess: dict = {}
    for sid, tdf in it:
        try:
            turns, _ = build_turns(tdf)
        except Exception:
            turns = []
        axis = _family_axis(turns)
        if not axis:
            continue
        T = _predict(tclf, [tutor_text(turns, i, _tt) for i in axis
                            if turns[i]["teacher"].strip()])
        tmask = np.array([bool(turns[i]["teacher"].strip()) for i in axis])
        Tal = np.zeros((len(axis), len(tclf["families"])), np.float32)
        Tal[tmask] = T
        S = _predict(sclf, [student_text(turns, i, _st) for i in axis])
        smask = np.ones(len(axis), bool)

        base = _families_for(Tal, S, tmask, smask, tclf, sclf, None,
                             [f for f in families if f not in ("lowin", "soft")])
        if not need_lo:
            sess[sid] = {None: base}
            continue
        words = [{w.lower() for w in _WORD_FAM.findall(f"{t['teacher']} {t['student']}")}
                 for t in turns]
        per_lo = {}
        for lo_text in set(lo_by_sess.get(sid, ())):
            _lo = A_lo(Tal, S, tclf, sclf, words, axis, lo_text)
            if "lowin" not in families:
                _lo = {k: v for k, v in _lo.items() if k.endswith("_soft_cov")}
            if "soft" not in families:
                _lo = {k: v for k, v in _lo.items() if not k.endswith("_soft_cov")}
            per_lo[lo_text] = dict(base, **_lo)
        sess[sid] = per_lo

    blank = {c: np.nan for c in cols}
    rows = []
    for s, l in zip(feats["session_id"], feats["learning_objective"]):
        d = sess.get(s)
        if d is None:
            rows.append(blank)
        elif need_lo:
            rows.append(d.get(l, blank))
        else:
            rows.append(d[None])
    out = pd.DataFrame(rows, index=pd.Index(feats["response_id"], name="response_id"))
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].astype(float)


import re as _re  # noqa: E402

_WORD_FAM = _re.compile(r"[A-Za-z']+")

#: Must equal `scripts/build_v2_families.AXES_T` / `AXES_S`. Duplicated rather than imported
#: because `scripts/` is not importable inside the runtime container; asserted equal by
#: `scripts/check_v2_family_parity.py`. The `None` entries are filled by `_init_axes`.
_AXES_T = {
    "ask": None, "tell": None, "fb_pos": ["FEEDBACK_CORRECT", "GIVING_PRAISE"],
    "fb_neg": ["FEEDBACK_INCORRECT"],
    "hint": ["GIVING_HINT", "SIMPLIFYING_TO_SUBPROBLEM", "NORMALIZING_DIFFICULTY"],
    "admin": None,
}
_AXES_S = {"reason": None, "struggle": None, "bare": ["ANSWER_ONLY"]}


# Fill the `None` entries of _AXES_T / _AXES_S from the code groups in magnificat.v2_agg.
def _init_axes():
    from . import v2_agg as A
    _AXES_T["ask"], _AXES_T["tell"], _AXES_T["admin"] = A.ASK, A.TELL, A.TUTOR_RESID
    _AXES_S["reason"], _AXES_S["struggle"] = A.STU_ENGAGE, A.STU_STRUGGLE


_init_axes()


#: Window pad, in paired turns, around each turn that names the objective. Must equal
#: `scripts/build_v2_families.LO_PAD_TURNS`, and asserted equal by
#: `scripts/check_v2_family_parity.py`.
LO_PAD_TURNS = 2


def A_lo(T, S, tclf, sclf, words, axis, lo_text) -> dict:
    """Compute the `lowin` and `soft` families for one objective on the family turn axis.

    ``words`` is indexed by raw turn index and ``axis`` maps the family axis back onto it, so
    the mask is built over every turn and then subset, as the training builder does
    (`build_v2_families._window` indexes `full[keep]`). Building it directly over the axis would
    move the window whenever a turn without student text sits inside it.

    Args:
        T: tutor probabilities on the family axis.
        S: student probabilities on the family axis.
        tclf: the saved tutor classifier.
        sclf: the saved student classifier.
        words: per raw turn, the set of lower-cased words in its tutor and student text.
        axis: raw turn index of each family-axis position (from :func:`_family_axis`).
        lo_text: the learning-objective text.

    Returns:
        Dict of ``v2w_*`` column name to float (the hit and window masks passed on are None
        when the objective's words never occur on the axis).
    """
    from . import v2_agg as A
    from .histpos_features import _lo_words

    lw = _lo_words(lo_text)
    h = w = None
    if lw:
        raw = np.array([bool(lw & ws) for ws in words])
        if raw.any():
            h = raw[np.asarray(axis)]
            if h.any():
                full = np.zeros(len(words), bool)
                for i in np.where(raw)[0]:
                    full[max(0, i - LO_PAD_TURNS):
                         min(len(words), i + LO_PAD_TURNS + 1)] = True
                w = full[np.asarray(axis)]
                w = w if w.any() else None
            else:
                h = None
    out = A.lo_window_features(T, tclf["families"], S, sclf["families"], w, "v2w")
    out.update(A.lo_soft_coverage(h, T.shape[0], "v2w"))
    return out


#: Which per-code prefixes each denominator writes, and which output prefix each width uses.
#: Mirrors `scripts/build_v2_narrow`; asserted equal by `scripts/check_v2_family_parity.py`.
_NARROW_SOURCE = {"rate": ("rate", "v2t", "v2s"), "instr": ("instr", "v2ti", "v2si"),
                  "trim": ("trim", "v2tt", "v2st")}
_NARROW_PREFIX = {"narrow": "v2n", "narrows": "v2m", "narrowt": "v2k", "narrowstu": "v2u"}


def build_v2_narrow_matrix(feats: pd.DataFrame, store=None, tclf: dict | None = None,
                           sclf: dict | None = None, which: str = "narrows",
                           source: str = "rate", extra_families=(),
                           show_progress: bool = False) -> pd.DataFrame:
    """Build a v2 block collapsed onto pedagogical axes, keyed by ``response_id``.

    The per-code rates are summed onto pedagogical axes: 17 columns (`narrows`) or 11
    (`narrowt`).

    The collapse is exact: an axis rate is the sum of its member rates, because all of them are
    means over the same turn set under the same denominator. Nothing is re-estimated, so the
    inference path and `scripts/build_v2_narrow.py` cannot disagree by construction.

    Args:
        feats: rows to score (see :func:`build_v2_family_matrix`).
        store: transcript store; defaults to :func:`magnificat.data.default_store`.
        tclf: the saved tutor classifier.
        sclf: the saved student classifier. If either is None, the (empty) wide frame is
            returned.
        which: output width: ``"narrow"`` (tutor axes + per-code student rates),
            ``"narrows"`` (tutor and student axes), ``"narrowt"`` (tutor axes only) or
            ``"narrowstu"`` (student axes only).
        source: per-code rates to collapse: ``"rate"``, ``"instr"`` or ``"trim"``.
        extra_families: further families computed alongside and appended (``v2b_``,
            ``v2sl_``, ``v2d_``, ``v2x_``, ``v2r_``, ``v2q_``, ``v2w_`` columns).
        show_progress: show a tqdm progress bar when tqdm is installed.

    Returns:
        DataFrame indexed by ``response_id`` with the axis columns, then any extra-family
        columns.
    """
    import sys as _sys
    src, tpre, spre = _NARROW_SOURCE[source]
    fams = [src] + [f for f in extra_families if f]
    wide = build_v2_family_matrix(feats, store=store, tclf=tclf, sclf=sclf,
                                  families=fams, show_progress=show_progress)
    if tclf is None or sclf is None:
        return wide

    # `scripts/` is not importable inside the runtime container, so the axis definitions are
    # vendored here rather than imported. `check_v2_family_parity.py` asserts the two agree.
    out_pre = _NARROW_PREFIX[which]
    keep = []
    for axis, codes in _TUTOR_AXES.items():
        cols = [f"{tpre}_{c.lower()}_rate" for c in codes]
        wide[f"{out_pre}t_{axis}_rate"] = wide[[c for c in cols if c in wide.columns]].sum(axis=1)
        keep.append(f"{out_pre}t_{axis}_rate")
    for r in ("elicit_to_tell", "feedback_positive_ratio"):
        wide[f"{out_pre}t_{r}"] = wide[f"{tpre}_{r}"]
        keep.append(f"{out_pre}t_{r}")
    if which in ("narrow",):
        for c in [c for c in wide.columns if c.startswith(f"{spre}_")]:
            n = c.replace(spre, f"{out_pre}s", 1)
            wide[n] = wide[c]
            keep.append(n)
    elif which in ("narrows", "narrowstu"):
        for axis, codes in _STUDENT_AXES.items():
            cols = [f"{spre}_{c.lower()}_rate" for c in codes]
            wide[f"{out_pre}s_{axis}_rate"] = wide[
                [c for c in cols if c in wide.columns]].sum(axis=1)
            keep.append(f"{out_pre}s_{axis}_rate")
    if which == "narrowt":
        keep = [c for c in keep if c.startswith(f"{out_pre}t_")]
    elif which == "narrowstu":
        keep = [c for c in keep if c.startswith(f"{out_pre}s_")]
    # Extra families are passed through; the pipeline selects columns by name, so anything the
    # booster was not trained on is ignored.
    keep += [c for c in wide.columns
             if c.startswith(("v2b_", "v2sl_", "v2d_", "v2x_", "v2r_", "v2q_", "v2w_"))]
    return wide[keep]


_TUTOR_AXES = {
    "elicit": ["PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP",
               "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
               "PROMPTING_SELF_CORRECTION", "RELEASE_HANDOFF"],
    "ask": ["ASKING_QUESTION", "TRANSFER_PROBE", "OFFERING_CHOICE"],
    "tell": ["EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
             "WORKED_DEMONSTRATION"],
    "scaffold": ["GIVING_HINT", "SIMPLIFYING_TO_SUBPROBLEM", "NORMALIZING_DIFFICULTY",
                 "GIVING_EXAMPLE"],
    "fbpos": ["FEEDBACK_CORRECT", "GIVING_PRAISE"],
    "fbneg": ["FEEDBACK_INCORRECT"],
    "uptake": ["REVOICING", "RESTATING", "SUMMARIZING_PROGRESS", "FEEDBACK_NEUTRAL"],
    "manage": ["GUIDING_SESSION", "EXPLAINING_TOOL", "MOTIVATING_RELEVANCE", "TUTOR_OTHER"],
    "admin": ["TUTOR_SOCIAL", "TUTOR_TECHNICAL", "TUTOR_UNINTELLIGIBLE", "TUTOR_NO_MOVE"],
}
_STUDENT_AXES = {
    "bare": ["ANSWER_ONLY"],
    "reason": ["REASONING_SHOWN", "SELF_CORRECTING"],
    "insight": ["INSIGHT_DELIGHT"],
    "struggle": ["CONFUSION_EXPRESSED", "FRUSTRATION"],
    "seek": ["SEEKING_CLARIFICATION"],
    "sresid": ["STUDENT_SOCIAL", "STUDENT_TECHNICAL", "STUDENT_UNINTELLIGIBLE",
               "STUDENT_NO_MOVE", "STUDENT_OTHER"],
}
